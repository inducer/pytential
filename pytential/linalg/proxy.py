__copyright__ = "Copyright (C) 2018 Alexandru Fikl"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import numpy.linalg as la

from pytools.obj_array import make_obj_array
from pytools import memoize_method
from sumpy.tools import BlockIndexRanges

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


__doc__ = """
Proxy Point Generation
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProxyGenerator

.. autofunction:: partition_by_nodes
.. autofunction:: gather_block_neighbor_points
"""


# {{{ point index partitioning

def partition_by_nodes(actx, discr,
        tree_kind="adaptive-level-restricted", max_nodes_in_box=None):
    """Generate equally sized ranges of nodes. The partition is created at the
    lowest level of granularity, i.e. nodes. This results in balanced ranges
    of points, but will split elements across different ranges.

    :arg discr: a :class:`meshmode.discretization.Discretization`.
    :arg tree_kind: if not *None*, it is passed to :class:`boxtree.TreeBuilder`.
    :arg max_nodes_in_box: passed to :class:`boxtree.TreeBuilder`.

    :return: a :class:`sumpy.tools.BlockIndexRanges`.
    """

    if max_nodes_in_box is None:
        # FIXME: this is just an arbitrary value
        max_nodes_in_box = 32

    if tree_kind is not None:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        builder = TreeBuilder(actx.context)

        from arraycontext import thaw
        from meshmode.dof_array import flatten
        tree, _ = builder(actx.queue,
                flatten(thaw(discr.nodes(), actx)),
                max_particles_in_box=max_nodes_in_box,
                kind=tree_kind)

        tree = tree.get(actx.queue)
        leaf_boxes, = (tree.box_flags
                       & box_flags_enum.HAS_CHILDREN == 0).nonzero()

        indices = np.empty(len(leaf_boxes), dtype=object)
        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]
            indices[i] = tree.user_source_ids[box_start:box_end]

        ranges = actx.from_numpy(
                np.cumsum([0] + [box.shape[0] for box in indices])
                )
        indices = actx.from_numpy(np.hstack(indices))
    else:
        indices = actx.from_numpy(np.arange(0, discr.ndofs, dtype=np.int64))
        ranges = actx.from_numpy(np.arange(
            0,
            discr.ndofs + 1,
            max_nodes_in_box, dtype=np.int64))

    assert ranges[-1] == discr.ndofs

    return BlockIndexRanges(actx.context,
        actx.freeze(indices), actx.freeze(ranges))

# }}}


# {{{ proxy point generator

def _generate_unit_sphere(ambient_dim, approx_npoints):
    """Generate uniform points on a unit sphere.

    :arg ambient_dim: dimension of the ambient space.
    :arg approx_npoints: approximate number of points to generate. If the
        ambient space is 3D, this will not generate the exact number of points.
    :return: array of shape ``(ambient_dim, npoints)``, where ``npoints``
        will not generally be the same as ``approx_npoints``.
    """

    if ambient_dim == 2:
        t = np.linspace(0.0, 2.0 * np.pi, approx_npoints)
        points = np.vstack([np.cos(t), np.sin(t)])
    elif ambient_dim == 3:
        # https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
        # code by Matt Wala from
        # https://github.com/mattwala/gigaqbx-accuracy-experiments/blob/d56ed063ffd7843186f4fe05d2a5b5bfe6ef420c/translation_accuracy.py#L23
        a = 4.0 * np.pi / approx_npoints
        m_theta = int(np.round(np.pi / np.sqrt(a)))
        d_theta = np.pi / m_theta
        d_phi = a / d_theta

        points = []
        for m in range(m_theta):
            theta = np.pi * (m + 0.5) / m_theta
            m_phi = int(np.round(2.0 * np.pi * np.sin(theta) / d_phi))

            for n in range(m_phi):
                phi = 2.0 * np.pi * n / m_phi
                points.append(np.array([np.sin(theta) * np.cos(phi),
                                        np.sin(theta) * np.sin(phi),
                                        np.cos(theta)]))

        for i in range(ambient_dim):
            for sign in [-1, 1]:
                pole = np.zeros(ambient_dim)
                pole[i] = sign
                points.append(pole)

        points = np.array(points).T
    else:
        raise ValueError("ambient_dim > 3 not supported.")

    return points


class ProxyGenerator:
    r"""
    .. attribute:: places

        A :class:`~pytential.GeometryCollection`
        containing the geometry on which the proxy balls are generated.

    .. attribute:: nproxy

        Number of proxy points in a single proxy ball.

    .. attribute:: radius_factor

        A factor used to compute the proxy ball radius. The radius
        is computed in the :math:`\ell^2` norm, resulting in a circle or
        sphere of proxy points. For QBX, we have two radii of interest
        for a set of points: the radius :math:`r_{block}` of the
        smallest ball containing all the points and the radius
        :math:`r_{qbx}` of the smallest ball containing all the QBX
        expansion balls in the block. If the factor :math:`\theta \in
        [0, 1]`, then the radius of the proxy ball is

        .. math::

            r = (1 - \theta) r_{block} + \theta r_{qbx}.

        If the factor :math:`\theta > 1`, the the radius is simply

        .. math::

            r = \theta r_{qbx}.

    .. attribute:: ref_points

        Reference points on a unit ball. Can be used to construct the points
        of a proxy ball :math:`i` by translating them to ``center[i]`` and
        scaling by ``radii[i]``, as obtained by :meth:`__call__`.

    .. automethod:: __call__
    """

    def __init__(self, places, approx_nproxy=None, radius_factor=None):
        from pytential import GeometryCollection
        if not isinstance(places, GeometryCollection):
            places = GeometryCollection(places)

        self.places = places
        self.ambient_dim = places.ambient_dim
        self.radius_factor = 1.1 if radius_factor is None else radius_factor

        approx_nproxy = 32 if approx_nproxy is None else approx_nproxy
        self.ref_points = \
                _generate_unit_sphere(self.ambient_dim, approx_nproxy)

    @property
    def nproxy(self):
        return self.ref_points.shape[1]

    @memoize_method
    def get_kernel(self):
        if self.radius_factor < 1.0:
            radius_expr = "(1.0 - {f}) * rblk + {f} * rqbx"
        else:
            radius_expr = "{f} * rqbx"
        radius_expr = radius_expr.format(f=self.radius_factor)

        # NOTE: centers of mass are computed using a second-order approximation
        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < dim}"
            ],
            ["""
            for irange
                <> ioffset = srcranges[irange]
                <> npoints = srcranges[irange + 1] - srcranges[irange]

                proxy_center[idim, irange] = 1.0 / npoints * \
                    reduce(sum, i, sources[idim, srcindices[i + ioffset]]) \
                        {{dup=idim:i}}

                <> rblk = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         sources[idim, srcindices[i + ioffset]]) ** 2)))

                <> rqbx_int = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_int[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]])
                <> rqbx_ext = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_ext[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]])
                <> rqbx = rqbx_int if rqbx_ext < rqbx_int else rqbx_ext

                proxy_radius[irange] = {radius_expr}
            end
            """.format(radius_expr=radius_expr)],
            [
                lp.GlobalArg("sources", None,
                    shape=(self.ambient_dim, "nsources"), dim_tags="sep,C"),
                lp.GlobalArg("center_int", None,
                    shape=(self.ambient_dim, "nsources"), dim_tags="sep,C"),
                lp.GlobalArg("center_ext", None,
                    shape=(self.ambient_dim, "nsources"), dim_tags="sep,C"),
                lp.GlobalArg("proxy_center", None,
                    shape=(self.ambient_dim, "nranges")),
                lp.GlobalArg("proxy_radius", None,
                    shape="nranges"),
                lp.ValueArg("nsources", np.int64),
                "..."
            ],
            name="find_proxy_radii_knl",
            assumptions="dim>=1 and nranges>=1",
            fixed_parameters=dict(dim=self.ambient_dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.tag_inames(knl, "idim*:unr")
        return knl

    @memoize_method
    def get_optimized_kernel(self):
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "irange", 128, outer_tag="g.0")

        return knl

    def __call__(self, actx, source_dd, indices, **kwargs):
        """Generate proxy points for each given range of source points in
        the discretization in *source_dd*.

        :arg actx: a :class:`~arraycontext.PyOpenCLArrayContext`.
        :arg source_dd: a :class:`~pytential.symbolic.primitives.DOFDescriptor`
            for the discretization on which the proxy points are to be
            generated.
        :arg indices: a :class:`sumpy.tools.BlockIndexRanges`.

        :return: a tuple of ``(proxies, pxyranges, pxycenters, pxyranges)``,
            where each element is a :class:`pyopencl.array.Array`. The
            sizes of the arrays are as follows: ``pxycenters`` is of size
            ``(2, nranges)``, ``pxyradii`` is of size ``(nranges,)``,
            ``pxyranges`` is of size ``(nranges + 1,)`` and ``proxies`` is
            of size ``(2, nranges * nproxy)``. The proxy points in a range
            :math:`i` can be obtained by a slice
            ``proxies[pxyranges[i]:pxyranges[i + 1]]`` and are all at a
            distance ``pxyradii[i]`` from the range center ``pxycenters[i]``.
        """

        def _affine_map(v, A, b):
            return np.dot(A, v) + b

        from pytential import bind, sym
        source_dd = sym.as_dofdesc(source_dd)
        discr = self.places.get_discretization(
                source_dd.geometry, source_dd.discr_stage)

        radii = bind(self.places, sym.expansion_radii(
            self.ambient_dim, dofdesc=source_dd))(actx)
        center_int = bind(self.places, sym.expansion_centers(
            self.ambient_dim, -1, dofdesc=source_dd))(actx)
        center_ext = bind(self.places, sym.expansion_centers(
            self.ambient_dim, +1, dofdesc=source_dd))(actx)

        from arraycontext import thaw
        from meshmode.dof_array import flatten
        knl = self.get_kernel()
        _, (centers_dev, radii_dev,) = knl(actx.queue,
            sources=flatten(thaw(discr.nodes(), actx)),
            center_int=flatten(center_int),
            center_ext=flatten(center_ext),
            expansion_radii=flatten(radii),
            srcindices=indices.indices,
            srcranges=indices.ranges, **kwargs)

        from meshmode.dof_array import flatten_to_numpy
        centers = flatten_to_numpy(actx, centers_dev, strict=False)
        radii = flatten_to_numpy(actx, radii_dev, strict=False)
        proxies = np.empty(indices.nblocks, dtype=object)
        for i in range(indices.nblocks):
            proxies[i] = _affine_map(self.ref_points,
                    A=(radii[i] * np.eye(self.ambient_dim)),
                    b=centers[:, i].reshape(-1, 1))

        pxyranges = actx.from_numpy(np.arange(
            0,
            proxies.shape[0] * proxies[0].shape[1] + 1,
            proxies[0].shape[1],
            dtype=indices.ranges.dtype))
        proxies = make_obj_array([
            actx.freeze(actx.from_numpy(np.hstack([p[idim] for p in proxies])))
            for idim in range(self.ambient_dim)
            ])
        centers = make_obj_array([
            actx.freeze(centers_dev[idim])
            for idim in range(self.ambient_dim)
            ])

        assert pxyranges[-1] == proxies[0].shape[0]
        return proxies, actx.freeze(pxyranges), centers, actx.freeze(radii_dev)


def gather_block_neighbor_points(actx, discr, indices, pxycenters, pxyradii,
        max_nodes_in_box=None):
    """Generate a set of neighboring points for each range of points in
    *discr*. Neighboring points of a range :math:`i` are defined
    as all the points inside the proxy ball :math:`i` that do not also
    belong to the range itself.

    :arg discr: a :class:`meshmode.discretization.Discretization`.
    :arg indices: a :class:`sumpy.tools.BlockIndexRanges`.
    :arg pxycenters: an array containing the center of each proxy ball.
    :arg pxyradii: an array containing the radius of each proxy ball.

    :return: a :class:`sumpy.tools.BlockIndexRanges`.
    """

    if max_nodes_in_box is None:
        # FIXME: this is a fairly arbitrary value
        max_nodes_in_box = 32

    indices = indices.get(actx.queue)

    # NOTE: this is constructed for multiple reasons:
    #   * TreeBuilder takes object arrays
    #   * `srcindices` can be a small subset of nodes, so this will save
    #   some work
    #   * `srcindices` may reorder the array returned by nodes(), so this
    #   makes sure that we have the same order in tree.user_source_ids
    #   and friends
    from meshmode.dof_array import flatten_to_numpy
    sources = flatten_to_numpy(actx, discr.nodes())
    sources = make_obj_array([
        actx.from_numpy(sources[idim][indices.indices])
        for idim in range(discr.ambient_dim)])

    # construct tree
    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)
    tree, _ = builder(actx.queue, sources,
            max_particles_in_box=max_nodes_in_box)

    from boxtree.area_query import AreaQueryBuilder
    builder = AreaQueryBuilder(actx.context)
    query, _ = builder(actx.queue, tree, pxycenters, pxyradii)

    # find nodes inside each proxy ball
    tree = tree.get(actx.queue)
    query = query.get(actx.queue)

    pxycenters = np.vstack([
        actx.to_numpy(pxycenters[idim])
        for idim in range(discr.ambient_dim)
        ])
    pxyradii = actx.to_numpy(pxyradii)

    nbrindices = np.empty(indices.nblocks, dtype=object)
    for iproxy in range(indices.nblocks):
        # get list of boxes intersecting the current ball
        istart = query.leaves_near_ball_starts[iproxy]
        iend = query.leaves_near_ball_starts[iproxy + 1]
        iboxes = query.leaves_near_ball_lists[istart:iend]

        # get nodes inside the boxes
        istart = tree.box_source_starts[iboxes]
        iend = istart + tree.box_source_counts_cumul[iboxes]
        isources = np.hstack([np.arange(s, e)
                              for s, e in zip(istart, iend)])
        nodes = np.vstack([tree.sources[idim][isources]
                           for idim in range(discr.ambient_dim)])
        isources = tree.user_source_ids[isources]

        # get nodes inside the ball but outside the current range
        center = pxycenters[:, iproxy].reshape(-1, 1)
        radius = pxyradii[iproxy]
        mask = ((la.norm(nodes - center, axis=0) < radius)
                & ((isources < indices.ranges[iproxy])
                    | (indices.ranges[iproxy + 1] <= isources)))

        nbrindices[iproxy] = indices.indices[isources[mask]]

    nbrranges = actx.from_numpy(np.cumsum([0] + [n.shape[0] for n in nbrindices]))
    nbrindices = actx.from_numpy(np.hstack(nbrindices))

    return BlockIndexRanges(actx.context,
            actx.freeze(nbrindices), actx.freeze(nbrranges))

# }}}

# vim: foldmethod=marker
