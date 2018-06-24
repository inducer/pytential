from __future__ import division, absolute_import

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

import pyopencl as cl
import pyopencl.array # noqa
from pyopencl.array import to_device

from pytools.obj_array import make_obj_array
from pytools import memoize_method, memoize

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


__doc__ = """
Proxy Point Generation
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProxyGenerator

.. autofunction:: partition_by_nodes

.. autofunction:: partition_by_elements

.. autofunction:: partition_from_coarse

.. autofunction:: build_neighbor_list

.. autofunction:: build_skeleton_list
"""


# {{{ point index partitioning

def _element_node_range(group, ielement):
    istart = group.node_nr_base + group.nunit_nodes * ielement
    iend = group.node_nr_base + group.nunit_nodes * (ielement + 1)

    return np.arange(istart, iend)


def partition_by_nodes(queue, discr,
                       use_tree=True,
                       max_nodes_in_box=None):
    """Generate clusters / ranges of nodes. The partition is created at the
    lowest level of granularity, i.e. nodes. This results in balanced ranges
    of points, but will split elements across different ranges.

    :arg queue: a :class:`pyopencl.CommandQueue`.
    :arg discr: a :class:`meshmode.discretization.Discretization`.
    :arg use_tree: if `True`, node partitions are generated using a
        :class:`boxtree.TreeBuilder`, which leads to geometrically close
        points to belong to the same partition. If `False`, a simple linear
        partition is constructed.
    :arg max_nodes_in_box: passed to :class:`boxtree.TreeBuilder`.

    :return: a tuple `(indices, ranges)` of :class:`pyopencl.array.Array`
        integer arrays. The indices in a range can be retrieved using
        `indices[ranges[i]:ranges[i + 1]]`.
    """

    if max_nodes_in_box is None:
        # FIXME: this is just an arbitrary value
        max_nodes_in_box = 32

    if use_tree:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        builder = TreeBuilder(discr.cl_context)

        tree, _ = builder(queue, discr.nodes(),
            max_particles_in_box=max_nodes_in_box)

        tree = tree.get(queue)
        leaf_boxes, = (tree.box_flags &
                       box_flags_enum.HAS_CHILDREN == 0).nonzero()

        indices = np.empty(len(leaf_boxes), dtype=np.object)
        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]
            indices[i] = tree.user_source_ids[box_start:box_end]

        ranges = to_device(queue,
            np.cumsum([0] + [box.shape[0] for box in indices]))
        indices = to_device(queue, np.hstack(indices))
    else:
        indices = cl.array.arange(queue, 0, discr.nnodes,
                                  dtype=np.int)
        ranges = cl.array.arange(queue, 0, discr.nnodes + 1,
                                 discr.nnodes // max_nodes_in_box,
                                 dtype=np.int)

    assert ranges[-1] == discr.nnodes
    return indices, ranges


def partition_by_elements(queue, discr,
                          use_tree=True,
                          max_elements_in_box=None):
    """Generate clusters / ranges of points. The partition is created at the
    element level, so that all the nodes belonging to an element belong to
    the same range. This can result in slightly larger differences in size
    between the ranges, but can be very useful when the individual partitions
    need to be resampled, integrated, etc.

    :arg queue: a :class:`pyopencl.CommandQueue`.
    :arg discr: a :class:`meshmode.discretization.Discretization`.
    :arg use_tree: if True, node partitions are generated using a
        :class:`boxtree.TreeBuilder`, which leads to geometrically close
        points to belong to the same partition. If False, a simple linear
        partition is constructed.
    :arg max_elements_in_box: passed to :class:`boxtree.TreeBuilder`.

    :return: a tuple `(indices, ranges)` of :class:`pyopencl.array.Array`
        integer arrays. The indices in a range can be retrieved using
        `indices[ranges[i]:ranges[i + 1]]`.
    """

    if max_elements_in_box is None:
        # NOTE: keep in sync with partition_by_nodes
        max_nodes_in_box = 32

        nunit_nodes = int(np.mean([g.nunit_nodes for g in discr.groups]))
        max_elements_in_box = max_nodes_in_box // nunit_nodes

    if use_tree:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        builder = TreeBuilder(discr.cl_context)

        from pytential.qbx.utils import element_centers_of_mass
        elranges = np.cumsum([group.nelements for group in discr.mesh.groups])
        elcenters = element_centers_of_mass(discr)

        tree, _ = builder(queue, elcenters,
            max_particles_in_box=max_elements_in_box)

        groups = discr.groups
        tree = tree.get(queue)
        leaf_boxes, = (tree.box_flags &
                       box_flags_enum.HAS_CHILDREN == 0).nonzero()

        indices = np.empty(len(leaf_boxes), dtype=np.object)
        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]

            ielement = tree.user_source_ids[box_start:box_end]
            igroup = np.digitize(ielement, elranges)

            indices[i] = np.hstack([_element_node_range(groups[j], k)
                                    for j, k in zip(igroup, ielement)])
    else:
        nelements = discr.mesh.nelements
        elements = np.array_split(np.arange(0, nelements),
                                  nelements // max_elements_in_box)

        elranges = np.cumsum([g.nelements for g in discr.groups])
        elgroups = [np.digitize(elements[i], elranges)
                    for i in range(len(elements))]

        indices = np.empty(len(elements), dtype=np.object)
        for i in range(indices.shape[0]):
            indices[i] = np.hstack([_element_node_range(discr.groups[j], k)
                                    for j, k in zip(elgroups[i], elements[i])])

    ranges = to_device(queue,
            np.cumsum([0] + [box.shape[0] for box in indices]))
    indices = to_device(queue, np.hstack(indices))

    assert ranges[-1] == discr.nnodes
    return indices, ranges


def partition_from_coarse(queue, resampler, from_indices, from_ranges):
    """Generate a partition of nodes from an existing partition on a
    coarser discretization. The new partition is generated based on element
    refinement relationships in :attr:`resampler`, so the existing partition
    needs to be created using :func:`partition_by_element`, since we assume
    that each range contains all the nodes in an element.

    The new partition will have the same number of ranges as the old partition.
    The nodes inside each range in the new partition are all the nodes in
    :attr:`resampler.to_discr` that belong to the same region as the nodes
    in the same range from :attr:`resampler.from_discr`.

    :arg queue: a :class:`pyopencl.CommandQueue`.
    :arg resampler: a
        :class:`meshmode.discretization.connection.DirectDiscretizationConnection`.
    :arg from_indices: a set of indices into the nodes in
        :attr:`resampler.from_discr`.
    :arg from_ranges: array used to index into :attr:`from_indices`.

    :return: a tuple `(indices, ranges)` of :class:`pyopencl.array.Array`
        integer arrays. The indices in a range can be retrieved using
        `indices[ranges[i]:ranges[i + 1]]`.
    """

    if not hasattr(resampler, "groups"):
        raise ValueError("resampler must be a DirectDiscretizationConnection.")

    if isinstance(from_ranges, cl.array.Array):
        from_indices = from_indices.get(queue)
        from_ranges = from_ranges.get(queue)

    # construct ranges
    from_discr = resampler.from_discr
    from_grp_ranges = np.cumsum([0] +
            [grp.nelements for grp in from_discr.mesh.groups])
    from_el_ranges = np.hstack([
        np.arange(grp.node_nr_base, grp.nnodes + 1, grp.nunit_nodes)
        for grp in from_discr.groups])

    # construct coarse element arrays in each from_range
    el_indices = np.empty(from_ranges.shape[0] - 1, dtype=np.object)
    el_ranges = np.full(from_grp_ranges[-1], -1, dtype=np.int)
    for i in range(from_ranges.shape[0] - 1):
        irange = np.s_[from_ranges[i]:from_ranges[i + 1]]
        el_indices[i] = \
            np.unique(np.digitize(from_indices[irange], from_el_ranges)) - 1
        el_ranges[el_indices[i]] = i
    el_indices = np.hstack(el_indices)

    # construct lookup table
    to_el_table = [np.full(g.nelements, -1, dtype=np.int)
                   for g in resampler.to_discr.groups]

    for igrp, grp in enumerate(resampler.groups):
        for batch in grp.batches:
            to_el_table[igrp][batch.to_element_indices.get(queue)] = \
                from_grp_ranges[igrp] + batch.from_element_indices.get(queue)

    # construct fine node index list
    indices = [np.empty(0, dtype=np.int)
               for _ in range(from_ranges.shape[0] - 1)]
    for igrp in range(len(resampler.groups)):
        to_element_indices = \
                np.where(np.isin(to_el_table[igrp], el_indices))[0]

        for i, j in zip(el_ranges[to_el_table[igrp][to_element_indices]],
                        to_element_indices):
            indices[i] = np.hstack([indices[i],
                _element_node_range(resampler.to_discr.groups[igrp], j)])

    ranges = to_device(queue,
            np.cumsum([0] + [box.shape[0] for box in indices]))
    indices = to_device(queue, np.hstack(indices))

    return indices, ranges

# }}}


# {{{ proxy point generator

class ProxyGenerator(object):
    r"""
    :arg discr: a :class:`pytential.qbx.QBXLayerPotentialSource`.
    :arg nproxy: number of proxy points.
    :arg ratio: a ratio used to compute the proxy point radius. The radius
        is computed in the :math:`L_2` norm, resulting in a circle or
        sphere of proxy points. For QBX, we have two radii of interest
        for a set of points: the radius :math:`r_{block}` of the
        smallest ball containing all the points and the radius
        :math:`r_{qbx}` of the smallest ball containing all the QBX
        expansion balls in the block. If the ratio :math:`\theta \in
        [0, 1]`, then the radius of the proxy ball is

        .. math::

            r = (1 - \theta) r_{block} + \theta r_{qbx}.

        If the ratio :math:`\theta > 1`, the the radius is simply

        .. math::

            r = \theta r_{qbx}.

    .. automethod:: __call__
    """

    def __init__(self, source, nproxy=30, ratio=1.1, **kwargs):
        self.source = source
        self.ratio = abs(ratio)
        self.nproxy = int(abs(nproxy))
        self.ambient_dim = source.density_discr.ambient_dim

        if self.ambient_dim == 2:
            from meshmode.mesh.generation import ellipse, make_curve_mesh

            self.ref_mesh = make_curve_mesh(lambda t: ellipse(1.0, t),
                    np.linspace(0.0, 1.0, self.nproxy + 1),
                    self.nproxy)
        elif self.ambient_dim == 3:
            from meshmode.mesh.generation import generate_icosphere

            self.ref_mesh = generate_icosphere(1, self.nproxy)
        else:
            raise ValueError("unsupported ambient dimension")

    @memoize_method
    def get_kernel(self):
        if self.ratio < 1.0:
            radius_expr = "(1.0 - {ratio}) * rblk + {ratio} * rqbx"
        else:
            radius_expr = "{ratio} * rqbx"
        radius_expr = radius_expr.format(ratio=self.ratio)

        # NOTE: centers of mass are computed using a second-order approximation
        # that currently matches what is in `element_centers_of_mass`.
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
                    reduce(sum, i, nodes[idim, srcindices[i + ioffset]]) \
                        {{dup=idim:i}}

                <> rblk = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         nodes[idim, srcindices[i + ioffset]]) ** 2)))

                <> rqbx_int = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_int[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]])
                <> rqbx_ext = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_ext[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]])
                <> rqbx = if(rqbx_ext < rqbx_int, rqbx_int, rqbx_ext)

                proxy_radius[irange] = {radius_expr}
            end
            """.format(radius_expr=radius_expr)],
            [
                lp.GlobalArg("nodes", None,
                    shape=(self.ambient_dim, "nnodes")),
                lp.GlobalArg("center_int", None,
                    shape=(self.ambient_dim, "nnodes"), dim_tags="sep,C"),
                lp.GlobalArg("center_ext", None,
                    shape=(self.ambient_dim, "nnodes"), dim_tags="sep,C"),
                lp.GlobalArg("expansion_radii", None,
                    shape="nnodes"),
                lp.GlobalArg("srcranges", None,
                    shape="nranges + 1"),
                lp.GlobalArg("srcindices", None,
                    shape="nindices"),
                lp.GlobalArg("proxy_center", None,
                    shape=(self.ambient_dim, "nranges")),
                lp.GlobalArg("proxy_radius", None,
                    shape="nranges"),
                lp.ValueArg("nnodes", np.int64),
                lp.ValueArg("nranges", None),
                lp.ValueArg("nindices", np.int64)
            ],
            name="proxy_generator_knl",
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

    def __call__(self, queue, srcindices, srcranges, **kwargs):
        """Generate proxy points for each given range of source points in
        the discretization in :attr:`source`.

        :arg queue: a :class:`pyopencl.CommandQueue`.
        :arg srcindices: a :class:`pyopencl.array.Array` of indices around
            which to construct proxy balls.
        :arg srcranges: an :class:`pyopencl.array.Array` of size `(nranges + 1,)`
            used to index into :attr:`srcindices`. Each one of the `nranges`
            ranges will get a proxy ball.

        :return: a tuple of `(proxies, pxyranges, pxycenters, pxyranges)`, where
            each element is a :class:`pyopencl.array.Array`. The
            sizes of the arrays are as follows: `pxycenters` is of size
            `(2, nranges)`, `pxyradii` is of size `(nranges,)`, `pxyranges` is
            of size `(nranges + 1,)` and `proxies` is of size
            `(2, nranges * nproxy)`. The proxy points in a range :math:`i`
            can be obtained by a slice `proxies[pxyranges[i]:pxyranges[i + 1]]`
            and are all at a distance `pxyradii[i]` from the range center
            `pxycenters[i]`.
        """

        from pytential.qbx.utils import get_centers_on_side

        knl = self.get_kernel()
        _, (centers_dev, radii_dev,) = knl(queue,
            nodes=self.source.density_discr.nodes(),
            center_int=get_centers_on_side(self.source, -1),
            center_ext=get_centers_on_side(self.source, +1),
            expansion_radii=self.source._expansion_radii("nsources"),
            srcindices=srcindices, srcranges=srcranges, **kwargs)
        centers = centers_dev.get()
        radii = radii_dev.get()

        from meshmode.mesh.processing import affine_map
        proxies = np.empty(srcranges.shape[0] - 1, dtype=np.object)

        for i in range(srcranges.shape[0] - 1):
            mesh = affine_map(self.ref_mesh,
                A=(radii[i] * np.eye(self.ambient_dim)),
                b=centers[:, i].reshape(-1))
            proxies[i] = mesh.vertices

        pxyranges = cl.array.arange(queue, 0,
                proxies.shape[0] * proxies[0].shape[-1] + 1, proxies[0].shape[-1],
                dtype=srcranges.dtype)
        proxies = make_obj_array([
            cl.array.to_device(queue, np.hstack([p[idim] for p in proxies]))
            for idim in range(self.ambient_dim)])
        centers = make_obj_array([
            centers_dev[idim].with_queue(queue).copy()
            for idim in range(self.ambient_dim)])

        assert pxyranges[-1] == proxies[0].shape[0]
        return proxies, pxyranges, centers, radii_dev


def build_neighbor_list(discr, srcindices, srcranges, pxycenters, pxyradii,
                        max_nodes_in_box=None, **kwargs):
    """Generate a set of neighboring points for each range of points in
    :attr:`discr`. Neighboring points of a range :math:`i` are defined
    as all the points inside the proxy ball :math:`i` that do not also
    belong to the range itself.

    :arg discr: a :class:`meshmode.discretization.Discretization`.
    :arg srcindices: an array of indices for a subset of the nodes in
        :attr:`discr`.
    :arg srcranges: an array used to index into the :attr:`srcindices` array.
    :arg pxycenters: an array containing the center of each proxy ball.
    :arg pxyradii: an array containing the radius of each proxy ball.

    :return: a tuple `(nbrindices, nbrranges)`, where each value is a
        :class:`pyopencl.array.Array`. For a range :math:`i`, we can
        get the slice using `nbrindices[nbrranges[i]:nbrranges[i + 1]]`.
    """

    if max_nodes_in_box is None:
        # FIXME: this is a fairly arbitrary value
        max_nodes_in_box = 32

    with cl.CommandQueue(discr.cl_context) as queue:
        if isinstance(srcindices, cl.array.Array):
            srcindices = srcindices.get(queue)
        if isinstance(srcranges, cl.array.Array):
            srcranges = srcranges.get(queue)

        # NOTE: this is used for multiple reasons:
        #   * TreeBuilder takes object arrays
        #   * `srcndices` can be a small subset of nodes, so this will save
        #   some work
        #   * `srcindices` may reorder the array returned by nodes(), so this
        #   makes sure that we have the same order in tree.user_source_ids
        #   and friends
        sources = discr.nodes().get(queue)
        sources = make_obj_array([
            cl.array.to_device(queue, sources[idim, srcindices])
            for idim in range(discr.ambient_dim)])

        # construct tree
        from boxtree import TreeBuilder
        builder = TreeBuilder(discr.cl_context)
        tree, _ = builder(queue, sources,
                          max_particles_in_box=max_nodes_in_box)

        from boxtree.area_query import AreaQueryBuilder
        builder = AreaQueryBuilder(discr.cl_context)
        query, _ = builder(queue, tree, pxycenters, pxyradii)

        # find nodes inside each proxy ball
        tree = tree.get(queue)
        query = query.get(queue)

        if isinstance(pxycenters[0], cl.array.Array):
            pxycenters = np.vstack([pxycenters[idim].get(queue)
                                    for idim in range(discr.ambient_dim)])
        if isinstance(pxyradii, cl.array.Array):
            pxyradii = pxyradii.get(queue)

        nbrindices = np.empty(srcranges.shape[0] - 1, dtype=np.object)
        for iproxy in range(srcranges.shape[0] - 1):
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
            mask = (la.norm(nodes - center, axis=0) < radius) & \
                   ((isources < srcranges[iproxy]) |
                    (srcranges[iproxy + 1] <= isources))

            nbrindices[iproxy] = srcindices[isources[mask]]

        nbrranges = to_device(queue,
                np.cumsum([0] + [n.shape[0] for n in nbrindices]))
        nbrindices = to_device(queue, np.hstack(nbrindices))

    return nbrindices, nbrranges


def build_skeleton_list(source, srcindices, srcranges, **kwargs):
    """Generate sets of skeleton points for each given range of indices
    in the :attr:`source` discretization. Skeleton points are meant to
    model the interactions of a set of points. They are composed of two
    parts:

    - a set of proxy points (or balls) around a given range, which
      models farfield interactions.

    - a set of neighboring points that are inside the proxy balls, but
      do not belong to the given range, which model nearby interactions.

    :arg source: a :class:`pytential.qbx.QBXLayerPotentialSource`.
    :arg srcindices: a :class:`pyopencl.array.Array` of indices for points
        in :attr:`source`.
    :arg srcranges: a :class:`pyopencl.array.Array` describing the ranges
        from :attr:`srcindices` around which to build proxy points. For each
        range, this builds a ball of proxy points centered
        at the center of mass of the points in the range with a radius
        defined by :attr:`ratio`.
    :arg kwargs: additional arguments passed to :class:`ProxyGenerator`
        or :func:`build_neighbor_list`.

    :returns: a tuple `(skeletons, sklranges)`, where each value is a
        :class:`pyopencl.array.Array`. For a range :math:`i`, we can
        get the slice using `skeletons[sklranges[i]:sklranges[i + 1]]`.
    """

    @memoize
    def knl():
        loopy_knl = lp.make_kernel([
            "{[irange, idim]: 0 <= irange < nranges and \
                              0 <= idim < dim}",
            "{[ipxy, ingb]: 0 <= ipxy < npxyblock and \
                            0 <= ingb < nngbblock}"
            ],
            """
            for irange
                <> pxystart = pxyranges[irange]
                <> pxyend = pxyranges[irange + 1]
                <> npxyblock = pxyend - pxystart

                <> ngbstart = nbrranges[irange]
                <> ngbend = nbrranges[irange + 1]
                <> nngbblock = ngbend - ngbstart

                <> sklstart = pxyranges[irange] + nbrranges[irange]
                skeletons[idim, sklstart + ipxy] = \
                    proxies[idim, pxystart + ipxy] \
                    {id_prefix=write_pxy,nosync=write_ngb}
                skeletons[idim, sklstart + npxyblock + ingb] = \
                    sources[idim, nbrindices[ngbstart + ingb]] \
                    {id_prefix=write_ngb,nosync=write_pxy}
                sklranges[irange + 1] = sklranges[irange] + \
                        npxyblock + nngbblock
            end
            """,
            [
                lp.GlobalArg("sources", None,
                    shape=(source.ambient_dim, "nsources")),
                lp.GlobalArg("proxies", None,
                    shape=(source.ambient_dim, "nproxies"), dim_tags="sep,C"),
                lp.GlobalArg("nbrindices", None,
                    shape="nnbrindices"),
                lp.GlobalArg("pxyranges", None,
                    shape="nranges + 1"),
                lp.GlobalArg("nbrranges", None,
                    shape="nranges + 1"),
                lp.GlobalArg("skeletons", None,
                    shape=(source.ambient_dim, "nproxies + nnbrindices")),
                lp.GlobalArg("sklranges", None,
                    shape="nranges + 1"),
                lp.ValueArg("nsources", np.int32),
                lp.ValueArg("nproxies", np.int32),
                lp.ValueArg("nnbrindices", np.int32),
                "..."
            ],
            name="concat_skl",
            default_offset=lp.auto,
            silenced_warnings="write_race(write_*)",
            fixed_parameters=dict(dim=source.ambient_dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.split_iname(loopy_knl, "irange", 128, outer_tag="g.0")

        return loopy_knl

    with cl.CommandQueue(source.cl_context) as queue:
        proxy = ProxyGenerator(source, **kwargs)
        proxies, pxyranges, pxycenters, pxyradii = \
                proxy(queue, srcindices, srcranges)

        nbrindices, nbrranges = build_neighbor_list(source.density_discr,
                srcindices, srcranges, pxycenters, pxyradii, **kwargs)

        sklranges = cl.array.zeros(queue, srcranges.shape, dtype=np.int)
        _, (skeletons, sklranges) = knl()(queue,
                sources=source.density_discr.nodes(),
                proxies=proxies,
                pxyranges=pxyranges,
                nbrindices=nbrindices,
                nbrranges=nbrranges,
                sklranges=sklranges)

    return skeletons, sklranges

# }}}

# vim: foldmethod=marker
