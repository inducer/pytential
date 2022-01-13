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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.linalg as la

from arraycontext import PyOpenCLArrayContext, flatten
from meshmode.discretization import Discretization
from meshmode.dof_array import DOFArray

from pytools import memoize_in
from pytential.linalg.utils import BlockIndexRanges

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


__doc__ = """
Proxy Point Generation
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pytential.linalg

.. autoclass:: BlockProxyPoints
.. autoclass:: ProxyGeneratorBase
.. autoclass:: ProxyGenerator
.. autoclass:: QBXProxyGenerator

.. autofunction:: partition_by_nodes
.. autofunction:: gather_block_neighbor_points
"""


# {{{ point index partitioning

def partition_by_nodes(
        actx: PyOpenCLArrayContext, discr: Discretization, *,
        tree_kind: Optional[str] = "adaptive-level-restricted",
        max_particles_in_box: Optional[int] = None) -> BlockIndexRanges:
    """Generate equally sized ranges of nodes. The partition is created at the
    lowest level of granularity, i.e. nodes. This results in balanced ranges
    of points, but will split elements across different ranges.

    :arg tree_kind: if not *None*, it is passed to :class:`boxtree.TreeBuilder`.
    :arg max_particles_in_box: passed to :class:`boxtree.TreeBuilder`.
    """

    if max_particles_in_box is None:
        # FIXME: this is just an arbitrary value
        max_particles_in_box = 32

    if tree_kind is not None:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        builder = TreeBuilder(actx.context)

        tree, _ = builder(actx.queue,
                particles=flatten(discr.nodes(), actx, leaf_class=DOFArray),
                max_particles_in_box=max_particles_in_box,
                kind=tree_kind)

        tree = tree.get(actx.queue)
        leaf_boxes, = (tree.box_flags & box_flags_enum.HAS_CHILDREN == 0).nonzero()

        indices = np.empty(len(leaf_boxes), dtype=object)
        ranges = None

        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]
            indices[i] = tree.user_source_ids[box_start:box_end]
    else:
        if discr.ambient_dim != 2 and discr.dim == 1:
            raise ValueError("only curves are supported for 'tree_kind=None'")

        nblocks = max(discr.ndofs // max_particles_in_box, 2)
        indices = np.arange(0, discr.ndofs, dtype=np.int64)
        ranges = np.linspace(0, discr.ndofs, nblocks + 1, dtype=np.int64)
        assert ranges[-1] == discr.ndofs

    from pytential.linalg import make_block_index_from_array
    return make_block_index_from_array(indices, ranges=ranges)

# }}}


# {{{ proxy point generator

def _generate_unit_sphere(ambient_dim: int, approx_npoints: int) -> np.ndarray:
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


@dataclass(frozen=True)
class BlockProxyPoints:
    """
    .. attribute:: srcindices

        A :class:`~pytential.linalg.BlockIndexRanges` describing which block of
        points each proxy ball was created from.

    .. attribute:: indices

        A :class:`~pytential.linalg.BlockIndexRanges` describing which proxies
        belong to which block.

    .. attribute:: points

        A concatenated list of all the proxy points. Can be sliced into
        using :attr:`indices` (shape ``(dim, nproxy_per_block * nblocks)``).

    .. attribute:: centers

        A list of all the proxy ball centers (shape ``(dim, nblocks)``).

    .. attribute:: radii

        A list of all the proxy ball radii (shape ``(nblocks,)``).

    .. attribute:: nblocks
    .. attribute:: nproxy_per_block
    .. automethod:: to_numpy
    """

    srcindices: BlockIndexRanges
    indices: BlockIndexRanges
    points: np.ndarray
    centers: np.ndarray
    radii: np.ndarray

    @property
    def nblocks(self) -> int:
        return self.srcindices.nblocks

    @property
    def nproxy_per_block(self) -> int:
        return self.points[0].shape[0] // self.nblocks

    def to_numpy(self, actx: PyOpenCLArrayContext) -> "BlockProxyPoints":
        from arraycontext import to_numpy
        from dataclasses import replace
        return replace(self,
                points=to_numpy(self.points, actx),
                centers=to_numpy(self.centers, actx),
                radii=to_numpy(self.radii, actx))


def make_compute_block_centers_knl(
        actx: PyOpenCLArrayContext, ndim: int, norm_type: str) -> lp.LoopKernel:
    @memoize_in(actx, (make_compute_block_centers_knl, ndim, norm_type))
    def prg():
        if norm_type == "l2":
            insns = """
            proxy_center[idim, irange] = 1.0 / npoints \
                    * simul_reduce(sum, i, sources[idim, srcindices[i + ioffset]])
            """
        elif norm_type == "linf":
            insns = """
            <> bbox_max = \
                    simul_reduce(max, i, sources[idim, srcindices[i + ioffset]])
            <> bbox_min = \
                    simul_reduce(min, i, sources[idim, srcindices[i + ioffset]])

            proxy_center[idim, irange] = (bbox_max + bbox_min) / 2.0
            """
        else:
            raise ValueError(f"unknown norm type: '{norm_type}'")

        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < ndim}"
            ],
            """
            for irange
                <> ioffset = srcranges[irange]
                <> npoints = srcranges[irange + 1] - srcranges[irange]

                %(insns)s
            end
            """ % dict(insns=insns), [
                lp.GlobalArg("sources", None, shape=(ndim, "nsources")),
                lp.ValueArg("nsources", np.int64),
                ...
                ],
            name="compute_block_centers_knl",
            assumptions="ndim>=1 and nranges>=1",
            fixed_parameters=dict(ndim=ndim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "irange", 64, outer_tag="g.0")

        return knl

    return prg()


class ProxyGeneratorBase:
    r"""
    .. attribute:: nproxy

        Number of proxy points in a single proxy ball.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, places,
            approx_nproxy: Optional[int] = None,
            radius_factor: Optional[float] = None,
            norm_type: str = "linf"):
        """
        :param approx_nproxy: desired number of proxy points. In higher
            dimensions, it is not always possible to construct a proxy ball
            with exactly this number of proxy points. The exact number of
            proxy points can be retrieved with :attr:`nproxy`.
        :param radius_factor: Factor multiplying the block radius (i.e radius
            of the bounding box) to get the proxy ball radius.
        :param norm_type: type of the norm used to compute the centers of
            each block. Supported values are ``"linf"`` and ``"l2"``.
        """
        from pytential import GeometryCollection
        if not isinstance(places, GeometryCollection):
            places = GeometryCollection(places)

        self.places = places
        self.radius_factor = 1.1 if radius_factor is None else radius_factor
        self.norm_type = norm_type

        approx_nproxy = 32 if approx_nproxy is None else approx_nproxy
        self.ref_points = _generate_unit_sphere(self.ambient_dim, approx_nproxy)

    @property
    def ambient_dim(self) -> int:
        return self.places.ambient_dim

    @property
    def nproxy(self) -> int:
        return self.ref_points.shape[1]

    def get_centers_knl(self, actx: PyOpenCLArrayContext) -> lp.LoopKernel:
        return make_compute_block_centers_knl(actx, self.ambient_dim, self.norm_type)

    def get_radii_knl(self, actx: PyOpenCLArrayContext) -> lp.LoopKernel:
        raise NotImplementedError

    def __call__(self,
            actx: PyOpenCLArrayContext,
            source_dd,
            indices: BlockIndexRanges, **kwargs) -> BlockProxyPoints:
        """Generate proxy points for each block in *indices* with nodes in
        the discretization *source_dd*.

        :arg source_dd: a :class:`~pytential.symbolic.primitives.DOFDescriptor`
            for the discretization on which the proxy points are to be
            generated.
        """
        from pytential import sym
        source_dd = sym.as_dofdesc(source_dd)
        discr = self.places.get_discretization(
                source_dd.geometry, source_dd.discr_stage)

        # {{{ get proxy centers and radii

        sources = flatten(discr.nodes(), actx, leaf_class=DOFArray)

        knl = self.get_centers_knl(actx)
        _, (centers_dev,) = knl(actx.queue,
                sources=sources,
                srcindices=indices.indices,
                srcranges=indices.ranges)

        knl = self.get_radii_knl(actx)
        _, (radii_dev,) = knl(actx.queue,
                sources=sources,
                srcindices=indices.indices,
                srcranges=indices.ranges,
                radius_factor=self.radius_factor,
                proxy_centers=centers_dev,
                **kwargs)

        # }}}

        # {{{ build proxy points for each block

        from arraycontext import to_numpy
        centers = np.vstack(to_numpy(centers_dev, actx))
        radii = to_numpy(radii_dev, actx)

        nproxy = self.nproxy * indices.nblocks
        proxies = np.empty((self.ambient_dim, nproxy), dtype=centers.dtype)
        pxy_nr_base = 0

        for i in range(indices.nblocks):
            bball = radii[i] * self.ref_points + centers[:, i].reshape(-1, 1)
            proxies[:, pxy_nr_base:pxy_nr_base + self.nproxy] = bball

            pxy_nr_base += self.nproxy

        # }}}

        pxyindices = np.arange(0, nproxy, dtype=indices.indices.dtype)
        pxyranges = np.arange(0, nproxy + 1, self.nproxy)

        from arraycontext import freeze, from_numpy
        from pytential.linalg import make_block_index_from_array
        return BlockProxyPoints(
                srcindices=indices,
                indices=make_block_index_from_array(pxyindices, pxyranges),
                points=freeze(from_numpy(proxies, actx), actx),
                centers=freeze(centers_dev, actx),
                radii=freeze(radii_dev, actx),
                )


def make_compute_block_radii_knl(
        actx: PyOpenCLArrayContext, ndim: int) -> lp.LoopKernel:
    @memoize_in(actx, (make_compute_block_radii_knl, ndim))
    def prg():
        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < ndim}"
            ],
            """
            for irange
                <> ioffset = srcranges[irange]
                <> npoints = srcranges[irange + 1] - srcranges[irange]
                <> rblk = reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_centers[idim, irange]
                        - sources[idim, srcindices[i + ioffset]]) ** 2)
                        ))

                proxy_radius[irange] = radius_factor * rblk
            end
            """, [
                lp.GlobalArg("sources", None, shape=(ndim, "nsources")),
                lp.ValueArg("nsources", np.int64),
                lp.ValueArg("radius_factor", np.float64),
                ...
                ],
            name="compute_block_radii_knl",
            assumptions="ndim>=1 and nranges>=1",
            fixed_parameters=dict(ndim=ndim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "irange", 64, outer_tag="g.0")

        return knl

    return prg()


class ProxyGenerator(ProxyGeneratorBase):
    """A proxy point generator that only considers the points in the current
    block when determining the radius of the proxy ball.

    Inherits from :class:`ProxyGeneratorBase`.
    """

    def get_radii_knl(self, actx: PyOpenCLArrayContext) -> lp.LoopKernel:
        return make_compute_block_radii_knl(actx, self.ambient_dim)


def make_compute_block_qbx_radii_knl(
        actx: PyOpenCLArrayContext, ndim: int) -> lp.LoopKernel:
    @memoize_in(actx, (make_compute_block_qbx_radii_knl, ndim))
    def prg():
        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < ndim}"
            ],
            """
            for irange
                <> ioffset = srcranges[irange]
                <> npoints = srcranges[irange + 1] - srcranges[irange]
                <> rqbx_int = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_centers[idim, irange] -
                         center_int[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]])
                <> rqbx_ext = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_centers[idim, irange] -
                         center_ext[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]]) \
                         {dup=idim}
                <> rqbx = rqbx_int if rqbx_ext < rqbx_int else rqbx_ext

                proxy_radius[irange] = radius_factor * rqbx
            end
            """, [
                lp.GlobalArg("sources", None, shape=(ndim, "nsources")),
                lp.GlobalArg("center_int", None, shape=(ndim, "nsources")),
                lp.GlobalArg("center_ext", None, shape=(ndim, "nsources")),
                lp.ValueArg("nsources", np.int64),
                lp.ValueArg("radius_factor", np.float64),
                ...
                ],
            name="compute_block_qbx_radii_knl",
            assumptions="ndim>=1 and nranges>=1",
            fixed_parameters=dict(ndim=ndim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "irange", 64, outer_tag="g.0")

        return knl

    return prg()


class QBXProxyGenerator(ProxyGeneratorBase):
    """A proxy point generator that also considers the QBX expansion
    when determining the radius of the proxy ball.

    Inherits from :class:`ProxyGeneratorBase`.
    """

    def get_radii_knl(self, actx: PyOpenCLArrayContext) -> lp.LoopKernel:
        return make_compute_block_qbx_radii_knl(actx, self.ambient_dim)

    def __call__(self,
            actx: PyOpenCLArrayContext,
            source_dd,
            indices: BlockIndexRanges, **kwargs) -> BlockProxyPoints:
        from pytential import bind, sym
        source_dd = sym.as_dofdesc(source_dd)

        radii = bind(self.places, sym.expansion_radii(
            self.ambient_dim, dofdesc=source_dd))(actx)
        center_int = bind(self.places, sym.expansion_centers(
            self.ambient_dim, -1, dofdesc=source_dd))(actx)
        center_ext = bind(self.places, sym.expansion_centers(
            self.ambient_dim, +1, dofdesc=source_dd))(actx)

        return super().__call__(actx, source_dd, indices,
                expansion_radii=flatten(radii, actx),
                center_int=flatten(center_int, actx, leaf_class=DOFArray),
                center_ext=flatten(center_ext, actx, leaf_class=DOFArray),
                **kwargs)

# }}}


# {{{ gather_block_neighbor_points

def gather_block_neighbor_points(
        actx: PyOpenCLArrayContext, discr: Discretization, pxy: BlockProxyPoints,
        max_particles_in_box: Optional[int] = None) -> BlockIndexRanges:
    """Generate a set of neighboring points for each range of points in
    *discr*. Neighboring points of a range :math:`i` are defined
    as all the points inside the proxy ball :math:`i` that do not also
    belong to the range itself.
    """

    if max_particles_in_box is None:
        # FIXME: this is a fairly arbitrary value
        max_particles_in_box = 32

    # {{{ get only sources in indices

    @memoize_in(actx,
            (gather_block_neighbor_points, discr.ambient_dim, "picker_knl"))
    def prg():
        knl = lp.make_kernel(
            "{[idim, i]: 0 <= idim < ndim and 0 <= i < npoints}",
            """
            result[idim, i] = ary[idim, srcindices[i]]
            """, [
                lp.GlobalArg("ary", None, shape=(discr.ambient_dim, "ndofs")),
                lp.ValueArg("ndofs", np.int64),
                ...],
            name="picker_knl",
            assumptions="ndim>=1 and npoints>=1",
            fixed_parameters=dict(ndim=discr.ambient_dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "i", 64, outer_tag="g.0")

        return knl

    _, (sources,) = prg()(actx.queue,
            ary=flatten(discr.nodes(), actx, leaf_class=DOFArray),
            srcindices=pxy.srcindices.indices)

    # }}}

    # {{{ perform area query

    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)
    tree, _ = builder(actx.queue, sources,
            max_particles_in_box=max_particles_in_box)

    from boxtree.area_query import AreaQueryBuilder
    builder = AreaQueryBuilder(actx.context)
    query, _ = builder(actx.queue, tree, pxy.centers, pxy.radii)

    # find nodes inside each proxy ball
    tree = tree.get(actx.queue)
    query = query.get(actx.queue)

    # }}}

    # {{{ retrieve results

    from arraycontext import to_numpy
    pxycenters = to_numpy(pxy.centers, actx)
    pxyradii = to_numpy(pxy.radii, actx)
    indices = pxy.srcindices

    nbrindices = np.empty(indices.nblocks, dtype=object)
    for iproxy in range(indices.nblocks):
        # get list of boxes intersecting the current ball
        istart = query.leaves_near_ball_starts[iproxy]
        iend = query.leaves_near_ball_starts[iproxy + 1]
        iboxes = query.leaves_near_ball_lists[istart:iend]

        if (iend - istart) <= 0:
            nbrindices[iproxy] = np.empty(0, dtype=np.int64)
            continue

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

    # }}}

    from pytential.linalg import make_block_index_from_array
    return make_block_index_from_array(indices=nbrindices)

# }}}

# vim: foldmethod=marker
