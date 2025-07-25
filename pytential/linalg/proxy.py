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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.linalg as la

from arraycontext import Array, ArrayContainer, PyOpenCLArrayContext, flatten
from meshmode.discretization import Discretization
from meshmode.dof_array import DOFArray

from numpy.typing import NDArray
from pytools import memoize_in
from pytential import GeometryCollection, bind, sym
from pytential.symbolic.dof_desc import DOFDescriptorLike
from pytential.linalg.utils import IndexList
from pytential.source import PointPotentialSource
from pytential.target import PointsTarget
from pytential.qbx import QBXLayerPotentialSource

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


__doc__ = """
Proxy Point Generation
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pytential.linalg

.. autoclass:: ProxyPointSource
.. autoclass:: ProxyPointTarget
.. autoclass:: ProxyClusterGeometryData

.. autoclass:: ProxyGeneratorBase
.. autoclass:: ProxyGenerator
.. autoclass:: QBXProxyGenerator

.. autofunction:: partition_by_nodes
.. autofunction:: gather_cluster_neighbor_points
"""

# FIXME: this is just an arbitrary value
_DEFAULT_MAX_PARTICLES_IN_BOX = 32


# {{{ point index partitioning

def partition_by_nodes(
        actx: PyOpenCLArrayContext, places: GeometryCollection, *,
        dofdesc: DOFDescriptorLike | None = None,
        tree_kind: str | None = "adaptive-level-restricted",
        max_particles_in_box: int | None = None) -> IndexList:
    """Generate equally sized ranges of nodes. The partition is created at the
    lowest level of granularity, i.e. nodes. This results in balanced ranges
    of points, but will split elements across different ranges.

    :arg dofdesc: a :class:`~pytential.symbolic.dof_desc.DOFDescriptor` for
        the geometry in *places* which should be partitioned.
    :arg tree_kind: if not *None*, it is passed to :class:`boxtree.TreeBuilder`.
    :arg max_particles_in_box: value used to control the number of points
        in each partition (and thus the number of partitions). See the documentation
        in :class:`boxtree.TreeBuilder`.
    """
    if dofdesc is None:
        dofdesc = places.auto_source
    dofdesc = sym.as_dofdesc(dofdesc)

    if max_particles_in_box is None:
        max_particles_in_box = _DEFAULT_MAX_PARTICLES_IN_BOX

    from pytential.source import LayerPotentialSourceBase

    lpot_source = places.get_geometry(dofdesc.geometry)
    assert isinstance(lpot_source, LayerPotentialSourceBase)

    discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    assert isinstance(discr, Discretization)

    if tree_kind is not None:
        from pytential.qbx.utils import tree_code_container
        tcc = tree_code_container(lpot_source._setup_actx)

        tree, _ = tcc.build_tree()(actx.queue,
                particles=flatten(
                    actx.thaw(discr.nodes()), actx, leaf_class=DOFArray
                    ),
                max_particles_in_box=max_particles_in_box,
                kind=tree_kind)

        from boxtree import box_flags_enum
        tree = tree.get(actx.queue)
        # FIXME maybe this should use IS_LEAF once available?
        leaf_boxes, = (
                tree.box_flags & box_flags_enum.HAS_SOURCE_OR_TARGET_CHILD_BOXES == 0
                ).nonzero()

        indices: np.ndarray = np.empty(len(leaf_boxes), dtype=object)
        starts: np.ndarray | None = None

        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]
            indices[i] = tree.user_source_ids[box_start:box_end]
    else:
        if discr.ambient_dim != 2 and discr.dim == 1:
            raise ValueError("only curves are supported for 'tree_kind=None'")

        nclusters = max(discr.ndofs // max_particles_in_box, 2)
        indices = np.arange(0, discr.ndofs, dtype=np.int64)
        starts = np.linspace(0, discr.ndofs, nclusters + 1, dtype=np.int64)

        # FIXME: I'm not sure why mypy can't figure this out.
        assert starts is not None

        assert starts[-1] == discr.ndofs

    from pytential.linalg import make_index_list
    return make_index_list(indices, starts=starts)

# }}}


# {{{ proxy points

class ProxyPointSource(PointPotentialSource):
    """
    .. automethod:: get_expansion_for_qbx_direct_eval
    """

    def __init__(self,
            lpot_source: QBXLayerPotentialSource,
            proxies: Array) -> None:
        """
        :arg lpot_source: the layer potential for which the proxy are constructed.
        :arg proxies: an array of shape ``(ambient_dim, nproxies)`` containing
            the proxy points.
        """
        assert proxies.shape[0] == lpot_source.ambient_dim

        super().__init__(proxies)
        self.lpot_source = lpot_source

    def get_expansion_for_qbx_direct_eval(self, base_kernel, target_kernels):
        """Wrapper around
        ``pytential.qbx.QBXLayerPotentialSource.get_expansion_for_qbx_direct_eval``
        to allow this class to be used by the matrix builders.
        """
        return self.lpot_source.get_expansion_for_qbx_direct_eval(
                base_kernel, target_kernels)


class ProxyPointTarget(PointsTarget):
    def __init__(self,
            lpot_source: QBXLayerPotentialSource,
            proxies: Array) -> None:
        """
        :arg lpot_source: the layer potential for which the proxy are constructed.
            This argument is kept for symmetry with :class:`ProxyPointSource`.
        :arg proxies: an array of shape ``(ambient_dim, nproxies)`` containing
            the proxy points.
        """
        assert proxies.shape[0] == lpot_source.ambient_dim

        super().__init__(proxies)
        self.lpot_source = lpot_source


@dataclass(frozen=True)
class ProxyClusterGeometryData:
    """
    .. attribute:: srcindex

        A :class:`~pytential.linalg.IndexList` describing which cluster
        of points each proxy ball was created from.

    .. attribute:: pxyindex

        A :class:`~pytential.linalg.IndexList` describing which proxies
        belong to which cluster.

    .. attribute:: points

        A concatenated array of all the proxy points. Can be sliced into
        using :attr:`pxyindex` (shape ``(dim, nproxies)``).

    .. attribute:: centers

        An array of all the proxy ball centers (shape ``(dim, nclusters)``).

    .. attribute:: radii

        An array of all the proxy ball radii (shape ``(nclusters,)``).

    .. attribute:: nclusters

    .. automethod:: __init__
    .. automethod:: to_numpy
    .. automethod:: as_sources
    .. automethod:: as_targets
    """

    places: GeometryCollection
    dofdesc: sym.DOFDescriptor

    srcindex: IndexList
    pxyindex: IndexList

    points: NDArray[Any]
    centers: NDArray[Any]
    radii: NDArray[Any]

    _cluster_radii: np.ndarray | None = None

    @property
    def nclusters(self) -> int:
        return self.srcindex.nclusters

    @property
    def discr(self):
        return self.places.get_discretization(
                self.dofdesc.geometry,
                self.dofdesc.discr_stage)

    def to_numpy(self, actx: PyOpenCLArrayContext) -> "ProxyClusterGeometryData":
        if self._cluster_radii is not None:
            cluster_radii = actx.to_numpy(self._cluster_radii)
        else:
            cluster_radii = None

        from dataclasses import replace
        return replace(self,
                points=np.stack(actx.to_numpy(self.points)),
                centers=np.stack(actx.to_numpy(self.centers)),
                radii=actx.to_numpy(self.radii),
                _cluster_radii=cluster_radii)

    def as_sources(self) -> ProxyPointSource:
        lpot_source = self.places.get_geometry(self.dofdesc.geometry)
        assert isinstance(lpot_source, QBXLayerPotentialSource)

        return ProxyPointSource(lpot_source, cast("Array", self.points))

    def as_targets(self) -> ProxyPointTarget:
        lpot_source = self.places.get_geometry(self.dofdesc.geometry)
        assert isinstance(lpot_source, QBXLayerPotentialSource)

        return ProxyPointTarget(lpot_source, cast("Array", self.points))

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
        t = np.linspace(0.0, 2.0 * np.pi, approx_npoints, endpoint=False)
        points = np.vstack([np.cos(t), np.sin(t)])
    elif ambient_dim == 3:
        from pytools import sphere_sample_equidistant
        points = sphere_sample_equidistant(approx_npoints, r=1)
    else:
        raise ValueError("ambient_dim > 3 not supported.")

    return points


def make_compute_cluster_centers_kernel_ex(
        actx: PyOpenCLArrayContext, ndim: int, norm_type: str) -> lp.ExecutorBase:
    @memoize_in(actx, (make_compute_cluster_centers_kernel_ex, ndim, norm_type))
    def prg():
        if norm_type == "l2":
            # NOTE: computes first-order approximation of the source centroids
            insns = """
            proxy_center[idim, icluster] = 1.0 / npoints \
                    * simul_reduce(sum, i, sources[idim, srcindices[i + ioffset]])
            """
        elif norm_type == "linf":
            # NOTE: computes the centers of the bounding box
            insns = """
            <> bbox_max = \
                    simul_reduce(max, i, sources[idim, srcindices[i + ioffset]])
            <> bbox_min = \
                    simul_reduce(min, i, sources[idim, srcindices[i + ioffset]])

            proxy_center[idim, icluster] = (bbox_max + bbox_min) / 2.0
            """
        else:
            raise ValueError(f"unknown norm type: '{norm_type}'")

        knl = lp.make_kernel([
            "{[icluster]: 0 <= icluster < nclusters}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < ndim}"
            ],
            f"""
            for icluster
                <> ioffset = srcstarts[icluster]
                <> npoints = srcstarts[icluster + 1] - ioffset

                {insns}
            end
            """, [
                lp.GlobalArg("sources", None,
                    shape=(ndim, "nsources"), dim_tags="sep,C", offset=lp.auto),
                lp.ValueArg("nsources", np.int64),
                ...
                ],
            name="compute_cluster_centers_knl",
            assumptions="ndim>=1 and nclusters>=1",
            fixed_parameters={"ndim": ndim},
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "icluster", 64, outer_tag="g.0")

        return knl.executor(actx.context)

    return prg()


class ProxyGeneratorBase:
    r"""
    .. attribute:: nproxy

        Number of proxy points in a single proxy ball.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, places: GeometryCollection,
            approx_nproxy: int | None = None,
            radius_factor: float | None = None,
            norm_type: str = "linf",

            _generate_ref_proxies: Callable[[int], np.ndarray] | None = None,
            ) -> None:
        """
        :param approx_nproxy: desired number of proxy points. In higher
            dimensions, it is not always possible to construct a proxy ball
            with exactly this number of proxy points. The exact number of
            proxy points can be retrieved with :attr:`nproxy`.
        :param radius_factor: Factor multiplying the cluster radius (i.e radius
            of the bounding box) to get the proxy ball radius.
        :param norm_type: type of the norm used to compute the centers of
            each cluster. Supported values are ``"linf"`` and ``"l2"``.
        """
        generate_ref_proxies = _generate_ref_proxies
        if generate_ref_proxies is None:
            from functools import partial
            generate_ref_proxies = partial(_generate_unit_sphere, places.ambient_dim)

        from pytential import GeometryCollection
        if not isinstance(places, GeometryCollection):
            places = GeometryCollection(places)

        if norm_type not in ("l2", "linf"):
            raise ValueError(
                    f"unsupported norm type: '{norm_type}' "
                    + "(expected one of 'l2' or 'linf')")

        if radius_factor is None:
            # FIXME: this is a fairly arbitrary value
            radius_factor = 1.1

        if approx_nproxy is None:
            # FIXME: this is a fairly arbitrary value
            approx_nproxy = 32

        self.places = places
        self.radius_factor = radius_factor
        self.norm_type = norm_type

        self.ref_points = generate_ref_proxies(approx_nproxy)

    @property
    def ambient_dim(self) -> int:
        return self.places.ambient_dim

    @property
    def nproxy(self) -> int:
        return self.ref_points.shape[1]

    def get_centers_kernel_ex(self, actx: PyOpenCLArrayContext) -> lp.ExecutorBase:
        return make_compute_cluster_centers_kernel_ex(
                actx, self.ambient_dim, self.norm_type)

    def get_radii_kernel_ex(self, actx: PyOpenCLArrayContext) -> lp.ExecutorBase:
        raise NotImplementedError

    def __call__(self,
            actx: PyOpenCLArrayContext,
            source_dd: DOFDescriptorLike | None,
            dof_index: IndexList,
            **kwargs: Any) -> ProxyClusterGeometryData:
        """Generate proxy points for each cluster in *dof_index_set* with nodes in
        the discretization *source_dd*.

        :arg source_dd: a :class:`~pytential.symbolic.dof_desc.DOFDescriptor`
            for the discretization on which the proxy points are to be
            generated.
        """
        if source_dd is None:
            source_dd = self.places.auto_source
        source_dd = sym.as_dofdesc(source_dd)

        discr = self.places.get_discretization(
                source_dd.geometry, source_dd.discr_stage)
        assert isinstance(discr, Discretization)

        include_cluster_radii = kwargs.pop("include_cluster_radii", False)

        # {{{ get proxy centers and radii

        sources = flatten(discr.nodes(), actx, leaf_class=DOFArray)

        knl = self.get_centers_kernel_ex(actx)
        _, (centers_dev,) = knl(actx.queue,
                sources=sources,
                srcindices=dof_index.indices,
                srcstarts=dof_index.starts)

        knl = self.get_radii_kernel_ex(actx)
        _, (radii_dev,) = knl(actx.queue,
                sources=sources,
                srcindices=dof_index.indices,
                srcstarts=dof_index.starts,
                radius_factor=self.radius_factor,
                proxy_centers=centers_dev,
                **kwargs)

        if include_cluster_radii:
            knl = make_compute_cluster_radii_kernel_ex(actx, self.ambient_dim)
            _, (cluster_radii,) = knl(actx.queue,
                    sources=sources,
                    srcindices=dof_index.indices,
                    srcstarts=dof_index.starts,
                    radius_factor=1.0,
                    proxy_centers=centers_dev)
            cluster_radii = actx.freeze(cluster_radii)
        else:
            cluster_radii = None

        # }}}

        # {{{ build proxy points for each cluster

        centers = np.vstack(actx.to_numpy(centers_dev))
        radii = actx.to_numpy(radii_dev)

        nproxy = self.nproxy * dof_index.nclusters
        proxies = np.empty((self.ambient_dim, nproxy), dtype=centers.dtype)
        pxy_nr_base = 0

        for i in range(dof_index.nclusters):
            points = radii[i] * self.ref_points + centers[:, i].reshape(-1, 1)
            proxies[:, pxy_nr_base:pxy_nr_base + self.nproxy] = points

            pxy_nr_base += self.nproxy

        # }}}

        pxyindices = np.arange(0, nproxy, dtype=dof_index.indices.dtype)
        pxystarts = np.arange(0, nproxy + 1, self.nproxy)

        from pytential.linalg import make_index_list
        return ProxyClusterGeometryData(
                places=self.places,
                dofdesc=source_dd,
                srcindex=dof_index,
                pxyindex=make_index_list(pxyindices, pxystarts),
                points=actx.freeze(actx.from_numpy(proxies)),
                centers=actx.freeze(centers_dev),
                radii=actx.freeze(radii_dev),
                _cluster_radii=cluster_radii
                )


def make_compute_cluster_radii_kernel_ex(
        actx: PyOpenCLArrayContext, ndim: int) -> lp.ExecutorBase:
    @memoize_in(actx, (make_compute_cluster_radii_kernel_ex, ndim))
    def prg():
        knl = lp.make_kernel([
            "{[icluster]: 0 <= icluster < nclusters}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < ndim}"
            ],
            """
            for icluster
                <> ioffset = srcstarts[icluster]
                <> npoints = srcstarts[icluster + 1] - ioffset
                <> cluster_radius = reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_centers[idim, icluster]
                        - sources[idim, srcindices[i + ioffset]]) ** 2)
                        ))

                proxy_radius[icluster] = radius_factor * cluster_radius
            end
            """, [
                lp.GlobalArg("sources", None,
                    shape=(ndim, "nsources"), dim_tags="sep,C", offset=lp.auto),
                lp.ValueArg("nsources", np.int64),
                lp.ValueArg("radius_factor", np.float64),
                ...
                ],
            name="compute_cluster_radii_knl",
            assumptions="ndim>=1 and nclusters>=1",
            fixed_parameters={"ndim": ndim},
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "icluster", 64, outer_tag="g.0")

        return knl.executor(actx.context)

    return prg()


class ProxyGenerator(ProxyGeneratorBase):
    """A proxy point generator that only considers the points in the current
    cluster when determining the radius of the proxy ball.

    Inherits from :class:`ProxyGeneratorBase`.
    """

    def get_radii_kernel_ex(self, actx: PyOpenCLArrayContext) -> lp.ExecutorBase:
        return make_compute_cluster_radii_kernel_ex(actx, self.ambient_dim)


def make_compute_cluster_qbx_radii_kernel_ex(
        actx: PyOpenCLArrayContext, ndim: int) -> lp.ExecutorBase:
    @memoize_in(actx, (make_compute_cluster_qbx_radii_kernel_ex, ndim))
    def prg():
        knl = lp.make_kernel([
            "{[icluster]: 0 <= icluster < nclusters}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < ndim}"
            ],
            """
            for icluster
                <> ioffset = srcstarts[icluster]
                <> npoints = srcstarts[icluster + 1] - ioffset

                <> rqbx_int = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_centers[idim, icluster] -
                         center_int[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]])
                <> rqbx_ext = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_centers[idim, icluster] -
                         center_ext[idim, srcindices[i + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[i + ioffset]]) \
                         {dup=idim}

                <> rqbx = rqbx_int if rqbx_ext < rqbx_int else rqbx_ext

                proxy_radius[icluster] = radius_factor * rqbx
            end
            """, [
                lp.GlobalArg("sources", None,
                    shape=(ndim, "nsources"), dim_tags="sep,C", offset=lp.auto),
                lp.GlobalArg("center_int", None,
                    shape=(ndim, "nsources"), dim_tags="sep,C", offset=lp.auto),
                lp.GlobalArg("center_ext", None,
                    shape=(ndim, "nsources"), dim_tags="sep,C", offset=lp.auto),
                lp.ValueArg("nsources", np.int64),
                lp.ValueArg("radius_factor", np.float64),
                ...
                ],
            name="compute_cluster_qbx_radii_knl",
            assumptions="ndim>=1 and nclusters>=1",
            fixed_parameters={"ndim": ndim},
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "icluster", 64, outer_tag="g.0")

        return knl.executor(actx.context)

    return prg()


class QBXProxyGenerator(ProxyGeneratorBase):
    """A proxy point generator that also considers the QBX expansion
    when determining the radius of the proxy ball.

    Inherits from :class:`ProxyGeneratorBase`.
    """

    def get_radii_kernel_ex(self, actx: PyOpenCLArrayContext) -> lp.ExecutorBase:
        return make_compute_cluster_qbx_radii_kernel_ex(actx, self.ambient_dim)

    def __call__(self,
            actx: PyOpenCLArrayContext,
            source_dd: DOFDescriptorLike | None,
            dof_index: IndexList, **kwargs) -> ProxyClusterGeometryData:
        if source_dd is None:
            source_dd = self.places.auto_source
        source_dd = sym.as_dofdesc(source_dd)

        radii = cast(ArrayContainer, bind(self.places, sym.expansion_radii(
            self.ambient_dim, dofdesc=source_dd))(actx))
        center_int = cast(ArrayContainer, bind(self.places, sym.expansion_centers(
            self.ambient_dim, -1, dofdesc=source_dd))(actx))
        center_ext = cast(ArrayContainer, bind(self.places, sym.expansion_centers(
            self.ambient_dim, +1, dofdesc=source_dd))(actx))

        return super().__call__(actx, source_dd, dof_index,
                expansion_radii=flatten(radii, actx),
                center_int=flatten(center_int, actx, leaf_class=DOFArray),
                center_ext=flatten(center_ext, actx, leaf_class=DOFArray),
                **kwargs)

# }}}


# {{{ gather_cluster_neighbor_points

def gather_cluster_neighbor_points(
        actx: PyOpenCLArrayContext, pxy: ProxyClusterGeometryData, *,
        max_particles_in_box: int | None = None) -> IndexList:
    """Generate a set of neighboring points for each cluster of points in
    *pxy*. Neighboring points of a cluster :math:`i` are defined
    as all the points inside the proxy ball :math:`i` that do not also
    belong to the cluster itself.
    """

    if max_particles_in_box is None:
        max_particles_in_box = _DEFAULT_MAX_PARTICLES_IN_BOX

    from pytential.source import LayerPotentialSourceBase

    dofdesc = pxy.dofdesc
    lpot_source = pxy.places.get_geometry(dofdesc.geometry)
    assert isinstance(lpot_source, LayerPotentialSourceBase)

    discr = pxy.places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    assert isinstance(discr, Discretization)

    # {{{ get only sources in the current cluster set

    @memoize_in(actx,
            (gather_cluster_neighbor_points, discr.ambient_dim, "picker_knl"))
    def prg() -> lp.ExecutorBase:
        knl = lp.make_kernel(
            "{[idim, i]: 0 <= idim < ndim and 0 <= i < npoints}",
            """
            result[idim, i] = ary[idim, srcindices[i]]
            """, [
                lp.GlobalArg("ary", None,
                    shape=(discr.ambient_dim, "ndofs"), dim_tags="sep,C"),
                lp.ValueArg("ndofs", np.dtype(np.int64)),
                ...],
            name="picker_knl",
            assumptions="ndim>=1 and npoints>=1",
            fixed_parameters={"ndim": discr.ambient_dim},
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.split_iname(knl, "i", 64, outer_tag="g.0")

        return knl.executor(actx.context)

    _, (sources,) = prg()(actx.queue,
            ary=flatten(discr.nodes(), actx, leaf_class=DOFArray),
            srcindices=pxy.srcindex.indices)

    # }}}

    # {{{ perform area query

    from pytential.qbx.utils import tree_code_container
    tcc = tree_code_container(lpot_source._setup_actx)

    tree, _ = tcc.build_tree()(actx.queue, sources,
            max_particles_in_box=max_particles_in_box)
    query, _ = tcc.build_area_query()(actx.queue, tree, pxy.centers, pxy.radii)

    tree = tree.get(actx.queue)
    query = query.get(actx.queue)

    # }}}

    # {{{ retrieve results

    pxycenters = actx.to_numpy(pxy.centers)
    pxyradii = actx.to_numpy(pxy.radii)
    srcindex = pxy.srcindex

    nbrindices: np.ndarray = np.empty(srcindex.nclusters, dtype=object)
    for icluster in range(srcindex.nclusters):
        # get list of boxes intersecting the current ball
        istart = query.leaves_near_ball_starts[icluster]
        iend = query.leaves_near_ball_starts[icluster + 1]
        iboxes = query.leaves_near_ball_lists[istart:iend]

        if (iend - istart) <= 0:
            nbrindices[icluster] = np.empty(0, dtype=np.int64)
            continue

        # get nodes inside the boxes
        istart = tree.box_source_starts[iboxes]
        iend = istart + tree.box_source_counts_cumul[iboxes]
        isources = np.hstack([
            np.arange(s, e) for s, e in zip(istart, iend, strict=True)])
        nodes = np.vstack([s[isources] for s in tree.sources])
        isources = tree.user_source_ids[isources]

        # get nodes inside the ball but outside the current cluster
        # FIXME: this assumes that only the points in `pxy.secindex` should
        # count as neighbors, not all the nodes in the discretization.
        # FIXME: it also assumes that all the indices are sorted?
        center = pxycenters[:, icluster].reshape(-1, 1)
        radius = pxyradii[icluster]
        mask = ((la.norm(nodes - center, axis=0) < radius)
                & ((isources < srcindex.starts[icluster])
                    | (srcindex.starts[icluster + 1] <= isources)))

        nbrindices[icluster] = srcindex.indices[isources[mask]]

    # }}}

    from pytential.linalg import make_index_list
    return make_index_list(indices=nbrindices)

# }}}

# vim: foldmethod=marker
