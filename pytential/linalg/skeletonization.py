__copyright__ = "Copyright (C) 2018-2022 Alexandru Fikl"

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
from typing import (
        Any, Callable, Dict, Hashable, Iterable, Optional, Tuple, Union)

import numpy as np

from arraycontext import PyOpenCLArrayContext
from pytools.obj_array import make_obj_array

from pytential import GeometryCollection, sym
from pytential.linalg.utils import IndexList, TargetAndSourceClusterList
from pytential.linalg.proxy import ProxyGeneratorBase, ProxyClusterGeometryData
from pytential.symbolic.mappers import IdentityMapper, LocationTagger
from sumpy.kernel import TargetTransformationRemover, SourceTransformationRemover


__doc__ = """
.. currentmodule:: pytential.linalg

Skeletonization
---------------

.. autoclass:: SkeletonizationWrangler
.. autoclass:: make_skeletonization_wrangler

.. autoclass:: SkeletonizationResult
.. autofunction:: skeletonize_by_proxy
"""


# {{{ symbolic

class PROXY_SKELETONIZATION_SOURCE:             # noqa: N801
    pass


class PROXY_SKELETONIZATION_TARGET:             # noqa: N801
    pass


class NonOperatorRemover(IdentityMapper):
    r"""A mapper that removes any terms that do not contain an
    :class:`~pytential.symbolic.primitives.IntG`. It can only handle
    expressions of the form

    .. math::

        \sum F_i(\mathbf{x}, \sigma(\mathbf{x}))
        + \sum \int_\Sigma
            G_j(\mathbf{x} - \mathbf{y}) \sigma(\mathbf{y}) \mathrm{d}S_y

    and removes all the :math:`F_i` terms that are diagonal in terms of
    the density :math:`\sigma`.
    """

    def map_sum(self, expr):
        from pytential.symbolic.mappers import OperatorCollector
        children = []
        for child in expr.children:
            rec_child = self.rec(child)
            ops = OperatorCollector()(rec_child)
            if ops:
                children.append(rec_child)

        from pymbolic.primitives import flattened_sum
        return flattened_sum(children)

    def map_int_g(self, expr):
        return expr


class KernelTransformationRemover(IdentityMapper):
    r"""A mapper that removes the transformations from the kernel of all
    :class:`~pytential.symbolic.primitives.IntG`\ s in the expression.

    This is used when evaluating the proxy-target or proxy-source
    interactions because

    * Evaluating a single-layer vs a double-layer does not make a difference
      there (proxies are assumed to be far enough from the surface)
    * Kernel arguments, such as the normal, are not necessarily available at
      the proxies.
    """

    def __init__(self):
        self.sxr = SourceTransformationRemover()
        self.txr = TargetTransformationRemover()

    def map_int_g(self, expr):
        target_kernel = self.txr(expr.target_kernel)
        source_kernels = tuple([self.sxr(kernel) for kernel in expr.source_kernels])
        if (target_kernel == expr.target_kernel
                and source_kernels == expr.source_kernels):
            return expr

        source_args = {
            arg.name for kernel in expr.source_kernels
            for arg in kernel.get_source_args()}
        kernel_arguments = {
            name: self.rec(arg) for name, arg in expr.kernel_arguments.items()
            if name not in source_args
        }

        return expr.copy(target_kernel=target_kernel,
                         source_kernels=source_kernels,
                         kernel_arguments=kernel_arguments)


class LocationReplacer(LocationTagger):
    def _default_dofdesc(self, dofdesc):
        return self.default_target

    def map_int_g(self, expr):
        return type(expr)(
                expr.target_kernel, expr.source_kernels,
                densities=self.operand_rec(expr.densities),
                qbx_forced_limit=expr.qbx_forced_limit,
                source=self.default_source, target=self.default_target,
                kernel_arguments={
                    name: self.operand_rec(arg_expr)
                    for name, arg_expr in expr.kernel_arguments.items()
                    }
                )


class DOFDescriptorReplacer(LocationReplacer):
    r"""Mapper that replaces all the
    :class:`~pytential.symbolic.primitives.DOFDescriptor`\ s in the expression
    with the given ones.

    .. automethod:: __init__
    """

    def __init__(self, source, target):
        """
        :param source: a descriptor for all expressions to be evaluated on
            the source geometry.
        :param target: a descriptor for all expressions to be evaluate on
            the target geometry.
        """
        super().__init__(target, default_source=source)
        self.operand_rec = LocationReplacer(source, default_source=source)


def _prepare_neighbor_expr(places, exprs, auto_where=None):
    from pytential.symbolic.execution import _prepare_expr
    return make_obj_array([
        _prepare_expr(places, expr, auto_where=auto_where)
        for expr in exprs])


def prepare_proxy_expr(places, exprs, auto_where=None):
    def _prepare_expr(expr):
        # remove all diagonal / non-operator terms in the expression
        expr = NonOperatorRemover()(expr)
        # ensure all IntGs remove all the kernel derivatives
        expr = KernelTransformationRemover()(expr)
        # ensure all IntGs have their source and targets set
        expr = DOFDescriptorReplacer(auto_where[0], auto_where[1])(expr)

        return expr

    return make_obj_array([_prepare_expr(expr) for expr in exprs])

# }}}


# {{{ wrangler

def _approximate_geometry_waa_magnitude(actx, places, nbrindex, domain):
    from pytools import memoize_in
    from pytential import bind, sym

    @memoize_in(actx, (_approximate_geometry_waa_magnitude, "mean_over_cluster"))
    def prg():
        import loopy as lp
        knl = lp.make_kernel([
            "{[icluster]: 0 <= icluster < nclusters}",
            "{[i]: 0 <= i < npoints}",
            ],
            """
            <> ioffset = starts[icluster]
            <> npoints = starts[icluster + 1] - ioffset
            result[icluster] = reduce(sum, i, waa[indices[i + ioffset]]) / npoints
            """)

        return knl

    waa = bind(places, sym.weights_and_area_elements(
        places.ambient_dim, dofdesc=domain))(actx)
    result = actx.zeros((nbrindex.nclusters,), dtype=waa.entry_dtype)

    from arraycontext import flatten
    _, (waa_per_cluster,) = prg()(actx.queue,         # pylint: disable=not-callable
            waa=flatten(waa, actx),
            result=result,
            indices=nbrindex.indices,
            starts=nbrindex.starts)

    return waa_per_cluster


def _apply_weights(actx, mat, places, tgt_pxy_index, nbrindex, domain):
    assert tgt_pxy_index.nclusters == nbrindex.nclusters
    waa = actx.to_numpy(
            _approximate_geometry_waa_magnitude(actx, places, nbrindex, domain)
            )

    for i in range(tgt_pxy_index.nclusters):
        istart, iend = tgt_pxy_index._flat_cluster_starts[i:i + 2]
        mat[istart:iend] *= waa[i]

    return mat


@dataclass(frozen=True)
class SkeletonizationWrangler:
    """
    .. attribute:: exprs

        An :class:`~numpy.ndarray` of expressions (layer potentials)
        that correspond to the output blocks of the matrix. These expressions
        are tagged for nearfield neighbor evalution.

    .. attribute:: source_proxy_exprs
    .. attribute:: target_proxy_exprs

        Like :attr:`exprs`, but stripped down for farfield proxy
        evaluation.

    .. attribute:: input_exprs

        A :class:`tuple` of densities that correspond to the input blocks
        of the matrix.

    .. attribute:: domains

        A :class:`tuple` of the same length as *input_exprs* defining the
        domain of each input.

    .. attribute:: context

        A :class:`dict` with additional parameters required to evaluate the
        expressions.

    The following attributes and methods are internal and used for skeletonization.

    .. attribute:: weighted_sources

        A flag which if *True* adds a weight to the source to proxy evaluation.

    .. attribute:: weighted_targets

        A flag which if *True* adds a weight to the proxy to target evaluation.

    .. attribute:: proxy_source_cluster_builder
    .. attribute:: proxy_target_cluster_builder

        A callable that is used to evaluate farfield proxy interactions.
        This should follow the calling convention of the constructor to
        :class:`pytential.symbolic.matrix.P2PClusterMatrixBuilder`.

    .. attribute:: neighbor_cluster_builder

        A callable that is used to evaluate nearfield neighbour interactions.
        This should follow the calling convention of the constructor to
        :class:`pytential.symbolic.matrix.QBXClusterMatrixBuilder`.

    .. automethod:: evaluate_source_proxy_interaction
    .. automethod:: evaluate_target_proxy_interaction
    .. automethod:: evaluate_source_neighbor_interaction
    .. automethod:: evaluate_target_neighbor_interaction
    """

    # operator
    exprs: np.ndarray
    input_exprs: Tuple[sym.var, ...]
    domains: Tuple[Hashable, ...]
    context: Dict[str, Any]

    neighbor_cluster_builder: Callable[..., np.ndarray]

    # target skeletonization
    weighted_targets: bool
    target_proxy_exprs: np.ndarray
    proxy_target_cluster_builder: Callable[..., np.ndarray]

    # source skeletonization
    weighted_sources: bool
    source_proxy_exprs: np.ndarray
    proxy_source_cluster_builder: Callable[..., np.ndarray]

    @property
    def nrows(self):
        return len(self.exprs)

    @property
    def ncols(self):
        return len(self.input_exprs)

    def _evaluate_expr(self,
            actx, places, eval_mapper_cls, tgt_src_index, expr, idomain,
            **kwargs):
        domain = self.domains[idomain]
        dep_source = places.get_geometry(domain.geometry)
        dep_discr = places.get_discretization(domain.geometry, domain.discr_stage)

        return eval_mapper_cls(actx,
                dep_expr=self.input_exprs[idomain],
                other_dep_exprs=(
                    self.input_exprs[:idomain]
                    + self.input_exprs[idomain+1:]),
                dep_source=dep_source,
                dep_discr=dep_discr,
                places=places,
                tgt_src_index=tgt_src_index,
                context=self.context,
                **kwargs)(expr)

    # {{{ nearfield

    def evaluate_source_neighbor_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData, nbrindex: IndexList, *,
            ibrow: int, ibcol: int) -> Tuple[np.ndarray, TargetAndSourceClusterList]:
        nbr_src_index = TargetAndSourceClusterList(nbrindex, pxy.srcindex)

        eval_mapper_cls = self.neighbor_cluster_builder
        expr = self.exprs[ibrow]
        mat = self._evaluate_expr(
                actx, places, eval_mapper_cls, nbr_src_index, expr,
                idomain=ibcol, _weighted=self.weighted_sources)

        return mat, nbr_src_index

    def evaluate_target_neighbor_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData, nbrindex: IndexList, *,
            ibrow: int, ibcol: int) -> Tuple[np.ndarray, TargetAndSourceClusterList]:
        tgt_nbr_index = TargetAndSourceClusterList(pxy.srcindex, nbrindex)

        eval_mapper_cls = self.neighbor_cluster_builder
        expr = self.exprs[ibrow]
        mat = self._evaluate_expr(
                actx, places, eval_mapper_cls, tgt_nbr_index, expr,
                idomain=ibcol, _weighted=self.weighted_sources)

        return mat, tgt_nbr_index

    # }}}

    # {{{ proxy

    def evaluate_source_proxy_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData, nbrindex: IndexList, *,
            ibrow: int, ibcol: int) -> Tuple[np.ndarray, TargetAndSourceClusterList]:
        from pytential.collection import add_geometry_to_collection
        pxy_src_index = TargetAndSourceClusterList(pxy.pxyindex, pxy.srcindex)
        places = add_geometry_to_collection(
                places, {PROXY_SKELETONIZATION_TARGET: pxy.as_targets()}
                )

        eval_mapper_cls = self.proxy_source_cluster_builder
        expr = self.source_proxy_exprs[ibrow]
        mat = self._evaluate_expr(
                actx, places, eval_mapper_cls, pxy_src_index, expr,
                idomain=ibcol,
                _weighted=self.weighted_sources,
                exclude_self=False)

        return mat, pxy_src_index

    def evaluate_target_proxy_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData, nbrindex: IndexList, *,
            ibrow: int, ibcol: int) -> Tuple[np.ndarray, TargetAndSourceClusterList]:
        from pytential.collection import add_geometry_to_collection
        tgt_pxy_index = TargetAndSourceClusterList(pxy.srcindex, pxy.pxyindex)
        places = add_geometry_to_collection(
                places, {PROXY_SKELETONIZATION_SOURCE: pxy.as_sources()}
                )

        eval_mapper_cls = self.proxy_target_cluster_builder
        expr = self.target_proxy_exprs[ibrow]
        mat = self._evaluate_expr(
                actx, places, eval_mapper_cls, tgt_pxy_index, expr,
                idomain=ibcol,
                _weighted=False,
                exclude_self=False)

        if self.weighted_targets:
            mat = _apply_weights(
                    actx, mat, places,
                    tgt_pxy_index, nbrindex, self.domains[ibcol])

        return mat, tgt_pxy_index

    # }}}


def make_skeletonization_wrangler(
        places: GeometryCollection,
        exprs: Union[sym.var, Iterable[sym.var]],
        input_exprs: Union[sym.var, Iterable[sym.var]], *,
        domains: Optional[Iterable[Hashable]] = None,
        context: Optional[Dict[str, Any]] = None,
        auto_where: Optional[Union[Hashable, Tuple[Hashable, Hashable]]] = None,

        # internal
        _weighted_proxy: Optional[Union[bool, Tuple[bool, bool]]] = None,
        _proxy_source_cluster_builder: Optional[Callable[..., np.ndarray]] = None,
        _proxy_target_cluster_builder: Optional[Callable[..., np.ndarray]] = None,
        _neighbor_cluster_builder: Optional[Callable[..., np.ndarray]] = None):
    if context is None:
        context = {}

    # {{{ setup expressions

    try:
        exprs = list(exprs)
    except TypeError:
        exprs = [exprs]

    try:
        input_exprs = list(input_exprs)
    except TypeError:
        input_exprs = [input_exprs]

    from pytential.symbolic.execution import _prepare_auto_where
    auto_where = _prepare_auto_where(auto_where, places)
    from pytential.symbolic.execution import _prepare_domains
    domains = _prepare_domains(len(input_exprs), places, domains, auto_where[0])

    exprs = _prepare_neighbor_expr(places, exprs, auto_where)
    source_proxy_exprs = prepare_proxy_expr(
            places, exprs, (auto_where[0], PROXY_SKELETONIZATION_TARGET))
    target_proxy_exprs = prepare_proxy_expr(
            places, exprs, (PROXY_SKELETONIZATION_SOURCE, auto_where[1]))

    # }}}

    # {{{ weighting

    if _weighted_proxy is None:
        weighted_sources = weighted_targets = True
    elif isinstance(_weighted_proxy, bool):
        weighted_sources = weighted_targets = _weighted_proxy
    elif isinstance(_weighted_proxy, tuple):
        weighted_sources, weighted_targets = _weighted_proxy
    else:
        raise ValueError(f"unknown value for weighting: '{_weighted_proxy}'")

    # }}}

    # {{{ builders

    from pytential.symbolic.matrix import (
            P2PClusterMatrixBuilder, QBXClusterMatrixBuilder)

    if _neighbor_cluster_builder is None:
        _neighbor_cluster_builder = QBXClusterMatrixBuilder

    if _proxy_source_cluster_builder is None:
        _proxy_source_cluster_builder = P2PClusterMatrixBuilder

    if _proxy_target_cluster_builder is None:
        _proxy_target_cluster_builder = QBXClusterMatrixBuilder

    # }}}

    return SkeletonizationWrangler(
            # operator
            exprs=exprs,
            input_exprs=tuple(input_exprs),
            domains=tuple(domains),
            context=context,
            neighbor_cluster_builder=_neighbor_cluster_builder,
            # source
            weighted_sources=weighted_sources,
            source_proxy_exprs=source_proxy_exprs,
            proxy_source_cluster_builder=_proxy_source_cluster_builder,
            # target
            weighted_targets=weighted_targets,
            target_proxy_exprs=target_proxy_exprs,
            proxy_target_cluster_builder=_proxy_target_cluster_builder,
            )

# }}}


# {{{ skeletonize_block_by_proxy

@dataclass(frozen=True)
class _ProxyNeighborEvaluationResult:
    pxy: ProxyClusterGeometryData

    pxymat: np.ndarray
    pxyindex: TargetAndSourceClusterList

    nbrmat: np.ndarray
    nbrindex: TargetAndSourceClusterList

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        shape = self.nbrindex.cluster_shape(i, i)
        nbrmat_i = self.nbrindex.flat_cluster_take(self.nbrmat, i).reshape(*shape)

        shape = self.pxyindex.cluster_shape(i, i)
        pxymat_i = self.pxyindex.flat_cluster_take(self.pxymat, i).reshape(*shape)

        return [pxymat_i, nbrmat_i]


def _make_block_proxy_skeleton(
        actx, ibrow, ibcol,
        places, proxy_generator, wrangler, cluster_index,
        evaluate_proxy, evaluate_neighbor,
        max_particles_in_box=None):
    """Builds a block matrix that can be used to skeletonize the
    rows (targets) or columns (sources) of the symbolic matrix block
    described by ``(ibrow, ibcol)``.
    """

    if cluster_index.nclusters == 1:
        raise ValueError("cannot make a proxy skeleton for a single cluster")

    # {{{ generate proxies

    domain = wrangler.domains[ibcol]
    pxy = proxy_generator(actx, domain, cluster_index)

    # }}}

    # {{{ evaluate proxy and neighbor interactions

    from pytential.linalg import gather_cluster_neighbor_points
    nbrindex = gather_cluster_neighbor_points(
            actx, pxy,
            max_particles_in_box=max_particles_in_box)

    pxymat, pxy_geo_index = evaluate_proxy(
            actx, places, pxy, nbrindex, ibrow=ibrow, ibcol=ibcol)
    nbrmat, nbr_geo_index = evaluate_neighbor(
            actx, places, pxy, nbrindex, ibrow=ibrow, ibcol=ibcol)

    # }}}

    return _ProxyNeighborEvaluationResult(
            pxy=pxy,
            pxymat=pxymat, pxyindex=pxy_geo_index,
            nbrmat=nbrmat, nbrindex=nbr_geo_index)


def _skeletonize_block_by_proxy_with_mats(
        actx: PyOpenCLArrayContext, ibrow: int, ibcol: int,
        places: GeometryCollection,
        proxy_generator: ProxyGeneratorBase,
        wrangler: SkeletonizationWrangler,
        tgt_src_index: TargetAndSourceClusterList, *,
        id_eps: Optional[float] = None, id_rank: Optional[int] = None,
        max_particles_in_box: Optional[int] = None
        ):
    nclusters = tgt_src_index.nclusters
    if nclusters == 1:
        raise ValueError("cannot make a proxy skeleton for a single cluster")

    # construct proxy matrices to skeletonize
    from functools import partial
    make_proxy_skeleton = partial(_make_block_proxy_skeleton,
            actx, ibrow, ibcol, places, proxy_generator, wrangler,
            max_particles_in_box=max_particles_in_box)

    src_result = make_proxy_skeleton(
            tgt_src_index.sources,
            evaluate_proxy=wrangler.evaluate_source_proxy_interaction,
            evaluate_neighbor=wrangler.evaluate_source_neighbor_interaction,
            )
    tgt_result = make_proxy_skeleton(
            tgt_src_index.targets,
            evaluate_proxy=wrangler.evaluate_target_proxy_interaction,
            evaluate_neighbor=wrangler.evaluate_target_neighbor_interaction,
            )

    src_skl_indices = np.empty(nclusters, dtype=object)
    tgt_skl_indices = np.empty(nclusters, dtype=object)
    skel_starts = np.zeros(nclusters + 1, dtype=np.int32)

    L = np.full((nclusters, nclusters), 0, dtype=object)
    R = np.full((nclusters, nclusters), 0, dtype=object)

    from pytential.linalg import interp_decomp
    for i in range(nclusters):
        k = id_rank
        src_mat = np.vstack(src_result[i])
        tgt_mat = np.hstack(tgt_result[i])

        assert np.all(np.isfinite(tgt_mat)), np.where(np.isfinite(tgt_mat))
        assert np.all(np.isfinite(src_mat)), np.where(np.isfinite(src_mat))

        # skeletonize target points
        k, idx, interp = interp_decomp(tgt_mat.T, k, id_eps)
        assert k > 0

        L[i, i] = interp.T
        tgt_skl_indices[i] = tgt_src_index.targets.cluster_indices(i)[idx[:k]]

        # skeletonize source points
        k, idx, interp = interp_decomp(src_mat, k, id_eps)
        assert k > 0

        R[i, i] = interp
        src_skl_indices[i] = tgt_src_index.sources.cluster_indices(i)[idx[:k]]

        skel_starts[i + 1] = skel_starts[i] + k
        assert R[i, i].shape == (k, src_mat.shape[1])
        assert L[i, i].shape == (tgt_mat.shape[0], k)

    from pytential.linalg import make_index_list
    src_skl_index = make_index_list(np.hstack(src_skl_indices), skel_starts)
    tgt_skl_index = make_index_list(np.hstack(tgt_skl_indices), skel_starts)
    skel_tgt_src_index = TargetAndSourceClusterList(tgt_skl_index, src_skl_index)

    return L, R, skel_tgt_src_index, src_result, tgt_result

# }}}


# {{{ skeletonize_by_proxy

@dataclass(frozen=True)
class SkeletonizationResult:
    r"""Result of a skeletonization procedure.

    A matrix :math:`A` can be reconstructed using:

    .. math::

        A \approx L S R

    where :math:`S = A_{I, J}` for a subset :math:`I` and :math:`J` of the
    rows and columns of :math:`A`, respectively. This applies to each cluster
    in :attr:`tgt_src_index`.

    .. attribute:: nclusters

        Number of clusters that have been skeletonized.

    .. attribute:: L

        A block diagonal :class:`~numpy.ndarray` object array.

    .. attribute:: R

        A block diagonal :class:`~numpy.ndarray` object array.

    .. attribute:: tgt_src_index

        A :class:`~pytential.linalg.TargetAndSourceClusterList` representing the
        indices in the original matrix :math:`A` that have been skeletonized.

    .. attribute:: skel_tgt_src_index

        A :class:`~pytential.linalg.TargetAndSourceClusterList` representing a
        subset of :attr:`tgt_src_index` of the matrix :math:`S`, i.e. the
        skeleton of each cluster of :math:`A`.
    """

    L: np.ndarray
    R: np.ndarray

    tgt_src_index: TargetAndSourceClusterList
    skel_tgt_src_index: TargetAndSourceClusterList

    def __post_init__(self):
        if __debug__:
            nclusters = self.tgt_src_index.nclusters
            shape = (nclusters, nclusters)

            if self.tgt_src_index.nclusters != self.skel_tgt_src_index.nclusters:
                raise ValueError("'tgt_src_index' and 'skel_tgt_src_index' have "
                        f"different number of blocks: {self.tgt_src_index.nclusters}"
                        f" vs {self.skel_tgt_src_index.nclusters}")

            if self.L.shape != shape:
                raise ValueError(f"'L' has shape {self.L.shape}, expected {shape}")

            if self.R.shape != shape:
                raise ValueError(f"'R' has shape {self.R.shape}, expected {shape}")

    @property
    def nclusters(self):
        return self.tgt_src_index.nclusters


def skeletonize_by_proxy(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection,

        tgt_src_index: TargetAndSourceClusterList,
        exprs: Union[sym.var, Iterable[sym.var]],
        input_exprs: Union[sym.var, Iterable[sym.var]], *,
        domains: Optional[Iterable[Hashable]] = None,
        context: Optional[Dict[str, Any]] = None,

        approx_nproxy: Optional[int] = None,
        proxy_radius_factor: Optional[float] = None,

        id_eps: Optional[float] = None,
        id_rank: Optional[int] = None,
        max_particles_in_box: Optional[int] = None) -> np.ndarray:
    r"""Evaluate and skeletonize a symbolic expression using proxy-based methods.

    :arg tgt_src_index: a :class:`~pytential.linalg.TargetAndSourceClusterList`
        indicating which indices participate in the skeletonization.

    :arg exprs: see :func:`make_skeletonization_wrangler`.
    :arg input_exprs: see :func:`make_skeletonization_wrangler`.
    :arg domains: see :func:`make_skeletonization_wrangler`.
    :arg context: see :func:`make_skeletonization_wrangler`.

    :arg approx_nproxy: see :class:`~pytential.linalg.ProxyGenerator`.
    :arg proxy_radius_factor: see :class:`~pytential.linalg.ProxyGenerator`.

    :arg id_eps: a floating point value used as a tolerance in skeletonizing
        each block in *tgt_src_index*.
    :arg id_rank: an alternative to *id_eps*, which fixes the rank of each
        skeletonization.
    :arg max_particles_in_box: passed to :class:`boxtree.TreeBuilder` as necessary.

    :returns: an :class:`~numpy.ndarray` of :class:`SkeletonizationResult` of
        shape ``(len(exprs), len(input_exprs))``.
    """
    from pytential.linalg.proxy import QBXProxyGenerator
    wrangler = make_skeletonization_wrangler(
            places, exprs, input_exprs,
            domains=domains, context=context)
    proxy = QBXProxyGenerator(places,
            approx_nproxy=approx_nproxy,
            radius_factor=proxy_radius_factor)

    skels = np.empty((wrangler.nrows, wrangler.ncols), dtype=object)
    for ibrow in range(wrangler.nrows):
        for ibcol in range(wrangler.ncols):
            L, R, skel_tgt_src_index, _, _ = _skeletonize_block_by_proxy_with_mats(
                    actx, ibrow, ibcol, places, proxy, wrangler, tgt_src_index,
                    id_eps=id_eps, id_rank=id_rank,
                    max_particles_in_box=max_particles_in_box)

            skels[ibrow, ibcol] = SkeletonizationResult(
                    L=L, R=R,
                    tgt_src_index=tgt_src_index,
                    skel_tgt_src_index=skel_tgt_src_index)

    return skels

# }}}
