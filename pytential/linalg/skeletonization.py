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

from arraycontext import PyOpenCLArrayContext, Array, ArrayT

from pytential import GeometryCollection, sym
from pytential.linalg.utils import IndexList, TargetAndSourceClusterList
from pytential.linalg.proxy import ProxyGeneratorBase, ProxyClusterGeometryData
from pytential.linalg.direct_solver_symbolic import (
        PROXY_SKELETONIZATION_TARGET, PROXY_SKELETONIZATION_SOURCE,
        prepare_expr, prepare_proxy_expr)


__doc__ = """
.. currentmodule:: pytential.linalg

Skeletonization
---------------

.. autoclass:: SkeletonizationWrangler
.. autoclass:: make_skeletonization_wrangler

.. autoclass:: SkeletonizationResult
.. autofunction:: skeletonize_by_proxy
"""


# {{{ wrangler

def _approximate_geometry_waa_magnitude(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection,
        cluster_index: "IndexList",
        domain: sym.DOFDescriptor) -> Array:
    """
    :arg cluster_index: a :class:`~pytential.linalg.IndexList` representing the
        clusters on which to operate. In practice, this can be the cluster
        indices themselves or the indices of neighboring points to each
        cluster (i.e. points inside the proxy ball).

    :returns: an array of size ``(nclusters,)`` with a characteristic magnitude
        of the quadrature weights and area elements in the cluster. Currently,
        this is a simple average.
    """
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
    result = actx.zeros((cluster_index.nclusters,), dtype=waa.entry_dtype)

    from arraycontext import flatten
    _, (waa_per_cluster,) = prg()(actx.queue,         # pylint: disable=not-callable
            waa=flatten(waa, actx),
            result=result,
            indices=cluster_index.indices,
            starts=cluster_index.starts)

    return waa_per_cluster


def _apply_weights(
        actx: PyOpenCLArrayContext,
        mat: ArrayT,
        places: GeometryCollection,
        tgt_pxy_index: TargetAndSourceClusterList,
        cluster_index: IndexList,
        domain: sym.DOFDescriptor) -> ArrayT:
    """Computes the weights using :func:`_approximate_geometry_waa_magnitude`
    and multiplies each cluster in *mat* by its corresponding weight.

    :returns: *mat* multiplied by the weights.
    """
    assert tgt_pxy_index.nclusters == cluster_index.nclusters
    waa = actx.to_numpy(
            _approximate_geometry_waa_magnitude(actx, places, cluster_index, domain)
            )

    result = actx.np.zeros_like(mat)
    for i in range(tgt_pxy_index.nclusters):
        istart, iend = tgt_pxy_index._flat_cluster_starts[i:i + 2]
        result[istart:iend] = mat[istart:iend] * waa[i]

    return result


@dataclass(frozen=True)
class SkeletonizationWrangler:
    """
    .. attribute:: nrows

        Number of output :attr:`exprs` in the operator.

    .. attribute:: ncols

        Number of :attr:`input_exprs` in the operator.

    .. attribute:: exprs

        An :class:`~numpy.ndarray` of shape ``(nrows,)`` of expressions
        (layer potentials) that correspond to the output blocks of the matrix.
        These expressions are tagged for nearfield neighbor evalution.

    .. attribute:: source_proxy_exprs
    .. attribute:: target_proxy_exprs

        Like :attr:`exprs`, but stripped down for farfield proxy evaluation.

    .. attribute:: input_exprs

        A :class:`tuple` of size ``(ncols,)`` of densities that correspond to
        the input blocks of the matrix.

    .. attribute:: domains

        A :class:`tuple` of the same length as *input_exprs* defining the
        domain of each input.

    .. attribute:: context

        A :class:`dict` with additional parameters required to evaluate the
        expressions.

    The following attributes and methods are internal and used for skeletonization.

    .. attribute:: weighted_sources

        A flag which if *True* adds a weight to the source to proxy evaluation.
        This can only be meaningfully set to *False* when skeletonizing direct
        P2P interactions.

    .. attribute:: weighted_targets

        A flag which if *True* adds a weight to the proxy to target evaluation.
        This can only be meaningfully set to *False* when skeletonizing direct
        P2P interactions.

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
    domains: Tuple[sym.DOFDescriptor, ...]
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
    def nrows(self) -> int:
        return len(self.exprs)

    @property
    def ncols(self) -> int:
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
                idomain=ibcol, _weighted=self.weighted_targets)

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
        _neighbor_cluster_builder: Optional[Callable[..., np.ndarray]] = None,
        ) -> SkeletonizationWrangler:
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

    exprs = prepare_expr(places, exprs, auto_where)
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
    """
    .. attribute:: pxy

        A :class:`~pytential.linalg.ProxyClusterGeometryData` containing the
        proxy points from which :attr:`pxymat` is obtained. This data is also
        used to construct :attr:`nbrindex` and evaluate :attr:`nbrmat`.

    .. attribute:: pxymat

        Interaction matrix between the proxy points and the source or
        target points. This matrix is flattened to a shape of ``(nsize,)``,
        which is consistent with the sum of the cluster sizes in :attr:`pxyindex`,
        as obtained from
        :meth:`~pytential.linalg.TargetAndSourceClusterList.cluster_size`.

    .. attribute:: pxyindex

        A :class:`~pytential.linalg.TargetAndSourceClusterList` used to describe
        the cluster interactions in :attr:`pxymat`.

    .. attribute:: nbrmat

        Interaction matrix between the neighboring points and the source or
        target points. This matrix is flattened to a shape of ``(nsize,)``,
        which is consistent with the sum of the cluster sizes in :attr:`nbrindex`,
        as obtained from
        :meth:`~pytential.linalg.TargetAndSourceClusterList.cluster_size`.

    .. attribute:: nbrindex

        A :class:`~pytential.linalg.TargetAndSourceClusterList` used to describe
        the cluster interactions in :attr:`nbrmat`.

    .. automethod:: __getitem__
    """

    pxy: ProxyClusterGeometryData

    pxymat: np.ndarray
    pxyindex: TargetAndSourceClusterList

    nbrmat: np.ndarray
    nbrindex: TargetAndSourceClusterList

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        :returns: a :class:`tuple` of ``(pxymat, nbrmat)`` containing the
            :math:`i`-th cluster interactions. The matrices are reshaped into
            their full sizes.
        """

        shape = self.nbrindex.cluster_shape(i, i)
        nbrmat_i = self.nbrindex.flat_cluster_take(self.nbrmat, i).reshape(*shape)

        shape = self.pxyindex.cluster_shape(i, i)
        pxymat_i = self.pxyindex.flat_cluster_take(self.pxymat, i).reshape(*shape)

        return pxymat_i, nbrmat_i


def _evaluate_proxy_skeletonization_interaction(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection,
        proxy_generator: ProxyGeneratorBase,
        wrangler: SkeletonizationWrangler,
        cluster_index: IndexList, *,
        evaluate_proxy: Callable[...,
            Tuple[np.ndarray, TargetAndSourceClusterList]],
        evaluate_neighbor: Callable[...,
            Tuple[np.ndarray, TargetAndSourceClusterList]],
        dofdesc: Optional[sym.DOFDescriptor] = None,
        max_particles_in_box: Optional[int] = None,
        ) -> _ProxyNeighborEvaluationResult:
    """Evaluate the proxy to cluster and neighbor to cluster interactions for
    each cluster in *cluster_index*.
    """

    if cluster_index.nclusters == 1:
        raise ValueError("cannot make a proxy skeleton for a single cluster")

    from pytential.linalg import gather_cluster_neighbor_points
    pxy = proxy_generator(actx, dofdesc, cluster_index)
    nbrindex = gather_cluster_neighbor_points(
            actx, pxy,
            max_particles_in_box=max_particles_in_box)

    pxymat, pxy_cluster_index = evaluate_proxy(actx, places, pxy, nbrindex)
    nbrmat, nbr_cluster_index = evaluate_neighbor(actx, places, pxy, nbrindex)

    return _ProxyNeighborEvaluationResult(
            pxy=pxy,
            pxymat=pxymat, pxyindex=pxy_cluster_index,
            nbrmat=nbrmat, nbrindex=nbr_cluster_index)


def _skeletonize_block_by_proxy_with_mats(
        actx: PyOpenCLArrayContext, ibrow: int, ibcol: int,
        places: GeometryCollection,
        proxy_generator: ProxyGeneratorBase,
        wrangler: SkeletonizationWrangler,
        tgt_src_index: TargetAndSourceClusterList, *,
        id_eps: Optional[float] = None, id_rank: Optional[int] = None,
        max_particles_in_box: Optional[int] = None
        ) -> "SkeletonizationResult":
    nclusters = tgt_src_index.nclusters
    if nclusters == 1:
        raise ValueError("cannot make a proxy skeleton for a single cluster")

    # construct proxy matrices to skeletonize
    from functools import partial
    evaluate_skeletonization_interaction = partial(
            _evaluate_proxy_skeletonization_interaction,
            actx, places, proxy_generator, wrangler,
            dofdesc=wrangler.domains[ibcol],
            max_particles_in_box=max_particles_in_box)

    src_result = evaluate_skeletonization_interaction(
            tgt_src_index.sources,
            evaluate_proxy=partial(
                wrangler.evaluate_source_proxy_interaction,
                ibrow=ibrow, ibcol=ibcol),
            evaluate_neighbor=partial(
                wrangler.evaluate_source_neighbor_interaction,
                ibrow=ibrow, ibcol=ibcol),
            )
    tgt_result = evaluate_skeletonization_interaction(
            tgt_src_index.targets,
            evaluate_proxy=partial(
                wrangler.evaluate_target_proxy_interaction,
                ibrow=ibrow, ibcol=ibcol),
            evaluate_neighbor=partial(
                wrangler.evaluate_target_neighbor_interaction,
                ibrow=ibrow, ibcol=ibcol)
            )

    src_skl_indices = np.empty(nclusters, dtype=object)
    tgt_skl_indices = np.empty(nclusters, dtype=object)
    skel_starts = np.zeros(nclusters + 1, dtype=np.int32)

    L = np.empty(nclusters, dtype=object)
    R = np.empty(nclusters, dtype=object)

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

        L[i] = interp.T
        tgt_skl_indices[i] = tgt_src_index.targets.cluster_indices(i)[idx[:k]]

        # skeletonize source points
        k, idx, interp = interp_decomp(src_mat, k, id_eps)
        assert k > 0

        R[i] = interp
        src_skl_indices[i] = tgt_src_index.sources.cluster_indices(i)[idx[:k]]

        skel_starts[i + 1] = skel_starts[i] + k
        assert R[i].shape == (k, src_mat.shape[1])
        assert L[i].shape == (tgt_mat.shape[0], k)

    from pytential.linalg import make_index_list
    src_skl_index = make_index_list(np.hstack(src_skl_indices), skel_starts)
    tgt_skl_index = make_index_list(np.hstack(tgt_skl_indices), skel_starts)
    skel_tgt_src_index = TargetAndSourceClusterList(tgt_skl_index, src_skl_index)

    return SkeletonizationResult(
            L=L, R=R,
            tgt_src_index=tgt_src_index, skel_tgt_src_index=skel_tgt_src_index,
            _src_eval_result=src_result, _tgt_eval_result=tgt_result)

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
    in :attr:`tgt_src_index`. In particular, for a cluster pair :math:`(i, j)`,
    we can reconstruct the matrix entries as follows

    .. code:: python

        Aij = tgt_src_index.cluster_take(A, i, j)
        Sij = skel_tgt_src_index.cluster_take(A, i, j)
        Aij_approx = L[i] @ Sij @ R[j]

    .. attribute:: nclusters

        Number of clusters that have been skeletonized.

    .. attribute:: L

        An object :class:`~numpy.ndarray` of size ``(nclusters,)``.

    .. attribute:: R

        An object :class:`~numpy.ndarray` of size ``(nclusters,)``.

    .. attribute:: tgt_src_index

        A :class:`~pytential.linalg.TargetAndSourceClusterList` representing the
        indices in the original matrix :math:`A` that have been skeletonized.

    .. attribute:: skel_tgt_src_index

        A :class:`~pytential.linalg.TargetAndSourceClusterList` representing a
        subset of :attr:`tgt_src_index`, i.e. the skeleton of each cluster of
        :math:`A`. These indices can be used to reconstruct the :math:`S`
        matrix.
    """

    L: np.ndarray
    R: np.ndarray

    tgt_src_index: TargetAndSourceClusterList
    skel_tgt_src_index: TargetAndSourceClusterList

    # NOTE: these are meant only for testing! They contain the interactions
    # between the source / target points and their proxies / neighbors.
    _src_eval_result: Optional[_ProxyNeighborEvaluationResult] = None
    _tgt_eval_result: Optional[_ProxyNeighborEvaluationResult] = None

    def __post_init__(self):
        if __debug__:
            nclusters = self.tgt_src_index.nclusters
            shape = (nclusters,)

            if self.tgt_src_index.nclusters != self.skel_tgt_src_index.nclusters:
                raise ValueError("'tgt_src_index' and 'skel_tgt_src_index' have "
                        f"different number of clusters: {nclusters}"
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
            skels[ibrow, ibcol] = _skeletonize_block_by_proxy_with_mats(
                    actx, ibrow, ibcol, places, proxy, wrangler, tgt_src_index,
                    id_eps=id_eps, id_rank=id_rank,
                    max_particles_in_box=max_particles_in_box)

    return skels

# }}}
