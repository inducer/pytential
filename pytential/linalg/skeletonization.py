from __future__ import annotations


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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from meshmode.discretization import Discretization
from pytools import log_process, memoize_in, obj_array

from pytential import GeometryCollection, bind, sym
from pytential.linalg.cluster import ClusterTree, cluster
from pytential.linalg.direct_solver_symbolic import (
    PROXY_SKELETONIZATION_SOURCE,
    PROXY_SKELETONIZATION_TARGET,
    prepare_expr,
    prepare_proxy_expr,
)
from pytential.linalg.utils import IndexList, TargetAndSourceClusterList


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    import optype.numpy as onp

    from arraycontext import Array, PyOpenCLArrayContext
    from pymbolic.typing import ArithmeticExpression

    from pytential.linalg.proxy import ProxyClusterGeometryData, ProxyGeneratorBase
    from pytential.symbolic.matrix import ClusterMatrixBuilderBase


logger = logging.getLogger(__name__)

__doc__ = """
Skeletonization
---------------

.. autoclass:: SkeletonizationWrangler
.. autoclass:: make_skeletonization_wrangler

.. autoclass:: SkeletonizationResult
.. autofunction:: skeletonize_by_proxy
.. autofunction:: rec_skeletonize_by_proxy
"""


# {{{ wrangler

# Adding weights to proxy evaluation: Why? How? Huh?
# --------------------------------------------------
#
# We have a couple of interaction evaluations that need to happen for the
# proxy-based skeletonization, namely
# 1. with the proxies + neighbors when the current cluster is the source
# 2. with the proxies + neighbors when the current cluster is the target
#
# The operator that we want to skeletonize at the end of the day is the full
# layer potential, that contains quadrature weights and area elements. That
# means that it is in our interest to include some of that information in
# the proxy + neighbor interaction matrices that get passed to the ID
# (interpolative decomposition). This helps on two fronts:
# * first, it gives the ID a better chance at guessing a good range. This is more
#   important for the nearfield than the farfield.
# * second, as the ID constructs an approximation until a relative error is
#   reached, columns with a larger norm will be preferred. However, since
#   we ultimately reconstruct both nearfield and farfield, we have no reason to
#   weigh the proxy or neighbor interactions more heavily. Ideally both sets of
#   interactions are weighted similarly, which is what we try to achieve below.
#
# In the first case, the geometry is the source and we can directly
# right multiply with the same weights, e.g.
#
#   P_w <- P @ W
#   N_w <- N @ W
#
# where `P` and `N` just evaluate the direct proxy interactions (P2P) and
# the neighbor interactions (QBX) without any other changes. Then, the matrix
# `hstack([P_w, N_w])` is the one that gets ID-ed. As the proxies and neighbors
# are at a comparable distance from the cluster, multiplying both by the same
# weights is expected to ensure similar norms. Adding these weights is
# controlled by `SkeletonizationWrangler.weighted_sources`.
#
# In the second case, the source points are either the proxy points or the
# neighboring points. For neighboring points, we retrieve the weights on the
# geometry (same as above). However, for the proxy points we generally do not
# construct (or wish to) a full discretization with a quadrature scheme. From a
# linear algebra point of view, we want to find a matrix `W_p` in
#
#   P_w <- P @ W_p
#   N_w <- N @ W
#
# such that `P_w` and `N_w` have comparable norms. For now, we approximate `W_p`
# by taking the average of `W`, i.e. `W_p <- avg(W) * I`. As `W` is diagonal,
# we could easily approximate the `\ell^2` or Frobenius norm, but the average
# is thought of as reasonable compromise. Adding these weights is controlled by
# `SkeletonizationWrangler.weighted_targets` and computed below in
# `_approximate_geometry_waa_magnitude`.


def _approximate_geometry_waa_magnitude(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection,
        cluster_index: IndexList,
        domain: sym.DOFDescriptor) -> Array:
    """
    :arg cluster_index: a :class:`~pytential.linalg.utils.IndexList` representing
        the clusters on which to operate. In practice, this can be the cluster
        indices themselves or the indices of neighboring points to each
        cluster (i.e. points inside the proxy ball).

    :returns: an array of size ``(nclusters,)`` with a characteristic magnitude
        of the quadrature weights and area elements in the cluster. Currently,
        this is a simple average.
    """
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
            result[icluster] = (
                reduce(sum, i, abs(waa[indices[i + ioffset]])) / npoints
                if npoints > 0 else 1.0)
            """,
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            )

        return knl.executor(actx.context)

    from meshmode.dof_array import DOFArray
    waa = bind(places, sym.weights_and_area_elements(
        places.ambient_dim, dofdesc=domain))(actx)
    assert isinstance(waa, DOFArray)
    result = actx.np.zeros((cluster_index.nclusters,), dtype=waa.entry_dtype)

    from arraycontext import flatten
    _, (waa_per_cluster,) = prg()(actx.queue,
            waa=flatten(waa, actx),
            result=result,
            indices=cluster_index.indices,
            starts=cluster_index.starts)

    return waa_per_cluster


def _apply_weights(
        actx: PyOpenCLArrayContext,
        mat: onp.Array1D[np.inexact],
        places: GeometryCollection,
        tgt_pxy_index: TargetAndSourceClusterList,
        cluster_index: IndexList,
        domain: sym.DOFDescriptor) -> onp.Array1D[np.inexact]:
    """Computes the weights using :func:`_approximate_geometry_waa_magnitude`
    and multiplies each cluster in *mat* by its corresponding weight.

    :returns: *mat* multiplied by the weights.
    """
    assert tgt_pxy_index.nclusters == cluster_index.nclusters
    waa = actx.to_numpy(
            _approximate_geometry_waa_magnitude(actx, places, cluster_index, domain)
            )

    result = np.zeros_like(mat)
    for i in range(tgt_pxy_index.nclusters):
        istart, iend = tgt_pxy_index._flat_cluster_starts[i:i + 2]
        result[istart:iend] = mat[istart:iend] * waa[i]

    return result


@dataclass(frozen=True)
class SkeletonizationWrangler:
    """
    .. autoproperty:: nrows
    .. autoproperty:: ncols
    .. autoattribute:: exprs
    .. autoattribute:: source_proxy_exprs
    .. autoattribute:: target_proxy_exprs
    .. autoattribute:: input_exprs
    .. autoattribute:: domains
    .. autoattribute:: context

    The following attributes and methods are internal and used for skeletonization.

    .. autoattribute:: weighted_sources
    .. autoattribute:: weighted_targets

    .. autoattribute:: proxy_source_cluster_builder
    .. autoattribute:: proxy_target_cluster_builder
    .. autoattribute:: neighbor_cluster_builder

    .. automethod:: evaluate_source_proxy_interaction
    .. automethod:: evaluate_target_proxy_interaction
    .. automethod:: evaluate_source_neighbor_interaction
    .. automethod:: evaluate_target_neighbor_interaction
    """

    # operator
    exprs: obj_array.ObjectArray1D[ArithmeticExpression]
    """An :class:`~numpy.ndarray` of shape ``(nrows,)`` of expressions
    (layer potentials) that correspond to the output blocks of the operator.
    These expressions must be tagged for nearfield neighbor evaluation.
    """
    input_exprs: tuple[sym.var, ...]
    """A :class:`tuple` of size ``(ncols,)`` of densities that correspond to
    the input blocks of the matrix.
    """
    domains: tuple[sym.DOFDescriptor, ...]
    """A :class:`tuple` of the same length as *input_exprs* defining the
    domain of each input.
    """
    context: dict[str, Any]
    """A :class:`dict` with additional parameters required to evaluate the
    expressions.
    """

    # skeletonization
    neighbor_cluster_builder: type[ClusterMatrixBuilderBase]
    """A callable that is used to evaluate nearfield neighbour interactions.
    This should follow the calling convention of the constructor to
    :class:`pytential.symbolic.matrix.QBXClusterMatrixBuilder`.
    """

    # target skeletonization
    weighted_targets: bool
    """A flag which if *True* adds a weight to the proxy to target evaluation.
    This can only be meaningfully set to *False* when skeletonizing direct
    P2P interactions.
    """
    target_proxy_exprs: obj_array.ObjectArray1D[ArithmeticExpression]
    """Like :attr:`exprs`, but stripped down for farfield proxy evaluation."""
    proxy_target_cluster_builder: type[ClusterMatrixBuilderBase]
    """A callable that is used to evaluate farfield proxy interactions.
    This should follow the calling convention of the constructor to
    :class:`pytential.symbolic.matrix.P2PClusterMatrixBuilder`.
    """

    # source skeletonization
    weighted_sources: bool
    """A flag which if *True* adds a weight to the source to proxy evaluation.
    This can only be meaningfully set to *False* when skeletonizing direct
    P2P interactions.
    """
    source_proxy_exprs: obj_array.ObjectArray1D[ArithmeticExpression]
    """Like :attr:`exprs`, but stripped down for farfield proxy evaluation."""
    proxy_source_cluster_builder: type[ClusterMatrixBuilderBase]
    """A callable that is used to evaluate farfield proxy interactions.
    This should follow the calling convention of the constructor to
    :class:`pytential.symbolic.matrix.P2PClusterMatrixBuilder`.
    """

    @property
    def nrows(self) -> int:
        """Number of output :attr:`exprs` in the operator."""
        return len(self.exprs)

    @property
    def ncols(self) -> int:
        """Number of :attr:`input_exprs` in the operator."""
        return len(self.input_exprs)

    def _evaluate_expr(
            self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            eval_mapper_cls: type[ClusterMatrixBuilderBase],
            tgt_src_index: TargetAndSourceClusterList,
            expr: ArithmeticExpression,
            idomain: int,
            **kwargs: Any) -> onp.Array1D[np.inexact]:
        domain = self.domains[idomain]
        dep_discr = places.get_discretization(domain.geometry, domain.discr_stage)
        assert isinstance(dep_discr, Discretization)

        return eval_mapper_cls(
                actx,
                dep_expr=self.input_exprs[idomain],
                other_dep_exprs=(
                    self.input_exprs[:idomain]
                    + self.input_exprs[idomain+1:]),
                dep_discr=dep_discr,
                places=places,
                tgt_src_index=tgt_src_index,
                context=self.context,
                **kwargs)(expr)

    def evaluate_self(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            tgt_src_index: TargetAndSourceClusterList,
            ibrow: int, ibcol: int,
            ) -> onp.Array1D[Any]:
        cls = self.neighbor_cluster_builder
        return self._evaluate_expr(
            actx, places, cls, tgt_src_index, self.exprs[ibrow],
            idomain=ibcol, _weighted=True)

    # {{{ nearfield

    @log_process(logger)
    def evaluate_source_neighbor_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData,
            nbrindex: IndexList, *,
            ibrow: int, ibcol: int,
        ) -> tuple[onp.Array1D[np.inexact], TargetAndSourceClusterList]:
        nbr_src_index = TargetAndSourceClusterList(nbrindex, pxy.srcindex)

        eval_mapper_cls = self.neighbor_cluster_builder
        expr = self.exprs[ibrow]
        mat = self._evaluate_expr(
                actx, places, eval_mapper_cls, nbr_src_index, expr,
                idomain=ibcol, _weighted=True)

        return mat, nbr_src_index

    @log_process(logger)
    def evaluate_target_neighbor_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData,
            nbrindex: IndexList, *,
            ibrow: int, ibcol: int,
        ) -> tuple[onp.Array1D[np.inexact], TargetAndSourceClusterList]:
        tgt_nbr_index = TargetAndSourceClusterList(pxy.srcindex, nbrindex)

        eval_mapper_cls = self.neighbor_cluster_builder
        expr = self.exprs[ibrow]
        mat = self._evaluate_expr(
                actx, places, eval_mapper_cls, tgt_nbr_index, expr,
                idomain=ibcol, _weighted=True)

        return mat, tgt_nbr_index

    # }}}

    # {{{ proxy

    @log_process(logger)
    def evaluate_source_proxy_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData,
            nbrindex: IndexList, *,
            ibrow: int, ibcol: int,
        ) -> tuple[onp.Array1D[np.inexact], TargetAndSourceClusterList]:
        from pytential.collection import add_geometry_to_collection
        pxy_src_index = TargetAndSourceClusterList(pxy.pxyindex, pxy.srcindex)
        places = add_geometry_to_collection(
                places, {PROXY_SKELETONIZATION_TARGET: pxy.as_targets()}
                )

        if not self.weighted_sources:
            logger.warning("Source-Proxy weighting is turned off. This will not give "
                           "good results for skeletonization.", stacklevel=3)

        eval_mapper_cls = self.proxy_source_cluster_builder
        expr = self.source_proxy_exprs[ibrow]
        mat = self._evaluate_expr(
                actx, places, eval_mapper_cls, pxy_src_index, expr,
                idomain=ibcol,
                _weighted=self.weighted_sources,
                exclude_self=False)

        return mat, pxy_src_index

    @log_process(logger)
    def evaluate_target_proxy_interaction(self,
            actx: PyOpenCLArrayContext,
            places: GeometryCollection,
            pxy: ProxyClusterGeometryData, nbrindex: IndexList, *,
            ibrow: int, ibcol: int,
        ) -> tuple[onp.Array1D[np.inexact], TargetAndSourceClusterList]:
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
        else:
            logger.warning("Target-Proxy weighting is turned off. This will not give "
                           "good results for skeletonization.", stacklevel=3)

        return mat, tgt_pxy_index

    # }}}


def make_skeletonization_wrangler(
        places: GeometryCollection,
        exprs: ArithmeticExpression | Sequence[ArithmeticExpression],
        input_exprs: sym.var | Sequence[sym.var], *,
        domains: Sequence[Hashable] | None = None,
        context: dict[str, Any] | None = None,
        auto_where: Hashable | tuple[Hashable, Hashable] | None = None,

        # internal
        _weighted_proxy: bool | tuple[bool, bool] | None = None,
        _proxy_source_cluster_builder: type[ClusterMatrixBuilderBase] | None = None,
        _proxy_target_cluster_builder: type[ClusterMatrixBuilderBase] | None = None,
        _neighbor_cluster_builder: type[ClusterMatrixBuilderBase] | None = None,
        ) -> SkeletonizationWrangler:
    if context is None:
        context = {}

    # {{{ setup expressions

    from pymbolic.primitives import is_arithmetic_expression

    if is_arithmetic_expression(exprs):
        lpot_exprs = [exprs]
    else:
        lpot_exprs = list(exprs)

    if isinstance(input_exprs, sym.var):
        input_exprs = [input_exprs]
    else:
        input_exprs = list(input_exprs)

    from pytential.symbolic.execution import _prepare_auto_where, _prepare_domains

    auto_where = _prepare_auto_where(auto_where, places)
    domains = _prepare_domains(len(input_exprs), places, domains, auto_where[0])

    prepared_lpot_exprs = prepare_expr(places, lpot_exprs, auto_where)
    source_proxy_exprs = prepare_proxy_expr(
            places, prepared_lpot_exprs, (auto_where[0], PROXY_SKELETONIZATION_TARGET))
    target_proxy_exprs = prepare_proxy_expr(
            places, prepared_lpot_exprs, (PROXY_SKELETONIZATION_SOURCE, auto_where[1]))

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
        P2PClusterMatrixBuilder,
        QBXClusterMatrixBuilder,
    )

    neighbor_cluster_builder = _neighbor_cluster_builder
    if neighbor_cluster_builder is None:
        neighbor_cluster_builder = QBXClusterMatrixBuilder

    proxy_source_cluster_builder = _proxy_source_cluster_builder
    if proxy_source_cluster_builder is None:
        proxy_source_cluster_builder = P2PClusterMatrixBuilder

    proxy_target_cluster_builder = _proxy_target_cluster_builder
    if proxy_target_cluster_builder is None:
        proxy_target_cluster_builder = QBXClusterMatrixBuilder

    # }}}

    return SkeletonizationWrangler(
            # operator
            exprs=prepared_lpot_exprs,
            input_exprs=tuple(input_exprs),
            domains=tuple(domains),
            context=context,
            neighbor_cluster_builder=neighbor_cluster_builder,
            # source
            weighted_sources=weighted_sources,
            source_proxy_exprs=source_proxy_exprs,
            proxy_source_cluster_builder=proxy_source_cluster_builder,
            # target
            weighted_targets=weighted_targets,
            target_proxy_exprs=target_proxy_exprs,
            proxy_target_cluster_builder=proxy_target_cluster_builder,
            )

# }}}


# {{{ skeletonize_block_by_proxy

@dataclass(frozen=True)
class _ProxyNeighborEvaluationResult:
    """
    .. autoattribute:: pxy
    .. autoattribute:: pxymat
    .. autoattribute:: pxyindex
    .. autoattribute:: nbrmat
    .. autoattribute:: nbrindex

    .. automethod:: __getitem__
    """

    pxy: ProxyClusterGeometryData
    """A :class:`~pytential.linalg.utils.ProxyClusterGeometryData` containing the
    proxy points from which :attr:`pxymat` is obtained. This data is also
    used to construct :attr:`nbrindex` and evaluate :attr:`nbrmat`.
    """
    pxymat: onp.Array1D[np.inexact]
    """Interaction matrix between the proxy points and the source or
    target points. This matrix is flattened to a shape of ``(nsize,)``,
    which is consistent with the sum of the cluster sizes in :attr:`pxyindex`,
    as obtained from
    :meth:`~pytential.linalg.utils.TargetAndSourceClusterList.cluster_size`.
    """
    pxyindex: TargetAndSourceClusterList
    """A :class:`~pytential.linalg.utils.TargetAndSourceClusterList` used to
    describe the cluster interactions in :attr:`pxymat`.
    """

    nbrmat: onp.Array1D[np.inexact]
    """Interaction matrix between the neighboring points and the source or
    target points. This matrix is flattened to a shape of ``(nsize,)``,
    which is consistent with the sum of the cluster sizes in :attr:`nbrindex`,
    as obtained from
    :meth:`~pytential.linalg.utils.TargetAndSourceClusterList.cluster_size`.
    """
    nbrindex: TargetAndSourceClusterList
    """A :class:`~pytential.linalg.utils.TargetAndSourceClusterList` used to
    describe the cluster interactions in :attr:`nbrmat`.
    """

    def __getitem__(self, i: int) -> tuple[onp.Array2D[np.inexact],
                                           onp.Array2D[np.inexact]]:
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
        source_index: IndexList,
        target_index: IndexList, *,
        evaluate_proxy: Callable[...,
            tuple[onp.Array1D[np.inexact], TargetAndSourceClusterList]],
        evaluate_neighbor: Callable[...,
            tuple[onp.Array1D[np.inexact], TargetAndSourceClusterList]],
        dofdesc: sym.DOFDescriptor | None = None,
        max_particles_in_box: int | None = None,
        ) -> _ProxyNeighborEvaluationResult:
    """Evaluate the proxy to cluster and neighbor to cluster interactions for
    each cluster in *cluster_index*.
    """

    if source_index.nclusters == 1:
        raise ValueError("cannot make a proxy skeleton for a single cluster")

    from pytential.linalg.proxy import gather_cluster_neighbor_points
    pxy = proxy_generator(actx, dofdesc, source_index)
    nbrindex = gather_cluster_neighbor_points(
            actx, pxy, target_index,
            max_particles_in_box=max_particles_in_box)

    pxymat, pxy_cluster_index = evaluate_proxy(actx, places, pxy, nbrindex)
    nbrmat, nbr_cluster_index = evaluate_neighbor(actx, places, pxy, nbrindex)
    result = _ProxyNeighborEvaluationResult(
            pxy=pxy,
            pxymat=pxymat, pxyindex=pxy_cluster_index,
            nbrmat=nbrmat, nbrindex=nbr_cluster_index)

    return result


def _skeletonize_block_by_proxy_with_mats(
        actx: PyOpenCLArrayContext, ibrow: int, ibcol: int,
        places: GeometryCollection,
        proxy_generator: ProxyGeneratorBase,
        wrangler: SkeletonizationWrangler,
        tgt_src_index: TargetAndSourceClusterList, *,
        id_eps: float | None = None,
        id_rank: int | None = None,
        max_particles_in_box: int | None = None,
        rng: np.random.Generator | None = None,
        ) -> SkeletonizationResult:
    nclusters = tgt_src_index.nclusters
    if nclusters == 1:
        raise ValueError("cannot make a proxy skeleton for a single cluster")

    # construct proxy matrices to skeletonize
    from functools import partial
    evaluate_skeletonization_interaction = partial(
            _evaluate_proxy_skeletonization_interaction,
            actx, places, proxy_generator,
            dofdesc=wrangler.domains[ibcol],
            max_particles_in_box=max_particles_in_box)

    from pytential.linalg.utils import interp_decomp
    skel_src_indices = np.empty(nclusters, dtype=object)
    skel_tgt_indices = np.empty(nclusters, dtype=object)
    skel_starts = np.zeros(nclusters + 1, dtype=np.int32)

    L = np.empty(nclusters, dtype=object)
    R = np.empty(nclusters, dtype=object)

    from pytools import ProcessLogger

    with ProcessLogger(
            logger,
            f"_skeletonize_block_by_proxy_with_mats_{ibrow}_{ibcol}"):
        src_result = evaluate_skeletonization_interaction(
                tgt_src_index.sources, tgt_src_index.targets,
                evaluate_proxy=partial(
                    wrangler.evaluate_source_proxy_interaction,
                    ibrow=ibrow, ibcol=ibcol),
                evaluate_neighbor=partial(
                    wrangler.evaluate_source_neighbor_interaction,
                    ibrow=ibrow, ibcol=ibcol),
                )
        tgt_result = evaluate_skeletonization_interaction(
                tgt_src_index.targets, tgt_src_index.sources,
                evaluate_proxy=partial(
                    wrangler.evaluate_target_proxy_interaction,
                    ibrow=ibrow, ibcol=ibcol),
                evaluate_neighbor=partial(
                    wrangler.evaluate_target_neighbor_interaction,
                    ibrow=ibrow, ibcol=ibcol)
                )

        for i in range(nclusters):
            k = id_rank
            src_mat = np.vstack(src_result[i])
            tgt_mat = np.hstack(tgt_result[i])
            max_allowable_rank = min(*src_mat.shape, *tgt_mat.shape)

            if __debug__:
                isfinite = np.isfinite(tgt_mat)
                assert np.all(isfinite), np.where(isfinite)
                isfinite = np.isfinite(src_mat)
                assert np.all(isfinite), np.where(isfinite)

            # skeletonize target points
            k, idx, interp = interp_decomp(tgt_mat.T, rank=k, eps=id_eps, rng=rng)
            assert 0 < k <= len(idx)

            if k > max_allowable_rank:
                k = max_allowable_rank
                interp = interp[:k, :]

            L[i] = interp.T
            skel_tgt_indices[i] = tgt_src_index.targets.cluster_indices(i)[idx[:k]]
            assert interp.shape == (k, tgt_mat.shape[0])

            # skeletonize source points
            k, idx, interp = interp_decomp(src_mat, rank=k, eps=None, rng=rng)
            assert 0 < k <= len(idx)

            R[i] = interp
            skel_src_indices[i] = tgt_src_index.sources.cluster_indices(i)[idx[:k]]
            assert interp.shape == (k, src_mat.shape[1])

            skel_starts[i + 1] = skel_starts[i] + k
            assert skel_tgt_indices[i].shape == skel_src_indices[i].shape

    # evaluate diagonal
    from pytential.linalg.utils import make_flat_cluster_diag
    mat = wrangler.evaluate_self(actx, places, tgt_src_index, ibrow, ibcol)
    D = make_flat_cluster_diag(mat, tgt_src_index)

    from pytential.linalg import make_index_list
    skel_src_index = make_index_list(np.hstack(list(skel_src_indices)), skel_starts)
    skel_tgt_index = make_index_list(np.hstack(list(skel_tgt_indices)), skel_starts)
    skel_tgt_src_index = TargetAndSourceClusterList(skel_tgt_index, skel_src_index)

    return SkeletonizationResult(
            L=L, R=R, D=D,
            tgt_src_index=tgt_src_index, skel_tgt_src_index=skel_tgt_src_index,
            _src_eval_result=src_result, _tgt_eval_result=tgt_result)


def _evaluate_root(
        actx: PyOpenCLArrayContext, ibrow: int, ibcol: int,
        places: GeometryCollection,
        wrangler: SkeletonizationWrangler,
        tgt_src_index: TargetAndSourceClusterList
        ) -> SkeletonizationResult:
    assert tgt_src_index.nclusters == 1

    from pytential.linalg.utils import make_flat_cluster_diag
    mat = wrangler.evaluate_self(actx, places, tgt_src_index, ibrow, ibcol)
    D = make_flat_cluster_diag(mat, tgt_src_index)

    return SkeletonizationResult(
        L=obj_array.new_1d([np.eye(*D[0].shape)]),
        R=obj_array.new_1d([np.eye(*D[0].shape)]),
        D=D,
        tgt_src_index=tgt_src_index, skel_tgt_src_index=tgt_src_index,
        _src_eval_result=None, _tgt_eval_result=None,
        )

# }}}


# {{{ skeletonize_by_proxy

@dataclass(frozen=True)
class SkeletonizationResult:
    r"""Result of a skeletonization procedure.

    A matrix :math:`A` can be reconstructed using:

    .. math::

        A \approx D + L S R

    where :math:`S = A_{I, J}` for a subset :math:`I` and :math:`J` of the
    rows and columns of :math:`A`, respectively. This applies to each cluster
    in :attr:`tgt_src_index`. In particular, for a cluster pair :math:`(i, j)`,
    we can reconstruct the matrix entries as follows

    .. code:: python

        Aij = tgt_src_index.cluster_take(A, i, j)
        Sij = skel_tgt_src_index.cluster_take(A, i, j)
        Aij_approx = L[i] @ Sij @ R[j]

    .. autoproperty:: nclusters

    .. autoattribute:: L
    .. autoattribute:: R
    .. autoattribute:: D
    .. autoattribute:: tgt_src_index
    .. autoattribute:: skel_tgt_src_index
    """

    L: obj_array.ObjectArray1D[onp.Array2D[Any]]
    """An object :class:`~numpy.ndarray` of size ``(nclusters,)`` that contains
    the left block interpolation matrices."""
    R: obj_array.ObjectArray1D[onp.Array2D[Any]]
    """An object :class:`~numpy.ndarray` of size ``(nclusters,)`` that contains
    the right block interpolation matrices."""
    D: obj_array.ObjectArray1D[onp.Array2D[Any]]
    """An object :class:`~numpy.ndarray` of size ``(nclusters,)`` that contains
    the dense diagonal blocks."""

    tgt_src_index: TargetAndSourceClusterList
    """A :class:`~pytential.linalg.utils.TargetAndSourceClusterList` representing
    the indices in the original matrix :math:`A` that have been skeletonized.
    """
    skel_tgt_src_index: TargetAndSourceClusterList
    """A :class:`~pytential.linalg.utils.TargetAndSourceClusterList` representing
    a subset of :attr:`tgt_src_index`, i.e. the skeleton of each cluster of
    :math:`A`. These indices can be used to reconstruct the :math:`S` matrix.
    """

    # NOTE: these are meant only for testing! They contain the interactions
    # between the source / target points and their proxies / neighbors.
    _src_eval_result: _ProxyNeighborEvaluationResult | None = None
    _tgt_eval_result: _ProxyNeighborEvaluationResult | None = None

    def __post_init__(self) -> None:
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
    def nclusters(self) -> int:
        """Number of clusters that have been skeletonized."""
        return self.tgt_src_index.nclusters


@log_process(logger)
def skeletonize_by_proxy(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection,

        tgt_src_index: TargetAndSourceClusterList,
        exprs: ArithmeticExpression | Sequence[ArithmeticExpression],
        input_exprs: sym.var | Sequence[sym.var], *,
        domains: Sequence[Hashable] | None = None,
        context: dict[str, Any] | None = None,
        auto_where: Any = None,

        approx_nproxy: int | None = None,
        proxy_radius_factor: float | None = None,

        id_eps: float | None = None,
        id_rank: int | None = None,
        rng: np.random.Generator | None = None,
        max_particles_in_box: int | None = None,
    ) -> obj_array.ObjectArray2D[SkeletonizationResult]:
    r"""Evaluate and skeletonize a symbolic expression using proxy-based methods.

    :arg tgt_src_index: a :class:`~pytential.linalg.utils.TargetAndSourceClusterList`
        indicating which indices participate in the skeletonization.

    :arg exprs: see :func:`make_skeletonization_wrangler`.
    :arg input_exprs: see :func:`make_skeletonization_wrangler`.
    :arg domains: see :func:`make_skeletonization_wrangler`.
    :arg context: see :func:`make_skeletonization_wrangler`.

    :arg approx_nproxy: see :class:`~pytential.linalg.proxy.ProxyGenerator`.
    :arg proxy_radius_factor: see :class:`~pytential.linalg.proxy.ProxyGenerator`.

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
            domains=domains, context=context, auto_where=auto_where)
    proxy = QBXProxyGenerator(places,
            approx_nproxy=approx_nproxy,
            radius_factor=proxy_radius_factor)

    from itertools import product

    skels = np.empty((wrangler.nrows, wrangler.ncols), dtype=object)
    for ibrow, ibcol in product(range(wrangler.nrows), range(wrangler.ncols)):
        skels[ibrow, ibcol] = _skeletonize_block_by_proxy_with_mats(
                actx, ibrow, ibcol, places, proxy, wrangler, tgt_src_index,
                id_eps=id_eps, id_rank=id_rank,
                max_particles_in_box=max_particles_in_box,
                rng=rng)

    return skels

# }}}


# {{{ recursive skeletonization by proxy

@log_process(logger)
def rec_skeletonize_by_proxy(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection,

        ctree: ClusterTree,
        tgt_src_index: TargetAndSourceClusterList,
        exprs: ArithmeticExpression | Sequence[ArithmeticExpression],
        input_exprs: sym.var | Sequence[sym.var], *,
        domains: Sequence[Hashable] | None = None,
        context: dict[str, Any] | None = None,
        auto_where: Any = None,

        approx_nproxy: int | None = None,
        proxy_radius_factor: float | None = None,

        id_eps: float | None = None,
        rng: np.random.Generator | None = None,
        max_particles_in_box: int | None = None,

        _wrangler: SkeletonizationWrangler | None = None,
        _proxy: ProxyGeneratorBase | None = None,
    ) -> obj_array.ObjectArray1D[SkeletonizationResult]:
    r"""Performs recursive skeletonization based on :func:`skeletonize_by_proxy`.

    :returns: an object :class:`~numpy.ndarray` of :class:`SkeletonizationResult`\ s,
        one per level in *ctree*.
    """

    assert ctree.nclusters == tgt_src_index.nclusters

    if id_eps is None:
        id_eps = 1.0e-8

    if _proxy is None:
        from pytential.linalg.proxy import QBXProxyGenerator
        proxy: ProxyGeneratorBase = QBXProxyGenerator(places,
                approx_nproxy=approx_nproxy,
                radius_factor=proxy_radius_factor)
    else:
        proxy = _proxy

    if _wrangler is None:
        wrangler = make_skeletonization_wrangler(
                places, exprs, input_exprs,
                domains=domains, context=context, auto_where=auto_where)
    else:
        wrangler = _wrangler

    if wrangler.nrows != 1 or wrangler.ncols != 1:
        raise NotImplementedError("support for block matrices")

    from itertools import product

    skel_per_level = np.empty(ctree.nlevels, dtype=object)
    for i, clevel in enumerate(ctree.levels[:-1]):
        for ibrow, ibcol in product(range(wrangler.nrows), range(wrangler.ncols)):
            skeleton = _skeletonize_block_by_proxy_with_mats(
                actx, ibrow, ibcol, proxy.places, proxy, wrangler, tgt_src_index,
                id_eps=id_eps,
                # NOTE: we probably never want to set the rank here?
                id_rank=None,
                rng=rng,
                max_particles_in_box=max_particles_in_box)

        skel_per_level[i] = skeleton
        tgt_src_index = cluster(skeleton.skel_tgt_src_index, clevel)

    assert tgt_src_index.nclusters == 1
    assert not isinstance(skel_per_level[-1], SkeletonizationResult)

    # evaluate the full root cluster (no skeletonization or anything)
    skeleton = _evaluate_root(actx, 0, 0, places, wrangler, tgt_src_index)
    skel_per_level[-1] = skeleton

    return skel_per_level

# }}}

# vim: foldmethod=marker
