from __future__ import annotations


__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

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
import numpy.linalg as la
from scipy.sparse.linalg import LinearOperator

from arraycontext import ArrayOrContainerT, PyOpenCLArrayContext, flatten, unflatten
from meshmode.dof_array import DOFArray
from pytools import ProcessLogger, log_process, obj_array


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from pytential import GeometryCollection, sym
    from pytential.linalg.cluster import ClusterLevel, ClusterTree
    from pytential.linalg.proxy import ProxyGeneratorBase
    from pytential.linalg.skeletonization import (
        SkeletonizationResult,
        SkeletonizationWrangler,
    )
    from pytential.linalg.utils import IndexList, TargetAndSourceClusterList

logger = logging.getLogger(__name__)


__doc__ = """
Hierarical Matrix Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProxyHierarchicalMatrixWrangler
.. autoclass:: ProxyHierarchicalMatrix
.. autoclass:: ProxyHierarchicalForwardMatrix
.. autoclass:: ProxyHierarchicalBackwardMatrix

.. autofunction:: build_hmatrix_by_proxy
"""


# {{{ error model

def hmatrix_error_from_param(
        ambient_dim: int,
        *,
        id_eps: float,
        id_rank: int,
        min_proxy_radius: float,
        max_cluster_radius: float,
        nproxies: int,
        nsources: int,
        ntargets: int,
        c: float = 1.0e-3) -> float:
    import math

    # FIXME: This is horribly out of date right now. Need to get the updated version
    # from https://github.com/alexfikl/qbx-ds-paper-experiments
    if ambient_dim == 2:
        p = int(0.5 * id_rank)
    elif ambient_dim == 3:
        p = int((math.sqrt(1 + 4 * id_rank) - 1) / 2)
    else:
        raise ValueError(f"unsupported ambient dimension: '{ambient_dim}'")

    rho = alpha = max_cluster_radius / min_proxy_radius
    return float(
        c * rho ** (p + 1) / (1 - rho)
        + math.sqrt(nsources / nproxies)
        * (1 - alpha ** (p + 1)) / (1 - alpha) * id_eps
        )

# }}}


# {{{ update diagonals

def _update_skeleton_diagonal(
        skeleton: SkeletonizationResult,
        parent: SkeletonizationResult | None,
        clevel: ClusterLevel | None,
        diagonal: NDArray[np.inexact] | None = None) -> SkeletonizationResult:
    """Due to the evaluation in :func:`_skeletonize_block_by_proxy_with_mats`,
    the diagonal matrix in *skeleton* also contains the indices from its
    parent. In particular, at a level :math:`l` we need the diagonal block::

        0               D_{i, j + 1}        D_{i, j + 2}
        D_{i + 1, j}    0                   D_{i + 1, j + 2}
        D_{i + 2, j}    D_{i + 2, j + 1}    0

    but the version in *skeleton* also fills in the 0 blocks in there. This
    routine goes through them and zeros them out.
    """

    if clevel is None:
        return skeleton

    assert parent is not None
    assert skeleton.tgt_src_index.shape == parent.skel_tgt_src_index.shape

    if diagonal is None:
        diagonal = np.zeros(parent.nclusters)

    from numbers import Number
    if isinstance(diagonal, Number):
        diagonal = np.full(parent.nclusters, diagonal, dtype=skeleton.dtype)

    assert diagonal.size == parent.nclusters
    targets, sources = parent.skel_tgt_src_index

    # FIXME: nicer way to do this?
    mat = np.empty(skeleton.nclusters, dtype=object)
    for k in range(skeleton.nclusters):
        D = skeleton.D[k].copy()

        i = j = 0
        for icluster in clevel.parent_map[k]:
            di = targets.cluster_size(icluster)
            dj = sources.cluster_size(icluster)
            D[np.s_[i:i + di], np.s_[j:j + dj]] = diagonal[icluster]

            i += di
            j += dj

        assert D.shape == (i, j)
        mat[k] = D

    from dataclasses import replace
    return replace(skeleton, D=mat)


@log_process(logger)
def _update_skeletons_diagonal(
        wrangler: ProxyHierarchicalMatrixWrangler,
        forward: bool = True,
        ) -> NDArray[np.inexact]:
    skeletons = np.empty(wrangler.skeletons.shape, dtype=object)
    skeletons[0] = wrangler.skeletons[0]

    for i in range(1, wrangler.ctree.nlevels):
        diagonal = None if forward else skeletons[i - 1].Dhat

        skeletons[i] = _update_skeleton_diagonal(
            wrangler.skeletons[i],
            wrangler.skeletons[i - 1],
            wrangler.ctree.levels[i - 1],
            diagonal=diagonal)

    return skeletons

# }}}


# {{{ ProxyHierarchicalMatrix

@dataclass(frozen=True)
class ProxyHierarchicalMatrixWrangler:
    """
    .. automethod:: get_forward
    .. automethod:: get_backward
    """

    wrangler: SkeletonizationWrangler
    proxy: ProxyGeneratorBase
    ctree: ClusterTree
    skeletons: obj_array.ObjectArray1D[SkeletonizationResult]

    @property
    def tgt_src_index(self) -> TargetAndSourceClusterList:
        return self.skeletons[0].tgt_src_index

    def get_forward(self) -> ProxyHierarchicalForwardMatrix:
        return ProxyHierarchicalForwardMatrix(
            ctree=self.ctree,
            skeletons=_update_skeletons_diagonal(self, forward=True),
            )

    def get_backward(self) -> ProxyHierarchicalBackwardMatrix:
        return ProxyHierarchicalBackwardMatrix(
            ctree=self.ctree,
            skeletons=_update_skeletons_diagonal(self, forward=False))


@dataclass(frozen=True)
class ProxyHierarchicalMatrix(LinearOperator):
    """
    .. autoattribute:: ctree
    .. autoattribute:: skeletons

    This class implements the :class:`scipy.sparse.linalg.LinearOperator`
    interface. In particular, the following attributes and methods:

    .. autoproperty:: shape
    .. autoproperty:: dtype

    .. automethod:: matvec
    .. automethod:: __matmul__
    """

    ctree: ClusterTree
    """A tree structure that describes the hierarchy of the solver."""
    skeletons: obj_array.ObjectArray1D[SkeletonizationResult]
    """An :class:`~numpy.ndarray` containing skeletonization information
    for each level of the hierarchy. For additional details, see
    :class:`~pytential.linalg.skeletonization.SkeletonizationResult`.
    """

    @property
    def shape(self) -> tuple[int, int]:
        """A :class:`tuple` that gives the size of the skeletonized operator."""
        return self.skeletons[0].tgt_src_index.shape

    @property
    def dtype(self) -> np.dtype[np.inexact]:
        """The :class:`numpy.dtype` of the skeletonized operator."""
        # FIXME: assert that everyone has this dtype?
        return self.skeletons[0].R[0].dtype

    @property
    def nlevels(self) -> int:
        return self.skeletons.size

    @property
    def nclusters(self) -> int:
        return self.skeletons[0].nclusters

    def __matmul__(self, x: ArrayOrContainerT) -> ArrayOrContainerT:
        """Same as :meth:`_matvec`."""
        return self._matvec(x)

    def _matmat(self, mat):
        raise NotImplementedError

    def _adjoint(self, x):
        raise NotImplementedError

# }}}


# {{{ forward

@dataclass(frozen=True)
class ProxyHierarchicalForwardMatrix(ProxyHierarchicalMatrix):
    def _matvec(self, x: ArrayOrContainerT) -> ArrayOrContainerT:
        if isinstance(x, DOFArray):
            from arraycontext import get_container_context_recursively_opt
            actx = get_container_context_recursively_opt(x)
            if actx is None:
                raise ValueError("input array is frozen")

            ary = actx.to_numpy(flatten(x, actx))
        elif isinstance(x, np.ndarray) and x.dtype.char != "O":
            ary = x
        else:
            raise TypeError(f"unsupported input type: {type(x)}")

        assert actx is None or isinstance(actx, PyOpenCLArrayContext)
        result = apply_skeleton_forward_matvec(self, ary)

        if isinstance(x, DOFArray):
            assert actx is not None
            result = unflatten(x, actx.from_numpy(result), actx)

        return result


@log_process(logger)
def apply_skeleton_forward_matvec(
        hmat: ProxyHierarchicalMatrix,
        ary: ArrayOrContainerT,
        ) -> ArrayOrContainerT:
    from pytential.linalg.cluster import split_array
    targets, sources = hmat.skeletons[0].tgt_src_index
    x = split_array(ary, sources)   # type: ignore[arg-type]

    # NOTE: this computes a telescoping product of the form
    #
    #   A x_0 = (D0 + L0 (D1 + L1 (...) R1) R0) x_0
    #
    # with arbitrary numbers of levels. When recursing down, we compute
    #
    #   x_{k + 1} = R_k x_k
    #   z_{k + 1} = D_k x_k
    #
    # and, at the root level, we have
    #
    #   x_{N + 1} = z_{N + 1} = D_N x_N.
    #
    # When recursing back up, we take `b_{N + 1} = x_{N + 1}` and
    #
    #   b_{k - 1} = z_k + L_k b_k
    #
    # which gives back the desired product when we reach the leaf level again.

    d_dot_x = np.empty(hmat.nlevels, dtype=object)

    # {{{ recurse down

    from pytential.linalg.cluster import cluster

    with ProcessLogger(logger, "apply_skeleton_forward_matvec (compress)"):
        for k, clevel in enumerate(hmat.ctree.levels):
            skeleton = hmat.skeletons[k]
            assert x.shape == (skeleton.nclusters,)
            assert skeleton.tgt_src_index.shape[1] == sum(xi.size for xi in x)

            d_dot_x_k = np.empty(skeleton.nclusters, dtype=object)
            r_dot_x_k = np.empty(skeleton.nclusters, dtype=object)

            for i in range(skeleton.nclusters):
                r_dot_x_k[i] = skeleton.R[i] @ x[i]
                d_dot_x_k[i] = skeleton.D[i] @ x[i]

            d_dot_x[k] = d_dot_x_k
            x = cluster(r_dot_x_k, clevel)

    # }}}

    # {{{ root

    # NOTE: at root level, we just multiply with the full diagonal
    b = d_dot_x[hmat.nlevels - 1]
    assert b.shape == (1,)

    # }}}

    # {{{ recurse up

    from pytential.linalg.cluster import uncluster

    with ProcessLogger(logger, "apply_skeleton_forward_matvec (inflate)"):
        for k, clevel in reversed(list(enumerate(hmat.ctree.levels[:-1]))):
            skeleton = hmat.skeletons[k]
            d_dot_x_k = d_dot_x[k]
            assert d_dot_x_k.shape == (skeleton.nclusters,)

            b = uncluster(b, skeleton.skel_tgt_src_index.targets, clevel)
            for i in range(skeleton.nclusters):
                b[i] = d_dot_x_k[i] + skeleton.L[i] @ b[i]

    assert b.shape == (hmat.nclusters,)

    # }}}

    return np.concatenate(b)[np.argsort(targets.indices)]

# }}}


# {{{ backward

@dataclass(frozen=True)
class ProxyHierarchicalBackwardMatrix(ProxyHierarchicalMatrix):
    def _matvec(self, x: ArrayOrContainerT) -> ArrayOrContainerT:
        if isinstance(x, DOFArray):
            from arraycontext import get_container_context_recursively_opt
            actx = get_container_context_recursively_opt(x)
            if actx is None:
                raise ValueError("input array is frozen")

            ary = actx.to_numpy(flatten(x, actx))
        elif isinstance(x, np.ndarray) and x.dtype.char != "O":
            ary = x
        else:
            raise TypeError(f"unsupported input type: {type(x)}")

        assert actx is None or isinstance(actx, PyOpenCLArrayContext)
        result = apply_skeleton_backward_matvec(actx, self, ary)

        if isinstance(x, DOFArray):
            assert actx is not None
            result = unflatten(x, actx.from_numpy(result), actx)

        return result


@log_process(logger)
def apply_skeleton_backward_matvec(
        actx: PyOpenCLArrayContext | None,
        hmat: ProxyHierarchicalMatrix,
        ary: ArrayOrContainerT,
        ) -> ArrayOrContainerT:
    from pytential.linalg.cluster import split_array
    targets, sources = hmat.skeletons[0].tgt_src_index

    b = split_array(ary, targets)   # type: ignore[arg-type]
    r_dot_b = np.empty(hmat.nlevels, dtype=object)

    # {{{ recurse down

    # NOTE: this solves a telescoping product of the form
    #
    #   A x_0 = (D0 + L0 (D1 + L1 (...) R1) R0) x_0 = b_0
    #
    # with arbitrary numbers of levels. When recursing down, we compute
    #
    #   b_{k + 1} = \hat{D}_k R_k D_k^{-1} b_k
    #   \hat{D}_k = (R_k D_k^{-1} L_k)^{-1}
    #
    # and, at the root level, we solve
    #
    #   D_N x_N = b_N.
    #
    # When recursing back up, we take `b_{N + 1} = x_{N + 1}` and
    #
    #   x_{k} = D_k^{-1} (b_k - L_k b_{k + 1} + L_k \hat{D}_k x_{k + 1})
    #
    # which gives back the desired product when we reach the leaf level again.

    from pytential.linalg.cluster import cluster

    with ProcessLogger(logger, "apply_skeleton_backward_matvec (compress)"):
        for k, clevel in enumerate(hmat.ctree.levels):
            skeleton = hmat.skeletons[k]
            assert b.shape == (skeleton.nclusters,)
            assert skeleton.tgt_src_index.shape[0] == sum(bi.size for bi in b)

            dhat_dot_b_k = np.empty(skeleton.nclusters, dtype=object)
            for i in range(skeleton.nclusters):
                dhat_dot_b_k[i] = (
                    skeleton.Dhat[i] @ (skeleton.R[i] @ (skeleton.invD[i] @ b[i]))
                    )

            r_dot_b[k] = b
            b = cluster(dhat_dot_b_k, clevel)

    # }}}

    # {{{ root

    assert b.shape == (1,)

    with ProcessLogger(logger,
                       f"apply_skeleton_backward_matvec (root solve: {b[0].size}): "):
        x = obj_array.new_1d([
            la.solve(D, bi) for D, bi in zip(hmat.skeletons[-1].D, b, strict=True)
            ])

    # }}}

    # {{{ recurse up

    from pytential.linalg.cluster import uncluster

    with ProcessLogger(logger, "apply_skeleton_backward_matvec (inflate)"):
        for k, clevel in reversed(list(enumerate(hmat.ctree.levels[:-1]))):
            skeleton = hmat.skeletons[k]
            b0 = r_dot_b[k]
            b1 = r_dot_b[k + 1]
            assert b0.shape == (skeleton.nclusters,)

            x = uncluster(x, skeleton.skel_tgt_src_index.sources, clevel)
            b1 = uncluster(b1, skeleton.skel_tgt_src_index.targets, clevel)

            for i in range(skeleton.nclusters):
                sx = b1[i] - skeleton.Dhat[i] @ x[i]
                x[i] = skeleton.invD[i] @ (b0[i] - skeleton.L[i] @ sx)

    assert x.shape == (hmat.nclusters,)

    # }}}

    return np.concatenate(x)[np.argsort(sources.indices)]

# }}}


# {{{ build_hmatrix_by_proxy

def build_hmatrix_by_proxy(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection,
        exprs: sym.Expression | Sequence[sym.Expression],
        input_exprs: sym.Variable | Sequence[sym.Variable], *,
        auto_where: sym.DOFDescriptorLike | None = None,
        domains: Sequence[sym.DOFDescriptorLike] | None = None,
        context: dict[str, Any] | None = None,
        id_eps: float = 1.0e-8,
        rng: np.random.Generator | None = None,

        # NOTE: these are dev variables and can disappear at any time!
        _tree_kind: str | None = "adaptive-level-restricted",
        _weighted_proxy: bool | tuple[bool, bool] | None = None,

        # TODO: plugin in error model to get an estimate for:
        #   * how many points we want per cluster?
        #   * how many proxy points we want?
        #   * how far away should the proxy points be?
        # based on id_eps. How many of these should be user tunable?
        _max_particles_in_box: int | None = None,
        _approx_nproxy: int | None = None,
        _proxy_radius_factor: float | None = None,
        ) -> ProxyHierarchicalMatrixWrangler:
    from pytential.linalg.skeletonization import make_skeletonization_wrangler
    from pytential.symbolic.matrix import P2PClusterMatrixBuilder

    def P2PClusterMatrixBuilderWithDiagonal(*args, **kwargs):
        kwargs["exclude_self"] = True
        return P2PClusterMatrixBuilder(*args, **kwargs)

    wrangler = make_skeletonization_wrangler(
            places, exprs, input_exprs,
            domains=domains, context=context, auto_where=auto_where,
            _weighted_proxy=_weighted_proxy,
            # _remove_source_transforms=True,
            # _neighbor_cluster_builder=P2PClusterMatrixBuilderWithDiagonal,
            # _proxy_source_cluster_builder=P2PClusterMatrixBuilder,
            # _proxy_target_cluster_builder=P2PClusterMatrixBuilder,
            )

    if wrangler.nrows != 1 or wrangler.ncols != 1:
        raise ValueError("multi-block operators are not supported")

    from pytential.linalg.proxy import QBXProxyGenerator
    proxy = QBXProxyGenerator(places,
            approx_nproxy=_approx_nproxy,
            radius_factor=_proxy_radius_factor)

    from pytential.linalg.cluster import partition_by_nodes
    cluster_index, ctree = partition_by_nodes(
        actx, places,
        dofdesc=wrangler.domains[0],
        tree_kind=_tree_kind,
        max_particles_in_box=_max_particles_in_box)

    logger.info("tree levels: %d", ctree.nlevels)
    logger.info("cluster count: %d", cluster_index.nclusters)
    logger.info("leaf cluster sizes: %s", [
        # NOTE: making into a list so that they all get printed
        int(s) for s in np.diff(cluster_index.starts)
        ])

    from pytential.linalg.utils import TargetAndSourceClusterList
    tgt_src_index = TargetAndSourceClusterList(
        targets=cluster_index, sources=cluster_index)

    from pytential.linalg.skeletonization import rec_skeletonize_by_proxy
    skeletons = rec_skeletonize_by_proxy(
        actx, places, ctree, tgt_src_index, exprs, input_exprs,
        id_eps=id_eps,
        rng=rng,
        max_particles_in_box=_max_particles_in_box,
        _proxy=proxy,
        _wrangler=wrangler,
        )

    if __debug__:
        def _get_cluster_avg_size(idx: IndexList) -> str:
            d = np.diff(idx.starts)
            return f"{np.mean(d):.2f} ± {np.std(d):.2f}"

        logger.info("avg cluster size: %s", " ".join(
            _get_cluster_avg_size(sk.tgt_src_index.sources)
            for sk in skeletons
            ))

    return ProxyHierarchicalMatrixWrangler(
        wrangler=wrangler, proxy=proxy, ctree=ctree, skeletons=skeletons
        )

# }}}

# vim: foldmethod=marker
