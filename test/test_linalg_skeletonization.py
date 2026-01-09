from __future__ import annotations


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

import logging
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import extra_matrix_data as extra
import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts
from meshmode.mesh.generation import NArmedStarfish, ellipse

from pytential import GeometryCollection, sym
from pytential.array_context import PytestPyOpenCLArrayContextFactory
from pytential.utils import pytest_teardown_function as teardown_function  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])

SKELETONIZE_TEST_CASES: Sequence[extra.CurveTestCase | extra.TorusTestCase] = [
        extra.CurveTestCase(
            name="ellipse",
            op_type="scalar",
            target_order=7,
            curve_fn=partial(ellipse, 3.0)),
        extra.CurveTestCase(
            name="starfish",
            op_type="scalar",
            target_order=4,
            curve_fn=NArmedStarfish(5, 0.25),
            resolutions=[32]),
        extra.TorusTestCase(
            target_order=4,
            op_type="scalar",
            resolutions=[0])
        ]


def _plot_skeleton_with_proxies(name, sources, pxy, srcindex, sklindex):
    import matplotlib.pyplot as pt

    fig, ax = pt.subplots(1, figsize=(10, 10), dpi=300)
    ax.plot(sources[0][srcindex.indices], sources[1][srcindex.indices],
            "ko", alpha=0.5)

    for i in range(srcindex.nclusters):
        iskl = sklindex.cluster_indices(i)
        pt.plot(sources[0][iskl], sources[1][iskl], "o")

        c = pt.Circle(pxy.centers[:, i], pxy.radii[i], color="k", alpha=0.1)
        ax.add_artist(c)
        ax.text(*pxy.centers[:, i], f"{i}",
                fontweight="bold", ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    fig.savefig(f"test_skeletonize_by_proxy_{name}")
    pt.close(fig)


# {{{ test_skeletonize_symbolic

@pytest.mark.parametrize("case", [
    # Laplace
    replace(SKELETONIZE_TEST_CASES[0], op_type="single", knl_class_or_helmholtz_k=0),
    replace(SKELETONIZE_TEST_CASES[0], op_type="double", knl_class_or_helmholtz_k=0),
    # Helmholz
    replace(SKELETONIZE_TEST_CASES[0], op_type="single", knl_class_or_helmholtz_k=5),
    replace(SKELETONIZE_TEST_CASES[0], op_type="double", knl_class_or_helmholtz_k=5),
    ])
def test_skeletonize_symbolic(actx_factory: ArrayContextFactory, case, visualize=False):
    """Tests that the symbolic manipulations work for different kernels / IntGs.
    This tests that `prepare_expr` and `prepare_proxy_expr` can "clean" the
    given integral equations and that the result can be evaluated and skeletonized.
    """

    actx = actx_factory()
    rng = np.random.default_rng(42)

    if visualize:
        logging.basicConfig(level=logging.INFO)
    logger.info("\n%s", case)

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[0], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    tgt_src_index, ctree = case.get_tgt_src_cluster_index(actx, places, dd)

    logger.info("nclusters %3d ndofs %7d",
            tgt_src_index.nclusters, density_discr.ndofs)

    # }}}

    from pytential.linalg.skeletonization import rec_skeletonize_by_proxy

    sym_u, sym_op = case.get_operator(places.ambient_dim)
    rec_skeletonize_by_proxy(
        actx, places, ctree, tgt_src_index, sym_op, sym_u,
        context=case.knl_concrete_kwargs,
        auto_where=dd,
        id_eps=1.0e-8,
        rng=rng,
    )

# }}}


# {{{ test_skeletonize_by_proxy


def run_skeletonize_by_proxy(actx, case, resolution,
                             places=None, mat=None,
                             ctol=None, rtol=None,
                             tgt_src_index=None,
                             rng=None,
                             suffix="", visualize=False):
    from pytools import ProcessTimer

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    if places is None:
        qbx = case.get_layer_potential(actx, resolution, case.target_order)
        places = GeometryCollection(qbx, auto_where=dd)
    else:
        qbx = places.get_geometry(dd.geometry)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    if tgt_src_index is None:
        tgt_src_index, _ = case.get_tgt_src_cluster_index(actx, places, dd)

    logger.info("nclusters %3d ndofs %7d",
            tgt_src_index.nclusters, density_discr.ndofs)

    # }}}

    # {{{ wranglers

    proxy_approx_count = case.proxy_approx_count
    if proxy_approx_count is None:
        # FIXME: replace this with an estimate from an error model
        proxy_approx_count = int(1.5 * np.max(np.diff(tgt_src_index.targets.starts)))

    logger.info("proxy factor %.2f count %7d",
                case.proxy_radius_factor, proxy_approx_count)

    from pytential.linalg.proxy import QBXProxyGenerator
    proxy_generator = QBXProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=proxy_approx_count)

    from pytential.linalg.skeletonization import make_skeletonization_wrangler
    sym_u, sym_op = case.get_operator(places.ambient_dim)
    wrangler = make_skeletonization_wrangler(places, sym_op, sym_u,
            context=case.knl_concrete_kwargs,
            auto_where=dd,
            _weighted_proxy=case.weighted_proxy,
            _proxy_source_cluster_builder=case.proxy_source_cluster_builder,
            _proxy_target_cluster_builder=case.proxy_target_cluster_builder,
            _neighbor_cluster_builder=case.neighbor_cluster_builder)

    # }}}

    # {{{ check proxy id decomposition

    # NOTE: ideally we would use the 2-norm everywhere (proper matrix norm), but
    # large matrices take a VERY long time to do the SVD, so Frobenius it is!
    max_ndofs = 4096
    ord = "fro" if density_discr.ndofs > max_ndofs else 2

    if mat is None:
        from pytential.symbolic.execution import _prepare_expr
        expr = _prepare_expr(places, sym_op)

        # dense matrix
        from pytential.symbolic.matrix import MatrixBuilder
        with ProcessTimer() as p:
            mat = MatrixBuilder(
                actx,
                dep_expr=sym_u,
                other_dep_exprs=[],
                dep_discr=density_discr,
                places=places,
                context=case.knl_concrete_kwargs,
                _weighted=wrangler.weighted_sources)(expr)

        logger.info("[time] dense matrix construction: %s", p)

    # skeleton
    from pytential.linalg.skeletonization import _skeletonize_block_by_proxy_with_mats
    with ProcessTimer() as p:
        skeleton = _skeletonize_block_by_proxy_with_mats(
                actx, 0, 0, places, proxy_generator, wrangler, tgt_src_index,
                id_eps=case.id_eps,
                max_particles_in_box=case.max_particles_in_box,
                rng=rng)

    logger.info("[time] skeletonization by proxy: %s", p)

    def intersect1d(x, y):
        return np.where((x.reshape(1, -1) - y.reshape(-1, 1)) == 0)[1]

    L, R = skeleton.L, skeleton.R
    for i in range(tgt_src_index.nclusters):
        # targets (rows)
        bi = intersect1d(
            tgt_src_index.targets.cluster_indices(i),
            skeleton.skel_tgt_src_index.targets.cluster_indices(i),
            )

        A = np.hstack(skeleton._tgt_eval_result[i])
        S = A[bi, :]
        tgt_error = la.norm(A - L[i] @ S, ord=ord) / la.norm(A, ord=ord)

        # sources (columns)
        bj = intersect1d(
            tgt_src_index.sources.cluster_indices(i),
            skeleton.skel_tgt_src_index.sources.cluster_indices(i),
            )

        A = np.vstack(skeleton._src_eval_result[i])
        S = A[:, bj]
        src_error = la.norm(A - S @ R[i], ord=ord) / la.norm(A, ord=ord)

        logger.info("[%04d] id_eps %.5e src %.5e tgt %.5e rank %d/%d",
                i, case.id_eps,
                src_error, tgt_error, R[i].shape[0], R[i].shape[1])

        if ctol is not None:
            assert src_error < ctol
            assert tgt_error < ctol

    # }}}

    # {{{ check skeletonize

    from pytential.linalg.utils import (
        cluster_skeletonization_error,
        skeletonization_error,
    )

    with ProcessTimer() as p:
        blk_err_l, blk_err_r = cluster_skeletonization_error(
                mat, skeleton, ord=ord, relative=True)

        err_l = la.norm(blk_err_l, np.inf)
        err_r = la.norm(blk_err_r, np.inf)

    logger.info("[time] cluster error: %s", p)

    if density_discr.ndofs > max_ndofs:
        err_f = max(err_l, err_r)
    else:
        with ProcessTimer() as p:
            err_f = skeletonization_error(mat, skeleton, ord=ord, relative=True)

        logger.info("[time] full error: %s", p)

    logger.info("error: id_eps %.5e R %.5e L %.5e F %.5e (rtol %.5e)",
            case.id_eps, err_r, err_l, err_f,
            rtol if rtol is not None else 0.0)

    if rtol:
        assert err_l < rtol
        assert err_r < rtol
        assert err_f < rtol

    # }}}

    # {{{ visualize

    if visualize:
        import matplotlib.pyplot as pt
        pt.imshow(np.log10(blk_err_l + 1.0e-16))
        pt.colorbar()
        pt.savefig(f"test_skeletonize_by_proxy_err_l{suffix}")
        pt.clf()

        pt.imshow(np.log10(blk_err_r + 1.0e-16))
        pt.colorbar()
        pt.savefig(f"test_skeletonize_by_proxy_err_r{suffix}")
        pt.clf()

        if places.ambient_dim == 2:
            pxy = proxy_generator(
                    actx, wrangler.domains[0], tgt_src_index.targets
                    ).to_numpy(actx)

            from arraycontext import flatten
            sources = actx.to_numpy(
                    flatten(density_discr.nodes(), actx)
                    ).reshape(places.ambient_dim, -1)

            _plot_skeleton_with_proxies(f"sources{suffix}", sources, pxy,
                    tgt_src_index.sources, skeleton.skel_tgt_src_index.sources)
            _plot_skeleton_with_proxies(f"targets{suffix}", sources, pxy,
                    tgt_src_index.targets, skeleton.skel_tgt_src_index.targets)
        else:
            # TODO: would be nice to figure out a way to visualize some of these
            # skeletonization results in 3D. Probably need to teach the
            # visualizers to spit out point clouds
            pass

    # }}}

    return err_f, (places, mat, skeleton)


@pytest.mark.parametrize("case", [
    SKELETONIZE_TEST_CASES[0],
    SKELETONIZE_TEST_CASES[1],
    SKELETONIZE_TEST_CASES[2],
    ])
def test_skeletonize_by_proxy(actx_factory: ArrayContextFactory, case, visualize=False):
    r"""Test multilevel skeletonization accuracy. Checks that the error for
    every level satisfies :math:`e < c \epsilon_{id}` for a fixed ID tolerance
    and an empirically determined (not too huge) :math:`c`.
    """

    import scipy.linalg.interpolative as sli

    sli.seed(42)
    rng = np.random.default_rng(42)

    actx = actx_factory()

    if visualize:
        logging.basicConfig(level=logging.INFO)

    case = replace(case, approx_cluster_count=6, id_eps=1.0e-8)
    logger.info("\n%s", case)

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[0], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    tgt_src_index, ctree = case.get_tgt_src_cluster_index(actx, places, dd)
    mat = None

    from pytential.linalg.cluster import cluster
    for clevel in ctree.levels[:-1]:
        logger.info("[%2d/%2d] nclusters %3d",
            clevel.level, ctree.nlevels, clevel.nclusters)

        _, (_, mat, skeleton) = run_skeletonize_by_proxy(
            actx, case, case.resolutions[0],
            ctol=10 * case.id_eps,
            # FIXME: why is the 3D error so large?
            rtol=10**case.ambient_dim * case.id_eps,
            places=places, mat=mat, rng=rng, tgt_src_index=tgt_src_index,
            visualize=visualize)

        tgt_src_index = cluster(skeleton.skel_tgt_src_index, clevel)

# }}}


# {{{ test_skeletonize_by_proxy_convergence

CONVERGENCE_TEST_CASES = [
        replace(SKELETONIZE_TEST_CASES[0], resolutions=[256]),
        replace(SKELETONIZE_TEST_CASES[1], resolutions=[256]),
        replace(SKELETONIZE_TEST_CASES[2], resolutions=[0]),
        extra.GMSHSphereTestCase(
            target_order=8,
            op_type="scalar",
            resolutions=[0.4]),
        ]


@pytest.mark.parametrize("case", [
    CONVERGENCE_TEST_CASES[0],
    CONVERGENCE_TEST_CASES[1],
    pytest.param(CONVERGENCE_TEST_CASES[2], marks=pytest.mark.slowtest),
    pytest.param(CONVERGENCE_TEST_CASES[3], marks=pytest.mark.slowtest),
    ])
def test_skeletonize_by_proxy_convergence(
        actx_factory, case, weighted=True,
        visualize=False):
    r"""Test single-level skeletonization accuracy. Checks that the
    accuracy of the skeletonization scales linearly with :math:`\epsilon_{id}`
    (the ID tolerance).
    """
    import scipy.linalg.interpolative as sli

    sli.seed(42)
    rng = np.random.default_rng(42)

    actx = actx_factory()

    if visualize:
        logging.basicConfig(level=logging.INFO)

    if case.ambient_dim == 2:
        nclusters = 6
    else:
        nclusters = 12

    case = replace(case, approx_cluster_count=nclusters)
    logger.info("\n%s", case)

    id_eps = 10.0 ** (-np.arange(2, 16))
    rec_error = np.zeros_like(id_eps)

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    r = case.resolutions[-1]
    w = "ww" if weighted or weighted is None else "nw"
    if isinstance(r, int):
        suffix = f"_{case.name}_{r:06d}_{w}"
    else:
        suffix = f"_{case.name}_{r:.3f}_{w}"

    was_zero = False
    places = mat = None
    for i in range(id_eps.size):
        case = replace(case, id_eps=id_eps[i], weighted_proxy=weighted)

        # NOTE: don't skeletonize anymore if we reached zero error, but we still
        # want to loop to do `eoc.add_data_point()`
        if not was_zero:
            rec_error[i], (places, mat, _) = run_skeletonize_by_proxy(
                actx, case, r, places=places, mat=mat,
                suffix=f"{suffix}_{i:04d}", rng=rng, visualize=False)

        was_zero = rec_error[i] == 0.0
        eoc.add_data_point(id_eps[i], rec_error[i])
        if was_zero:
            break

    logger.info("\n%s", eoc.pretty_print())

    if visualize:
        import matplotlib.pyplot as pt
        fig = pt.figure(figsize=(10, 10), dpi=300)
        ax = fig.gca()

        ax.loglog(id_eps, id_eps, "k--")
        ax.loglog(id_eps, rec_error)

        ax.grid(True)
        ax.set_xlabel(r"$\epsilon_{id}$")
        ax.set_ylabel("$Error$")
        ax.set_title(case.name)

        fig.savefig(f"test_skeletonize_by_proxy_convergence{suffix}")
        pt.close(fig)

    assert eoc.order_estimate() > 0.9

# }}}


if __name__ == "__main__":
    import sys

    from pytential.array_context import _acf  # noqa: F401

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
