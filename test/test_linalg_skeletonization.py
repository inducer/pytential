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

from dataclasses import replace
from functools import partial
import pytest

import numpy as np
import numpy.linalg as la

from pytential import sym
from pytential import GeometryCollection

from meshmode.mesh.generation import ellipse, NArmedStarfish

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import extra_matrix_data as extra
import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])

SKELETONIZE_TEST_CASES = [
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
def test_skeletonize_symbolic(actx_factory, case, visualize=False):
    """Tests that the symbolic manipulations work for different kernels / IntGs."""
    actx = actx_factory()

    if visualize:
        logging.basicConfig(level=logging.INFO)
    logger.info("\n%s", case)

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[0], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    tgt_src_index = case.get_tgt_src_cluster_index(actx, places, dd)

    logger.info("nclusters %3d ndofs %7d",
            tgt_src_index.nclusters, density_discr.ndofs)

    # }}}

    # {{{ wranglers

    from pytential.linalg import QBXProxyGenerator
    from pytential.linalg.skeletonization import make_skeletonization_wrangler
    proxy_generator = QBXProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=case.proxy_approx_count)

    sym_u, sym_op = case.get_operator(places.ambient_dim)
    wrangler = make_skeletonization_wrangler(places, sym_op, sym_u,
            domains=None,
            context=case.knl_concrete_kwargs,
            _weighted_proxy=case.weighted_proxy,
            _proxy_source_cluster_builder=case.proxy_source_cluster_builder,
            _proxy_target_cluster_builder=case.proxy_target_cluster_builder,
            _neighbor_cluster_builder=case.neighbor_cluster_builder)

    # }}}

    from pytential.linalg.skeletonization import (
            _skeletonize_block_by_proxy_with_mats)
    _skeletonize_block_by_proxy_with_mats(
        actx, 0, 0, places, proxy_generator, wrangler, tgt_src_index,
        id_eps=1.0e-8
    )

# }}}


# {{{ test_skeletonize_by_proxy


def run_skeletonize_by_proxy(actx, case, resolution,
                             places=None, mat=None,
                             force_assert=True,
                             suffix="", visualize=False):
    from pytools import ProcessTimer

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    if places is None:
        qbx = case.get_layer_potential(actx, resolution, case.target_order)
        places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    tgt_src_index = case.get_tgt_src_cluster_index(actx, places, dd)

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

    from pytential.linalg import QBXProxyGenerator
    from pytential.linalg.skeletonization import make_skeletonization_wrangler
    proxy_generator = QBXProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=proxy_approx_count)

    sym_u, sym_op = case.get_operator(places.ambient_dim)
    wrangler = make_skeletonization_wrangler(places, sym_op, sym_u,
            domains=None,
            context=case.knl_concrete_kwargs,
            _weighted_proxy=case.weighted_proxy,
            _proxy_source_cluster_builder=case.proxy_source_cluster_builder,
            _proxy_target_cluster_builder=case.proxy_target_cluster_builder,
            _neighbor_cluster_builder=case.neighbor_cluster_builder)

    # }}}

    # {{{ check proxy id decomposition

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
                dep_source=qbx,
                dep_discr=density_discr,
                places=places,
                context=case.knl_concrete_kwargs,
                _weighted=wrangler.weighted_sources)(expr)

        logger.info("[time] dense matrix construction: %s", p)

    # skeleton
    from pytential.linalg.skeletonization import \
            _skeletonize_block_by_proxy_with_mats
    with ProcessTimer() as p:
        L, R, skel_tgt_src_index, src, tgt = _skeletonize_block_by_proxy_with_mats(
                actx, 0, 0, places, proxy_generator, wrangler, tgt_src_index,
                id_eps=case.id_eps,
                max_particles_in_box=case.max_particles_in_box)

    logger.info("[time] skeletonization by proxy: %s", p)

    for i in range(tgt_src_index.nclusters):
        # targets (rows)
        bi = np.searchsorted(
            tgt_src_index.targets.cluster_indices(i),
            skel_tgt_src_index.targets.cluster_indices(i),
            )

        A = np.hstack(tgt[i])
        S = A[bi, :]
        tgt_error = la.norm(A - L[i, i] @ S, ord=2) / la.norm(A, ord=2)

        # sources (columns)
        bj = np.searchsorted(
            tgt_src_index.sources.cluster_indices(i),
            skel_tgt_src_index.sources.cluster_indices(i),
            )

        A = np.vstack(src[i])
        S = A[:, bj]
        src_error = la.norm(A - S @ R[i, i], ord=2) / la.norm(A, ord=2)

        logger.info("[%04d] id_eps %.5e src %.5e tgt %.5e rank %d/%d",
                i, case.id_eps,
                src_error, tgt_error, R[i, i].shape[0], R[i, i].shape[1])

        if force_assert:
            assert src_error < 6 * case.id_eps
            assert tgt_error < 6 * case.id_eps

    # }}}

    # {{{ check skeletonize

    from pytential.linalg import SkeletonizationResult
    skeleton = SkeletonizationResult(
            L=L, R=R,
            tgt_src_index=tgt_src_index,
            skel_tgt_src_index=skel_tgt_src_index)

    from pytential.linalg.utils import (
            cluster_skeletonization_error, skeletonization_error)

    with ProcessTimer() as p:
        blk_err_l, blk_err_r = cluster_skeletonization_error(
                mat, skeleton, ord=2, relative=True)

        err_l = la.norm(blk_err_l, np.inf)
        err_r = la.norm(blk_err_r, np.inf)

    logger.info("[time] cluster error: %s", p)

    if density_discr.ndofs > 4096:
        err_f = max(err_l, err_r)
    else:
        with ProcessTimer() as p:
            err_f = skeletonization_error(mat, skeleton, ord=2, relative=True)

        logger.info("[time] full error: %s", p)

    # FIXME: why is the 3D error so large?
    rtol = 10**places.ambient_dim * case.id_eps

    logger.info("error: id_eps %.5e R %.5e L %.5e F %.5e (rtol %.5e)",
            case.id_eps, err_r, err_l, err_f, rtol)

    if force_assert:
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
                    tgt_src_index.sources, skel_tgt_src_index.sources)
            _plot_skeleton_with_proxies(f"targets{suffix}", sources, pxy,
                    tgt_src_index.targets, skel_tgt_src_index.targets)
        else:
            # TODO: would be nice to figure out a way to visualize some of these
            # skeletonization results in 3D. Probably need to teach the
            # visualizers to spit out point clouds
            pass

    # }}}

    return err_f, (places, mat)


@pytest.mark.parametrize("case", SKELETONIZE_TEST_CASES)
def test_skeletonize_by_proxy(actx_factory, case, visualize=False):
    """Test single-level level skeletonization accuracy."""
    import scipy.linalg.interpolative as sli    # pylint:disable=no-name-in-module
    sli.seed(42)

    actx = actx_factory()

    if visualize:
        logging.basicConfig(level=logging.INFO)

    case = replace(case, approx_cluster_count=6, id_eps=1.0e-8)
    logger.info("\n%s", case)

    run_skeletonize_by_proxy(actx, case, case.resolutions[0], visualize=visualize)

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
    """Test single-level level skeletonization accuracy."""
    import scipy.linalg.interpolative as sli    # pylint:disable=no-name-in-module
    sli.seed(42)

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

        if not was_zero:
            rec_error[i], (places, mat) = run_skeletonize_by_proxy(
                actx, case, r, places=places, mat=mat,
                force_assert=False,
                suffix=f"{suffix}_{i:04d}", visualize=False)

        was_zero = rec_error[i] == 0.0
        eoc.add_data_point(id_eps[i], rec_error[i])

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
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
