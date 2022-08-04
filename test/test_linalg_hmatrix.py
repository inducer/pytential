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
from dataclasses import replace

import extra_matrix_data as extra
import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.mesh.generation import NArmedStarfish

from pytential import GeometryCollection, bind, sym
from pytential.array_context import PytestPyOpenCLArrayContextFactory
from pytential.utils import pytest_teardown_function as teardown_function  # noqa: F401


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


HMATRIX_TEST_CASES = [
        extra.CurveTestCase(
            name="starfish",
            op_type="scalar",
            target_order=4,
            curve_fn=NArmedStarfish(5, 0.25),
            resolutions=[512]),
        extra.CurveTestCase(
            name="starfish",
            op_type="double",
            target_order=4,
            curve_fn=NArmedStarfish(5, 0.25),
            resolutions=[512]),
        extra.TorusTestCase(
            target_order=4,
            op_type="scalar",
            resolutions=[0])
        ]


# {{{ test_hmatrix_forward_matvec_single_level

def hmatrix_matvec_single_level(mat, x, skeleton):
    from pytential.linalg.cluster import split_array
    targets, sources = skeleton.tgt_src_index
    y = split_array(x, sources)

    y_hat = np.empty(y.shape, dtype=object)

    for i in range(skeleton.nclusters):
        y_hat[i] = skeleton.R[i] @ y[i]

    from pytential.linalg.utils import skeletonization_matrix
    D, S = skeletonization_matrix(mat, skeleton)
    syhat = np.zeros(y.shape, dtype=object)

    from itertools import product
    for i, j in product(range(skeleton.nclusters), repeat=2):
        if i == j:
            continue

        syhat[i] = syhat[i] + S[i, j] @ y_hat[j]

    for i in range(skeleton.nclusters):
        y[i] = D[i] @ y[i] + skeleton.L[i] @ syhat[i]

    return np.concatenate(y)[np.argsort(targets.indices)]


@pytest.mark.parametrize("case", HMATRIX_TEST_CASES)
@pytest.mark.parametrize("discr_stage", [sym.QBX_SOURCE_STAGE1])
def test_hmatrix_forward_matvec_single_level(
        actx_factory, case, discr_stage, visualize=False):
    actx = actx_factory()
    rng = np.random.default_rng(42)

    if visualize:
        logging.basicConfig(level=logging.INFO)

    if case.ambient_dim == 2:
        kwargs = {"proxy_approx_count": 64, "proxy_radius_factor": 1.15}
    else:
        kwargs = {"proxy_approx_count": 256, "proxy_radius_factor": 1.25}

    case = replace(case, skel_discr_stage=discr_stage, **kwargs)
    logger.info("\n%s", case)

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    tgt_src_index, _ = case.get_tgt_src_cluster_index(actx, places, dd)

    logger.info("dd %s", dd)
    logger.info("nclusters %3d ndofs %7d",
            tgt_src_index.nclusters, density_discr.ndofs)

    # }}}

    # {{{ construct reference

    from pytential.linalg.direct_solver_symbolic import prepare_expr
    from pytential.symbolic.matrix import MatrixBuilder
    sym_u, sym_op = case.get_operator(places.ambient_dim)
    sym_op_prepr, = prepare_expr(places, [sym_op], (dd, dd))
    mat = MatrixBuilder(
        actx,
        dep_expr=sym_u,
        other_dep_exprs=[],
        dep_discr=density_discr,
        places=places,
        context={},
        )(sym_op_prepr)

    from arraycontext import flatten, unflatten
    x = actx.thaw(density_discr.nodes()[0])
    y = actx.to_numpy(flatten(x, actx))
    r_lpot = unflatten(x, actx.from_numpy(mat @ y), actx)

    # }}}

    # {{{ check matvec

    id_eps = 10.0 ** (-np.arange(2, 16))
    rec_error = np.zeros_like(id_eps)

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    from pytential.linalg.skeletonization import skeletonize_by_proxy
    for i in range(id_eps.size):
        skeleton = skeletonize_by_proxy(
            actx, places, tgt_src_index, sym_op, sym_u,
            domains=[dd], context={},
            approx_nproxy=case.proxy_approx_count,
            proxy_radius_factor=case.proxy_radius_factor,
            id_eps=id_eps[i],
            rng=rng,
            )
        r_hmat = hmatrix_matvec_single_level(mat, y, skeleton[0, 0])
        r_hmat = unflatten(x, actx.from_numpy(r_hmat), actx)

        from meshmode.dof_array import flat_norm
        rec_error[i] = actx.to_numpy(
            flat_norm(r_hmat - r_lpot) / flat_norm(r_lpot)
            )
        logger.info("id_eps %.2e error: %.12e", id_eps[i], rec_error[i])
        # assert rec_error[i] < 0.1

        eoc.add_data_point(id_eps[i], rec_error[i])

    logger.info("\n%s", eoc.pretty_print(
        abscissa_format="%.8e",
        error_format="%.8e",
        eoc_format="%.2f"))

    # }}}

    if not visualize:
        return

    import matplotlib.pyplot as pt
    fig = pt.figure(figsize=(10, 10), dpi=300)
    ax = fig.gca()

    ax.loglog(id_eps, id_eps, "k--")
    ax.loglog(id_eps, rec_error)

    ax.grid(True)
    ax.set_xlabel(r"$\epsilon_{id}$")
    ax.set_ylabel("$Error$")
    ax.set_title(case.name)

    basename = "linalg_hmatrix_single_matvec"
    fig.savefig(f"{basename}_{case.name}_{case.op_type}_convergence")

    if case.ambient_dim == 2:
        fig.clf()
        ax = fig.gca()

        from arraycontext import flatten
        r_hmap = actx.to_numpy(flatten(r_hmat, actx))
        r_lpot = actx.to_numpy(flatten(r_lpot, actx))

        ax.semilogy(r_hmap - r_lpot)
        ax.set_ylim([1.0e-16, 1.0])
        fig.savefig(f"{basename}_{case.name}_{case.op_type}_error")

    pt.close(fig)

# }}}


# {{{ test_hmatrix_forward_matvec

@pytest.mark.parametrize("case", [
    HMATRIX_TEST_CASES[0],
    HMATRIX_TEST_CASES[1],
    pytest.param(HMATRIX_TEST_CASES[2], marks=pytest.mark.slowtest),
    ])
@pytest.mark.parametrize("discr_stage", [
    sym.QBX_SOURCE_STAGE1,
    # sym.QBX_SOURCE_STAGE2
    ])
def test_hmatrix_forward_matvec(
        actx_factory, case, discr_stage, p2p=False, visualize=False):
    actx = actx_factory()
    rng = np.random.default_rng(42)

    if visualize:
        logging.basicConfig(level=logging.INFO)

    if case.ambient_dim == 2:
        kwargs = {"proxy_approx_count": 64, "proxy_radius_factor": 1.25}
    else:
        kwargs = {"proxy_approx_count": 256, "proxy_radius_factor": 1.25}

    case = replace(case, skel_discr_stage=discr_stage, **kwargs)
    logger.info("\n%s", case)

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    max_particles_in_box = case.max_particles_in_box_for_discr(density_discr)

    tgt_src_index, _ = case.get_tgt_src_cluster_index(
        actx, places, dd, max_particles_in_box=max_particles_in_box)

    logger.info("dd %s", dd)
    logger.info("nclusters %3d ndofs %7d",
            tgt_src_index.nclusters, density_discr.ndofs)

    # }}}

    # {{{ construct hmatrix

    from pytential.linalg.hmatrix import build_hmatrix_by_proxy
    sym_u, sym_op = case.get_operator(places.ambient_dim)

    x = actx.thaw(density_discr.nodes()[0])

    if p2p:
        # NOTE: this also needs changed in `build_hmatrix_by_proxy`
        # to actually evaluate the p2p interactions instead of qbx
        from pytential.linalg.direct_solver_symbolic import prepare_expr
        from pytential.symbolic.matrix import P2PMatrixBuilder
        mat = P2PMatrixBuilder(
            actx,
            dep_expr=sym_u,
            other_dep_exprs=[],
            dep_discr=density_discr,
            places=places,
            context={},
            )(prepare_expr(places, sym_op, (dd, dd)))

        from arraycontext import flatten, unflatten
        y = actx.to_numpy(flatten(x, actx))
        r_lpot = unflatten(x, actx.from_numpy(mat @ y), actx)
    else:
        r_lpot = bind(places, sym_op, auto_where=dd)(actx, u=x)

    from pytential.linalg.hmatrix import hmatrix_error_from_param
    id_eps = 10.0 ** (-np.arange(2, 16))
    rec_error = np.zeros_like(id_eps)
    model_error = np.zeros_like(id_eps)

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for i in range(id_eps.size):
        wrangler = build_hmatrix_by_proxy(
            actx, places, sym_op, sym_u,
            domains=[dd],
            context=case.knl_concrete_kwargs,
            id_eps=id_eps[i],
            rng=rng,
            _tree_kind=case.tree_kind,
            _max_particles_in_box=max_particles_in_box,
            _approx_nproxy=case.proxy_approx_count,
            _proxy_radius_factor=case.proxy_radius_factor,
            )
        hmat = wrangler.get_forward()

        # {{{ skeletonization error

        from meshmode.dof_array import flat_norm
        r_hmap = hmat @ x
        rec_error[i] = actx.to_numpy(
            flat_norm(r_hmap - r_lpot) / flat_norm(r_lpot)
            )

        # }}}

        # {{{ model error

        skeleton = hmat.skeletons[0]
        icluster = np.argmax(np.diff(skeleton.skel_tgt_src_index.targets.starts))

        proxy_radius = actx.to_numpy(
            skeleton._src_eval_result.pxy.radii[icluster]
            )
        cluster_radius = actx.to_numpy(
            skeleton._src_eval_result.pxy.cluster_radii[icluster]
            )

        model_error[i] = hmatrix_error_from_param(
            places.ambient_dim,
            id_eps=id_eps[i],
            min_proxy_radius=proxy_radius,
            max_cluster_radius=cluster_radius,
            id_rank=skeleton.skel_tgt_src_index.targets.cluster_size(icluster),
            nproxies=skeleton._src_eval_result.pxy.pxyindex.cluster_size(icluster),
            ntargets=skeleton.tgt_src_index.targets.cluster_size(icluster),
            nsources=skeleton.tgt_src_index.targets.cluster_size(icluster),
            c=1.0e-8
            )

        # }}}

        logger.info("id_eps %.2e error: %.12e (%.12e)",
            id_eps[i], rec_error[i], model_error[i])
        eoc.add_data_point(id_eps[i], rec_error[i])

    logger.info("\n%s", eoc.pretty_print(
        abscissa_format="%.8e",
        error_format="%.8e",
        eoc_format="%.2f"))

    if not visualize:
        assert eoc.order_estimate() > 0.6

    # }}}

    if not visualize:
        return

    import matplotlib.pyplot as pt
    fig = pt.figure(figsize=(10, 10), dpi=300)
    ax = fig.gca()

    ax.loglog(id_eps, id_eps, "k--")
    ax.loglog(id_eps, rec_error)
    ax.loglog(id_eps, model_error)

    ax.grid(True)
    ax.set_xlabel(r"$\epsilon_{id}$")
    ax.set_ylabel("$Error$")
    ax.set_title(case.name)

    lpot_name = "p2p" if p2p else "qbx"
    basename = f"linalg_hmatrix_{lpot_name}_matvec"
    fig.savefig(f"{basename}_{case.name}_{case.op_type}_convergence")

    if case.ambient_dim == 2:
        fig.clf()
        ax = fig.gca()

        from arraycontext import flatten
        r_hmap = actx.to_numpy(flatten(r_hmap, actx))
        r_lpot = actx.to_numpy(flatten(r_lpot, actx))

        ax.semilogy(r_hmap - r_lpot)
        ax.set_ylim([1.0e-16, 1.0])
        fig.savefig(f"{basename}_{case.name}_{case.op_type}_error")

    pt.close(fig)

# }}}


# {{{ test_hmatrix_backward_matvec

@pytest.mark.parametrize("case", [
    HMATRIX_TEST_CASES[0],
    HMATRIX_TEST_CASES[1],
    pytest.param(HMATRIX_TEST_CASES[2], marks=pytest.mark.slowtest),
    ])
@pytest.mark.parametrize("discr_stage", [
    sym.QBX_SOURCE_STAGE1,
    # sym.QBX_SOURCE_STAGE2
    ])
def test_hmatrix_backward_matvec(actx_factory, case, discr_stage, visualize=False):
    actx = actx_factory()
    rng = np.random.default_rng(42)

    if visualize:
        logging.basicConfig(level=logging.INFO)

    if case.ambient_dim == 2:
        kwargs = {"proxy_approx_count": 64, "proxy_radius_factor": 1.25}
    else:
        kwargs = {"proxy_approx_count": 64, "proxy_radius_factor": 1.25}

    case = replace(case, skel_discr_stage=discr_stage, **kwargs)
    logger.info("\n%s", case)

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    max_particles_in_box = case.max_particles_in_box_for_discr(density_discr)

    tgt_src_index, _ = case.get_tgt_src_cluster_index(
        actx, places, dd, max_particles_in_box=max_particles_in_box)

    logger.info("dd %s", dd)
    logger.info("nclusters %3d ndofs %7d",
            tgt_src_index.nclusters, density_discr.ndofs)

    # }}}

    # {{{

    sym_u, sym_op = case.get_operator(places.ambient_dim)

    if visualize:
        from pytential.linalg.direct_solver_symbolic import prepare_expr
        from pytential.symbolic.matrix import MatrixBuilder
        mat = MatrixBuilder(
            actx,
            dep_expr=sym_u,
            other_dep_exprs=[],
            dep_discr=density_discr,
            places=places,
            context={},
            )(prepare_expr(places, sym_op, (dd, dd)))

        import pytential.linalg.utils as hla
        eigs_ref = hla.eigs(mat, k=5)
        kappa_ref = np.linalg.cond(mat, p=2)

    # }}}

    # {{{ construct hmatrix

    from pytential.linalg.hmatrix import build_hmatrix_by_proxy
    sym_u, sym_op = case.get_operator(places.ambient_dim)

    x_ref = actx.thaw(density_discr.nodes()[0])
    b_ref = bind(places, sym_op, auto_where=dd)(actx, u=x_ref)

    id_eps = 10.0 ** (-np.arange(2, 16))
    rec_error = np.zeros_like(id_eps)

    if visualize:
        rec_eigs = np.zeros((id_eps.size, eigs_ref.size), dtype=np.complex128)
        rec_kappa = np.zeros(id_eps.size)

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for i in range(id_eps.size):
        wrangler = build_hmatrix_by_proxy(
            actx, places, sym_op, sym_u,
            domains=[dd],
            context=case.knl_concrete_kwargs,
            id_eps=id_eps[i],
            rng=rng,
            _tree_kind=case.tree_kind,
            _max_particles_in_box=max_particles_in_box,
            _approx_nproxy=case.proxy_approx_count,
            _proxy_radius_factor=case.proxy_radius_factor,
            )

        hmat_inv = wrangler.get_backward()
        x_hmat = hmat_inv @ b_ref

        if visualize:
            hmat = wrangler.get_forward()
            rec_eigs[i, :] = hla.eigs(hmat, k=5, tol=1.0e-6)
            rec_kappa[i] = hla.cond(hmat, p=2, tol=1.0e-6)

            logger.info("eigs: %s %s", eigs_ref, rec_eigs[i])
            logger.info("kappa %.12e %.12e", kappa_ref, rec_kappa[i])

        from meshmode.dof_array import flat_norm
        rec_error[i] = actx.to_numpy(
            flat_norm(x_hmat - x_ref) / flat_norm(x_ref)
            )
        logger.info("id_eps %.2e error: %.12e", id_eps[i], rec_error[i])
        eoc.add_data_point(id_eps[i], rec_error[i])

    logger.info("\n%s", eoc.pretty_print(
        abscissa_format="%.8e",
        error_format="%.8e",
        eoc_format="%.2f"))

    if not visualize:
        assert eoc.order_estimate() > 0.6

    # }}}

    if not visualize:
        return

    import matplotlib.pyplot as pt
    fig = pt.figure(figsize=(10, 10), dpi=300)

    # {{{ convergence

    ax = fig.gca()
    ax.loglog(id_eps, id_eps, "k--")
    ax.loglog(id_eps, rec_error)

    ax.grid(True)
    ax.set_xlabel(r"$\epsilon_{id}$")
    ax.set_ylabel("$Error$")
    ax.set_title(case.name)

    fig.savefig(f"linalg_hmatrix_inverse_{case.name}_{case.op_type}_convergence")
    fig.clf()

    # }}}

    # {{{ eigs

    ax = fig.gca()
    ax.plot(np.real(eigs_ref), np.imag(eigs_ref), "ko")
    for i in range(id_eps.size):
        ax.plot(np.real(rec_eigs[i]), np.imag(rec_eigs[i]), "v")

    ax.grid(True)
    ax.set_xlabel(r"$\Re \lambda$")
    ax.set_ylabel(r"$\Im \lambda$")

    fig.savefig(f"linalg_hmatrix_inverse_{case.name}_{case.op_type}_eigs")
    fig.clf()

    # }}}

    if case.ambient_dim == 2:
        ax = fig.gca()

        from arraycontext import flatten
        x_hmat = actx.to_numpy(flatten(x_hmat, actx))
        x_ref = actx.to_numpy(flatten(x_ref, actx))

        ax.semilogy(x_hmat - x_ref)
        ax.set_ylim([1.0e-16, 1.0])
        fig.savefig(f"linalg_hmatrix_inverse_{case.name}_{case.op_type}_error")
        fig.clf()

    pt.close(fig)

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
