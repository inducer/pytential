from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2014 Shidong Jiang, Andreas Kloeckner"

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
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

import pytest

from pytools.obj_array import make_obj_array

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from six.moves import range

from pytential import bind, sym, norm  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)


def run_dielectric_test(cl_ctx, queue, nelements, qbx_order,
        k0=3, k1=2.9, mesh_order=10,
        bdry_quad_order=None, bdry_ovsmp_quad_order=None,
        fmm_order=None, visualize=False):

    if fmm_order is None:
        fmm_order = qbx_order * 2
    if bdry_quad_order is None:
        bdry_quad_order = mesh_order
    if bdry_ovsmp_quad_order is None:
        bdry_ovsmp_quad_order = 4*bdry_quad_order

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial
    mesh = make_curve_mesh(
            partial(ellipse, 3),
            np.linspace(0, 1, nelements+1),
            mesh_order)

    density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    logger.info("%d elements" % mesh.nelements)

    # from meshmode.discretization.visualization import make_visualizer
    # bdry_vis = make_visualizer(queue, density_discr, 20)

    # {{{ solve bvp

    from sumpy.kernel import HelmholtzKernel, AxisTargetDerivative
    kernel = HelmholtzKernel(2)

    beta = 2.5
    K0 = np.sqrt(k0**2-beta**2)  # noqa
    K1 = np.sqrt(k1**2-beta**2)  # noqa

    from pytential.symbolic.pde.scalar import TEMDielectric2DBoundaryOperator
    pde_op = TEMDielectric2DBoundaryOperator(
            k_vacuum=1,
            interfaces=((0, 1, sym.DEFAULT_SOURCE),),
            domain_k_exprs=(k0, k1),
            beta=beta)

    op_unknown_sym = pde_op.make_unknown("unknown")

    representation0_sym = pde_op.representation(op_unknown_sym, 0)
    representation1_sym = pde_op.representation(op_unknown_sym, 1)

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(
            density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order
            )

    bound_pde_op = bind(qbx, pde_op.operator(op_unknown_sym))

    e_sources_0 = make_obj_array(list(np.array([
        [0.1, 0.2]
        ]).T.copy()))
    e_strengths_0 = np.array([1])
    e_sources_1 = make_obj_array(list(np.array([
        [4, 4]
        ]).T.copy()))
    e_strengths_1 = np.array([1])

    h_sources_0 = make_obj_array(list(np.array([
        [0.2, 0.1]
        ]).T.copy()))
    h_strengths_0 = np.array([1])
    h_sources_1 = make_obj_array(list(np.array([
        [4, 5]
        ]).T.copy()))
    h_strengths_1 = np.array([1])

    kernel_grad = [
        AxisTargetDerivative(i, kernel) for i in range(density_discr.ambient_dim)]

    from sumpy.p2p import P2P
    pot_p2p = P2P(cl_ctx, [kernel], exclude_self=False)
    pot_p2p_grad = P2P(cl_ctx, kernel_grad, exclude_self=False)

    normal = bind(density_discr, sym.normal())(queue).as_vector(np.object)
    tangent = bind(
        density_discr,
        sym.pseudoscalar()/sym.area_element())(queue).as_vector(np.object)

    _, (E0,) = pot_p2p(queue, density_discr.nodes(), e_sources_0, [e_strengths_0],
                    out_host=False, k=K0)
    _, (E1,) = pot_p2p(queue, density_discr.nodes(), e_sources_1, [e_strengths_1],
                    out_host=False, k=K1)
    _, (grad0_E0, grad1_E0) = pot_p2p_grad(
        queue, density_discr.nodes(), e_sources_0, [e_strengths_0],
        out_host=False, k=K0)
    _, (grad0_E1, grad1_E1) = pot_p2p_grad(
        queue, density_discr.nodes(), e_sources_1, [e_strengths_1],
        out_host=False, k=K1)

    _, (H0,) = pot_p2p(queue, density_discr.nodes(), h_sources_0, [h_strengths_0],
                    out_host=False, k=K0)
    _, (H1,) = pot_p2p(queue, density_discr.nodes(), h_sources_1, [h_strengths_1],
                    out_host=False, k=K1)
    _, (grad0_H0, grad1_H0) = pot_p2p_grad(
        queue, density_discr.nodes(), h_sources_0, [h_strengths_0],
        out_host=False, k=K0)
    _, (grad0_H1, grad1_H1) = pot_p2p_grad(
        queue, density_discr.nodes(), h_sources_1, [h_strengths_1],
        out_host=False, k=K1)

    E0_dntarget = (grad0_E0*normal[0] + grad1_E0*normal[1])  # noqa
    E1_dntarget = (grad0_E1*normal[0] + grad1_E1*normal[1])  # noqa

    H0_dntarget = (grad0_H0*normal[0] + grad1_H0*normal[1])  # noqa
    H1_dntarget = (grad0_H1*normal[0] + grad1_H1*normal[1])  # noqa

    E0_dttarget = (-grad0_E0*normal[1] + grad1_E0*normal[0])  # noqa
    E1_dttarget = (-grad0_E1*normal[1] + grad1_E1*normal[0])  # noqa

    H0_dttarget = (-grad0_H0*normal[1] + grad1_H0*normal[0])  # noqa
    H1_dttarget = (-grad0_H1*normal[1] + grad1_H1*normal[0])  # noqa

    sqrt_w = bind(density_discr, sym.sqrt_jac_q_weight())(queue)

    bvp_rhs = np.zeros(len(pde_op.bcs), dtype=np.object)
    for i_bc, terms in enumerate(pde_op.bcs):
        for term in terms:
            assert term.i_interface == 0
            if term.field_kind == pde_op.field_kind_e:

                if term.direction == pde_op.dir_none:
                    bvp_rhs[i_bc] += (
                        term.coeff_outer * E0
                        + term.coeff_inner * E1)
                elif term.direction == pde_op.dir_normal:
                    bvp_rhs[i_bc] += (
                        term.coeff_outer * E0_dntarget
                        + term.coeff_inner * E1_dntarget)
                elif term.direction == pde_op.dir_tangential:
                    bvp_rhs[i_bc] += (
                        term.coeff_outer * E0_dttarget
                        + term.coeff_inner * E1_dttarget)
                else:
                    raise NotImplementedError("direction spec in RHS")

                bvp_rhs[i_bc] *= sqrt_w
            elif term.field_kind == pde_op.field_kind_h:

                if term.direction == pde_op.dir_none:
                    bvp_rhs[i_bc] += (
                        term.coeff_outer * H0
                        + term.coeff_inner * H1)
                elif term.direction == pde_op.dir_normal:
                    bvp_rhs[i_bc] += (
                        term.coeff_outer * H0_dntarget
                        + term.coeff_inner * H1_dntarget)
                elif term.direction == pde_op.dir_tangential:
                    bvp_rhs[i_bc] += (
                        term.coeff_outer * H0_dttarget
                        + term.coeff_inner * H1_dttarget)
                else:
                    raise NotImplementedError("direction spec in RHS")

                bvp_rhs[i_bc] *= sqrt_w

    scipy_op = bound_pde_op.scipy_op(queue, "unknown",
            domains=[sym.DEFAULT_TARGET]*4, K0=K0, K1=K1)

    if 0:
        from pytential.gmres import gmres
        gmres_result = gmres(scipy_op,
                bvp_rhs, tol=1e-14, progress=True,
                hard_failure=True, stall_iterations=0)

        unknown = gmres_result.solution
    else:
        from sumpy.tools import build_matrix
        mat = build_matrix(scipy_op)

        print("condition number: %g" % la.cond(mat))
        if 0:
            ev = la.eigvals(mat)
            import matplotlib.pyplot as pt
            pt.plot(ev.real, ev.imag, "o")
            pt.show()

        bvp_rhs_flat = np.array([
            rhs_i.get()
            for rhs_i in bvp_rhs]).reshape(-1)
        unknown = la.solve(mat, bvp_rhs_flat)

    # }}}

    targets_0 = make_obj_array(list(np.array([
        [3.2 + t, -4]
        for t in [0, 0.5, 1]
        ]).T.copy()))
    targets_1 = make_obj_array(list(np.array([
        [t*-0.3, t*-0.2]
        for t in [0, 0.5, 1]
        ]).T.copy()))

    from pytential.target import PointsTarget
    F0_tgt = bind(  # noqa
            (qbx, PointsTarget(targets_0)),
            representation0_sym)(queue, unknown=unknown, K0=K0, K1=K1).get()

    F1_tgt = bind(  # noqa
            (qbx, PointsTarget(targets_1)),
            representation1_sym)(queue, unknown=unknown, K0=K0, K1=K1).get()

    _, (E0_tgt_true,) = pot_p2p(queue, targets_0, e_sources_0, [e_strengths_0],
                    out_host=True, k=K0)
    _, (E1_tgt_true,) = pot_p2p(queue, targets_1, e_sources_1, [e_strengths_1],
                    out_host=True, k=K1)

    _, (H0_tgt_true,) = pot_p2p(queue, targets_0, h_sources_0, [h_strengths_0],
                    out_host=True, k=K0)
    _, (H1_tgt_true,) = pot_p2p(queue, targets_1, h_sources_1, [h_strengths_1],
                    out_host=True, k=K1)

    F0_tgt_true = np.array([E0_tgt_true, H0_tgt_true])
    F1_tgt_true = np.array([E1_tgt_true, H1_tgt_true])

    err_F0 = la.norm((F0_tgt - F0_tgt_true).reshape(-1))/la.norm(F0_tgt_true.reshape(-1))  # noqa
    err_F1 = la.norm((F1_tgt - F1_tgt_true).reshape(-1))/la.norm(F1_tgt_true.reshape(-1))  # noqa

    print("Err F0", err_F0)
    print("Err F1", err_F1)

    if visualize:
        from sumpy.visualization import FieldPlotter
        fplot = FieldPlotter(np.zeros(2), extent=5, npoints=300)
        from pytential.target import PointsTarget
        fld0 = bind(
                (qbx, PointsTarget(fplot.points)),
                representation0_sym)(queue, unknown=unknown, K0=K0).get()
        fld1 = bind(
                (qbx, PointsTarget(fplot.points)),
                representation1_sym)(queue, unknown=unknown, K1=K1).get()

        e_fld0, h_fld0 = fld0
        e_fld1, h_fld1 = fld1

        _, (e_fld0_true,) = pot_p2p(queue, fplot.points, e_sources_0, [e_strengths_0],
                        out_host=True, k=K0)
        _, (e_fld1_true,) = pot_p2p(queue, fplot.points, e_sources_1, [e_strengths_1],
                        out_host=True, k=K1)
        _, (h_fld0_true,) = pot_p2p(queue, fplot.points, h_sources_0, [h_strengths_0],
                        out_host=True, k=K0)
        _, (h_fld1_true,) = pot_p2p(queue, fplot.points, h_sources_1, [h_strengths_1],
                        out_host=True, k=K1)

        #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
        fplot.write_vtk_file(
                "potential-n%d.vts" % nelements,
                [
                    ("e_fld0", e_fld0),
                    ("e_fld1", e_fld1),
                    ("e_fld0_true", e_fld0_true),
                    ("e_fld1_true", e_fld1_true),

                    ("h_fld0", h_fld0),
                    ("h_fld1", h_fld1),
                    ("h_fld0_true", h_fld0_true),
                    ("h_fld1_true", h_fld1_true),
                    ]
                )

    return err_F0, err_F1


@pytest.mark.parametrize("qbx_order", [4])
def test_dielectric(ctx_getter, qbx_order, visualize=False):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    import logging
    logging.basicConfig(level=logging.INFO)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for nelements in [30, 50, 70]:
        # prevent sympy cache 'splosion
        from sympy.core.cache import clear_cache
        clear_cache()

        errs = run_dielectric_test(
                cl_ctx, queue,
                nelements=nelements, qbx_order=qbx_order,
                visualize=visualize)

        eoc_rec.add_data_point(1/nelements, la.norm(list(errs)))

    print(eoc_rec)
    assert eoc_rec.order_estimate() > qbx_order - 0.5


# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
