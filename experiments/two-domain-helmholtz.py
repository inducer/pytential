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

from pytools import obj_array

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym, norm  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    nelements = 60
    qbx_order = 3
    k_fac = 4
    k0 = 3*k_fac
    k1 = 2.9*k_fac
    mesh_order = 10
    bdry_quad_order = mesh_order
    bdry_ovsmp_quad_order = bdry_quad_order * 4
    fmm_order = qbx_order * 2

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

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

    from sumpy.kernel import HelmholtzKernel
    kernel = HelmholtzKernel(2)

    beta = 2.5*k_fac
    K0 = np.sqrt(k0**2-beta**2)
    K1 = np.sqrt(k1**2-beta**2)

    from pytential.symbolic.pde.scalar import DielectricSDRep2DBoundaryOperator
    pde_op = DielectricSDRep2DBoundaryOperator(
            mode='tm',
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

    # in inner domain
    sources_1 = obj_array.new_1d(list(np.array([
        [-1.5, 0.5]
        ]).T.copy()))
    strengths_1 = np.array([1])

    from sumpy.p2p import P2P
    pot_p2p = P2P(cl_ctx, [kernel], exclude_self=False)

    _, (Einc,) = pot_p2p(queue, density_discr.nodes(), sources_1, [strengths_1],
                    out_host=False, k=K0)

    sqrt_w = bind(density_discr, sym.sqrt_jac_q_weight())(queue)

    bvp_rhs = np.zeros(len(pde_op.bcs), dtype=object)
    for i_bc, terms in enumerate(pde_op.bcs):
        for term in terms:
            assert term.i_interface == 0
            assert term.field_kind == pde_op.field_kind_e

            if term.direction == pde_op.dir_none:
                bvp_rhs[i_bc] += (
                        term.coeff_outer * (-Einc)
                        )
            elif term.direction == pde_op.dir_normal:
                # no jump in normal derivative
                bvp_rhs[i_bc] += 0*Einc
            else:
                raise NotImplementedError("direction spec in RHS")

        bvp_rhs[i_bc] *= sqrt_w

    from pytential.linalg.gmres import gmres
    gmres_result = gmres(
            bound_pde_op.scipy_op(queue, "unknown", dtype=np.complex128,
                domains=[sym.DEFAULT_TARGET]*2, K0=K0, K1=K1),
            bvp_rhs, tol=1e-6, progress=True,
            hard_failure=True, stall_iterations=0)

    # }}}

    unknown = gmres_result.solution

    # {{{ visualize

    from pytential.qbx import QBXLayerPotentialSource
    lap_qbx = QBXLayerPotentialSource(
            density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=qbx_order
            )

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=300)
    from pytential.target import PointsTarget
    fld0 = bind(
            (qbx, PointsTarget(fplot.points)),
            representation0_sym)(queue, unknown=unknown, K0=K0).get()
    fld1 = bind(
            (qbx, PointsTarget(fplot.points)),
            representation1_sym)(queue, unknown=unknown, K1=K1).get()
    ones = cl.array.empty(queue, density_discr.nnodes, np.float64)
    dom1_indicator = -bind(
            (lap_qbx, PointsTarget(fplot.points)),
            sym.D(0, sym.var("sigma")))(
                    queue, sigma=ones.fill(1)).get()
    _, (fld_inc_vol,) = pot_p2p(queue, fplot.points, sources_1, [strengths_1],
                    out_host=True, k=K0)

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file(
            "potential.vts",
            [
                ("fld0", fld0),
                ("fld1", fld1),
                ("fld_inc_vol", fld_inc_vol),
                ("fld_total", (
                    (fld_inc_vol + fld0)*(1-dom1_indicator)
                    +
                    fld1*dom1_indicator
                    )),
                ("dom1_indicator", dom1_indicator),
                ]
            )

    # }}}


if __name__ == "__main__":
    main()

# vim: fdm=marker
