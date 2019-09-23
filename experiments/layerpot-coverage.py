from sumpy import set_caching_enabled
set_caching_enabled(False)

import logging
if 0:
    logging.basicConfig(level='INFO')

import pyopencl as cl

from functools import partial
import numpy as np
from meshmode.mesh.generation import make_curve_mesh

def get_ellipse_mesh(resolution, aspect_ratio, mesh_order):
    from meshmode.mesh.generation import ellipse
    curve_func = partial(ellipse, aspect_ratio)
    return make_curve_mesh(curve_func,
                           np.linspace(0, 1, resolution+1),
                           mesh_order)

def run_test(cl_ctx, queue):

    q_order = 5
    qbx_order = q_order
    fmm_backend = "sumpy"
    mesh = get_ellipse_mesh(20, 40, mesh_order=5)
    a = 1
    b = 1 / 40

    if 0:
        from meshmode.mesh.visualization import draw_curve
        import matplotlib.pyplot as plt
        draw_curve(mesh)
        plt.axes().set_aspect('equal')
        plt.show()

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(q_order))

    refiner_extra_kwargs = {
            # "_expansion_disturbance_tolerance": 0.05,
            "_scaled_max_curvature_threshold": 1,
            "maxiter": 10,
            }

    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr,
            fine_order=4 * q_order,
            qbx_order=qbx_order,
            fmm_backend=fmm_backend,
            fmm_order=qbx_order + 5,
            ).with_refinement(**refiner_extra_kwargs)

    if 1:
        print("%d stage-1 elements after refinement"
                % qbx.density_discr.mesh.nelements)
        print("%d stage-2 elements after refinement"
                % qbx.stage2_density_discr.mesh.nelements)
        print("quad stage-2 elements have %d nodes"
                % qbx.quad_stage2_density_discr.groups[0].nunit_nodes)

    def reference_solu(rvec):
        # a harmonic function
        x, y = rvec
        return 2.1 * x * y + (x**2 - y**2) * 0.5 + x

    bvals = reference_solu(qbx.density_discr.nodes().with_queue(queue))

    from pytential.symbolic.pde.scalar import DirichletOperator
    from sumpy.kernel import LaplaceKernel
    from pytential import sym, bind
    op = DirichletOperator(LaplaceKernel(2), -1)

    bound_op = bind(
            qbx.copy(target_association_tolerance=0.5),
            op.operator(sym.var('sigma')))
    rhs = bind(qbx.density_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bvals)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.float64),
            rhs,
            tol=1e-12,
            progress=True,
            hard_failure=True,
            stall_iterations=50, no_progress_factor=1.05)


    from sumpy.visualization import FieldPlotter
    from pytential.target import PointsTarget
    pltsize = b * 1.5
    fplot = FieldPlotter(np.array([-1 + pltsize * 0.5, 0]), extent=pltsize * 1.05, npoints=500)
    plt_targets = cl.array.to_device(queue, fplot.points)

    interior_pts = (fplot.points[0]**2 / a**2 + fplot.points[1]**2 / b**2) < 0.99

    exact_vals = reference_solu(fplot.points)
    out_errs = []

    for assotol in [0.05]:

        qbx_stick_out = qbx.copy(target_association_tolerance=0.05)

        vol_solution = bind((qbx_stick_out, PointsTarget(plt_targets)),
                op.representation(sym.var('sigma')))(
                        queue, sigma=gmres_result.solution).get()


        interior_error_linf = (
                np.linalg.norm(
                    np.abs(vol_solution - exact_vals)[interior_pts], ord=np.inf)
                /
                np.linalg.norm(exact_vals[interior_pts], ord=np.inf)
                )

        interior_error_l2 = (
                np.linalg.norm(
                    np.abs(vol_solution - exact_vals)[interior_pts], ord=2)
                /
                np.linalg.norm(exact_vals[interior_pts], ord=2)
                )

        print("\nassotol = %f" % assotol)
        print("L_inf Error = %e " % interior_error_linf)
        print("L_2 Error = %e " % interior_error_l2)

        out_errs.append(("error-%f" % assotol, np.abs(vol_solution - exact_vals)))

    if 1:
        fplot.write_vtk_file("results.vts", out_errs)


if __name__ == '__main__':
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    run_test(cl_ctx, queue)

