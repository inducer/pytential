import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym, norm  # noqa
from pytential.target import PointsTarget

# {{{ set some constants for use below

nelements = 20
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 4*bdry_quad_order
fmm_order = 3

# }}}


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_torus

    rout = 10
    rin = 1
    if 1:
        base_mesh = generate_torus(
                rout, rin, 40, 4,
                mesh_order)

        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
        # nx = 1
        # ny = 1
        nz = 1
        dz = 0
        meshes = [
                affine_map(
                    base_mesh,
                    A=np.diag([1, 1, 1]),
                    b=np.array([0, 0, iz*dz]))
                for iz in range(nz)]

        mesh = merge_disjoint_meshes(meshes, single_group=True)

        if 0:
            from meshmode.mesh.visualization import draw_curve
            draw_curve(mesh)
            import matplotlib.pyplot as plt
            plt.show()

    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import (
            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            ).with_refinement()
    density_discr = qbx.density_discr

    # {{{ describe bvp

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(3)

    cse = sym.cse

    sigma_sym = sym.var("sigma")
    #sqrt_w = sym.sqrt_jac_q_weight(3)
    sqrt_w = 1
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = +1

    bdry_op_sym = (loc_sign*0.5*sigma_sym
            + sqrt_w*(
                sym.S(kernel, inv_sqrt_w_sigma)
                + sym.D(kernel, inv_sqrt_w_sigma)
                ))

    # }}}

    bound_op = bind(qbx, bdry_op_sym)

    # {{{ fix rhs and solve

    nodes = density_discr.nodes().with_queue(queue)
    source = np.array([rout, 0, 0])

    def u_incoming_func(x):
        #        return 1/cl.clmath.sqrt( (x[0] - source[0])**2
        #                                +(x[1] - source[1])**2
        #                                +(x[2] - source[2])**2 )
        return 1.0/la.norm(x.get()-source[:, None], axis=0)

    bc = cl.array.to_device(queue, u_incoming_func(nodes))

    bvp_rhs = bind(qbx, sqrt_w*sym.var("bc"))(queue, bc=bc)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.float64),
            bvp_rhs, tol=1e-14, progress=True,
            stall_iterations=0,
            hard_failure=True)

    sigma = bind(qbx, sym.var("sigma")/sqrt_w)(queue, sigma=gmres_result.solution)

    # }}}

    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(queue, density_discr, 20)
    bdry_vis.write_vtk_file("laplace.vtu", [
        ("sigma", sigma),
        ])

    # {{{ postprocess/visualize

    repr_kwargs = dict(qbx_forced_limit=None)
    representation_sym = (
            sym.S(kernel, inv_sqrt_w_sigma, **repr_kwargs)
            + sym.D(kernel, inv_sqrt_w_sigma, **repr_kwargs))

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(3), extent=20, npoints=50)

    targets = cl.array.to_device(queue, fplot.points)

    qbx_stick_out = qbx.copy(target_stick_out_factor=0.2)

    try:
        fld_in_vol = bind(
                (qbx_stick_out, PointsTarget(targets)),
                representation_sym)(queue, sigma=sigma).get()
    except QBXTargetAssociationFailedException as e:
        fplot.write_vtk_file(
                "failed-targets.vts",
                [
                    ("failed", e.failed_target_flags.get(queue))
                    ]
                )
        raise

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file(
            "potential-laplace-3d.vts",
            [
                ("potential", fld_in_vol),
                ]
            )

    # }}}


if __name__ == "__main__":
    main()
