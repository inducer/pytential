import numpy as np
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym, norm  # noqa

# {{{ set some constants for use below

nelements = 20
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 4*bdry_quad_order
fmm_order = 25
k = 25

# }}}


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial

    if 0:
        mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)
    else:
        base_mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)

        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
        nx = 5
        ny = 5
        dx = 2 / nx
        meshes = [
                affine_map(
                    base_mesh,
                    A=np.diag([dx*0.25, dx*0.25]),
                    b=np.array([dx*(ix-nx/2), dx*(iy-ny/2)]))
                for ix in range(nx)
                for iy in range(ny)]

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
            fmm_order=fmm_order
            ).with_refinement()
    density_discr = qbx.density_discr

    # {{{ describe bvp

    from sumpy.kernel import HelmholtzKernel
    kernel = HelmholtzKernel(2)

    cse = sym.cse

    sigma_sym = sym.var("sigma")
    sqrt_w = sym.sqrt_jac_q_weight(2)
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)

    # Brakhage-Werner parameter
    alpha = 1j

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = +1

    bdry_op_sym = (-loc_sign*0.5*sigma_sym
            + sqrt_w*(
                alpha*sym.S(kernel, inv_sqrt_w_sigma, k=sym.var("k"))
                - sym.D(kernel, inv_sqrt_w_sigma, k=sym.var("k"))
                ))

    # }}}

    bound_op = bind(qbx, bdry_op_sym)

    # {{{ fix rhs and solve

    mode_nr = 3
    nodes = density_discr.nodes().with_queue(queue)
    angle = cl.clmath.atan2(nodes[1], nodes[0])

    bc = cl.clmath.cos(mode_nr*angle)

    bvp_rhs = bind(qbx, sqrt_w*sym.var("bc"))(queue, bc=bc)

    from pytential.solve import gmres
    # gmres_result = gmres(
    #         bound_op.scipy_op(queue, "sigma", dtype=np.complex128, k=k),
    #         bvp_rhs, tol=1e-8, progress=True,
    #         stall_iterations=0,
    #         hard_failure=True)

    # }}}

    # {{{ postprocess/visualize

    #sigma = gmres_result.solution
    sigma = bc

    repr_kwargs = dict(k=sym.var("k"), qbx_forced_limit=None)
    representation_sym = (
            alpha*sym.S(kernel, inv_sqrt_w_sigma, **repr_kwargs)
            - sym.D(kernel, inv_sqrt_w_sigma, **repr_kwargs))

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1000)

    targets = cl.array.to_device(queue, fplot.points)

    qbx_stick_out = qbx.copy(target_stick_out_factor=0.05)

    from pytential.target import PointsTarget
    try:
        fld_in_vol = bind(
                (qbx_stick_out, PointsTarget(targets)),
                representation_sym)(queue, sigma=sigma, k=k).get()
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
            "potential.vts",
            [
                ("potential", fld_in_vol)
                ]
            )

    # }}}


if __name__ == "__main__":
    main()
