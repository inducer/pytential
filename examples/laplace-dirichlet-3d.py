import numpy as np

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym
from pytential.target import PointsTarget

# {{{ set some constants for use below

nelements = 20
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 4*bdry_quad_order
fmm_order = 3

# }}}


def main(mesh_name="torus", visualize=False):
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    if mesh_name == "torus":
        rout = 10
        rin = 1

        from meshmode.mesh.generation import generate_torus
        base_mesh = generate_torus(
                rout, rin, 40, 4,
                order=mesh_order)

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

        if visualize:
            from meshmode.mesh.visualization import draw_curve
            draw_curve(mesh)
            import matplotlib.pyplot as plt
            plt.show()
    else:
        raise ValueError(f"unknown mesh name: {mesh_name}")

    pre_density_discr = Discretization(
            actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import (
            QBXLayerPotentialSource, QBXTargetAssociationFailedError)
    qbx = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            )

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(3), extent=20, npoints=50)
    targets = actx.from_numpy(fplot.points)

    from pytential import GeometryCollection
    places = GeometryCollection({
        "qbx": qbx,
        "qbx_target_assoc": qbx.copy(target_association_tolerance=0.2),
        "targets": PointsTarget(targets)
        }, auto_where="qbx")
    density_discr = places.get_discretization("qbx")

    # {{{ describe bvp

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(3)

    sigma_sym = sym.var("sigma")
    # sqrt_w = sym.sqrt_jac_q_weight(3)
    sqrt_w = 1
    inv_sqrt_w_sigma = sym.cse(sigma_sym/sqrt_w)

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = +1

    bdry_op_sym = (loc_sign*0.5*sigma_sym
            + sqrt_w*(
                sym.S(kernel, inv_sqrt_w_sigma, qbx_forced_limit=+1)
                + sym.D(kernel, inv_sqrt_w_sigma, qbx_forced_limit="avg")
                ))

    # }}}

    bound_op = bind(places, bdry_op_sym)

    # {{{ fix rhs and solve

    nodes = actx.thaw(density_discr.nodes())
    source = np.array([rout, 0, 0], dtype=object)

    def u_incoming_func(x):
        dists = x - source
        return 1.0 / actx.np.sqrt(sum(dists**2))

    bc = u_incoming_func(nodes)
    bvp_rhs = bind(places, sqrt_w*sym.var("bc"))(actx, bc=bc)

    from pytential.linalg.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(actx, "sigma", dtype=np.float64),
            bvp_rhs, tol=1e-14, progress=True,
            stall_iterations=0,
            hard_failure=True)

    sigma = bind(places, sym.var("sigma")/sqrt_w)(
            actx, sigma=gmres_result.solution)

    # }}}

    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(actx, density_discr, 20)
    bdry_vis.write_vtk_file("laplace.vtu", [
        ("sigma", sigma),
        ])

    # {{{ postprocess/visualize

    repr_kwargs = {
            "source": "qbx_target_assoc",
            "target": "targets",
            "qbx_forced_limit": None}
    representation_sym = (
            sym.S(kernel, inv_sqrt_w_sigma, **repr_kwargs)
            + sym.D(kernel, inv_sqrt_w_sigma, **repr_kwargs))

    try:
        fld_in_vol = actx.to_numpy(
                bind(places, representation_sym)(actx, sigma=sigma))
    except QBXTargetAssociationFailedError as e:
        fplot.write_vtk_file("laplace-dirichlet-3d-failed-targets.vts", [
            ("failed", actx.to_numpy(e.failed_target_flags)),
            ])
        raise

    # fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file("laplace-dirichlet-3d-potential.vts", [
        ("potential", fld_in_vol),
        ])

    # }}}


if __name__ == "__main__":
    main()
