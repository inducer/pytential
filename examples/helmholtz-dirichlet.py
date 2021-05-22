import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from arraycontext import PyOpenCLArrayContext, thaw
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
fmm_order = 10
k = 3

# }}}


def main(mesh_name="ellipse", visualize=False):
    import logging
    logging.basicConfig(level=logging.INFO)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial

    if mesh_name == "ellipse":
        mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)
    elif mesh_name == "ellipse_array":
        base_mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)

        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
        nx = 2
        ny = 2
        dx = 2 / nx
        meshes = [
                affine_map(
                    base_mesh,
                    A=np.diag([dx*0.25, dx*0.25]),
                    b=np.array([dx*(ix-nx/2), dx*(iy-ny/2)]))
                for ix in range(nx)
                for iy in range(ny)]

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
            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    qbx = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order
            )

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)
    targets = actx.from_numpy(fplot.points)

    from pytential import GeometryCollection
    places = GeometryCollection({
        "qbx": qbx,
        "qbx_high_target_assoc_tol": qbx.copy(target_association_tolerance=0.05),
        "targets": PointsTarget(targets)
        }, auto_where="qbx")
    density_discr = places.get_discretization("qbx")

    # {{{ describe bvp

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    kernel = HelmholtzKernel(2)

    sigma_sym = sym.var("sigma")
    sqrt_w = sym.sqrt_jac_q_weight(2)
    inv_sqrt_w_sigma = sym.cse(sigma_sym/sqrt_w)

    # Brakhage-Werner parameter
    alpha = 1j

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = +1

    k_sym = sym.var("k")
    bdry_op_sym = (-loc_sign*0.5*sigma_sym
            + sqrt_w*(
                alpha*sym.S(kernel, inv_sqrt_w_sigma, k=k_sym,
                    qbx_forced_limit=+1)
                - sym.D(kernel, inv_sqrt_w_sigma, k=k_sym,
                    qbx_forced_limit="avg")
                ))

    # }}}

    bound_op = bind(places, bdry_op_sym)

    # {{{ fix rhs and solve

    nodes = thaw(density_discr.nodes(), actx)
    k_vec = np.array([2, 1])
    k_vec = k * k_vec / la.norm(k_vec, 2)

    def u_incoming_func(x):
        return actx.np.exp(
                1j * (x[0] * k_vec[0] + x[1] * k_vec[1]))

    bc = -u_incoming_func(nodes)

    bvp_rhs = bind(places, sqrt_w*sym.var("bc"))(actx, bc=bc)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(actx, sigma_sym.name, dtype=np.complex128, k=k),
            bvp_rhs, tol=1e-8, progress=True,
            stall_iterations=0,
            hard_failure=True)

    # }}}

    # {{{ postprocess/visualize

    repr_kwargs = dict(
            source="qbx_high_target_assoc_tol",
            target="targets",
            qbx_forced_limit=None)
    representation_sym = (
            alpha*sym.S(kernel, inv_sqrt_w_sigma, k=k_sym, **repr_kwargs)
            - sym.D(kernel, inv_sqrt_w_sigma, k=k_sym, **repr_kwargs))

    u_incoming = u_incoming_func(targets)
    ones_density = density_discr.zeros(actx)
    for elem in ones_density:
        elem.fill(1)

    indicator = actx.to_numpy(
            bind(places, sym.D(LaplaceKernel(2), sigma_sym, **repr_kwargs))(
                actx, sigma=ones_density))

    try:
        fld_in_vol = actx.to_numpy(
                bind(places, representation_sym)(
                    actx, sigma=gmres_result.solution, k=k))
    except QBXTargetAssociationFailedException as e:
        fplot.write_vtk_file("helmholtz-dirichlet-failed-targets.vts", [
            ("failed", e.failed_target_flags.get(queue))
            ])
        raise

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file("helmholtz-dirichlet-potential.vts", [
        ("potential", fld_in_vol),
        ("indicator", indicator),
        ("u_incoming", actx.to_numpy(u_incoming)),
        ])

    # }}}


if __name__ == "__main__":
    main()
