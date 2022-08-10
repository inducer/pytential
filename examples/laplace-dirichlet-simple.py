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
fmm_order = False
k = 3

# }}}


def main(mesh_name="starfish", visualize=False):
    import logging
    logging.basicConfig(level=logging.INFO)

    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    from meshmode.mesh.generation import ellipse, make_curve_mesh, starfish
    from functools import partial

    if mesh_name == "ellipse":
        mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)
    elif mesh_name == "starfish":
        mesh = make_curve_mesh(
                starfish,
                np.linspace(0, 1, nelements+1),
                mesh_order)
    else:
        raise ValueError(f"unknown mesh name: {mesh_name}")

    pre_density_discr = Discretization(
            actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import QBXLayerPotentialSource
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

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(2)

    sigma_sym = sym.var("sigma")

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = +1

    bdry_op_sym = (-loc_sign*0.5*sigma_sym
                - sym.D(kernel, sigma_sym, qbx_forced_limit="avg"))

    # }}}

    bound_op = bind(places, bdry_op_sym)

    # {{{ fix rhs and solve

    nodes = actx.thaw(density_discr.nodes())
    bvp_rhs = actx.np.sin(nodes[0])

    from pytential.linalg.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(actx, sigma_sym.name, dtype=np.float64),
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
            - sym.D(kernel, sigma_sym, **repr_kwargs))

    fld_in_vol = actx.to_numpy(
            bind(places, representation_sym)(
                actx, sigma=gmres_result.solution, k=k))

    if visualize:
        fplot.write_vtk_file("laplace-dirichlet-potential.vts", [
            ("potential", fld_in_vol),
            ])

    # }}}


if __name__ == "__main__":
    main(visualize=True)
