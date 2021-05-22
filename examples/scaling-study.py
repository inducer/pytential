import numpy as np
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
k = 0

# }}}


def make_mesh(nx, ny, visualize=False):
    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial

    base_mesh = make_curve_mesh(
            partial(ellipse, 1),
            np.linspace(0, 1, nelements+1),
            mesh_order)

    from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
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

    return mesh


def timing_run(nx, ny, visualize=False):
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    mesh = make_mesh(nx=nx, ny=ny, visualize=visualize)

    density_discr = Discretization(
            actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import (
            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    qbx = QBXLayerPotentialSource(
            density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order
            )

    places = {"qbx": qbx}
    if visualize:
        from sumpy.visualization import FieldPlotter
        fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1500)
        targets = PointsTarget(actx.from_numpy(fplot.points))

        places.update({
            "plot-targets": targets,
            "qbx-indicator": qbx.copy(
                target_association_tolerance=0.05,
                fmm_level_to_order=lambda lev: 7,
                qbx_order=2),
            "qbx-target-assoc": qbx.copy(target_association_tolerance=0.1)
            })

    from pytential import GeometryCollection
    places = GeometryCollection(places, auto_where="qbx")
    density_discr = places.get_discretization("qbx")

    # {{{ describe bvp

    from sumpy.kernel import HelmholtzKernel
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
    S_sym = sym.S(kernel, inv_sqrt_w_sigma, k=k_sym, qbx_forced_limit=+1)
    D_sym = sym.D(kernel, inv_sqrt_w_sigma, k=k_sym, qbx_forced_limit="avg")
    bdry_op_sym = -loc_sign*0.5*sigma_sym + sqrt_w*(alpha*S_sym + D_sym)

    # }}}

    bound_op = bind(places, bdry_op_sym)

    # {{{ fix rhs and solve

    mode_nr = 3

    nodes = thaw(density_discr.nodes(), actx)
    angle = actx.np.arctan2(nodes[1], nodes[0])

    sigma = actx.np.cos(mode_nr*angle)

    # }}}

    # {{{ postprocess/visualize

    repr_kwargs = dict(k=sym.var("k"), qbx_forced_limit=+1)

    sym_op = sym.S(kernel, sym.var("sigma"), **repr_kwargs)
    bound_op = bind(places, sym_op)

    print("FMM WARM-UP RUN 1: %5d elements" % mesh.nelements)
    bound_op(actx, sigma=sigma, k=k)
    queue.finish()

    print("FMM WARM-UP RUN 2: %5d elements" % mesh.nelements)
    bound_op(actx, sigma=sigma, k=k)
    queue.finish()

    from time import time
    t_start = time()
    bound_op(actx, sigma=sigma, k=k)
    actx.queue.finish()
    elapsed = time() - t_start

    print("FMM TIMING RUN:    %5d elements -> %g s"
            % (mesh.nelements, elapsed))

    if visualize:
        ones_density = density_discr.zeros(queue)
        ones_density.fill(1)
        indicator = bind(places, sym_op,
                auto_where=("qbx-indicator", "plot-targets"))(
                queue, sigma=ones_density).get()

        try:
            fld_in_vol = bind(places, sym_op,
                    auto_where=("qbx-target-assoc", "plot-targets"))(
                    queue, sigma=sigma, k=k).get()
        except QBXTargetAssociationFailedException as e:
            fplot.write_vtk_file("scaling-study-failed-targets.vts", [
                ("failed", e.failed_target_flags.get(queue)),
                ])
            raise

        fplot.write_vtk_file("scaling-study-potential.vts", [
            ("potential", fld_in_vol),
            ("indicator", indicator),
            ])

    return (mesh.nelements, elapsed)

    # }}}


if __name__ == "__main__":
    grid_sizes = [
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 5),
            (5, 5),
            (5, 6),
            (6, 6),
            (6, 7),
            (7, 7),
            (7, 8),
            (8, 8),
            (8, 9),
            (9, 9),
            (9, 10),
            (10, 10),
            ]

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for nx, ny in grid_sizes:
        npoints, t_elapsed = timing_run(nx, ny)
        eoc.add_data_point(npoints, t_elapsed)
    print(eoc.pretty_print(
        abscissa_label="Elements",
        error_label="Timing (s)"))
