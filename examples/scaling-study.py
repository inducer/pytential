import numpy as np
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
fmm_order = 10
k = 0

# }}}


def make_mesh(nx, ny):
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

    if 0:
        from meshmode.mesh.visualization import draw_curve
        draw_curve(mesh)
        import matplotlib.pyplot as plt
        plt.show()

    return mesh


def timing_run(nx, ny):
    import logging
    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    mesh = make_mesh(nx=nx, ny=ny)

    density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import (
            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    qbx = QBXLayerPotentialSource(
            density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order
            )

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

    sigma = cl.clmath.cos(mode_nr*angle)

    # }}}

    # {{{ postprocess/visualize

    repr_kwargs = dict(k=sym.var("k"), qbx_forced_limit=+1)

    sym_op = sym.S(kernel, sym.var("sigma"), **repr_kwargs)
    bound_op = bind(qbx, sym_op)

    print("FMM WARM-UP RUN 1: %d elements" % mesh.nelements)
    bound_op(queue, sigma=sigma, k=k)
    print("FMM WARM-UP RUN 2: %d elements" % mesh.nelements)
    bound_op(queue, sigma=sigma, k=k)
    queue.finish()
    print("FMM TIMING RUN: %d elements" % mesh.nelements)

    from time import time
    t_start = time()

    bound_op(queue, sigma=sigma, k=k)
    queue.finish()
    elapsed = time()-t_start

    print("FMM TIMING RUN DONE: %d elements -> %g s"
            % (mesh.nelements, elapsed))

    return (mesh.nelements, elapsed)

    if 0:
        from sumpy.visualization import FieldPlotter
        fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1500)

        targets = cl.array.to_device(queue, fplot.points)

        qbx_tgt_tol = qbx.copy(target_association_tolerance=0.05)

        indicator_qbx = qbx_tgt_tol.copy(
                fmm_level_to_order=lambda lev: 7, qbx_order=2)

        ones_density = density_discr.zeros(queue)
        ones_density.fill(1)
        indicator = bind(
                (indicator_qbx, PointsTarget(targets)),
                sym_op)(
                queue, sigma=ones_density).get()

        try:
            fld_in_vol = bind(
                    (qbx_stick_out, PointsTarget(targets)),
                    sym_op)(queue, sigma=sigma, k=k).get()
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
                    ("potential", fld_in_vol),
                    ("indicator", indicator)
                    ]
                )

    # }}}


if __name__ == "__main__":
    results = []
    for nx, ny in [
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
            ]:

        results.append(timing_run(nx, ny))

    for r in results:
        print(r)
