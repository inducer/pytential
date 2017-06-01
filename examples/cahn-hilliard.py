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
fmm_order = 8

# }}}


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial

    mesh = make_curve_mesh(
                partial(ellipse, 2),
                np.linspace(0, 1, nelements+1),
                mesh_order)

    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import (
            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            expansion_disks_in_tree_have_extent=True,
            ).with_refinement()
    density_discr = qbx.density_discr

    from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator
    chop = CahnHilliardOperator(b=5, c=1)

    unk = chop.make_unknown("sigma")
    bound_op = bind(qbx, chop.operator(unk))

    # {{{ fix rhs and solve

    nodes = density_discr.nodes().with_queue(queue)

    def g(xvec):
        x, y = xvec
        return cl.clmath.cos(5*cl.clmath.atan2(y, x))

    bc = sym.make_obj_array([
        # FIXME: Realistic BC
        5+g(nodes),
        3-g(nodes),
        ])

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.complex128),
            bc, tol=1e-8, progress=True,
            stall_iterations=0,
            hard_failure=True)

    sigma = gmres_result.solution

    # }}}

    # {{{ check pde

    def check_pde():
        from sumpy.point_calculus import CalculusPatch
        cp = CalculusPatch(np.zeros(2))
        targets = cl.array.to_device(queue, cp.points)

        u, v = bind(
                (qbx, PointsTarget(targets)),
                chop.representation(unk))(queue, sigma=sigma)

        u = u.get().real
        v = v.get().real

        print(la.norm(u), la.norm(v))
        print(la.norm(
            cp.laplace(cp.laplace(u)) - chop.b * cp.laplace(u) + chop.c*u))

        print(la.norm(
            v + cp.laplace(u) - chop.b*u))
        1/0

    check_pde()

    # }}}

    # {{{ postprocess/visualize

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)

    targets = cl.array.to_device(queue, fplot.points)

    indicator_qbx = qbx_stick_out.copy(qbx_order=2)

    from sumpy.kernel import LaplaceKernel
    ones_density = density_discr.zeros(queue)
    ones_density.fill(1)
    indicator = bind(
            (indicator_qbx, PointsTarget(targets)),
            sym.D(LaplaceKernel(2), sym.var("sigma")))(
            queue, sigma=ones_density).get()

    try:
        u, v = bind(
                (qbx_stick_out, PointsTarget(targets)),
                chop.representation(unk))(queue, sigma=sigma)
    except QBXTargetAssociationFailedException as e:
        fplot.write_vtk_file(
                "failed-targets.vts",
                [
                    ("failed", e.failed_target_flags.get(queue))
                    ]
                )
        raise
    u = u.get().real
    v = v.get().real

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file(
            "potential.vts",
            [
                ("u", u),
                ("v", v),
                ("indicator", indicator),
                ]
            )

    # }}}


if __name__ == "__main__":
    main()
