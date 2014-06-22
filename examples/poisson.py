import numpy as np  # noqa
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from meshmode.mesh.io import generate_gmsh, FileSource
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        QuadratureSimplexGroupFactory

from meshmode.discretization.visualization import make_visualizer

from pytential import bind, sym, norm  # noqa
from pytools.obj_array import make_obj_array

import pytential.symbolic.primitives as p

import logging
logger = logging.getLogger(__name__)


def sol_func(x, y):
    return 0.1*cl.clmath.sin(30*x)*cl.clmath.sin(20*y)


poisson_bc_func = sol_func


def rhs_func(x, y):
    return 0.1*-30*-20*cl.clmath.sin(30*x)*cl.clmath.sin(20*y)

mesh_order = 4
qbx_order = 3


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mesh = generate_gmsh(
            FileSource("blob-2d.step"), 2, order=mesh_order,
            force_ambient_dimension=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"]
            )

    logger.info("%d elements" % mesh.nelements)

    # {{{ discretizations and connections

    vol_discr = Discretization(ctx, mesh, QuadratureSimplexGroupFactory(mesh_order))

    from meshmode.discretization.connection import make_boundary_restriction
    bdry_mesh, bdry_discr, bdry_connection = make_boundary_restriction(
            queue, vol_discr, QuadratureSimplexGroupFactory(mesh_order))

    # }}}

    # {{{ visualizers

    vol_vis = make_visualizer(queue, vol_discr, 20)
    bdry_vis = make_visualizer(queue, bdry_discr, 20)

    # }}}

    vol_x = vol_discr.nodes().with_queue(queue)
    rhs = rhs_func(vol_x[0], vol_x[1])
    poisson_true_sol = sol_func(vol_x[0], vol_x[1])

    vol_vis.write_vtk_file("volume.vtu", [("f", rhs)])

    bdry_normals = bind(bdry_discr, p.normal())(queue).as_vector(dtype=object)
    bdry_vis.write_vtk_file("boundary.vtu", [
        ("normals", bdry_normals)
        ])

    bdry_nodes = bdry_discr.nodes().with_queue(queue)
    bdry_f = rhs_func(bdry_nodes[0], bdry_nodes[1])
    bdry_f_2 = bdry_connection(queue, rhs)

    bdry_vis.write_vtk_file("y.vtu", [("f", bdry_f_2)])

    if 0:
        vol_vis.show_scalar_in_mayavi(rhs, do_show=False)
        bdry_vis.show_scalar_in_mayavi(bdry_f - bdry_f_2, line_width=10,
                do_show=False)

        import mayavi.mlab as mlab
        mlab.colorbar()
        mlab.show()

    # {{{ compute volume potential

    from sumpy.qbx import LayerPotential
    from sumpy.expansion.local import LineTaylorLocalExpansion

    def get_kernel():
        from sumpy.symbolic import pymbolic_real_norm_2
        from pymbolic.primitives import (make_sym_vector, Variable as var)

        r = pymbolic_real_norm_2(make_sym_vector("d", 3))
        expr = var("log")(r)
        scaling = 1/(-2*var("pi"))

        from sumpy.kernel import ExpressionKernel
        return ExpressionKernel(
                dim=3,
                expression=expr,
                scaling=scaling,
                is_complex_valued=False)

    laplace_2d_in_3d_kernel = get_kernel()

    layer_pot = LayerPotential(ctx, [
        LineTaylorLocalExpansion(laplace_2d_in_3d_kernel,
            order=qbx_order)])

    targets = cl.array.zeros(queue, (3,) + vol_x.shape[1:], vol_x.dtype)
    targets[:2] = vol_x

    sources = targets
    centers = make_obj_array([ci.copy().reshape(vol_discr.nnodes) for ci in targets])
    centers[2].fill(0.1)

    vol_weights = bind(vol_discr, p.area_element() * p.QWeight())(queue)

    evt, (vol_pot,) = layer_pot(
            queue,
            targets=targets.reshape(3, vol_discr.nnodes),
            sources=sources.reshape(3, vol_discr.nnodes),
            centers=centers,
            strengths=((vol_weights*rhs).reshape(vol_discr.nnodes),)
            )

    # ??? FIXME
    vol_pot = 2*vol_pot

    vol_pot_bdry = bdry_connection(queue, vol_pot)

    # }}}

    # {{{ solve bvp

    from sumpy.kernel import LaplaceKernel
    from pytential.symbolic.pde.scalar import DirichletOperator
    op = DirichletOperator(LaplaceKernel(2), -1, use_l2_weighting=True)

    sym_sigma = sym.var("sigma")
    op_sigma = op.operator(sym_sigma)

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(
            bdry_discr, fine_order=4*mesh_order, qbx_order=qbx_order,
            fmm_order=3
            )

    bound_op = bind(qbx, op_sigma)

    poisson_bc = poisson_bc_func(bdry_nodes[0], bdry_nodes[1])
    bvp_bc = poisson_bc - vol_pot_bdry
    bdry_f = rhs_func(bdry_nodes[0], bdry_nodes[1])

    bvp_rhs = bind(bdry_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bvp_bc)

    from pytential.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma"),
            bvp_rhs, tol=1e-14, progress=True,
            hard_failure=False)

    sigma = gmres_result.solution
    print "gmres state:", gmres_result.state

    # }}}

    bvp_sol = bind(
            (qbx, vol_discr),
            op.representation(sym_sigma))(queue, sigma=sigma)

    poisson_sol = bvp_sol + vol_pot
    poisson_err = poisson_true_sol-poisson_sol

    bdry_vis.write_vtk_file("poisson-boundary.vtu", [
        ("vol_pot_bdry", vol_pot_bdry),
        ("sigma", sigma),
        ])

    vol_vis.write_vtk_file("poisson-volume.vtu", [
        ("bvp_sol", bvp_sol),
        ("poisson_sol", poisson_sol),
        ("poisson_true_sol", poisson_true_sol),
        ("poisson_err", poisson_err),
        ("vol_pot", vol_pot),
        ])

    print "rel error: %g" % (
            norm(vol_discr, queue, poisson_err)
            /
            norm(vol_discr, queue, poisson_true_sol))


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
