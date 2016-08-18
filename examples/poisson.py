from __future__ import absolute_import
from __future__ import print_function
import numpy as np  # noqa
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from meshmode.mesh.generation import generate_regular_rect_mesh
from meshmode.mesh.io import generate_gmsh, FileSource  # noqa
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from meshmode.discretization.visualization import make_visualizer

from pytential import bind, sym, norm  # noqa
from pytools.obj_array import make_obj_array

import pytential.symbolic.primitives as p

import logging
logger = logging.getLogger(__name__)


if 0:
    x_sin_factor = 30
    y_sin_factor = 10

    def sol_func(x, y):
        return 0.1*cl.clmath.sin(x_sin_factor*x)*cl.clmath.sin(y_sin_factor*y)

    poisson_bc_func = sol_func

    def rhs_func(x, y):
        return -x_sin_factor*-y_sin_factor*sol_func(x, y)

elif 0:
    def sol_func(x, y):
        return x+y

    poisson_bc_func = sol_func

    def rhs_func(x, y):
        return 0*x

elif 1:
    a = 300
    #xc = 0.2
    #yc = 0.1
    xc = 0
    yc = 0
    exp = cl.clmath.exp

    def sol_func(x, y):
        return exp(-a*(y-yc)**2-a*(x-xc)**2)

    poisson_bc_func = sol_func

    def rhs_func(x, y):
        base = sol_func(x, y)
        return (4*a**2*(y-yc)**2*base
                + 4*a**2*(x-xc)**2*base
                - 4*a*base)

h = 0.05
mesh_order = 2
vol_quad_order = 5
vol_ovsmp_quad_order = 4*vol_quad_order
bdry_quad_order = vol_quad_order
bdry_ovsmp_quad_order = 4*bdry_quad_order
qbx_order = 4
vol_qbx_order = 2
fmm_order = False


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    if 1:
        ext = 0.5
        mesh = generate_regular_rect_mesh(
                a=(-ext/2, -ext/2), b=(ext/2, ext/2), n=(int(ext/h), int(ext/h)))
    else:
        mesh = generate_gmsh(
                FileSource("circle.step"), 2, order=mesh_order,
                force_ambient_dim=2,
                other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h]
                )

    logger.info("%d elements" % mesh.nelements)

    # {{{ discretizations and connections

    vol_discr = Discretization(ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(vol_quad_order))
    ovsmp_vol_discr = Discretization(ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(vol_ovsmp_quad_order))

    from meshmode.mesh import BTAG_ALL
    from meshmode.discretization.connection import (
            make_face_restriction, make_same_mesh_connection)
    bdry_connection = make_face_restriction(
            vol_discr, InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order),
            BTAG_ALL)

    bdry_discr = bdry_connection.to_discr

    vol_to_ovsmp_vol = make_same_mesh_connection(ovsmp_vol_discr, vol_discr)

    # }}}

    # {{{ visualizers

    vol_vis = make_visualizer(queue, vol_discr, 20)
    bdry_vis = make_visualizer(queue, bdry_discr, 20)

    # }}}

    vol_x = vol_discr.nodes().with_queue(queue)
    ovsmp_vol_x = ovsmp_vol_discr.nodes().with_queue(queue)

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
        scaling = 1/(2*var("pi"))

        from sumpy.kernel import ExpressionKernel
        return ExpressionKernel(
                dim=3,
                expression=expr,
                scaling=scaling,
                is_complex_valued=False)

    laplace_2d_in_3d_kernel = get_kernel()

    layer_pot = LayerPotential(ctx, [
        LineTaylorLocalExpansion(laplace_2d_in_3d_kernel,
            order=vol_qbx_order)])

    targets = cl.array.zeros(queue, (3,) + vol_x.shape[1:], vol_x.dtype)
    targets[:2] = vol_x

    center_dist = 0.125*np.min(
            cl.clmath.sqrt(
                bind(vol_discr, p.area_element())(queue)).get())

    centers = make_obj_array([ci.copy().reshape(vol_discr.nnodes) for ci in targets])
    centers[2][:] = center_dist

    print(center_dist)

    sources = cl.array.zeros(queue, (3,) + ovsmp_vol_x.shape[1:], ovsmp_vol_x.dtype)
    sources[:2] = ovsmp_vol_x

    ovsmp_rhs = vol_to_ovsmp_vol(queue, rhs)
    ovsmp_vol_weights = bind(ovsmp_vol_discr, p.area_element() * p.QWeight())(queue)

    print("volume: %d source nodes, %d target nodes" % (
        ovsmp_vol_discr.nnodes, vol_discr.nnodes))
    evt, (vol_pot,) = layer_pot(
            queue,
            targets=targets.reshape(3, vol_discr.nnodes),
            centers=centers,
            sources=sources.reshape(3, ovsmp_vol_discr.nnodes),
            strengths=(
                (ovsmp_vol_weights*ovsmp_rhs).reshape(ovsmp_vol_discr.nnodes),)
            )

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
            bdry_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            enable_direct_close_evaluation=False
            )

    bound_op = bind(qbx, op_sigma)

    poisson_bc = poisson_bc_func(bdry_nodes[0], bdry_nodes[1])
    bvp_bc = poisson_bc - vol_pot_bdry
    bdry_f = rhs_func(bdry_nodes[0], bdry_nodes[1])

    bvp_rhs = bind(bdry_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bvp_bc)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.float64),
            bvp_rhs, tol=1e-14, progress=True,
            hard_failure=False)

    sigma = gmres_result.solution
    print("gmres state:", gmres_result.state)

    # }}}

    bvp_sol = bind(
            (qbx, vol_discr),
            op.representation(sym_sigma))(queue, sigma=sigma)

    poisson_sol = bvp_sol + vol_pot
    poisson_err = poisson_sol-poisson_true_sol

    rel_err = (
            norm(vol_discr, queue, poisson_err)
            /
            norm(vol_discr, queue, poisson_true_sol))
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
        ("rhs", rhs),
        ])

    print("h = %s" % h)
    print("mesh_order = %s" % mesh_order)
    print("vol_quad_order = %s" % vol_quad_order)
    print("vol_ovsmp_quad_order = %s" % vol_ovsmp_quad_order)
    print("bdry_quad_order = %s" % bdry_quad_order)
    print("bdry_ovsmp_quad_order = %s" % bdry_ovsmp_quad_order)
    print("qbx_order = %s" % qbx_order)
    print("vol_qbx_order = %s" % vol_qbx_order)
    print("fmm_order = %s" % fmm_order)
    print()
    print("rel err: %g" % rel_err)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
