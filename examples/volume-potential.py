from __future__ import absolute_import
from __future__ import print_function
import numpy as np  # noqa
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from meshmode.mesh.io import generate_gmsh, FileSource
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        QuadratureSimplexGroupFactory

from meshmode.discretization.visualization import make_visualizer

from pytential import bind, sym, norm, integral  # noqa
from pytools.obj_array import make_obj_array

import pytential.symbolic.primitives as p

import logging
logger = logging.getLogger(__name__)


h = 0.02
mesh_order = 3
vol_quad_order = 4
vol_ovsmp_quad_order = 4 * vol_quad_order
qbx_order = 3
vol_qbx_order = 3
fmm_order = 3

inner_rad = 0.1


def rhs_func(x, y):
    r = (x**2+y**2)**0.5
    return (r < inner_rad).astype(np.float64)


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mesh = generate_gmsh(
            FileSource("circle.step"), 2, order=mesh_order,
            force_ambient_dimension=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h]
            )

    logger.info("%d elements" % mesh.nelements)

    # {{{ discretizations and connections

    vol_discr = Discretization(ctx, mesh,
            QuadratureSimplexGroupFactory(vol_quad_order))
    ovsmp_vol_discr = Discretization(ctx, mesh,
            QuadratureSimplexGroupFactory(vol_ovsmp_quad_order))

    from meshmode.discretization.connection import make_same_mesh_connection
    vol_to_ovsmp_vol = make_same_mesh_connection(
            queue, ovsmp_vol_discr, vol_discr)

    # }}}

    # {{{ visualizers

    vol_vis = make_visualizer(queue, vol_discr, 20)
    ovsmp_vol_vis = make_visualizer(queue, ovsmp_vol_discr, 10)

    # }}}

    vol_x = vol_discr.nodes().with_queue(queue)
    ovsmp_vol_x = ovsmp_vol_discr.nodes().with_queue(queue)

    rhs = rhs_func(vol_x[0], vol_x[1])

    one = rhs.copy()
    one.fill(1)
    print("AREA", integral(vol_discr, queue, one), 0.25**2*np.pi)
    # FIXME area is wrong ?!

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

    center_dist = np.min(
            cl.clmath.sqrt(
                bind(vol_discr, p.area_element())(queue)).get())

    centers = make_obj_array([ci.copy().reshape(vol_discr.nnodes) for ci in targets])
    centers[2][:] = center_dist

    if 0:
        vol_zero = 0*rhs
        vol_vis.show_scalar_in_mayavi(vol_zero, do_show=False)

        import mayavi.mlab as mlab
        mlab.points3d(
                centers[0].get(),
                centers[1].get(),
                centers[2].get())
        mlab.show()

    sources = cl.array.zeros(queue, (3,) + ovsmp_vol_x.shape[1:], ovsmp_vol_x.dtype)
    sources[:2] = ovsmp_vol_x

    ovsmp_rhs = vol_to_ovsmp_vol(queue, rhs)
    ovsmp_vol_weights = bind(ovsmp_vol_discr, p.area_element() * p.QWeight())(queue)

    sources_reshaped = sources.reshape(3, ovsmp_vol_discr.nnodes)
    strengths_reshaped = \
                (ovsmp_vol_weights*ovsmp_rhs).reshape(ovsmp_vol_discr.nnodes)

    if 0:
        vol_zero = 0*rhs
        #vol_vis.show_scalar_in_mayavi(vol_zero, do_show=False)

        import mayavi.mlab as mlab
        mlab.points3d(
                sources_reshaped[0].get(),
                sources_reshaped[1].get(),
                sources_reshaped[2].get(),
                strengths_reshaped.get()
                )
        mlab.show()

    evt, (vol_pot,) = layer_pot(
            queue,
            targets=targets.reshape(3, vol_discr.nnodes),
            centers=centers,
            sources=sources_reshaped,
            strengths=(strengths_reshaped,),
            )

    np.set_printoptions(threshold=5000)
    print(vol_pot)
    print(vol_pot.shape)
    print(rhs.shape)

    # }}}

    ovsmp_vol_vis.write_vtk_file("volume-potential-o.vtu", [
        ("rhs", ovsmp_rhs),
        ("vol_weights", ovsmp_vol_weights)
        ])
    vol_vis.write_vtk_file("volume-potential.vtu", [
        ("vol_pot", vol_pot),
        ("rhs", rhs),
        ])

    print("h = %s" % h)
    print("mesh_order = %s" % mesh_order)
    print("vol_quad_order = %s" % vol_quad_order)
    print("qbx_order = %s" % qbx_order)
    print("vol_qbx_order = %s" % vol_qbx_order)
    print("fmm_order = %s" % fmm_order)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
