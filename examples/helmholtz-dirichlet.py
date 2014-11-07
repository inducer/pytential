import numpy as np  # noqa
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from meshmode.mesh.io import generate_gmsh, FileSource
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from meshmode.discretization.visualization import make_visualizer

from pytential import bind, sym, norm  # noqa
from pytools.obj_array import make_obj_array

import pytential.symbolic.primitives as p

import logging
logger = logging.getLogger(__name__)


nelements = 50
mesh_order = 3
bdry_quad_order = 10
bdry_ovsmp_quad_order = 4*bdry_quad_order
qbx_order = 3
fmm_order = 3
k = 3


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial
    mesh = make_curve_mesh(
            partial(ellipse, 3),
            np.linspace(0, 1, nelements+1),
            mesh_order)

    density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    logger.info("%d elements" % mesh.nelements)

    #bdry_vis = make_visualizer(queue, density_discr, 20)

    # {{{ solve bvp

    from sumpy.kernel import HelmholtzKernel
    kernel = HelmholtzKernel(2)

    cse = sym.cse

    sigma_sym = sym.var("sigma")
    sqrt_w = sym.sqrt_jac_q_weight()
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)

    alpha = 1j
    loc_sign = -1

    bdry_op_sym = (-loc_sign*0.5*sigma_sym
            + sqrt_w*(
                alpha*sym.S(kernel, inv_sqrt_w_sigma)
                - sym.D(kernel, inv_sqrt_w_sigma)
                ))

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(
            density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order
            )

    bound_op = bind(qbx, bdry_op_sym)

    mode_nr = 3
    nodes = density_discr.nodes().with_queue(queue)
    angle = cl.clmath.atan2(nodes[1], nodes[0])

    bc = cl.clmath.cos(mode_nr*angle)

    bvp_rhs = bind(qbx, sqrt_w*sym.var("bc"))(queue, bc=bc)

    from pytential.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", k=k),
            bvp_rhs, tol=1e-14, progress=True,
            hard_failure=True)

    # }}}

    sigma = gmres_result.solution

    representation_sym = (
            alpha*sym.S(kernel, inv_sqrt_w_sigma)
            - sym.D(kernel, inv_sqrt_w_sigma))

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1500)
    from pytential.target import PointsTarget
    fld_in_vol = bind(
            (qbx, PointsTarget(fplot.points)),
            representation_sym)(queue, sigma=sigma, k=k).get()

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file(
            "potential.vts",
            [
                ("potential", fld_in_vol)
                ]
            )


if __name__ == "__main__":
    main()
