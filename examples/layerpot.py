from __future__ import division, absolute_import

enable_mayavi = 0
if enable_mayavi:
    from mayavi import mlab  # noqa

import numpy as np
import pyopencl as cl
from sumpy.visualization import FieldPlotter
from sumpy.kernel import one_kernel_2d, LaplaceKernel, HelmholtzKernel  # noqa

from pytential import bind, sym
from six.moves import range

from meshmode.mesh.generation import starfish, ellipse, drop # noqa

target_order = 16
qbx_order = 3
nelements = 60
mode_nr = 3

k = 0


def main(curve_fn=starfish, visualize=True):
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import make_curve_mesh
    mesh = make_curve_mesh(
            curve_fn,
            np.linspace(0, 1, nelements+1),
            target_order)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    pre_density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx = QBXLayerPotentialSource(pre_density_discr, 4*target_order, qbx_order,
            fmm_order=qbx_order+3,
            target_association_tolerance=0.005)

    from pytential.target import PointsTarget
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1000)
    targets_dev = cl.array.to_device(queue, fplot.points)

    from pytential.symbolic.execution import GeometryCollection
    places = GeometryCollection({
        sym.DEFAULT_SOURCE: qbx,
        sym.DEFAULT_TARGET: qbx.density_discr,
        'targets': PointsTarget(targets_dev),
        })
    density_discr = places.get_discretization(places.auto_source)

    nodes = density_discr.nodes().with_queue(queue)
    angle = cl.clmath.atan2(nodes[1], nodes[0])

    if k:
        kernel = HelmholtzKernel(2)
        kernel_kwargs = {"k": sym.var("k")}
    else:
        kernel = LaplaceKernel(2)
        kernel_kwargs = {}

    def op(**kwargs):
        kwargs.update(kernel_kwargs)

        #op = sym.d_dx(sym.S(kernel, sym.var("sigma"), **kwargs))
        return sym.D(kernel, sym.var("sigma"), **kwargs)
        #op = sym.S(kernel, sym.var("sigma"), qbx_forced_limit=None, **kwargs)

    sigma = cl.clmath.cos(mode_nr*angle)
    if 0:
        sigma = 0*angle
        from random import randrange
        for i in range(5):
            sigma[randrange(len(sigma))] = 1

    if isinstance(kernel, HelmholtzKernel):
        sigma = sigma.astype(np.complex128)

    bound_bdry_op = bind(places, op())
    if visualize:
        fld_in_vol = bind(places, op(
            source=sym.DEFAULT_SOURCE,
            target='targets',
            qbx_forced_limit=None))(queue, sigma=sigma, k=k).get()

        if enable_mayavi:
            fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
        else:
            fplot.write_vtk_file("potential-2d.vts", [
                ("potential", fld_in_vol)
                ])

    if 0:
        def apply_op(density):
            return bound_bdry_op(
                    queue, sigma=cl.array.to_device(queue, density), k=k).get()

        from sumpy.tools import build_matrix
        n = len(sigma)
        mat = build_matrix(apply_op, dtype=np.float64, shape=(n, n))

        import matplotlib.pyplot as pt
        pt.imshow(mat)
        pt.colorbar()
        pt.show()

    if enable_mayavi:
        # {{{ plot boundary field

        fld_on_bdry = bound_bdry_op(queue, sigma=sigma, k=k).get()

        nodes_host = density_discr.nodes().get(queue=queue)
        mlab.points3d(nodes_host[0], nodes_host[1],
                fld_on_bdry.real, scale_factor=0.03)

        # }}}

    if enable_mayavi:
        mlab.colorbar()
        mlab.show()


if __name__ == "__main__":
    main()
