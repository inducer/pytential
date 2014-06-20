from __future__ import division
import numpy as np
import pyopencl as cl
from meshmode.mesh.generation import (  # noqa
        make_curve_mesh, starfish, ellipse, drop)
from sumpy.visualization import FieldPlotter
#from mayavi import mlab
from sumpy.kernel import one_kernel_2d, LaplaceKernel, HelmholtzKernel  # noqa

import faulthandler
faulthandler.enable()

import logging
logging.basicConfig(level=logging.INFO)

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 16
qbx_order = 3
nelements = 60
mode_nr = 1

k = 0
if k:
    kernel = HelmholtzKernel("k")
else:
    kernel = LaplaceKernel()
#kernel = OneKernel()

mesh = make_curve_mesh(
        #lambda t: ellipse(1, t),
        starfish,
        np.linspace(0, 1, nelements+1),
        target_order)

from pytential.qbx import QBXLayerPotentialSource
from meshmode.discretization.poly_element import \
        PolynomialQuadratureElementDiscretization

density_discr = PolynomialQuadratureElementDiscretization(
        cl_ctx, mesh, target_order)

qbx = QBXLayerPotentialSource(density_discr, 4*target_order, qbx_order,
        fmm_order=qbx_order)

nodes = density_discr.nodes().with_queue(queue)

angle = cl.clmath.atan2(nodes[1], nodes[0])

from pytential import bind, sym
d = sym.Derivative()
#op = d.nabla[0] * d(sym.S(kernel, sym.var("sigma")))
#op = sym.D(kernel, sym.var("sigma"))
op = sym.S(kernel, sym.var("sigma"))

sigma = cl.clmath.cos(mode_nr*angle)
if 0:
    sigma = 0*angle
    from random import randrange
    for i in range(5):
        sigma[randrange(len(sigma))] = 1

if isinstance(kernel, HelmholtzKernel):
    sigma = sigma.astype(np.complex128)

bound_bdry_op = bind(qbx, op)
#mlab.figure(bgcolor=(1, 1, 1))
if 1:
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1500)
    from pytential.target import PointsTarget
    fld_in_vol = bind(
            (qbx, PointsTarget(fplot.points)),
            op)(queue, sigma=sigma, k=k).get()

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file(
            "potential.vts",
            [
                ("potential", fld_in_vol)
                ]
            )

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

if 0:
    # {{{ plot boundary field

    fld_on_bdry = bound_bdry_op(queue, sigma=sigma, k=k).get()

    nodes_host = discr.nodes().get(queue=queue)
    #mlab.points3d(nodes_host[0], nodes_host[1], fld_on_bdry.real, scale_factor=0.03)

    # }}}

#mlab.colorbar()
#mlab.show()
