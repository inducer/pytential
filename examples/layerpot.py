import numpy as np
import pyopencl as cl
from pytential.mesh.generation import (  # noqa
        make_curve_mesh, starfish, drop)
from sumpy.visualization import FieldPlotter

import logging
logging.basicConfig(level=logging.INFO)

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 7
qbx_order = 5
nelements = 90
mode_nr = 0
k = 0

mesh = make_curve_mesh(drop,
        np.linspace(0, 1, nelements+1),
        target_order)

from pytential.discretization.qbx import make_upsampling_qbx_discr

discr = make_upsampling_qbx_discr(
        cl_ctx, mesh, target_order, qbx_order)

nodes = discr.nodes().with_queue(queue)

angle = cl.clmath.atan2(nodes[1], nodes[0])

from pytential import bind, sym
d = sym.Derivative()
#op = d.nabla[0] * d(sym.S("k" if k else 0, sym.var("sigma")))
op = sym.D("k" if k else 0, sym.var("sigma"))

sigma = cl.clmath.cos(mode_nr*angle)
if k:
    sigma = sigma.astype(np.complex128)

fplot = FieldPlotter(np.zeros(2), extent=5, npoints=100)
from pytential.discretization.target import PointsTarget
fld_in_vol = bind(
        (discr, PointsTarget(fplot.points)),
        op)(queue, sigma=sigma, k=k).get()

fld_on_bdry = bind(discr, op)(queue, sigma=sigma, k=k).get()

nodes = discr.nodes().get(queue=queue)

from mayavi import mlab
fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)

if 1:
    # {{{ plot boundary field

    mlab.points3d(nodes[0], nodes[1], fld_on_bdry.real, scale_factor=0.03)

    # }}}

mlab.colorbar()
mlab.show()
