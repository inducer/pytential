import numpy as np
import pyopencl as cl
from pytential.mesh.generation import (
        make_curve_mesh, starfish)
from sumpy.visualization import FieldPlotter

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 7
qbx_order = 5
nelements = 40
mode_nr = 2
k = 5

mesh = make_curve_mesh(starfish,
        np.linspace(0, 1, nelements+1),
        target_order)

from pytential.discretization.qbx import make_upsampling_qbx_discr

discr = make_upsampling_qbx_discr(
        cl_ctx, mesh, target_order, qbx_order)

nodes = discr.nodes().with_queue(queue)

angle = cl.clmath.atan2(nodes[1], nodes[0])

from pytential import bind, sym
op = sym.D(0, sym.var("sigma"))
bound_op = bind(discr, op)

sigma = cl.clmath.cos(mode_nr*angle)
op_sigma = bound_op(queue=queue, sigma=sigma)

fplot = FieldPlotter(np.zeros(2), extent=5, npoints=200)
from pytential.discretization.target import PointsTarget
fld_in_vol = bind(
        (discr, PointsTarget(fplot.points)),
        op)(queue, sigma=sigma).get()

fld_on_bdry = bind(discr, op)(queue, sigma=sigma).get()

nodes = discr.nodes().get(queue=queue)

from mayavi import mlab
mlab.points3d(nodes[0], nodes[1], fld_on_bdry, scale_factor=0.03)
fplot.show_scalar_in_mayavi(fld_in_vol, max_val=5)
mlab.show()
