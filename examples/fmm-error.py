from __future__ import division
import numpy as np
import pyopencl as cl
from pytential.mesh.generation import (  # noqa
        make_curve_mesh, starfish, ellipse, drop)
from sumpy.visualization import FieldPlotter
from sumpy.kernel import OneKernel, LaplaceKernel, HelmholtzKernel

import faulthandler
faulthandler.enable()

import logging
logging.basicConfig(level=logging.INFO)

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 16
qbx_order = 3
nelements = 60
mode_nr = 0

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

from pytential.discretization.qbx import make_upsampling_qbx_discr

discr = make_upsampling_qbx_discr(
        cl_ctx, mesh, target_order, qbx_order, fmm_order=qbx_order)
slow_discr = make_upsampling_qbx_discr(
        cl_ctx, mesh, target_order, qbx_order=5, fmm_order=False)
        #source_order=target_order)

nodes = discr.nodes().with_queue(queue)

angle = cl.clmath.atan2(nodes[1], nodes[0])

from pytential import bind, sym
d = sym.Derivative()
#op = d.nabla[0] * d(sym.S(kernel, sym.var("sigma")))
#op = sym.D(kernel, sym.var("sigma"))
op = sym.S(kernel, sym.var("sigma"))

sigma = cl.clmath.cos(mode_nr*angle)

if isinstance(kernel, HelmholtzKernel):
    sigma = sigma.astype(np.complex128)

bound_bdry_op = bind(discr, op)

fplot = FieldPlotter(np.zeros(2), extent=5, npoints=600)
from pytential.discretization.target import PointsTarget

fld_in_vol = bind(
        (slow_discr, PointsTarget(fplot.points)),
        op)(queue, sigma=sigma, k=k).get()

fmm_fld_in_vol = bind(
        (discr, PointsTarget(fplot.points)),
        op)(queue, sigma=sigma, k=k).get()

err = fmm_fld_in_vol-fld_in_vol
im = fplot.show_scalar_in_matplotlib(np.log10(np.abs(err)))

from matplotlib.colors import Normalize
im.set_norm(Normalize(vmin=-6, vmax=0))

import matplotlib.pyplot as pt
from matplotlib.ticker import NullFormatter
pt.gca().xaxis.set_major_formatter(NullFormatter())
pt.gca().yaxis.set_major_formatter(NullFormatter())

cb = pt.colorbar(shrink=0.9)
cb.set_label(r"$\log_{10}(\mathdefault{Error})$")

pt.savefig("fmm-error-order-%d.pdf" % qbx_order)
