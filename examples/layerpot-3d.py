from __future__ import division
import numpy as np
import pyopencl as cl
from sumpy.visualization import FieldPlotter
#from mayavi import mlab
from sumpy.kernel import one_kernel_2d, LaplaceKernel, HelmholtzKernel  # noqa

import faulthandler
from six.moves import range
faulthandler.enable()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 5
qbx_order = 3
mode_nr = 4

if 1:
    cad_file_name = "ellipsoid.step"
    h = 0.6
else:
    cad_file_name = "two-cylinders-smooth.step"
    h = 0.4

k = 0
if k:
    kernel = HelmholtzKernel(3)
else:
    kernel = LaplaceKernel(3)
#kernel = OneKernel()

from meshmode.mesh.io import generate_gmsh, FileSource
mesh = generate_gmsh(
        FileSource(cad_file_name), 2, order=2,
        other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h])

from meshmode.mesh.processing import find_bounding_box
bbox_min, bbox_max = find_bounding_box(mesh)
bbox_center = 0.5*(bbox_min+bbox_max)
bbox_size = max(bbox_max-bbox_min) / 2

logger.info("%d elements" % mesh.nelements)

from pytential.qbx import QBXLayerPotentialSource
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

density_discr = Discretization(
        cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

qbx, _ = QBXLayerPotentialSource(density_discr, 4*target_order, qbx_order,
        fmm_order=qbx_order + 3,
        target_association_tolerance=0.15).with_refinement()

nodes = density_discr.nodes().with_queue(queue)

angle = cl.clmath.atan2(nodes[1], nodes[0])

from pytential import bind, sym
#op = sym.d_dx(sym.S(kernel, sym.var("sigma"), qbx_forced_limit=None))
op = sym.D(kernel, sym.var("sigma"), qbx_forced_limit=None)
#op = sym.S(kernel, sym.var("sigma"), qbx_forced_limit=None)

sigma = cl.clmath.cos(mode_nr*angle)
if 0:
    sigma = 0*angle
    from random import randrange
    for i in range(5):
        sigma[randrange(len(sigma))] = 1

if isinstance(kernel, HelmholtzKernel):
    sigma = sigma.astype(np.complex128)

fplot = FieldPlotter(bbox_center, extent=3.5*bbox_size, npoints=150)

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

bdry_normals = bind(
        density_discr,
        sym.normal(density_discr.ambient_dim))(queue).as_vector(dtype=object)

from meshmode.discretization.visualization import make_visualizer
bdry_vis = make_visualizer(queue, density_discr, target_order)

bdry_vis.write_vtk_file("source.vtu", [
    ("sigma", sigma),
    ("bdry_normals", bdry_normals),
    ])
