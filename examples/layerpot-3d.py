from __future__ import division

import numpy as np
import pyopencl as cl

from sumpy.visualization import FieldPlotter
from sumpy.kernel import one_kernel_2d, LaplaceKernel, HelmholtzKernel  # noqa

from pytential import bind, sym
from six.moves import range

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 5
qbx_order = 3
mode_nr = 4

if 1:
    cad_file_name = "geometries/ellipsoid.step"
    h = 0.6
else:
    cad_file_name = "geometries/two-cylinders-smooth.step"
    h = 0.4

k = 0
if k:
    kernel = HelmholtzKernel(3)
else:
    kernel = LaplaceKernel(3)
#kernel = OneKernel()


def main():
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    from meshmode.mesh.io import generate_gmsh, FileSource
    mesh = generate_gmsh(
            FileSource(cad_file_name), 2, order=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h],
            target_unit="MM")

    from meshmode.mesh.processing import perform_flips
    # Flip elements--gmsh generates inside-out geometry.
    mesh = perform_flips(mesh, np.ones(mesh.nelements))

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

    from pytential.target import PointsTarget
    fplot = FieldPlotter(bbox_center, extent=3.5*bbox_size, npoints=150)

    from pytential.symbolic.execution import GeometryCollection
    places = GeometryCollection(qbx).places
    places.update({'targets': PointsTarget(fplot.points)})

    places = GeometryCollection(places)
    density_discr = places.get_discretization(places.auto_source)

    nodes = density_discr.nodes().with_queue(queue)
    angle = cl.clmath.atan2(nodes[1], nodes[0])

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

    fld_in_vol = bind(places, op, auto_where=(sym.DEFAULT_SOURCE, 'targets'))(
            queue, sigma=sigma, k=k).get()

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file(
            "potential-3d.vts",
            [
                ("potential", fld_in_vol)
                ]
            )

    bdry_normals = bind(places,
            sym.normal(density_discr.ambient_dim))(queue).as_vector(dtype=object)

    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(queue, density_discr, target_order)

    bdry_vis.write_vtk_file("source-3d.vtu", [
        ("sigma", sigma),
        ("bdry_normals", bdry_normals),
        ])


if __name__ == "__main__":
    main()
