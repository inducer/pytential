import numpy as np
import pyopencl as cl

from arraycontext import PyOpenCLArrayContext, thaw

from sumpy.visualization import FieldPlotter
from sumpy.kernel import one_kernel_2d, LaplaceKernel, HelmholtzKernel  # noqa

from pytential import bind, sym

target_order = 5
qbx_order = 3
mode_nr = 4
k = 0


def main(mesh_name="ellipsoid"):
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    if mesh_name == "ellipsoid":
        cad_file_name = "geometries/ellipsoid.step"
        h = 0.6
    elif mesh_name == "two-cylinders":
        cad_file_name = "geometries/two-cylinders-smooth.step"
        h = 0.4
    else:
        raise ValueError("unknown mesh name: %s" % mesh_name)

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
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx = QBXLayerPotentialSource(density_discr, 4*target_order, qbx_order,
            fmm_order=qbx_order + 3,
            target_association_tolerance=0.15)

    from pytential.target import PointsTarget
    fplot = FieldPlotter(bbox_center, extent=3.5*bbox_size, npoints=150)

    from pytential import GeometryCollection
    places = GeometryCollection({
        "qbx": qbx,
        "targets": PointsTarget(fplot.points)
        }, auto_where="qbx")
    density_discr = places.get_discretization("qbx")

    nodes = thaw(density_discr.nodes(), actx)
    angle = actx.np.arctan2(nodes[1], nodes[0])

    if k:
        kernel = HelmholtzKernel(3)
    else:
        kernel = LaplaceKernel(3)

    #op = sym.d_dx(sym.S(kernel, sym.var("sigma"), qbx_forced_limit=None))
    op = sym.D(kernel, sym.var("sigma"), qbx_forced_limit=None)
    #op = sym.S(kernel, sym.var("sigma"), qbx_forced_limit=None)

    sigma = actx.np.cos(mode_nr*angle)
    if 0:
        from meshmode.dof_array import flatten, unflatten
        sigma = flatten(0 * angle)
        from random import randrange
        for _ in range(5):
            sigma[randrange(len(sigma))] = 1
        sigma = unflatten(actx, density_discr, sigma)

    if isinstance(kernel, HelmholtzKernel):
        for i, elem in np.ndenumerate(sigma):
            sigma[i] = elem.astype(np.complex128)

    fld_in_vol = actx.to_numpy(
            bind(places, op, auto_where=("qbx", "targets"))(
                actx, sigma=sigma, k=k))

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file("layerpot-3d-potential.vts", [
        ("potential", fld_in_vol)
        ])

    bdry_normals = bind(places,
            sym.normal(density_discr.ambient_dim))(actx).as_vector(dtype=object)

    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(actx, density_discr, target_order)
    bdry_vis.write_vtk_file("layerpot-3d-density.vtu", [
        ("sigma", sigma),
        ("bdry_normals", bdry_normals),
        ])


if __name__ == "__main__":
    main()
