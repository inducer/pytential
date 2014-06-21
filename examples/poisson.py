import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from meshmode.mesh.io import generate_gmsh, FileSource
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        QuadratureSimplexGroupFactory

from meshmode.discretization.visualization import make_visualizer

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mesh = generate_gmsh(
        FileSource("blob-2d.step"), 2, order=4,
        force_ambient_dimension=2,
        other_options=["-string", "Mesh.CharacteristicLengthMax = 0.4;"]
        )

vol_discr = Discretization(ctx, mesh, QuadratureSimplexGroupFactory(10))

with cl.CommandQueue(vol_discr.cl_context) as queue:
    nodes = vol_discr.nodes().with_queue(queue).get()

vis = make_visualizer(queue, vol_discr, 20)

x = vol_discr.nodes()[0].with_queue(queue)
f = 0.1*cl.clmath.sin(30*x)

vis.write_vtk_file("x.vtu", [("f", f)])
#vis.show_scalar_in_mayavi(f, do_show=False)

from meshmode.discretization.connection import make_boundary_extractor
bdry_mesh, bdry_discr, bdry_connection = make_boundary_extractor(
        queue, vol_discr, QuadratureSimplexGroupFactory(10))

bdry_x = bdry_discr.nodes()[0].with_queue(queue)
bdry_f = 0.1*cl.clmath.sin(30*bdry_x)

bdry_vis = make_visualizer(queue, bdry_discr, 20)

#bdry_vis.show_scalar_in_mayavi(bdry_f, line_width=10)
bdry_vis.write_vtk_file("y.vtu", [("f", bdry_f)])
