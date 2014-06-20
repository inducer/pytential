import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from meshmode.mesh.io import generate_gmsh, FileSource
from meshmode.discretization.poly_element import \
        PolynomialQuadratureElementDiscretization

from meshmode.discretization.visualization import make_visualizer

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mesh = generate_gmsh(
        FileSource("blob-2d.step"), 2, order=4,
        force_ambient_dimension=2
        #other_options=["-string", "Mesh.CharacteristicLength = 0.02;"]
        )

vol_discr = PolynomialQuadratureElementDiscretization(
        ctx, mesh, 10)

with cl.CommandQueue(vol_discr.cl_context) as queue:
    nodes = vol_discr.nodes().with_queue(queue).get()

vis = make_visualizer(queue, vol_discr, 10)

# FIXME
x = vis.vis_discr.nodes()[0].with_queue(queue)

f = cl.clmath.sin(x)

vis.write_vtk_file("x.vtu", [("f", f.get())])
#vis.show_scalar_in_mayavi(f.get())
