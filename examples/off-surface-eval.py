__copyright__ = "Copyright (C) 2022 Hao Gao"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import pyopencl as cl
from arraycontext import PyOpenCLArrayContext
import numpy as np
import functools
import pytential
import matplotlib.pyplot as plt
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    cl_context = cl.create_some_context()
    queue = cl.CommandQueue(cl_context)
    actx = PyOpenCLArrayContext(queue)

    nelements = 30
    target_order = 8
    qbx_order = 3
    fmm_order = qbx_order

    discretization = None
    if mpi_rank == 0:
        from meshmode.mesh.generation import make_curve_mesh
        from meshmode.mesh.generation import ellipse
        mesh = make_curve_mesh(
            functools.partial(ellipse, 3.0),
            np.linspace(0, 1, nelements + 1),
            target_order)

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
        discretization = Discretization(
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

    # FIXME: Use GeometryCollection instead of DistributedQBXLayerPotentialSource
    from pytential.qbx.distributed import DistributedQBXLayerPotentialSource
    layer_pot_source = DistributedQBXLayerPotentialSource(
        comm,
        cl_context,
        discretization,
        fine_order=4 * target_order,
        qbx_order=qbx_order,
        fmm_order=fmm_order)

    from sumpy.kernel import LaplaceKernel
    op = pytential.sym.D(LaplaceKernel(2), pytential.sym.var("sigma"), qbx_forced_limit=-2)

    if mpi_rank == 0:
        sigma = layer_pot_source.density_discr.zeros(actx) + 1

        from sumpy.visualization import FieldPlotter
        from pytential.target import PointsTarget
        fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)
        targets = PointsTarget(fplot.points)

        places = (layer_pot_source, targets)
    else:
        sigma = None
        places = (layer_pot_source, None)

    from pytential.symbolic.execution import bind_distributed
    bound_op = bind_distributed(comm, places, op)
    potential = bound_op.eval(context={"sigma": sigma}, array_context=actx)

    if mpi_rank == 0:
        fplot.show_scalar_in_matplotlib(potential.get())
        plt.colorbar()
        plt.show()

    if mpi_rank == 0:
        from pytential.qbx import QBXLayerPotentialSource
        qbx = QBXLayerPotentialSource(
            discretization,
            4*target_order,
            qbx_order,
            fmm_order=fmm_order,
            fmm_backend="fmmlib"
            )

        from pytential import bind
        potential = bind((qbx, targets), op)(actx, sigma=sigma)

        fplot.show_scalar_in_matplotlib(potential.get())
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    main()
