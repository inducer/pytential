import pyopencl as cl
import numpy as np
import numpy.linalg as la  # noqa: F401

from pytential import bind, sym, norm  # noqa: F401


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 10

    from functools import partial
    nelements = 30
    qbx_order = 4

    from sumpy.kernel import LaplaceKernel
    from meshmode.mesh.generation import (  # noqa
            ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut,
            make_curve_mesh)
    mesh = make_curve_mesh(partial(ellipse, 1),
            np.linspace(0, 1, nelements+1),
            target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    density_discr = Discretization(cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(density_discr, 4*target_order,
            qbx_order, fmm_order=False)

    from pytools import obj_array
    sig_sym = sym.var("sig")
    knl = LaplaceKernel(2)
    op = obj_array.flat(
            sym.tangential_derivative(mesh.ambient_dim,
                sym.D(knl, sig_sym, qbx_forced_limit=+1)).as_scalar(),
            sym.tangential_derivative(mesh.ambient_dim,
                sym.D(knl, sig_sym, qbx_forced_limit=-1)).as_scalar(),
            )

    nodes = density_discr.nodes().with_queue(queue)
    angle = cl.clmath.atan2(nodes[1], nodes[0])
    n = 10
    sig = cl.clmath.sin(n*angle)
    dt_sig = n*cl.clmath.cos(n*angle)

    res = bind(qbx, op)(queue, sig=sig)

    extval = res[0].get()
    intval = res[1].get()
    pv = 0.5*(extval + intval)

    dt_sig_h = dt_sig.get()

    import matplotlib.pyplot as pt
    pt.plot(extval, label="+num")
    pt.plot(pv + dt_sig_h*0.5, label="+ex")
    pt.legend(loc="best")
    pt.show()


if __name__ == "__main__":
    main()
