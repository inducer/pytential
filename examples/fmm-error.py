from __future__ import division
import numpy as np
import pyopencl as cl
from meshmode.mesh.generation import (  # noqa
        make_curve_mesh, starfish, ellipse, drop)
from sumpy.visualization import FieldPlotter
from sumpy.kernel import LaplaceKernel, HelmholtzKernel


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 16
    qbx_order = 3
    nelements = 60
    mode_nr = 0

    k = 0
    if k:
        kernel = HelmholtzKernel(2)
    else:
        kernel = LaplaceKernel(2)
    #kernel = OneKernel()

    mesh = make_curve_mesh(
            #lambda t: ellipse(1, t),
            starfish,
            np.linspace(0, 1, nelements+1),
            target_order)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    slow_qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, fine_order=2*target_order,
            qbx_order=qbx_order, fmm_order=False,
            target_association_tolerance=.05
            ).with_refinement()
    qbx = slow_qbx.copy(fmm_order=10)
    density_discr = slow_qbx.density_discr

    nodes = density_discr.nodes().with_queue(queue)

    angle = cl.clmath.atan2(nodes[1], nodes[0])

    from pytential import bind, sym
    #op = sym.d_dx(sym.S(kernel, sym.var("sigma")), qbx_forced_limit=None)
    #op = sym.D(kernel, sym.var("sigma"), qbx_forced_limit=None)
    op = sym.S(kernel, sym.var("sigma"), qbx_forced_limit=None)

    sigma = cl.clmath.cos(mode_nr*angle)

    if isinstance(kernel, HelmholtzKernel):
        sigma = sigma.astype(np.complex128)

    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=600)
    from pytential.target import PointsTarget

    fld_in_vol = bind(
            (slow_qbx, PointsTarget(fplot.points)),
            op)(queue, sigma=sigma, k=k).get()

    fmm_fld_in_vol = bind(
            (qbx, PointsTarget(fplot.points)),
            op)(queue, sigma=sigma, k=k).get()

    err = fmm_fld_in_vol-fld_in_vol

    import matplotlib
    matplotlib.use('Agg')
    im = fplot.show_scalar_in_matplotlib(np.log10(np.abs(err) + 1e-17))

    from matplotlib.colors import Normalize
    im.set_norm(Normalize(vmin=-12, vmax=0))

    import matplotlib.pyplot as pt
    from matplotlib.ticker import NullFormatter
    pt.gca().xaxis.set_major_formatter(NullFormatter())
    pt.gca().yaxis.set_major_formatter(NullFormatter())

    cb = pt.colorbar(shrink=0.9)
    cb.set_label(r"$\log_{10}(\mathdefault{Error})$")

    pt.savefig("fmm-error-order-%d.pdf" % qbx_order)


if __name__ == "__main__":
    main()
