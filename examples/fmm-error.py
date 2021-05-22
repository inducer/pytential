import numpy as np
import pyopencl as cl

from arraycontext import PyOpenCLArrayContext, thaw
from meshmode.mesh.generation import (  # noqa
        make_curve_mesh, starfish, ellipse, drop)

from sumpy.visualization import FieldPlotter
from sumpy.kernel import LaplaceKernel, HelmholtzKernel


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    target_order = 16
    qbx_order = 3
    nelements = 60
    mode_nr = 0

    k = 0
    if k:
        kernel = HelmholtzKernel(2)
    else:
        kernel = LaplaceKernel(2)

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
            actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    unaccel_qbx = QBXLayerPotentialSource(
            pre_density_discr, fine_order=2*target_order,
            qbx_order=qbx_order, fmm_order=False,
            target_association_tolerance=.05,
            )

    from pytential.target import PointsTarget
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=600)

    from pytential import GeometryCollection
    places = GeometryCollection({
        "unaccel_qbx": unaccel_qbx,
        "qbx": unaccel_qbx.copy(fmm_order=10),
        "targets": PointsTarget(actx.freeze(actx.from_numpy(fplot.points)))
        })
    density_discr = places.get_discretization("unaccel_qbx")

    nodes = thaw(density_discr.nodes(), actx)
    angle = actx.np.arctan2(nodes[1], nodes[0])

    from pytential import bind, sym
    if k:
        kernel_kwargs = {"k": sym.var("k")}
    else:
        kernel_kwargs = {}

    def get_op():
        kwargs = dict(qbx_forced_limit=None)
        kwargs.update(kernel_kwargs)
        # return sym.d_dx(2, sym.S(kernel, sym.var("sigma"), **kwargs))
        # return sym.D(kernel, sym.var("sigma"), **kwargs)
        return sym.S(kernel, sym.var("sigma"), **kwargs)

    op = get_op()

    sigma = actx.np.cos(mode_nr*angle)

    if isinstance(kernel, HelmholtzKernel):
        for i, elem in np.ndenumerate(sigma):
            sigma[i] = elem.astype(np.complex128)

    fld_in_vol = bind(places, op, auto_where=("unaccel_qbx", "targets"))(
            actx, sigma=sigma, k=k).get()

    fmm_fld_in_vol = bind(places, op, auto_where=("qbx", "targets"))(
            actx, sigma=sigma, k=k).get()

    err = fmm_fld_in_vol-fld_in_vol

    try:
        import matplotlib
    except ImportError:
        return

    matplotlib.use("Agg")
    im = fplot.show_scalar_in_matplotlib(np.log10(np.abs(err) + 1e-17))

    from matplotlib.colors import Normalize
    im.set_norm(Normalize(vmin=-12, vmax=0))

    import matplotlib.pyplot as pt
    from matplotlib.ticker import NullFormatter
    pt.gca().xaxis.set_major_formatter(NullFormatter())
    pt.gca().yaxis.set_major_formatter(NullFormatter())

    cb = pt.colorbar(shrink=0.9)
    cb.set_label(r"$\log_{10}(\mathrm{Error})$")

    pt.savefig("fmm-error-order-%d.pdf" % qbx_order)


if __name__ == "__main__":
    main()
