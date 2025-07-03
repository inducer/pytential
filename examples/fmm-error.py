import numpy as np

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.mesh.generation import drop, ellipse, make_curve_mesh, starfish  # noqa
from sumpy.kernel import HelmholtzKernel, LaplaceKernel
from sumpy.visualization import FieldPlotter


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    import pyopencl as cl
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
            # lambda t: ellipse(1, t),
            starfish,
            np.linspace(0, 1, nelements+1),
            target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )

    from pytential.qbx import QBXLayerPotentialSource

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
        }, auto_where=("qbx", "targets"))
    density_discr = places.get_discretization("unaccel_qbx")

    nodes = actx.thaw(density_discr.nodes())
    angle = actx.np.arctan2(nodes[1], nodes[0])

    from pytential import bind, sym
    if k:
        kernel_kwargs = {"k": sym.var("k")}
    else:
        kernel_kwargs = {}

    def get_op():
        kwargs = {"qbx_forced_limit": None}
        kwargs.update(kernel_kwargs)
        # return sym.d_dx(2, sym.S(kernel, sym.var("sigma"), **kwargs))
        # return sym.D(kernel, sym.var("sigma"), **kwargs)
        return sym.S(kernel, sym.var("sigma"), **kwargs)

    op = get_op()

    sigma = actx.np.cos(mode_nr*angle)

    fld_in_vol = actx.to_numpy(
            bind(
                places, op, auto_where=("unaccel_qbx", "targets")
                )(actx, sigma=sigma, k=k)
            )

    fmm_fld_in_vol = actx.to_numpy(
            bind(
                places, op, auto_where=("qbx", "targets")
                )(actx, sigma=sigma, k=k)
            )

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
