import numpy as np

from meshmode.array_context import PyOpenCLArrayContext
from sumpy.visualization import FieldPlotter
from sumpy.kernel import one_kernel_2d, LaplaceKernel, HelmholtzKernel  # noqa: F401

from pytential import bind, sym

from meshmode.mesh.generation import starfish, ellipse, drop            # noqa: F401

target_order = 16
qbx_order = 3
nelements = 60
mode_nr = 3

k = 0


def main(curve_fn=starfish, visualize=True):
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    from meshmode.mesh.generation import make_curve_mesh
    mesh = make_curve_mesh(
            curve_fn,
            np.linspace(0, 1, nelements+1),
            target_order)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    pre_density_discr = Discretization(
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order, qbx_order,
            fmm_order=False,
            target_association_tolerance=0.005,
            #fmm_backend="fmmlib",
            )

    from pytential.target import PointsTarget
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=300)
    targets_dev = actx.from_numpy(fplot.points)

    from pytential import GeometryCollection
    places = GeometryCollection({
        "qbx": qbx,
        "targets": PointsTarget(targets_dev),
        }, auto_where="qbx")

    density_discr = places.get_discretization("qbx")

    nodes = actx.thaw(density_discr.nodes())
    angle = actx.np.arctan2(nodes[1], nodes[0])

    if k:
        kernel = HelmholtzKernel(2)
        kernel_kwargs = {"k": sym.var("k")}
    else:
        kernel = LaplaceKernel(2)
        kernel_kwargs = {}

    def op(**kwargs):
        kwargs.update(kernel_kwargs)

        #return sym.d_dx(2, sym.S(kernel, sym.var("sigma"), **kwargs))
        return sym.D(kernel, sym.var("sigma"), **kwargs)
        # return sym.S(kernel, sym.var("sigma"), **kwargs)

    if 0:
        from random import randrange
        sigma = actx.zeros(density_discr.ndofs, angle.entry_dtype)
        for _ in range(5):
            sigma[randrange(len(sigma))] = 1

        from arraycontext import unflatten
        sigma = unflatten(angle, sigma, actx)
    else:
        sigma = actx.np.cos(mode_nr*angle)

    bound_bdry_op = bind(places, op())
    if visualize:
        fld_in_vol = actx.to_numpy(
                bind(places, op(
                    source="qbx",
                    target="targets",
                    qbx_forced_limit=None))(actx, sigma=sigma, k=k))

        fplot.write_vtk_file("layerpot-potential.vts", [
            ("potential", fld_in_vol)
            ])

    if 0:
        apply_op = bound_bdry_op.scipy_op(actx, "sigma", np.float64, k=k)
        from sumpy.tools import build_matrix
        mat = build_matrix(apply_op)

        import matplotlib.pyplot as pt
        pt.imshow(mat)
        pt.colorbar()
        pt.show()


if __name__ == "__main__":
    main()
