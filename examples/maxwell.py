from __future__ import division
import numpy as np
import pyopencl as cl
from sumpy.visualization import FieldPlotter
#from mayavi import mlab
from sumpy.kernel import one_kernel_2d, LaplaceKernel, HelmholtzKernel  # noqa

import faulthandler
from six.moves import range
faulthandler.enable()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 5
qbx_order = 3
nelements = 60
mode_nr = 0

h = 1

k = 4


def main():
    from meshmode.mesh.io import generate_gmsh, FileSource
    mesh = generate_gmsh(
            FileSource("ellipsoid.step"), 2, order=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h])

    from meshmode.mesh.processing import perform_flips
    # Flip elements--gmsh generates inside-out geometry.
    mesh = perform_flips(mesh, np.ones(mesh.nelements))

    print("%d elements" % mesh.nelements)

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

    qbx = QBXLayerPotentialSource(density_discr, 4*target_order, qbx_order,
            fmm_order=qbx_order + 10, fmm_backend="fmmlib")

    from pytential.symbolic.pde.maxwell import PECAugmentedMFIEOperator
    pde_op = PECAugmentedMFIEOperator()
    from pytential import bind, sym

    rho_sym = sym.var("rho")
    # jt_sym = sym.make_sym_vector("jt", 2)
    # repr_op = pde_op.scattered_volume_field(jt_sym, rho_sym)

    # {{{ make a density

    nodes = density_discr.nodes().with_queue(queue)

    angle = cl.clmath.atan2(nodes[1], nodes[0])

    sigma = cl.clmath.cos(mode_nr*angle)
    if 0:
        sigma = 0*angle
        from random import randrange
        for i in range(5):
            sigma[randrange(len(sigma))] = 1

    sigma = sigma.astype(np.complex128)

    jt = sym.make_obj_array([sigma, sigma])
    rho = sigma

    # }}}

    #mlab.figure(bgcolor=(1, 1, 1))
    if 1:
        from meshmode.discretization.visualization import make_visualizer
        bdry_vis = make_visualizer(queue, density_discr, target_order)

        bdry_normals = bind(density_discr, sym.normal(3))(queue)\
                .as_vector(dtype=object)

        bdry_vis.write_vtk_file("source.vtu", [
            ("sigma", sigma),
            ("bdry_normals", bdry_normals),
            ])

        fplot = FieldPlotter(bbox_center, extent=2*bbox_size, npoints=(150, 150, 1))

        qbx_tgt_tol = qbx.copy(target_association_tolerance=0.1)
        from pytential.target import PointsTarget
        from pytential.qbx import TargetAssociationFailedException
        try:
            fld_in_vol = bind(
                    (qbx_tgt_tol, PointsTarget(fplot.points)),
                    sym.make_obj_array([
                        sym.S(pde_op.kernel, rho_sym, k=sym.var("k"),
                            qbx_forced_limit=None),
                        sym.d_dx(3, sym.S(pde_op.kernel, rho_sym, k=sym.var("k"),
                            qbx_forced_limit=None)),
                        sym.d_dy(3, sym.S(pde_op.kernel, rho_sym, k=sym.var("k"),
                            qbx_forced_limit=None)),
                        sym.d_dz(3, sym.S(pde_op.kernel, rho_sym, k=sym.var("k"),
                            qbx_forced_limit=None)),
                        ])
                    )(queue, jt=jt, rho=rho, k=k)
        except TargetAssociationFailedException as e:
            fplot.write_vtk_file(
                    "failed-targets.vts",
                    [
                        ("failed_targets", e.failed_target_flags.get(queue))
                        ])
            raise

        fld_in_vol = sym.make_obj_array(
            [fiv.get() for fiv in fld_in_vol])

        #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
        fplot.write_vtk_file(
                "potential.vts",
                [
                    ("potential", fld_in_vol[0]),
                    ("grad", fld_in_vol[1:]),
                    ]
                )


if __name__ == "__main__":
    main()
