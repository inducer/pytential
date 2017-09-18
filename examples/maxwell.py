# This is an untested/non-working work in progress. For a PEC Maxwell solver,
# see test/test_maxwell.py.

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
mode_nr = 0

h = 1

k = 4


def main():
    # cl.array.to_device(queue, numpy_array)
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

    from pytential.symbolic.pde.maxwell import MuellerAugmentedMFIEOperator
    pde_op = MuellerAugmentedMFIEOperator(
            omega=0.4,
            epss=[1.4, 1.0],
            mus=[1.2, 1.0],
            )
    from pytential import bind, sym

    unk = pde_op.make_unknown("sigma")
    sym_operator = pde_op.operator(unk)
    sym_rhs = pde_op.rhs(
            sym.make_sym_vector("Einc", 3),
            sym.make_sym_vector("Hinc", 3))
    sym_repr = pde_op.representation(0, unk)

    if 1:
        expr = sym_repr
        print(sym.pretty(expr))

        print("#"*80)
        from pytential.target import PointsTarget

        tgt_points=np.zeros((3,1))
        tgt_points[0,0] = 100
        tgt_points[1,0] = -200
        tgt_points[2,0] = 300

        bound_op = bind((qbx, PointsTarget(tgt_points)), expr)
        print(bound_op.code)

    if 1:

        def green3e(x,y,z,source,strength,k):
        # electric field corresponding to dyadic green's function
        # due to monochromatic electric dipole located at "source".
        # "strength" is the the intensity of the dipole.
        #  E = (I + Hess)(exp(ikr)/r) dot (strength)
        #
            dx = x - source[0]
            dy = y - source[1]
            dz = z - source[2]
            rr = np.sqrt(dx**2 + dy**2 + dz**2)

            fout = np.exp(1j*k*rr)/rr
            evec = fout*strength
            qmat = np.zeros((3,3),dtype=np.complex128)

            qmat[0,0]=(2*dx**2-dy**2-dz**2)*(1-1j*k*rr)
            qmat[1,1]=(2*dy**2-dz**2-dx**2)*(1-1j*k*rr)
            qmat[2,2]=(2*dz**2-dx**2-dy**2)*(1-1j*k*rr)

            qmat[0,0]=qmat[0,0]+(-k**2*dx**2*rr**2)
            qmat[1,1]=qmat[1,1]+(-k**2*dy**2*rr**2)
            qmat[2,2]=qmat[2,2]+(-k**2*dz**2*rr**2)

            qmat[0,1]=(3-k**2*rr**2-3*1j*k*rr)*(dx*dy)
            qmat[1,2]=(3-k**2*rr**2-3*1j*k*rr)*(dy*dz)
            qmat[2,0]=(3-k**2*rr**2-3*1j*k*rr)*(dz*dx)

            qmat[1,0]=qmat[0,1]
            qmat[2,1]=qmat[1,2]
            qmat[0,2]=qmat[2,0]

            fout=np.exp(1j*k*rr)/rr**5/k**2

            fvec = fout*np.dot(qmat,strength)
            evec = evec + fvec
            return evec

        def green3m(x,y,z,source,strength,k):
        # magnetic field corresponding to dyadic green's function
        # due to monochromatic electric dipole located at "source".
        # "strength" is the the intensity of the dipole.
        #  H = curl((I + Hess)(exp(ikr)/r) dot (strength)) = 
        #  strength \cross \grad (exp(ikr)/r)
        #
            dx = x - source[0]
            dy = y - source[1]
            dz = z - source[2]
            rr = np.sqrt(dx**2 + dy**2 + dz**2)

            fout=(1-1j*k*rr)*np.exp(1j*k*rr)/rr**3
            fvec = np.zeros(3,dtype=np.complex128)
            fvec[0] = fout*dx
            fvec[1] = fout*dy
            fvec[2] = fout*dz

            hvec = np.cross(strength,fvec)

            return hvec

        def dipole3e(x,y,z,source,strength,k):
        #
        #  evalaute electric and magnetic field due
        #  to monochromatic electric dipole located at "source"
        #  with intensity "strength"

            evec = green3e(x,y,z,source,strength,k)
            evec = evec*1j*k
            hvec = green3m(x,y,z,source,strength,k)
            return evec,hvec
            
        def dipole3m(x,y,z,source,strength,k):
        #
        #  evalaute electric and magnetic field due
        #  to monochromatic magnetic dipole located at "source"
        #  with intensity "strength"
            evec = green3m(x,y,z,source,strength,k)
            hvec = green3e(x,y,z,source,strength,k)
            hvec = -hvec*1j*k
            return evec,hvec
            

        def dipole3eall(x,y,z,sources,strengths,k):
            ns = len(strengths)
            evec = np.zeros(3,dtype=np.complex128)
            hvec = np.zeros(3,dtype=np.complex128)

            for i in range(ns):
                evect,hvect = dipole3e(x,y,z,sources[i],strengths[i],k)
                evec = evec + evect
                hvec = hvec + hvect

        nodes = density_discr.nodes().with_queue(queue).get()
        source = [0.01,-0.03,0.02]
#        source = cl.array.to_device(queue,np.zeros(3))
#        source[0] = 0.01
#        source[1] =-0.03
#        source[2] = 0.02
        strength = np.ones(3)
       
#        evec = cl.array.to_device(queue,np.zeros((3,len(nodes[0])),dtype=np.complex128))
#        hvec = cl.array.to_device(queue,np.zeros((3,len(nodes[0])),dtype=np.complex128))

        evec = np.zeros((3,len(nodes[0])),dtype=np.complex128)
        hvec = np.zeros((3,len(nodes[0])),dtype=np.complex128)
        for i in range(len(nodes[0])):
            evec[:,i],hvec[:,i] = dipole3e(nodes[0][i],nodes[1][i],nodes[2][i],source,strength,k)
        print(np.shape(hvec))
        print(type(evec))
        print(type(hvec))

        evec = cl.array.to_device(queue,evec)
        hvec = cl.array.to_device(queue,hvec)

        bvp_rhs = bind(qbx, sym_rhs)(queue,Einc=evec,Hinc=hvec)
        print(np.shape(bvp_rhs))
        print(type(bvp_rhs))
#        print(bvp_rhs)
        1/-1

        bound_op = bind(qbx, sym_operator)

        from pytential.solve import gmres
        if 0:
            gmres_result = gmres(
                bound_op.scipy_op(queue, "sigma", dtype=np.complex128, k=k),
                bvp_rhs, tol=1e-8, progress=True,
                stall_iterations=0,
                hard_failure=True)

            sigma = gmres_result.solution

        fld_at_tgt = bind((qbx, PointsTarget(tgt_points)), sym_repr)(queue,
        sigma=bvp_rhs,k=k)
        fld_at_tgt = np.array([
            fi.get() for fi in fld_at_tgt
            ])
        print(fld_at_tgt)
        1/0

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
        from pytential.qbx import QBXTargetAssociationFailedException

        rho_sym = sym.var("rho")

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
        except QBXTargetAssociationFailedException as e:
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
