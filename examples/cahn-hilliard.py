import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym, norm  # noqa
from pytential.target import PointsTarget

from pytools.obj_array import make_obj_array
import pytential.symbolic.primitives as p

from sumpy.kernel import ExpressionKernel
import loopy as lp

# {{{ set some constants for use below

# {{{ all kinds of orders
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 4 * bdry_quad_order
fmm_order = 8

vol_quad_order = 5
vol_qbx_order = 2
# }}}
# {{{ mesh generation
nelements = 20

from enum import Enum


class Geometry(Enum):
    RegularRectangle = 1
    Circle = 2


shape = Geometry.Circle
# }}}
# {{{ physical parameters
s = 1.5
epsilon = 0.01
delta_t = 0.05
final_t = delta_t * 1
theta_y = 60. / 180. * np.pi

b = s / (epsilon**2)
c = 1. / (epsilon * delta_t)
# }}}
# {{{ initial phi

# This phi function is also used to do PDE check
import pymbolic as pmbl
x = pmbl.var("x")
y = pmbl.var("y")

# FIXME: modify pymbolic to use tanh function
# phi = tanh(x / sqrt (2 * epsilon))
phi = x**2 + x * y + 2 * y**2
phi3 = phi**3
laphi  = pmbl.differentiate(pmbl.differentiate(phi, 'x'), 'x') + \
         pmbl.differentiate(pmbl.differentiate(phi, 'y'), 'y')
laphi3 = pmbl.differentiate(pmbl.differentiate(phi3, 'x'), 'x') + \
         pmbl.differentiate(pmbl.differentiate(phi3, 'y'), 'y')
f1_expr = c * phi - (1 + s) / epsilon**2 * laphi + 1 / epsilon**2 * laphi3
f2_expr = (phi3 - (1 + s) * phi) / epsilon**2


def f1_func(x, y):
    return pmbl.evaluate(f1_expr, {"x": x, "y": y})


def f2_func(x, y):
    return pmbl.evaluate(f2_expr, {"x": x, "y": y})


def initial_phi(x, y):
    return pmbl.evaluate(phi, {"x": x, "y": y})


#def initial_phi(x, y):
#   return np.tanh(x / np.sqrt(2. * initial_epsilon))
# }}}

# }}}

# {{{ a kernel class for G0
# FIXME: will the expressions work when lambda is complex?
# (may need conversion from 1j to var("I"))
from sumpy.kernel import ExpressionKernel


class ShidongKernel(ExpressionKernel):
    init_arg_names = ("dim", "lambda1", "lambda2")

    def __init__(self, dim=None, lambda1=0., lambda2=1.):
        """
        :arg lambda1,lambda2: The roots of the quadratic equation w.r.t
             laplacian.
         """
        # Assert against repeated roots.
        if abs(lambda1**2 - lambda2**2) < 1e-9:
            raise RuntimeError("illposed since input roots are too close")

# Based on http://mathworld.wolfram.com/ModifiedBesselFunctionoftheSecondKind.html
        if dim == 2:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr    = var("hankel_1")(0, var("I") * lambda1 * r) - \
                        var("hankel_1")(0, var("I") * lambda2 * r)
            scaling = 1. / (4. * var("I") * (lambda1**2 - lambda2**2))
        else:
            raise RuntimeError("unsupported dimensionality")

        ExpressionKernel.__init__(
            self,
            dim,
            expression=expr,
            scaling=scaling,
            is_complex_valued=True)

        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __getinitargs__(self):
        return (self._dim, self.lambda1, self.lambda2)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self._dim, self.lambda1, self.lambda2))

    def __repr__(self):
        if self._dim is not None:
            return "ShdgKnl%dD(%f, %f)" % (self._dim, self.lambda1,
                                           self.lambda2)
        else:
            return "ShdgKnl(%f, %f)" % (self.lambda1, self.lambda2)

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import (bessel_preamble_generator, bessel_mangler)
        loopy_knl = lp.register_function_manglers(loopy_knl, [bessel_mangler])
        loopy_knl = lp.register_preamble_generators(
            loopy_knl, [bessel_preamble_generator])
        return loopy_knl

    def get_args(self):
        k_dtype = np.complex128
        return [
            KernelArgument(
                loopy_arg=lp.ValueArg("shidong_kernel", k_dtype), )
        ]

    mapper_method = "map_shidong_kernel"


# }}}


# {{{ extended kernel getters
class HankelBasedKernel(ExpressionKernel):
    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import (bessel_preamble_generator, bessel_mangler)
        loopy_knl = lp.register_function_manglers(loopy_knl, [bessel_mangler])
        loopy_knl = lp.register_preamble_generators(
            loopy_knl, [bessel_preamble_generator])

        return loopy_knl


def get_extkernel_for_G0(lambda1, lambda2):
    from sumpy.symbolic import pymbolic_real_norm_2
    from pymbolic.primitives import make_sym_vector
    from pymbolic import var

    d = make_sym_vector("d", 3)
    r2 = pymbolic_real_norm_2(d[:-1])
    expr = var("hankel_1")(0, var("I") * lambda1 * r2
            + var("I") * d[-1]**2) \
            - var("hankel_1")(0, var("I") * lambda2 * r2
            + var("I") * d[-1]**2)
    scaling = 1. / (4. * var("I") * (lambda1**2 - lambda2**2))

    return HankelBasedKernel(
        dim=3, expression=expr, scaling=scaling, is_complex_valued=True)


def get_extkernel_for_G1(lamb):
    from sumpy.symbolic import pymbolic_real_norm_2
    from pymbolic.primitives import make_sym_vector
    from pymbolic import var

    d = make_sym_vector("d", 3)
    r2 = pymbolic_real_norm_2(d[:-1])
    expr = var("hankel_1")(0, var("I") * lamb * r2 + var("I") * d[-1]**2)
    scaling = -var("I") / 4.

    return HankelBasedKernel(
        dim=3, expression=expr, scaling=scaling, is_complex_valued=True)


# }}}


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ volume mesh generation

    if shape == Geometry.RegularRectangle:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        ext = 1.
        h = 0.05
        mesh = generate_regular_rect_mesh(
            a=(-ext / 2., -ext / 2.),
            b=(ext / 2., ext / 2.),
            n=(int(ext / h), int(ext / h)))
    elif shape == Geometry.Circle:
        from meshmode.mesh.io import generate_gmsh, FileSource
        h = 0.05
        mesh = generate_gmsh(
            FileSource("circle.step"),
            2,
            order=mesh_order,
            force_ambient_dim=2,
            other_options=[
                "-string", "Mesh.CharacteristicLengthMax = %g;" % h
            ])
    else:
        RuntimeError("unsupported geometry")

    logger.info("%d elements" % mesh.nelements)

    # }}}

    # {{{ discretization and connections

    vol_discr = Discretization(
        cl_ctx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(vol_quad_order))

    from meshmode.mesh import BTAG_ALL
    from meshmode.discretization.connection import make_face_restriction
    pre_density_connection = make_face_restriction(
        vol_discr,
        InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order), BTAG_ALL)
    pre_density_discr = pre_density_connection.to_discr

    from pytential.qbx import (QBXLayerPotentialSource,
                               QBXTargetAssociationFailedException)

    qbx, _ = QBXLayerPotentialSource(
        pre_density_discr,
        fine_order=bdry_ovsmp_quad_order,
        qbx_order=qbx_order,
        fmm_order=fmm_order,
        expansion_disks_in_tree_have_extent=True, ).with_refinement()

    density_discr = qbx.density_discr

    # composition of connetions
    # vol_discr --> pre_density_discr --> density_discr
    # via ChainedDiscretizationConnection

    #    from meshmode.mesh.generation import ellipse, make_curve_mesh
    #    from functools import partial
    #
    #    mesh = make_curve_mesh(
    #                partial(ellipse, 2),
    #                np.linspace(0, 1, nelements+1),
    #                mesh_order)
    #
    #    pre_density_discr = Discretization(
    #            cl_ctx, mesh,
    #            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))
    #
    #    from pytential.qbx import (
    #            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    #    qbx, _ = QBXLayerPotentialSource(
    #            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
    #            fmm_order=fmm_order,
    #            expansion_disks_in_tree_have_extent=True,
    #            ).with_refinement()
    #    density_discr = qbx.density_discr

    # }}}

    # {{{ setup operator and potentials

    #print("-- setup Cahn-Hilliard operator")
    from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator
    chop = CahnHilliardOperator(b=b, c=c)

    unk = chop.make_unknown("sigma")
    bound_op = bind(qbx, chop.operator(unk))

    #print ("-- construct kernels")
    yukawa_2d_in_3d_kernel_1 = get_extkernel_for_G1(chop.lambdas[0])
    shidong_2d_in_3d_kernel = get_extkernel_for_G0(chop.lambdas[0],
                                                   chop.lambdas[1])

    #print("-- construct layer potentials")
    from sumpy.qbx import LayerPotential
    from sumpy.expansion.local import LineTaylorLocalExpansion
    layer_pot_v0f1 = LayerPotential(cl_ctx, [
        LineTaylorLocalExpansion(shidong_2d_in_3d_kernel, order=vol_qbx_order)
    ])
    layer_pot_v1f1 = LayerPotential(cl_ctx, [
        LineTaylorLocalExpansion(
            yukawa_2d_in_3d_kernel_1, order=vol_qbx_order)
    ])
    # }}}

    # {{{ setup for volume integral

    vol_x = vol_discr.nodes().with_queue(queue)

    # target points
    targets = cl.array.zeros(queue, (3, ) + vol_x.shape[1:], vol_x.dtype)
    targets[:2] = vol_x

    # expansion centers
    center_dist = 0.125 * np.min(
        cl.clmath.sqrt(
            bind(vol_discr, p.area_element(mesh.ambient_dim, mesh.dim))(queue))
        .get())
    centers = make_obj_array(
        [ci.copy().reshape(vol_discr.nnodes) for ci in targets])
    centers[2][:] = center_dist
    print(center_dist)

    # source points
    # TODO: use over sampled source points?
    sources = cl.array.zeros(queue, (3, ) + vol_x.shape[1:], vol_x.dtype)
    sources[:2] = vol_x

    vol_weights = bind(
        vol_discr,
        p.area_element(mesh.ambient_dim, mesh.dim) * p.QWeight())(queue)

    print("volume: %d source nodes, %d target nodes" % (vol_discr.nnodes,
                                                        vol_discr.nnodes))

    # }}}

    # {{{ prepare for time stepping
    timestep_number = 0
    time = 0

    def get_vts_filename(tmstp_num):
        return "solution-" + '{0:03}'.format(tmstp_num) + ".vts"

    output_vts_filename = get_vts_filename(timestep_number)
    # }}}

    # {{{ [[TIME STEPPING]]
    while time < final_t:
        timestep_number += 1
        time += delta_t
        output_vts_filename = get_vts_filename(timestep_number)

        # a manufactured f1 function
        #x_sin_factor = 30
        #y_sin_factor = 10
        #def f1_func(x, y):
        #    return 0.1 * cl.clmath.sin(x_sin_factor*x) \
        #            * cl.clmath.sin(y_sin_factor*y)

        # get f1 to compute strengths
        f1 = f1_func(vol_x[0], vol_x[1])
        f2 = f2_func(vol_x[0], vol_x[1])

        evt, (vol_pot_v0f1, ) = layer_pot_v0f1(
            queue,
            targets=targets.reshape(3, vol_discr.nnodes),
            centers=centers,
            sources=sources.reshape(3, vol_discr.nnodes),
            strengths=((vol_weights * f1).reshape(vol_discr.nnodes), ))

        evt, (vol_pot_v1f1, ) = layer_pot_v1f1(
            queue,
            targets=targets.reshape(3, vol_discr.nnodes),
            centers=centers,
            sources=sources.reshape(3, vol_discr.nnodes),
            strengths=((vol_weights * f1).reshape(vol_discr.nnodes), ))

    # {{{ fix rhs and solve

    nodes = density_discr.nodes().with_queue(queue)

    def g(xvec):
        x, y = xvec
        return cl.clmath.cos(5 * cl.clmath.atan2(y, x))

    bc = sym.make_obj_array([
        # FIXME: Realistic BC
        5 + g(nodes),
        3 - g(nodes),
    ])

    from pytential.solve import gmres
    gmres_result = gmres(
        bound_op.scipy_op(queue, "sigma", dtype=np.complex128),
        bc,
        tol=1e-8,
        progress=True,
        stall_iterations=0,
        hard_failure=True)

    sigma = gmres_result.solution

    # }}}

    # {{{ check pde

    def check_pde():
        from sumpy.point_calculus import CalculusPatch
        vec_h = [1e-1, 1e-2, 1e-3, 1e-4]
        vec_ru = []
        vec_rv = []
        vec_rp = []
        vec_rm = []
        vec_rf1 = []
        vec_rf2 = []
        vec_rutld = []
        vec_rvtld = []
        for dx in vec_h:
            cp = CalculusPatch(np.zeros(2), order=4, h=dx)
            targets = cl.array.to_device(queue, cp.points)

            u, v = bind((qbx, PointsTarget(targets)),
                        chop.representation(unk))(
                            queue, sigma=sigma)

            u = u.get().real
            v = v.get().real

            lap_u = -(v - chop.b * u)

            # Check for homogeneous PDEs for u and v
            vec_ru.append(
                la.norm(
                    cp.laplace(lap_u) - chop.b * cp.laplace(u) + chop.c * u))
            vec_rv.append(la.norm(v + cp.laplace(u) - chop.b * u))

            # Check for inhomogeneous PDEs for phi and mu

            targets_in_3d = cl.array.zeros(queue, (3, ) + cp.points.shape[1:],
                                           vol_x.dtype)
            targets_in_3d[:2] = cp.points

            #center_dist = 0.125*np.min(
            #        cl.clmath.sqrt(
            #            bind(vol_discr,
            #                p.area_element(mesh.ambient_dim, mesh.dim))
            #            (queue)).get())

            centers_in_3d = make_obj_array([ci.copy() for ci in targets_in_3d])
            centers_in_3d[2][:] = center_dist

            evt, (v0f1, ) = layer_pot_v0f1(
                queue,
                targets=targets_in_3d,
                centers=centers_in_3d,
                sources=sources.reshape(3, vol_discr.nnodes),
                strengths=((vol_weights * f1).reshape(vol_discr.nnodes), ))
            v0f1 = v0f1.get().real

            evt, (v1f1, ) = layer_pot_v1f1(
                queue,
                targets=targets_in_3d,
                centers=centers_in_3d,
                sources=sources.reshape(3, vol_discr.nnodes),
                strengths=((vol_weights * f1).reshape(vol_discr.nnodes), ))
            v1f1 = v1f1.get().real

            f14ck = cl.array.to_device(queue,
                                       f1_func(cp.points[0], cp.points[1]))
            f24ck = cl.array.to_device(queue,
                                       f2_func(cp.points[0], cp.points[1]))
            phi_init = cl.array.to_device(queue,
                                          initial_phi(cp.points[0],
                                                      cp.points[1]))

            f14ck = f14ck.get().real
            f24ck = f24ck.get().real
            phi_init = phi_init.get().real
            vec_rf1.append(
                la.norm(chop.c * phi_init -
                        (1 + s) / epsilon**2 * cp.laplace(phi_init) +
                        1. / epsilon**2 * cp.laplace(phi_init**3) - f14ck))
            vec_rf2.append(
                la.norm((phi_init**3 - (1 + s) * phi_init) / epsilon**2 -
                        f24ck))

            utild = v0f1
            vtild = f24ck - v1f1 + chop.lambdas[0]**2 * v0f1
            lap_utild = -vtild + chop.b * utild + f24ck
            # FIXME: Not passing this check. (bugs in volume integral?)
            vec_rutld.append(
                la.norm(
                    cp.laplace(lap_utild) - chop.b * lap_utild +
                    chop.c * utild - f14ck))
            print(f14ck)
            print(f1[0:10])
            print(cp.laplace(lap_utild))
            print((lap_utild))
            print((utild))
            print(chop.b)
            print(chop.c)
            vec_rvtld.append(
                la.norm(vtild + cp.laplace(utild) - chop.b * utild - f24ck))

            ph = utild + u
            mu = epsilon * (vtild + v)
            lap_ph = f24ck + chop.b * ph - mu / epsilon

            vec_rp.append(
                la.norm(
                    cp.laplace(lap_ph) - chop.b * cp.laplace(ph) + chop.c * ph
                    - f14ck))
            vec_rm.append(
                la.norm(mu / epsilon + cp.laplace(ph) - chop.b * ph - f24ck))

        from tabulate import tabulate
        # overwrite if file exists
        with open('check_pde.dat', 'w') as f:
            print(
                "Residuals of PDE and numerical vs. symbolic differentiation:",
                file=f)
            print(
                tabulate([
                    ["h"] + vec_h,
                    ["residual_u"] + vec_ru,
                    ["residual_v"] + vec_rv,
                    ["residual_f1"] + vec_rf1,
                    ["residual_f2"] + vec_rf2,
                    ["residual_u_tld"] + vec_rutld,
                    ["residual_v_tld"] + vec_rvtld,
                    ["residual_phi"] + vec_rp,
                    ["residual_mu"] + vec_rm,
                ]),
                file=f)
        1 / 0

    check_pde()

    # }}}

    # {{{ postprocess/visualize

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=1.5, npoints=500)

    targets = cl.array.to_device(queue, fplot.points)

    qbx_stick_out = qbx.copy(target_stick_out_factor=0.05)
    indicator_qbx = qbx_stick_out.copy(qbx_order=2)

    from sumpy.kernel import LaplaceKernel
    ones_density = density_discr.zeros(queue)
    ones_density.fill(1)
    indicator = bind((indicator_qbx, PointsTarget(targets)),
                     sym.D(LaplaceKernel(2), sym.var("sigma")))(
                         queue, sigma=ones_density).get()

    # clean up the mess
    def clean_file(filename):
        import os
        try:
            os.remove(filename)
        except OSError:
            pass

    clean_file("failed-targets.vts")
    clean_file("potential.vts")

    try:
        u, v = bind((qbx_stick_out, PointsTarget(targets)),
                    chop.representation(unk))(
                        queue, sigma=sigma)
    except QBXTargetAssociationFailedException as e:
        fplot.write_vtk_file("failed-targets.vts",
                             [("failed", e.failed_target_flags.get(queue))])
        raise
    u = u.get().real
    v = v.get().real

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file("potential.vts", [
        ("u", u),
        ("v", v),
        ("indicator", indicator),
    ])

    # }}}

# }}}


if __name__ == "__main__":
    main()
