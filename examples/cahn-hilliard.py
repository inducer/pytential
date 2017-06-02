import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym, norm  # noqa
from pytential.target import PointsTarget
# {{{ set some constants for use below

nelements = 20
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 4*bdry_quad_order
fmm_order = 8

vol_quad_order = 5
vol_qbx_order  = 2

# }}}


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

# {{{ volume mesh generation

    from enum import Enum
    class Geometry(Enum):
        RegularRectangle = 1
        Circle           = 2

    shape = Geometry.Circle

    if shape == Geometry.RegularRectangle:
      from meshmode.mesh.generation import generate_regular_rect_mesh
      ext = 1.
      h   = 0.05
      mesh = generate_regular_rect_mesh(
              a=(-ext/2., -ext/2.),
              b=( ext/2.,  ext/2.),
              n=(int(ext/h), int(ext/h))
              )
    elif shape == Geometry.Circle:
        from meshmode.mesh.io import import generate_gmsh
        h    = 0.05
        mesh = generate_emsh(
                FileSource("circle.step"),
                2,
                order=mesh_order,
                force_ambient_dim=2,
                orther_options=["-string",
                                "Mesh.CharacteristicLengthMax = %g;" % h]
                )
    else:
        1/0

    logger.info("%d elements" % mesh.nelements)

# }}}

# {{{ discretization and connections

    vol_discr = Discretization(ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(vol_quad_order))

    from meshmode.mesh import BTAG_ALL
    from meshmode.discretization.connection import make_face_restriction
    pre_density_connection = make_face_restriction(
            vol_discr,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order),
            BTAG_ALL)
    pre_density_discr = pre_density_connection.to_discr

    from pytential.qbx import (
            QBXLayerPotentialSource, QBXTargetAssociationFailedException)

    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr,
            fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            expansion_disks_in_tree_have_extent=True,
            ).with_refinement()

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

# {{{ a kernel class for G0
# FIXME: will the expressions work when lambda is complex?
# (may need conversion from 1j to var("I"))
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
                r       = pymbolic_real_norm_2(make_sym_vector("d", dim))
                expr    = var("hankel_1")(0, var("I") * lambda1 * r) - \
                          var("hankel_1")(0, var("I") * lambda2 * r)
                scaling =  1. / ( 4. * var("I") * (lambda1**2 - lambda2**2) )
            else:
                raise RuntimeError("unsupported dimensionality")

            ExpressionKernel.__init__(
                    self,
                    dim,
                    expression=expr,
                    scaling = scaling,
                    is_complex_valued=True)

            self.lambda1 = lambda1
            self.lambda2 = lambda2

        def __getinitargs__(self):
            return(self._dim, self.lambda1, self.lambda2)

        def update_persistent_hash(self, key_hash, key_builder):
            key_hash.update(type(self).__name__.encode("utf8"))
            key_builder.rec(key_hash, 
                    (self._dim, self.lambda1, self.lambda2)
                    )

        def __repr__(self):
            if self._dim is not None:
                return "ShdgKnl%dD(%f, %f)" % (
                        self._dim, self.lambda1, self.lambda2)
            else:
                return "ShdgKnl(%f, %f)" % (self.lambda1, self.lambda2)

        def prepare_loopy_kernel(self, loopy_knl):
            from sumpy.codegen import (bessel_preamble_generator, bessel_mangler)
            loopy_knl = lp.register_function_manglers(loopy_knl,
                    [bessel_mangler])
            loopy_knl = lp.register_preamble_generators(loopy_knl,
                    [bessel_preamble_generator])
            return loopy_knl

        def get_args(self):
            k_dtype = np.complex128
            return [
                    KernelArgument(
                        loopy_arg=lp.ValueArg("shidong_kernel", k_dtype),
                        )]

        mapper_method = "map_shidong_kernel"

# }}}

# {{{ extended kernel getters
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
        scaling =  1. / ( 4. * var("I") * (lambda1**2 - lambda2**2) )

        from sumpy.kernel import ExpressionKernel
        return ExpressionKernel(
                dim=3,
                expression=expr,
                scaling=scaling,
                is_complex_valued=True)

    def get_extkernel_for_G1(lamb):
        from sumpy.symbolic import pymbolic_real_norm_2
        from pymbolic.primitives import make_sym_vector
        from pymbolic import var

        d = make_sym_vector("d", 3)
        r2 = pymbolic_real_norm_2(d[:-1])
        expr = var("hankel_1")(0, var("I") * lamb * r2
                + var("I") * d[-1]**2)
        scaling = - var("I") / 4.

        from sumpy.kernel import ExpressionKernel
        return ExpressionKernel(
                dim=3,
                expression=expr,
                scaling=scaling,
                is_complex_valued=True)


# }}}

# {{{ equation info

    s = 1.5
    epsilon = 0.01
    delta_t = 0.05

    b = s / (epsilon**2)
    c = 1. / (epsilon * delta_t)

    sqdet = np.sqrt( b**2 - 4. * c )
    assert np.abs(sqdet) > 1e-6

    lambda1 = ( b + sqdet ) / 2.
    lambda2 = ( b - sqdet ) / 2.

    from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator
    chop = CahnHilliardOperator(b=5, c=1)

    unk = chop.make_unknown("sigma")
    bound_op = bind(qbx, chop.operator(unk))

    yukawa_2d_in_3d_kernel_1 = get_extkernel_for_G1(lambda1)
    shidong_2d_in_3d_kernel  = get_extkernel_for_G0(lambda1, lambda2)

    from sumpy.qbx import LayerPotential
    from sumpy.qbx import LineTaylorLocalExpansion
    layer_pot_v0f1 = LayerPotential(ctx, [
        LineTaylorLocalExpansion(shidong_2d_in_3d_kernel,
            order=vol_qbx_order)])
    layer_pot_v1f1 = LayerPotential(ctx, [
        LineTaylorLocalExpansion(yukawa_2d_in_3d_kernel_1,
            order=vol_qbx_order)])
# }}}

# {{{ volume integral

    vol_x = vol_discr.nodes(),with_queue(queue)

    # target points
    targets = cl.array.zeros(queue, (3,) + vol_x.shape[1:], vol_x.dtype)
    targets[:2] = vol_x

    # expansion centers
    center_dist = 0.125*np.min(
            cl.clmath.sqrt(
                bind(vol_discr,
                    p.area_element(mesh.ambient_dim, mesh.dim))
                (queue)).get())
    centers = make_obj_array([ci.copy().reshape(vol_discr.nnodes) for ci in targets])
    centers[2][:] = center_dist
    print(center_dist)

    # source points
    # FIXME: use over sampled source points?
    sources = cl.array.zeros(queue, (3,) + vol_x.shape[1:], vol_x.dtype)
    sources[:2] = vol_x

    # a manufactured f1
    x_sin_factor = 30
    y_sin_factor = 10
    def f1_func(x, y)
        return 0.1 * cl.clmath.sin(x_sin_factor*x) * cl.clmath.sin(y_sin_factor*y)

    # strengths (with quadrature weights)
    f1 = f1_func(vol_x[0], vol_x[1])
    vol_weights = bind(vol_discr,
            p.area_element(mesh.ambient_dim, mesh.dim) * p.QWeight()
            )(queue)

    print("volume: %d source nodes, %d target nodes" % (
        vol_discr.nnodes, vol_discr.nnodes))

    evt, (vol_pot_v0f1,) = layer_pot_v0f1(
            queue,
            targets=targets.reshape(3, vol_discr.nnodes),
            centers=centers,
            sources=sources.reshape(3, vol_discr.nnodes),
            strengths=(
                (vol_weights * f1).reshape(vol_discr.nnodes),)
            )

    evt, (vol_pot_v1f1,) = layer_pot_v1f1(
            queue,
            targets=targets.reshape(3, vol_discr.nnodes),
            centers=centers,
            sources=sources.reshape(3, vol_discr.nnodes),
            strengths=(
                (vol_weights * f1).reshape(vol_discr.nnodes),)
            )

# }}}


    # {{{ fix rhs and solve

    nodes = density_discr.nodes().with_queue(queue)

    def g(xvec):
        x, y = xvec
        return cl.clmath.cos(5*cl.clmath.atan2(y, x))

    bc = sym.make_obj_array([
        # FIXME: Realistic BC
        5+g(nodes),
        3-g(nodes),
        ])

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.complex128),
            bc, tol=1e-8, progress=True,
            stall_iterations=0,
            hard_failure=True)

    sigma = gmres_result.solution

    # }}}

    # {{{ check pde

    def check_pde():
        from sumpy.point_calculus import CalculusPatch
        cp = CalculusPatch(np.zeros(2), order=4, h=0.1)
        targets = cl.array.to_device(queue, cp.points)

        u, v = bind(
                (qbx, PointsTarget(targets)),
                chop.representation(unk))(queue, sigma=sigma)

        u = u.get().real
        v = v.get().real

        lap_u = -(v - chop.b*u)

        print(la.norm(u), la.norm(v))

        print(la.norm(
            cp.laplace(lap_u) - chop.b * cp.laplace(u) + chop.c*u))

        print(la.norm(
            v + cp.laplace(u) - chop.b*u))
        1/0

    check_pde()

    # }}}

    # {{{ postprocess/visualize

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)

    targets = cl.array.to_device(queue, fplot.points)

    qbx_stick_out = qbx.copy(target_stick_out_factor=0.05)
    indicator_qbx = qbx_stick_out.copy(qbx_order=2)

    from sumpy.kernel import LaplaceKernel
    ones_density = density_discr.zeros(queue)
    ones_density.fill(1)
    indicator = bind(
            (indicator_qbx, PointsTarget(targets)),
            sym.D(LaplaceKernel(2), sym.var("sigma")))(
                    queue, sigma=ones_density).get()

    try:
        u, v = bind(
                (qbx_stick_out, PointsTarget(targets)),
                chop.representation(unk))(queue, sigma=sigma)
    except QBXTargetAssociationFailedException as e:
        fplot.write_vtk_file(
                "failed-targets.vts",
                [
                    ("failed", e.failed_target_flags.get(queue))
                    ]
                )
        raise
    u = u.get().real
    v = v.get().real

    #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file(
            "potential.vts",
            [
                ("u", u),
                ("v", v),
                ("indicator", indicator),
                ]
            )

    # }}}


if __name__ == "__main__":
    main()
