from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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


import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pytools import Record
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut,
        make_curve_mesh)
from sumpy.visualization import FieldPlotter
from pytential import bind, sym, norm

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

__all__ = [
        "pytest_generate_tests",

        # difficult curves not currently used for testing
        "drop", "n_gon", "cloverleaf"
        ]

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


def make_circular_point_group(npoints, radius,
        center=np.array([0., 0.]), func=lambda x: x):
    t = func(np.linspace(0, 1, npoints, endpoint=False)) * (2 * np.pi)
    center = np.asarray(center)
    result = center[:, np.newaxis] + radius*np.vstack((np.cos(t), np.sin(t)))
    return result


# {{{ geometry test

def test_geometry(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    nelements = 30
    order = 5

    mesh = make_curve_mesh(partial(ellipse, 1),
            np.linspace(0, 1, nelements+1),
            order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr = Discretization(cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    import pytential.symbolic.primitives as prim
    area_sym = prim.integral(1)

    area = bind(discr, area_sym)(queue)

    err = abs(area-2*np.pi)
    print(err)
    assert err < 1e-3

# }}}


# {{{ ellipse eigenvalues

@pytest.mark.parametrize(["ellipse_aspect", "mode_nr", "qbx_order"], [
    # Run with FMM
    (1, 5, 3),
    (1, 6, 3),
    # (2, 5, 3), /!\ FIXME: Does not achieve sufficient FMM precision

    # Run without FMM
    (1, 5, 4),
    (1, 6, 4),
    (1, 7, 5),
    (2, 5, 4),
    (2, 6, 4),
    (2, 7, 5),
    ])
def test_ellipse_eigenvalues(ctx_getter, ellipse_aspect, mode_nr, qbx_order):
    logging.basicConfig(level=logging.INFO)

    print("ellipse_aspect: %s, mode_nr: %d, qbx_order: %d" % (
            ellipse_aspect, mode_nr, qbx_order))

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 7

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    from pytools.convergence import EOCRecorder

    s_eoc_rec = EOCRecorder()
    d_eoc_rec = EOCRecorder()
    sp_eoc_rec = EOCRecorder()

    if ellipse_aspect != 1:
        nelements_values = [60, 100, 150, 200]
    else:
        nelements_values = [30, 70]

    # See
    #
    # [1] G. J. Rodin and O. Steinbach, "Boundary Element Preconditioners
    # for Problems Defined on Slender Domains", SIAM Journal on Scientific
    # Computing, Vol. 24, No. 4, pg. 1450, 2003.
    # http://dx.doi.org/10.1137/S1064827500372067

    for nelements in nelements_values:
        mesh = make_curve_mesh(partial(ellipse, ellipse_aspect),
                np.linspace(0, 1, nelements+1),
                target_order)

        fmm_order = qbx_order
        if fmm_order > 3:
            # FIXME: for now
            fmm_order = False

        density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
        qbx = QBXLayerPotentialSource(density_discr, 4*target_order,
                qbx_order, fmm_order=fmm_order)

        nodes = density_discr.nodes().with_queue(queue)

        if 0:
            # plot geometry, centers, normals
            centers = qbx.centers(density_discr, 1)
            nodes_h = nodes.get()
            centers_h = [centers[0].get(), centers[1].get()]
            pt.plot(nodes_h[0], nodes_h[1], "x-")
            pt.plot(centers_h[0], centers_h[1], "o")
            normal = bind(qbx, sym.normal())(queue).as_vector(np.object)
            pt.quiver(nodes_h[0], nodes_h[1],
                    normal[0].get(), normal[1].get())
            pt.gca().set_aspect("equal")
            pt.show()

        angle = cl.clmath.atan2(nodes[1]*ellipse_aspect, nodes[0])

        ellipse_fraction = ((1-ellipse_aspect)/(1+ellipse_aspect))**mode_nr

        # (2.6) in [1]
        J = cl.clmath.sqrt(  # noqa
                cl.clmath.sin(angle)**2
                + (1/ellipse_aspect)**2 * cl.clmath.cos(angle)**2)

        # {{{ single layer

        sigma = cl.clmath.cos(mode_nr*angle)/J

        s_sigma_op = bind(qbx, sym.S(0, sym.var("sigma")))
        s_sigma = s_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        s_eigval = 1/(2*mode_nr) * (1 + (-1)**mode_nr * ellipse_fraction)

        # (2.12) in [1]
        s_sigma_ref = s_eigval*J*sigma

        if 0:
            #pt.plot(s_sigma.get(), label="result")
            #pt.plot(s_sigma_ref.get(), label="ref")
            pt.plot((s_sigma_ref-s_sigma).get(), label="err")
            pt.legend()
            pt.show()

        s_err = (
                norm(density_discr, queue, s_sigma - s_sigma_ref)
                /
                norm(density_discr, queue, s_sigma_ref))
        s_eoc_rec.add_data_point(1/nelements, s_err)

        # }}}

        # {{{ double layer

        sigma = cl.clmath.cos(mode_nr*angle)

        d_sigma_op = bind(qbx, sym.D(0, sym.var("sigma")))
        d_sigma = d_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        d_eigval = -(-1)**mode_nr * 1/2*ellipse_fraction

        d_sigma_ref = d_eigval*sigma

        if 0:
            pt.plot(d_sigma.get(), label="result")
            pt.plot(d_sigma_ref.get(), label="ref")
            pt.legend()
            pt.show()

        if ellipse_aspect == 1:
            d_ref_norm = norm(density_discr, queue, sigma)
        else:
            d_ref_norm = norm(density_discr, queue, d_sigma_ref)

        d_err = (
                norm(density_discr, queue, d_sigma - d_sigma_ref)
                /
                d_ref_norm)
        d_eoc_rec.add_data_point(1/nelements, d_err)

        # }}}

        if ellipse_aspect == 1:
            # {{{ S'

            sigma = cl.clmath.cos(mode_nr*angle)

            sp_sigma_op = bind(qbx, sym.Sp(0, sym.var("sigma")))
            sp_sigma = sp_sigma_op(queue=queue, sigma=sigma)
            sp_eigval = 0

            sp_sigma_ref = sp_eigval*sigma

            sp_err = (
                    norm(density_discr, queue, sp_sigma - sp_sigma_ref)
                    /
                    norm(density_discr, queue, sigma))
            sp_eoc_rec.add_data_point(1/nelements, sp_err)

            # }}}

    print("Errors for S:")
    print(s_eoc_rec)
    required_order = qbx_order + 1
    assert s_eoc_rec.order_estimate() > required_order - 1.5

    print("Errors for D:")
    print(d_eoc_rec)
    required_order = qbx_order
    assert d_eoc_rec.order_estimate() > required_order - 1.5

    if ellipse_aspect == 1:
        print("Errors for S':")
        print(sp_eoc_rec)
        required_order = qbx_order
        assert sp_eoc_rec.order_estimate() > required_order - 1.5

# }}}


# {{{ integral equation test backend

def run_int_eq_test(
        cl_ctx, queue, curve_f, nelements, qbx_order, bc_type, loc_sign, k,
        target_order, source_order):

    mesh = make_curve_mesh(curve_f,
            np.linspace(0, 1, nelements+1),
            target_order)

    if 0:
        from pytential.visualization import show_mesh
        show_mesh(mesh)

        pt.gca().set_aspect("equal")
        pt.show()

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

    if source_order is None:
        source_order = 4*target_order

    qbx = QBXLayerPotentialSource(
            density_discr, fine_order=source_order, qbx_order=qbx_order,
            # Don't use FMM for now
            fmm_order=False)

    # {{{ set up operator

    from pytential.symbolic.pde.scalar import (
            DirichletOperator,
            NeumannOperator)

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel, AxisTargetDerivative
    if k:
        knl = HelmholtzKernel(2)
        knl_kwargs = {"k": k}
    else:
        knl = LaplaceKernel(2)
        knl_kwargs = {}

    if knl.is_complex_valued:
        dtype = np.complex128
    else:
        dtype = np.float64

    if bc_type == "dirichlet":
        op = DirichletOperator((knl, knl_kwargs), loc_sign, use_l2_weighting=True)
    elif bc_type == "neumann":
        op = NeumannOperator((knl, knl_kwargs), loc_sign, use_l2_weighting=True,
                 use_improved_operator=False)
    else:
        assert False

    op_u = op.operator(sym.var("u"))

    # }}}

    # {{{ set up test data

    inner_radius = 0.1
    outer_radius = 2

    if loc_sign < 0:
        test_src_geo_radius = outer_radius
        test_tgt_geo_radius = inner_radius
    else:
        test_src_geo_radius = inner_radius
        test_tgt_geo_radius = outer_radius

    point_sources = make_circular_point_group(10, test_src_geo_radius,
            func=lambda x: x**1.5)
    test_targets = make_circular_point_group(20, test_tgt_geo_radius)

    np.random.seed(22)
    source_charges = np.random.randn(point_sources.shape[1])
    source_charges[-1] = -np.sum(source_charges[:-1])
    source_charges = source_charges.astype(dtype)
    assert np.sum(source_charges) < 1e-15

    # }}}

    if 0:
        # show geometry, centers, normals
        nodes_h = density_discr.nodes().get(queue=queue)
        pt.plot(nodes_h[0], nodes_h[1], "x-")
        normal = bind(density_discr, sym.normal())(queue).as_vector(np.object)
        pt.quiver(nodes_h[0], nodes_h[1], normal[0].get(queue), normal[1].get(queue))
        pt.gca().set_aspect("equal")
        pt.show()

    # {{{ establish BCs

    from sumpy.p2p import P2P
    pot_p2p = P2P(cl_ctx,
            [knl], exclude_self=False, value_dtypes=dtype)

    evt, (test_direct,) = pot_p2p(
            queue, test_targets, point_sources, [source_charges],
            out_host=False, **knl_kwargs)

    nodes = density_discr.nodes()

    evt, (src_pot,) = pot_p2p(
            queue, nodes, point_sources, [source_charges],
            **knl_kwargs)

    grad_p2p = P2P(cl_ctx,
            [AxisTargetDerivative(0, knl), AxisTargetDerivative(1, knl)],
            exclude_self=False, value_dtypes=dtype)
    evt, (src_grad0, src_grad1) = grad_p2p(
            queue, nodes, point_sources, [source_charges],
            **knl_kwargs)

    if bc_type == "dirichlet":
        bc = src_pot
    elif bc_type == "neumann":
        normal = bind(density_discr, sym.normal())(queue).as_vector(np.object)
        bc = (src_grad0*normal[0] + src_grad1*normal[1])

    # }}}

    # {{{ solve

    bound_op = bind(qbx, op_u)

    rhs = bind(density_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bc)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "u", dtype, k=k),
            rhs, tol=1e-14, progress=True,
            hard_failure=False)

    u = gmres_result.solution
    print("gmres state:", gmres_result.state)

    if 0:
        # {{{ build matrix for spectrum check

        from sumpy.tools import build_matrix
        mat = build_matrix(bound_op.scipy_op("u", dtype=dtype, k=k))
        w, v = la.eig(mat)
        if 0:
            pt.imshow(np.log10(1e-20+np.abs(mat)))
            pt.colorbar()
            pt.show()

        #assert abs(s[-1]) < 1e-13, "h
        #assert abs(s[-2]) > 1e-7
        #from pudb import set_trace; set_trace()

        # }}}

    # }}}

    # {{{ error check

    from pytential.target import PointsTarget

    bound_tgt_op = bind((qbx, PointsTarget(test_targets)),
            op.representation(sym.var("u")))

    test_via_bdry = bound_tgt_op(queue, u=u, k=k)

    err = test_direct-test_via_bdry

    err = err.get()
    test_direct = test_direct.get()
    test_via_bdry = test_via_bdry.get()

    # {{{ remove effect of net source charge

    if k == 0 and bc_type == "neumann" and loc_sign == -1:
        # remove constant offset in interior Laplace Neumann error
        tgt_ones = np.ones_like(test_direct)
        tgt_ones = tgt_ones/la.norm(tgt_ones)
        err = err - np.vdot(tgt_ones, err)*tgt_ones

    # }}}

    rel_err_2 = la.norm(err)/la.norm(test_direct)
    rel_err_inf = la.norm(err, np.inf)/la.norm(test_direct, np.inf)

    # }}}

    print("rel_err_2: %g rel_err_inf: %g" % (rel_err_2, rel_err_inf))

    # {{{ test tangential derivative

    bound_t_deriv_op = bind(qbx,
            op.representation(
                sym.var("u"), map_potentials=sym.tangential_derivative,
                qbx_forced_limit=loc_sign))

    #print(bound_t_deriv_op.code)

    tang_deriv_from_src = bound_t_deriv_op(queue, u=u).as_scalar().get()

    tangent = bind(
            density_discr,
            sym.pseudoscalar()/sym.area_element())(queue).as_vector(np.object)

    tang_deriv_ref = (src_grad0 * tangent[0] + src_grad1 * tangent[1]).get()

    if 0:
        pt.plot(tang_deriv_ref.real)
        pt.plot(tang_deriv_from_src.real)
        pt.show()

    td_err = tang_deriv_from_src - tang_deriv_ref

    rel_td_err_inf = la.norm(td_err, np.inf)/la.norm(tang_deriv_ref, np.inf)

    print("rel_td_err_inf: %g" % rel_td_err_inf)

    # }}}

    # {{{ plotting

    if 0:
        fplot = FieldPlotter(np.zeros(2),
                extent=1.25*2*max(test_src_geo_radius, test_tgt_geo_radius),
                npoints=200)

        #pt.plot(u)
        #pt.show()

        evt, (fld_from_src,) = pot_p2p(
                queue, fplot.points, point_sources, [source_charges],
                **knl_kwargs)
        fld_from_bdry = bind(
                (qbx, PointsTarget(fplot.points)),
                op.representation(sym.var("u"))
                )(queue, u=u, k=k)
        fld_from_src = fld_from_src.get()
        fld_from_bdry = fld_from_bdry.get()

        nodes = density_discr.nodes().get(queue=queue)

        def prep():
            pt.plot(point_sources[0], point_sources[1], "o",
                    label="Monopole 'Point Charges'")
            pt.plot(test_targets[0], test_targets[1], "v",
                    label="Observation Points")
            pt.plot(nodes[0], nodes[1], "k-",
                    label=r"$\Gamma$")

        from matplotlib.cm import get_cmap
        cmap = get_cmap()
        cmap._init()
        if 0:
            cmap._lut[(cmap.N*99)//100:, -1] = 0  # make last percent transparent?

        prep()
        if 1:
            pt.subplot(131)
            pt.title("Field error (loc_sign=%s)" % loc_sign)
            log_err = np.log10(1e-20+np.abs(fld_from_src-fld_from_bdry))
            log_err = np.minimum(-3, log_err)
            fplot.show_scalar_in_matplotlib(log_err, cmap=cmap)

            #from matplotlib.colors import Normalize
            #im.set_norm(Normalize(vmin=-6, vmax=1))

            cb = pt.colorbar(shrink=0.9)
            cb.set_label(r"$\log_{10}(\mathdefault{Error})$")

        if 1:
            pt.subplot(132)
            prep()
            pt.title("Source Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_src.real, max_val=3)

            pt.colorbar(shrink=0.9)
        if 1:
            pt.subplot(133)
            prep()
            pt.title("Solved Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_bdry.real, max_val=3)

            pt.colorbar(shrink=0.9)

        # total field
        #fplot.show_scalar_in_matplotlib(
        #fld_from_src.real+fld_from_bdry.real, max_val=0.1)

        #pt.colorbar()

        pt.legend(loc="best", prop=dict(size=15))
        from matplotlib.ticker import NullFormatter
        pt.gca().xaxis.set_major_formatter(NullFormatter())
        pt.gca().yaxis.set_major_formatter(NullFormatter())

        pt.gca().set_aspect("equal")

        if 0:
            border_factor_top = 0.9
            border_factor = 0.3

            xl, xh = pt.xlim()
            xhsize = 0.5*(xh-xl)
            pt.xlim(xl-border_factor*xhsize, xh+border_factor*xhsize)

            yl, yh = pt.ylim()
            yhsize = 0.5*(yh-yl)
            pt.ylim(yl-border_factor_top*yhsize, yh+border_factor*yhsize)

        #pt.savefig("helmholtz.pdf", dpi=600)
        pt.show()

        # }}}

    class Result(Record):
        pass

    return Result(
            rel_err_2=rel_err_2,
            rel_err_inf=rel_err_inf,
            rel_td_err_inf=rel_td_err_inf,
            gmres_result=gmres_result)

# }}}


# {{{ integral equation test frontend

@pytest.mark.parametrize(("curve_name", "curve_f"), [
    # booo-ring.
    #("circle", partial(ellipse, 1)),

    ("3-to-1 ellipse", partial(ellipse, 3)),

    # underresolved at resolutions that take tolerable time
    #("starfish", starfish),
    ])
@pytest.mark.parametrize("k", [0, 1.2])
@pytest.mark.parametrize("bc_type", ["dirichlet", "neumann"])
@pytest.mark.parametrize("loc_sign", [+1, -1])
@pytest.mark.parametrize("qbx_order", [5])
# Sample test run:
# 'test_integral_equation(cl._csc, "circle", circle, 5, "dirichlet", +1, 5)'
def test_integral_equation(
        ctx_getter, curve_name, curve_f, qbx_order, bc_type, loc_sign, k,
        target_order=7, source_order=None):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from pytools.convergence import EOCRecorder
    print(("curve_name: %s, qbx_order: %d, bc_type: %s, loc_sign: %s, "
            "helmholtz_k: %s"
            % (curve_name, qbx_order, bc_type, loc_sign, k)))

    eoc_rec_target = EOCRecorder()
    eoc_rec_td = EOCRecorder()

    for nelements in [30, 40, 50]:
        result = run_int_eq_test(
                cl_ctx, queue, curve_f, nelements, qbx_order,
                bc_type, loc_sign, k, target_order=target_order,
                source_order=source_order)

        eoc_rec_target.add_data_point(1/nelements, result.rel_err_2)
        eoc_rec_td.add_data_point(1/nelements, result.rel_td_err_inf)

    if bc_type == "dirichlet":
        tgt_order = qbx_order
    elif bc_type == "neumann":
        tgt_order = qbx_order-1
    else:
        assert False

    print("TARGET ERROR:")
    print(eoc_rec_target)
    assert eoc_rec_target.order_estimate() > tgt_order - 1.3

    print("TANGENTIAL DERIVATIVE ERROR:")
    print(eoc_rec_td)
    assert eoc_rec_td.order_estimate() > tgt_order - 2.3

# }}}


# {{{ integral identity tester

d1 = sym.Derivative()
d2 = sym.Derivative()


@pytest.mark.parametrize(("curve_name", "curve_f"), [
    #("circle", partial(ellipse, 1)),
    ("3-to-1 ellipse", partial(ellipse, 3)),
    #("starfish", starfish),
    ])
@pytest.mark.parametrize("qbx_order", [5])
@pytest.mark.parametrize(("zero_op_name", "k"), [
    ("green", 0),
    ("green", 1.2),
    ("green_grad", 0),
    ("green_grad", 1.2),
    ("zero_calderon", 0),
    ])
# sample invocation to copy and paste:
# 'test_identities(cl._csc, "green", "circ", partial(ellipse, 1), 4, 0)'
def test_identities(ctx_getter, zero_op_name, curve_name, curve_f, qbx_order, k):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    target_order = 7

    u_sym = sym.var("u")
    grad_u_sym = sym.VectorVariable("grad_u")
    dn_u_sym = sym.var("dn_u")

    if k == 0:
        k_sym = 0
    else:
        k_sym = "k"

    zero_op_table = {
            "green":
            sym.S(k_sym, dn_u_sym) - sym.D(k_sym, u_sym) - 0.5*u_sym,

            "green_grad":
            d1.nabla * d1(sym.S(k_sym, dn_u_sym, qbx_forced_limit="avg"))
            - d2.nabla * d2(sym.D(k_sym, u_sym, qbx_forced_limit="avg"))
            - 0.5*grad_u_sym,

            # only for k==0:
            "zero_calderon":
            -sym.Dp(0, sym.S(0, u_sym))
            - 0.25*u_sym + sym.Sp(0, sym.Sp(0, u_sym))
            }
    order_table = {
            "green": qbx_order,
            "green_grad": qbx_order-1,
            "zero_calderon": qbx_order-1,
            }

    zero_op = zero_op_table[zero_op_name]

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for nelements in [30, 50, 70]:
        mesh = make_curve_mesh(curve_f,
                np.linspace(0, 1, nelements+1),
                target_order)

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        from pytential.qbx import QBXLayerPotentialSource
        density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))

        qbx = QBXLayerPotentialSource(density_discr, 4*target_order,
                qbx_order,
                # Don't use FMM for now
                fmm_order=False)

        # {{{ compute values of a solution to the PDE

        nodes_host = density_discr.nodes().get(queue)
        normal = bind(density_discr, sym.normal())(queue).as_vector(np.object)
        normal_host = [normal[0].get(), normal[1].get()]

        if k != 0:
            angle = 0.3
            wave_vec = np.array([np.cos(angle), np.sin(angle)])
            u = np.exp(1j*k*np.tensordot(wave_vec, nodes_host, axes=1))
            grad_u = 1j*k*wave_vec[:, np.newaxis]*u
        else:
            center = np.array([3, 1])
            diff = nodes_host - center[:, np.newaxis]
            dist_squared = np.sum(diff**2, axis=0)
            dist = np.sqrt(dist_squared)
            u = np.log(dist)
            grad_u = diff/dist_squared

        dn_u = normal_host[0]*grad_u[0] + normal_host[1]*grad_u[1]

        # }}}

        u_dev = cl.array.to_device(queue, u)
        dn_u_dev = cl.array.to_device(queue, dn_u)
        grad_u_dev = cl.array.to_device(queue, grad_u)

        key = (qbx_order, curve_name, nelements, zero_op_name)

        bound_op = bind(qbx, zero_op)
        error = bound_op(
                queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=k)
        if 0:
            pt.plot(error)
            pt.show()

        l2_error_norm = norm(density_discr, queue, error)
        print(key, l2_error_norm)

        eoc_rec.add_data_point(1/nelements, l2_error_norm)

    print(eoc_rec)
    tgt_order = order_table[zero_op_name]
    assert eoc_rec.order_estimate() > tgt_order - 1.3

# }}}


# {{{ test off-surface eval

@pytest.mark.parametrize("use_fmm", [True, False])
def test_off_surface_eval(ctx_getter, use_fmm, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    nelements = 30
    target_order = 8
    qbx_order = 3
    if use_fmm is True:
        fmm_order = qbx_order
    else:
        fmm_order = False

    mesh = make_curve_mesh(partial(ellipse, 3),
            np.linspace(0, 1, nelements+1),
            target_order)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    qbx = QBXLayerPotentialSource(density_discr, 4*target_order, qbx_order,
            fmm_order=fmm_order)

    from sumpy.kernel import LaplaceKernel
    op = sym.D(LaplaceKernel(), sym.var("sigma"), qbx_forced_limit=-2)

    sigma = density_discr.zeros(queue) + 1

    fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)
    from pytential.target import PointsTarget
    fld_in_vol = bind(
            (qbx, PointsTarget(fplot.points)),
            op)(queue, sigma=sigma)

    print(fld_in_vol)

    err = cl.clmath.fabs(fld_in_vol - (-1))

    if do_plot:
        fplot.show_scalar_in_matplotlib(fld_in_vol.get())
        import matplotlib.pyplot as pt
        pt.colorbar()
        pt.show()

    # FIXME: Why does the FMM only meet this sloppy tolerance?
    assert (err < 1e-2).get().all()

# }}}


# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
