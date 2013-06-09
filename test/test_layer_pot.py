from __future__ import division

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
from pytential.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon,
        make_curve_mesh)
from sumpy.visualization import FieldPlotter

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

    from pytential.discretization.poly_element import \
            PolynomialElementDiscretization

    discr = PolynomialElementDiscretization(cl_ctx, mesh, order)

    from pytential.symbolic.execution import bind
    import pytential.symbolic.primitives as prim
    area_sym = prim.integral(1)

    area = bind(discr, area_sym)(queue)

    err = abs(area-2*np.pi)
    print err
    assert err < 1e-3

# }}}


# {{{ ellipse eigenvalues

@pytest.mark.parametrize(["ellipse_aspect", "mode_nr", "qbx_order"], [
    (1, 5, 4),
    (1, 6, 4),
    (1, 7, 5),
    (2, 5, 4),
    (2, 6, 4),
    (2, 7, 5),
    ])
def test_ellipse_eigenvalues(ctx_getter, ellipse_aspect, mode_nr, qbx_order):
    #logging.basicConfig(level=logging.INFO)

    print "ellipse_aspect: %s, mode_nr: %d, qbx_order: %d" % (
            ellipse_aspect, mode_nr, qbx_order)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 7

    from pytential.discretization.qbx import make_upsampling_qbx_discr
    from pytential import bind, sym
    from pytools.convergence import EOCRecorder

    s_eoc_rec = EOCRecorder()
    d_eoc_rec = EOCRecorder()

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

        discr = make_upsampling_qbx_discr(
                cl_ctx, mesh, target_order, qbx_order)

        nodes = discr.nodes().with_queue(queue)

        if 0:
            centers = discr.source_discr.centers(discr, 1)
            nodes_h = nodes.get()
            centers_h = [centers[0].get(), centers[1].get()]
            pt.plot(nodes_h[0], nodes_h[1], "x-")
            pt.plot(centers_h[0], centers_h[1], "o")
            pt.show()

        angle = cl.clmath.atan2(nodes[1]*ellipse_aspect, nodes[0])

        ellipse_fraction = ((1-ellipse_aspect)/(1+ellipse_aspect))**mode_nr

        # (2.6) in [1]
        J = cl.clmath.sqrt(
                cl.clmath.sin(angle)**2
                + (1/ellipse_aspect)**2 * cl.clmath.cos(angle)**2)

        # {{{ single layer

        sigma = cl.clmath.cos(mode_nr*angle)/J

        s_sigma_op = bind(discr, sym.S(0, sym.var("sigma")))
        s_sigma = s_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        s_eigval = 1/(2*mode_nr) * (1 + (-1)**mode_nr * ellipse_fraction)

        # (2.12) in [1]
        s_sigma_ref = s_eigval*J*sigma

        if 0:
            pt.plot(s_sigma.get(), label="result")
            pt.plot(s_sigma_ref.get(), label="ref")
            pt.legend()
            pt.show()

        s_err = (
                discr.norm(queue, s_sigma - s_sigma_ref)
                /
                discr.norm(queue, s_sigma_ref))
        s_eoc_rec.add_data_point(1/nelements, s_err)

        # }}}

        # {{{ double layer

        sigma = cl.clmath.cos(mode_nr*angle)

        d_sigma_op = bind(discr, sym.D(0, sym.var("sigma")))
        d_sigma = d_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        d_eigval = -1 * (-1)**mode_nr * 1/2*ellipse_fraction

        d_sigma_ref = d_eigval*sigma

        if 0:
            pt.plot(d_sigma.get(), label="result")
            pt.plot(d_sigma_ref.get(), label="ref")
            pt.legend()
            pt.show()

        if ellipse_aspect == 1:
            d_ref_norm = discr.norm(queue, sigma)
        else:
            d_ref_norm = discr.norm(queue, d_sigma_ref)

        d_err = (
                discr.norm(queue, d_sigma - d_sigma_ref)
                /
                d_ref_norm)
        d_eoc_rec.add_data_point(1/nelements, d_err)

        # }}}

    print "Errors for S:"
    print s_eoc_rec
    target_order = qbx_order + 1
    #assert s_eoc_rec.order_estimate() > target_order - 1.5

    print "Errors for D:"
    print d_eoc_rec
    target_order = qbx_order
    assert d_eoc_rec.order_estimate() > target_order - 1.5

# }}}


# {{{ integral equation test backend

def run_int_eq_test(
        cl_ctx, queue, curve_f, nelements, qbx_order, bc_type, loc_sign, k):
    target_order = 7

    mesh = make_curve_mesh(curve_f,
            np.linspace(0, 1, nelements+1),
            target_order)

    if 0:
        from pytential.visualization import show_mesh
        show_mesh(mesh)

        pt.gca().set_aspect("equal")
        pt.show()

    from pytential.discretization.qbx import make_upsampling_qbx_discr

    discr = make_upsampling_qbx_discr(
            cl_ctx, mesh, target_order, qbx_order)

    # {{{ set up operator

    from pytential import sym, bind
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
        op = DirichletOperator(knl, loc_sign, use_l2_weighting=True)
    elif bc_type == "neumann":
        op = NeumannOperator(knl, loc_sign, use_l2_weighting=True,
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

    # {{{ establish BCs

    from sumpy.p2p import P2P
    pot_p2p = P2P(cl_ctx,
            [knl], exclude_self=False, value_dtypes=dtype)

    evt, (test_direct,) = pot_p2p(
            queue, test_targets, point_sources, [source_charges], **knl_kwargs)

    nodes = discr.nodes()
    if 0:
        n = nodes.get(queue=queue)
        pt.plot(n[0], n[1], "x-")
        pt.show()

    if bc_type == "dirichlet":
        evt, (bc,) = pot_p2p(
                queue, nodes, point_sources, [source_charges],
                **knl_kwargs)

    elif bc_type == "neumann":
        grad_p2p = P2P(cl_ctx,
                [AxisTargetDerivative(0, knl), AxisTargetDerivative(1, knl)],
                exclude_self=False, value_dtypes=dtype)
        evt, (grad0, grad1) = grad_p2p(
                queue, point_sources, [source_charges],
                **knl_kwargs)

        1/0  # FIXME use of normals
        bc = (
                grad0*discr.normals[0].reshape(-1) +
                grad1*discr.normals[1].reshape(-1))

    # }}}

    # {{{ solve

    bound_op = bind(discr, op_u)

    rhs = bind(discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bc)

    from pytential.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "u", k=k),
            rhs, tol=1e-14, progress=True,
            hard_failure=False)

    u = gmres_result.solution
    print "gmres state:", gmres_result.state

    if 0:
        # {{{ build matrix for spectrum check

        from sumpy.tools import build_matrix
        mat = build_matrix(bound_op.scipy_op("u"))
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

    from pytential.discretization.target import PointsTarget

    bound_tgt_op = bind((discr, PointsTarget(test_targets)),
            op.representation(sym.var("u")))

    test_via_bdry = bound_tgt_op(queue, u=u, k=k)

    err = test_direct-test_via_bdry

    if k == 0 and bc_type == "neumann" and loc_sign == -1:
        op_ones = bound_tgt_op(u=np.ones(len(discr), dtype=dtype))
        assert la.norm(op_ones) > 1e-4
        op_ones = op_ones/la.norm(op_ones)

        err = err - np.vdot(op_ones, err)*op_ones

    err = err.get()
    test_direct = test_direct.get()
    test_via_bdry = test_via_bdry.get()

    rel_err_2 = la.norm(err)/la.norm(test_direct)
    rel_err_inf = la.norm(err, np.inf)/la.norm(test_direct, np.inf)

    # }}}

    print "rel_err_2: %g rel_err_inf: %g" % (rel_err_2, rel_err_inf)

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
                (discr, PointsTarget(fplot.points)),
                op.representation(sym.var("u"))
                )(queue, u=u, k=k)
        fld_from_src = fld_from_src.get()
        fld_from_bdry = fld_from_bdry.get()

        nodes = discr.nodes().get(queue=queue)

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
        cmap._lut[(cmap.N*99)//100:, -1] = 0  # make last percent

        prep()
        if 1:
            pt.subplot(131)
            #pt.title("Field error (loc_sign=%s)" % loc_sign)
            log_err = np.log10(1e-20+np.abs(fld_from_src-fld_from_bdry))
            log_err = np.minimum(-3, log_err)
            fplot.show_scalar_in_matplotlib(log_err, cmap=cmap)

            #from matplotlib.colors import Normalize
            #im.set_norm(Normalize(vmin=-6, vmax=1))

            #cb = pt.colorbar(shrink=0.9)
            #cb.set_label(r"$\log_{10}(\mathdefault{Error})$")

        if 1:
            pt.subplot(132)
            prep()
            pt.title("Source Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_src.real, max_val=3)

        if 1:
            pt.subplot(133)
            prep()
            pt.title("Solved Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_bdry.real, max_val=3)

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
            gmres_result=gmres_result)

# }}}


# {{{ integral equation test frontend

@pytest.mark.parametrize(("curve_name", "curve_f"), [
    ("circle", partial(ellipse, 1)),
    ("3-to-1 ellipse", partial(ellipse, 3)),
    #("starfish", starfish),
    ])
@pytest.mark.parametrize("k", [0])
@pytest.mark.parametrize("bc_type", ["dirichlet"])
@pytest.mark.parametrize("loc_sign", [+1, -1])
@pytest.mark.parametrize("qbx_order", [3, 5, 7])
# Sample test run:
# 'test_integral_equation(cl._csc, circle, 30, 5, "dirichlet", 1, 5)'
def test_integral_equation(
        ctx_getter, curve_name, curve_f, qbx_order, bc_type, loc_sign, k):
    # logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from pytools.convergence import EOCRecorder
    print("curve_name: %s, qbx_order: %d, bc_type: %s, loc_sign: %s, "
            "helmholtz_k: %s"
            % (curve_name, qbx_order, bc_type, loc_sign, k))

    eoc_rec = EOCRecorder()
    for nelements in [30, 40, 50]:
        result = run_int_eq_test(
                cl_ctx, queue, curve_f, nelements, qbx_order,
                bc_type, loc_sign, k)

        eoc_rec.add_data_point(1/nelements, result.rel_err_2)

    print eoc_rec
    assert eoc_rec.order_estimate() > qbx_order - 1.3

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
