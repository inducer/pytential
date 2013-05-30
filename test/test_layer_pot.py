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
import pytest
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

from functools import partial
from pytential.mesh.generation import (
        ellipse, cloverleaf, starfish, drop, n_gon,
        make_curve_mesh)
from sumpy.visualization import FieldPlotter

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
    area_sym = prim.integral(prim.Ones())

    area = bind(discr, area_sym)(queue)

    err = abs(area-2*np.pi)
    print err
    assert err < 1e-3

# }}}


# {{{ integral equation test

@pytest.mark.parametrize(("curve_name", "curve_f"), [
    ("circle", partial(ellipse, 1)),
    ("5-to-1 ellipse", partial(ellipse, 5)),
    ("starfish", starfish),
    ])
@pytest.mark.parametrize("kernel", [0])
@pytest.mark.parametrize("bc_type", ["dirichlet", "neumann"])
@pytest.mark.parametrize("loc_sign", [+1, -1])
@pytest.mark.parametrize("qbx_order", [3, 5, 7])
# Sample test run:
# 'test_integral_equation(cl._csc, circle, 30, 5, "dirichlet", 1, 5)'
def test_integral_equation(
        ctx_getter, curve_f, nelements, qbx_order, bc_type, loc_sign, k):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    target_order = 7
    source_order = 63

    mesh = make_curve_mesh(curve_f,
            np.linspace(0, 1, nelements+1),
            target_order)

    if 0:
        from pytential.visualization import show_mesh
        show_mesh(mesh)

        pt.gca().set_aspect("equal")
        pt.show()

    from pytential.discretization.poly_element import \
            PolynomialElementDiscretization
    from pytential.discretization.qbx import QBXDiscretization

    src_discr = QBXDiscretization(
            cl_ctx, mesh, qbx_order, target_order, source_order)
    tgt_discr = PolynomialElementDiscretization(
            cl_ctx, mesh, target_order)

    # {{{ set up operator

    from pytential.symbolic.primitives import Variable
    from pytential.symbolic.pde.scalar import (
            DirichletOperator,
            NeumannOperator)

    if bc_type == "dirichlet":
        op = DirichletOperator(k, loc_sign, use_l2_weighting=True)
    elif bc_type == "neumann":
        op = NeumannOperator(k, loc_sign, use_l2_weighting=True,
                 use_improved_operator=False)
    else:
        assert False

    op_u = op.operator(Variable("u"))

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

    from pytential.symbolic.execution import bind

    np.random.seed(22)
    source_charges = np.random.randn(point_sources.shape[1])
    source_charges[-1] = -np.sum(source_charges[:-1])
    source_charges = source_charges.astype(np.complex128)
    assert np.sum(source_charges) < 1e-15

    # }}}

    # {{{ establish BCs

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel, TargetDerivative
    if k:
        knl = HelmholtzKernel(2)
        knl_kwargs = {"k": k}
    else:
        knl = LaplaceKernel(2)
        knl_kwargs = {}

    from sumpy.p2p import P2P
    pot_p2p = P2P(cl_ctx,
            [knl], exclude_self=False, value_dtypes=np.complex128)

    evt, (test_direct,) = pot_p2p(
            queue, test_targets, point_sources, [source_charges], **knl_kwargs)

    nodes = tgt_discr.nodes(queue)
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
                [TargetDerivative(0, knl), TargetDerivative(1, knl)],
                exclude_self=False, value_dtypes=np.complex128)
        evt, (grad0, grad1) = grad_p2p(
                queue, point_sources, [source_charges],
                **knl_kwargs)

        1/0  # FIXME use of normals
        bc = (
                grad0*tgt_discr.normals[0].reshape(-1) +
                grad1*tgt_discr.normals[1].reshape(-1))

    # }}}

    # {{{ solve

    rhs = bind(tgt_discr, op.prepare_rhs(Variable("bc")))(queue, bc=bc)

    bound_op = bind((src_discr, tgt_discr), op_u)

    from pytential.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "u"),
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

    bound_tgt_op = bind(op.representation(Variable("u")),
            (discr, PointsTarget(test_targets)), iprec=6)
    test_via_bdry = bound_tgt_op(u=u)

    err = test_direct-test_via_bdry

    if k == 0 and bc_type == "neumann" and loc_sign == -1:
        op_ones = bound_tgt_op(u=np.ones(len(discr), dtype=np.complex128))
        assert la.norm(op_ones) > 1e-4
        op_ones = op_ones/la.norm(op_ones)

        err = err - np.vdot(op_ones, err)*op_ones

    rel_err_2 = la.norm(err)/la.norm(test_direct)
    rel_err_inf = (
            la.norm(test_direct-test_via_bdry, np.inf)
            / la.norm(test_direct, np.inf))

    # }}}
    print "rel_err_2: %g rel_err_inf: %g" % (rel_err_2, rel_err_inf)

    # {{{ plotting

    fplot = FieldPlotter(np.zeros(2),
            extent=1.25*2*max(test_src_geo_radius, test_tgt_geo_radius),
            points=500)

    if 1:
        #pt.plot(u)
        #pt.show()

        evt, (fld_from_src,) = pot_p2p(
                queue, fplot.points, point_sources, [source_charges],
                **knl_kwargs)
        fld_from_bdry = bind(op.representation(Variable("u")),
            (discr, PointsTarget(fplot.points)), iprec=3)(u=u)

        def prep():
            pt.plot(point_sources[:, 0], point_sources[:, 1], "o",
                    label="Monopole 'Point Charges'")
            pt.plot(test_targets[:, 0], test_targets[:, 1], "v",
                    label="Observation Points")
            pt.plot(discr.nodes[:, 0], discr.nodes[:, 1], "k-",
                    label=r"$\Gamma$")

        from matplotlib.cm import get_cmap
        cmap = get_cmap()
        cmap._init()
        cmap._lut[(cmap.N*99)//100:, -1] = 0  # make last percent

        prep()
        if 1:
            #pt.subplot(131)
            #pt.title("Field error (loc_sign=%s)" % loc_sign)
            log_err = np.log10(1e-20+np.abs(fld_from_src-fld_from_bdry))
            log_err = np.minimum(-3, log_err)
            fplot.show_scalar_in_matplotlib(log_err, cmap=cmap)

            #from matplotlib.colors import Normalize
            #im.set_norm(Normalize(vmin=-6, vmax=1))

            #cb = pt.colorbar(shrink=0.9)
            #cb.set_label(r"$\log_{10}(\mathdefault{Error})$")

        if 0:
            pt.subplot(132)
            prep()
            pt.title("Source Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_src.real, maxval=3)

        if 0:
            pt.subplot(133)
            prep()
            pt.title("Solved Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_bdry.real, maxval=3)

        # total field
        #fplot.show_scalar_in_matplotlib(
                #fld_from_src.real+fld_from_bdry.real, maxval=0.1)

        #pt.colorbar()

        pt.legend(loc="best", prop=dict(size=15))
        from matplotlib.ticker import NullFormatter
        pt.gca().xaxis.set_major_formatter(NullFormatter())
        pt.gca().yaxis.set_major_formatter(NullFormatter())

        pt.gca().set_aspect("equal")

        border_factor_top = 0.9
        border_factor = 0.3

        xl, xh = pt.xlim()
        xhsize = 0.5*(xh-xl)
        pt.xlim(xl-border_factor*xhsize, xh+border_factor*xhsize)

        yl, yh = pt.ylim()
        yhsize = 0.5*(yh-yl)
        pt.ylim(yl-border_factor_top*yhsize, yh+border_factor*yhsize)

        pt.savefig("helmholtz.pdf", dpi=600)
        #pt.show()

        # }}}


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
