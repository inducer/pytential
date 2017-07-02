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
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh)
from sumpy.visualization import FieldPlotter
from pytential import bind, sym, norm

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


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
    area_sym = prim.integral(2, 1, 1)

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

    target_order = 8

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

        fmm_order = qbx_order + 3
        if fmm_order > 6:
            # FIXME: for now
            fmm_order = False

        pre_density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
        qbx, _ = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                qbx_order, fmm_order=fmm_order,
                _expansions_in_tree_have_extent=True,
                ).with_refinement()

        density_discr = qbx.density_discr
        nodes = density_discr.nodes().with_queue(queue)

        if 0:
            # plot geometry, centers, normals

            from pytential.qbx.utils import get_centers_on_side
            centers = get_centers_on_side(qbx, 1)

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

        from sumpy.kernel import LaplaceKernel
        lap_knl = LaplaceKernel(2)

        # {{{ single layer

        sigma = cl.clmath.cos(mode_nr*angle)/J

        s_sigma_op = bind(qbx, sym.S(lap_knl, sym.var("sigma")))
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
        s_eoc_rec.add_data_point(qbx.h_max, s_err)

        # }}}

        # {{{ double layer

        sigma = cl.clmath.cos(mode_nr*angle)

        d_sigma_op = bind(qbx,
                sym.D(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
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
        d_eoc_rec.add_data_point(qbx.h_max, d_err)

        # }}}

        if ellipse_aspect == 1:
            # {{{ S'

            sigma = cl.clmath.cos(mode_nr*angle)

            sp_sigma_op = bind(qbx,
                    sym.Sp(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
            sp_sigma = sp_sigma_op(queue=queue, sigma=sigma)
            sp_eigval = 0

            sp_sigma_ref = sp_eigval*sigma

            sp_err = (
                    norm(density_discr, queue, sp_sigma - sp_sigma_ref)
                    /
                    norm(density_discr, queue, sigma))
            sp_eoc_rec.add_data_point(qbx.h_max, sp_err)

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

    pre_density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr,
            4*target_order,
            qbx_order,
            fmm_order=fmm_order,
            ).with_refinement()

    density_discr = qbx.density_discr

    from sumpy.kernel import LaplaceKernel
    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=-2)

    sigma = density_discr.zeros(queue) + 1

    fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)
    from pytential.target import PointsTarget
    fld_in_vol = bind(
            (qbx, PointsTarget(fplot.points)),
            op)(queue, sigma=sigma)

    err = cl.clmath.fabs(fld_in_vol - (-1))

    linf_err = cl.array.max(err).get()
    print("l_inf error:", linf_err)

    if do_plot:
        fplot.show_scalar_in_matplotlib(fld_in_vol.get())
        import matplotlib.pyplot as pt
        pt.colorbar()
        pt.show()

    # FIXME: Why does the FMM only meet this sloppy tolerance?
    assert linf_err < 1e-2

# }}}


# {{{ test off-surface eval vs direct

def test_off_surface_eval_vs_direct(ctx_getter,  do_plot=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    nelements = 300
    target_order = 8
    qbx_order = 3

    mesh = make_curve_mesh(WobblyCircle.random(8, seed=30),
                np.linspace(0, 1, nelements+1),
                target_order)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    pre_density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    direct_qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order, qbx_order,
            fmm_order=False,
            target_association_tolerance=0.05,
            ).with_refinement()
    fmm_qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order, qbx_order,
            fmm_order=qbx_order + 3,
            _expansions_in_tree_have_extent=True,
            target_association_tolerance=0.05,
            ).with_refinement()

    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1000)
    from pytential.target import PointsTarget
    ptarget = PointsTarget(fplot.points)
    from sumpy.kernel import LaplaceKernel

    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=None)

    from pytential.qbx import QBXTargetAssociationFailedException
    try:
        direct_density_discr = direct_qbx.density_discr
        direct_sigma = direct_density_discr.zeros(queue) + 1
        direct_fld_in_vol = bind((direct_qbx, ptarget), op)(
                queue, sigma=direct_sigma)

    except QBXTargetAssociationFailedException as e:
        fplot.show_scalar_in_matplotlib(e.failed_target_flags.get(queue))
        import matplotlib.pyplot as pt
        pt.show()
        raise

    fmm_density_discr = fmm_qbx.density_discr
    fmm_sigma = fmm_density_discr.zeros(queue) + 1
    fmm_fld_in_vol = bind((fmm_qbx, ptarget), op)(queue, sigma=fmm_sigma)

    err = cl.clmath.fabs(fmm_fld_in_vol - direct_fld_in_vol)

    linf_err = cl.array.max(err).get()
    print("l_inf error:", linf_err)

    if do_plot:
        #fplot.show_scalar_in_mayavi(0.1*cl.clmath.log10(1e-15 + err).get(queue))
        fplot.show_scalar_in_mayavi(fmm_fld_in_vol.get(queue))
        import mayavi.mlab as mlab
        mlab.show()

    assert linf_err < 1e-3

# }}}


# {{{ unregularized tests


def test_unregularized_with_ones_kernel(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    nelements = 10
    order = 8

    mesh = make_curve_mesh(partial(ellipse, 1),
            np.linspace(0, 1, nelements+1),
            order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr = Discretization(cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    from pytential.unregularized import UnregularizedLayerPotentialSource
    lpot_src = UnregularizedLayerPotentialSource(discr)

    from sumpy.kernel import one_kernel_2d

    expr = sym.IntG(one_kernel_2d, sym.var("sigma"), qbx_forced_limit=None)

    from pytential.target import PointsTarget
    op_self = bind(lpot_src, expr)
    op_nonself = bind((lpot_src, PointsTarget(np.zeros((2, 1), dtype=float))), expr)

    with cl.CommandQueue(cl_ctx) as queue:
        sigma = cl.array.zeros(queue, discr.nnodes, dtype=float)
        sigma.fill(1)
        sigma.finish()

        result_self = op_self(queue, sigma=sigma)
        result_nonself = op_nonself(queue, sigma=sigma)

    assert np.allclose(result_self.get(), 2 * np.pi)
    assert np.allclose(result_nonself.get(), 2 * np.pi)

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
