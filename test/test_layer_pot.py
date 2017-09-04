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
        make_curve_mesh, NArmedStarfish)
from sumpy.visualization import FieldPlotter
from pytential import bind, sym

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)


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
        #fplot.show_scalar_in_mayavi(0.1*.get(queue))
        fplot.write_vtk_file("potential.vts", [
            ("fmm_fld_in_vol", fmm_fld_in_vol.get(queue)),
            ("direct_fld_in_vol", direct_fld_in_vol.get(queue))
            ])

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


def test_unregularized_off_surface_fmm_vs_direct(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    nelements = 300
    target_order = 8
    fmm_order = 4

    mesh = make_curve_mesh(WobblyCircle.random(8, seed=30),
                np.linspace(0, 1, nelements+1),
                target_order)

    from pytential.unregularized import UnregularizedLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    direct = UnregularizedLayerPotentialSource(
            density_discr,
            fmm_order=False,
            )
    fmm = direct.copy(fmm_level_to_order=lambda _: fmm_order)

    sigma = density_discr.zeros(queue) + 1

    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=100)
    from pytential.target import PointsTarget
    ptarget = PointsTarget(fplot.points)
    from sumpy.kernel import LaplaceKernel

    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=None)

    direct_fld_in_vol = bind((direct, ptarget), op)(queue, sigma=sigma)
    fmm_fld_in_vol = bind((fmm, ptarget), op)(queue, sigma=sigma)

    err = cl.clmath.fabs(fmm_fld_in_vol - direct_fld_in_vol)

    linf_err = cl.array.max(err).get()
    print("l_inf error:", linf_err)
    assert linf_err < 5e-3

# }}}


# {{{ test performance data gathering

def test_perf_data_gathering(ctx_getter, n_arms=5):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    target_order = 8

    starfish_func = NArmedStarfish(n_arms, 0.8)
    mesh = make_curve_mesh(
            starfish_func,
            np.linspace(0, 1, n_arms * 30),
            target_order)

    sigma_sym = sym.var("sigma")

    # The kernel doesn't really matter here
    from sumpy.kernel import LaplaceKernel
    k_sym = LaplaceKernel(mesh.ambient_dim)

    sym_op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)
    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    results = []

    def inspect_geo_data(insn, bound_expr, geo_data):
        from pytential.qbx.fmm import assemble_performance_data
        perf_data = assemble_performance_data(geo_data, uses_pde_expansions=True)
        results.append(perf_data)

        return False  # no need to do the actual FMM

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order,
            # qbx order and fmm order don't really matter
            10, fmm_order=10,
            _expansions_in_tree_have_extent=True,
            _expansion_stick_out_factor=0.5,
            geometry_data_inspector=inspect_geo_data,
            target_association_tolerance=1e-10,
            )

    lpot_source, _ = lpot_source.with_refinement()

    density_discr = lpot_source.density_discr

    if 0:
        from meshmode.discretization.visualization import draw_curve
        draw_curve(density_discr)
        import matplotlib.pyplot as plt
        plt.show()

    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    bind(lpot_source, sym_op)(queue, sigma=sigma)

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
