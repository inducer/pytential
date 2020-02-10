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
from pytential import bind, sym, norm

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)


# {{{ geometry test

def test_geometry(ctx_factory):
    cl_ctx = ctx_factory()
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
def test_off_surface_eval(ctx_factory, use_fmm, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    nelements = 30
    target_order = 8
    qbx_order = 3
    if use_fmm:
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

    assert linf_err < 1e-3

# }}}


# {{{ test off-surface eval vs direct

def test_off_surface_eval_vs_direct(ctx_factory,  do_plot=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
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

    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)
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


def test_unregularized_with_ones_kernel(ctx_factory):
    cl_ctx = ctx_factory()
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


def test_unregularized_off_surface_fmm_vs_direct(ctx_factory):
    cl_ctx = ctx_factory()
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
    fmm = direct.copy(
            fmm_level_to_order=lambda kernel, kernel_args, tree, level: fmm_order)

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


# {{{ test 3D jump relations

@pytest.mark.parametrize("relation", ["sp", "nxcurls", "div_s"])
def test_3d_jump_relations(ctx_factory, relation, visualize=False):
    # logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    if relation == "div_s":
        target_order = 3
    else:
        target_order = 4

    qbx_order = target_order

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for nel_factor in [6, 10, 14]:
        from meshmode.mesh.generation import generate_torus
        mesh = generate_torus(
                5, 2, order=target_order,
                n_major=2*nel_factor, n_minor=nel_factor)

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
        pre_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(3))

        from pytential.qbx import QBXLayerPotentialSource
        qbx, _ = QBXLayerPotentialSource(
                pre_discr, fine_order=4*target_order,
                qbx_order=qbx_order,
                fmm_order=qbx_order + 5,
                fmm_backend="fmmlib"
                ).with_refinement()

        from sumpy.kernel import LaplaceKernel
        knl = LaplaceKernel(3)

        def nxcurlS(qbx_forced_limit):

            return sym.n_cross(sym.curl(sym.S(
                knl,
                sym.cse(sym.tangential_to_xyz(density_sym), "jxyz"),
                qbx_forced_limit=qbx_forced_limit)))

        x, y, z = qbx.density_discr.nodes().with_queue(queue)
        m = cl.clmath

        if relation == "nxcurls":
            density_sym = sym.make_sym_vector("density", 2)

            jump_identity_sym = (
                    nxcurlS(+1)
                    - (nxcurlS("avg") + 0.5*sym.tangential_to_xyz(density_sym)))

            # The tangential coordinate system is element-local, so we can't just
            # conjure up some globally smooth functions, interpret their values
            # in the tangential coordinate system, and be done. Instead, generate
            # an XYZ function and project it.
            density = bind(
                    qbx,
                    sym.xyz_to_tangential(sym.make_sym_vector("jxyz", 3)))(
                            queue,
                            jxyz=sym.make_obj_array([
                                m.cos(0.5*x) * m.cos(0.5*y) * m.cos(0.5*z),
                                m.sin(0.5*x) * m.cos(0.5*y) * m.sin(0.5*z),
                                m.sin(0.5*x) * m.cos(0.5*y) * m.cos(0.5*z),
                                ]))

        elif relation == "sp":

            density = m.cos(2*x) * m.cos(2*y) * m.cos(z)
            density_sym = sym.var("density")

            jump_identity_sym = (
                    sym.Sp(knl, density_sym, qbx_forced_limit=+1)
                    - (sym.Sp(knl, density_sym, qbx_forced_limit="avg")
                        - 0.5*density_sym))

        elif relation == "div_s":

            density = m.cos(2*x) * m.cos(2*y) * m.cos(z)
            density_sym = sym.var("density")

            jump_identity_sym = (
                    sym.div(sym.S(knl, sym.normal(3).as_vector()*density_sym,
                        qbx_forced_limit="avg"))
                    + sym.D(knl, density_sym, qbx_forced_limit="avg"))

        else:
            raise ValueError("unexpected value of 'relation': %s" % relation)

        bound_jump_identity = bind(qbx, jump_identity_sym)
        jump_identity = bound_jump_identity(queue, density=density)

        h_max = bind(qbx, sym.h_max(qbx.ambient_dim))(queue)
        err = (
                norm(qbx, queue, jump_identity, np.inf)
                / norm(qbx, queue, density, np.inf))
        print("ERROR", h_max, err)

        eoc_rec.add_data_point(h_max, err)

        # {{{ visualization

        if visualize and relation == "nxcurls":
            nxcurlS_ext = bind(qbx, nxcurlS(+1))(queue, density=density)
            nxcurlS_avg = bind(qbx, nxcurlS("avg"))(queue, density=density)
            jtxyz = bind(qbx, sym.tangential_to_xyz(density_sym))(
                    queue, density=density)

            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(queue, qbx.density_discr, target_order+3)

            bdry_normals = bind(qbx, sym.normal(3))(queue)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % nel_factor, [
                ("jt", jtxyz),
                ("nxcurlS_ext", nxcurlS_ext),
                ("nxcurlS_avg", nxcurlS_avg),
                ("bdry_normals", bdry_normals),
                ])

        if visualize and relation == "sp":
            sp_ext = bind(qbx, sym.Sp(knl, density_sym, qbx_forced_limit=+1))(
                    queue, density=density)
            sp_avg = bind(qbx, sym.Sp(knl, density_sym, qbx_forced_limit="avg"))(
                    queue, density=density)

            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(queue, qbx.density_discr, target_order+3)

            bdry_normals = bind(qbx, sym.normal(3))(queue)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % nel_factor, [
                ("density", density),
                ("sp_ext", sp_ext),
                ("sp_avg", sp_avg),
                ("bdry_normals", bdry_normals),
                ])

        # }}}

    print(eoc_rec)

    assert eoc_rec.order_estimate() >= qbx_order - 1.5

# }}}


# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
