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
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh, NArmedStarfish)
from sumpy.visualization import FieldPlotter

from pytential import bind, sym, norm
from pytential import GeometryCollection

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)


# {{{ geometry test

def test_geometry(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nelements = 30
    order = 5

    mesh = make_curve_mesh(partial(ellipse, 1),
            np.linspace(0, 1, nelements+1),
            order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    import pytential.symbolic.primitives as prim
    area_sym = prim.integral(2, 1, 1)

    area = bind(discr, area_sym)(actx)

    err = abs(area-2*np.pi)
    print(err)
    assert err < 1e-3

# }}}


# {{{ test off-surface eval

@pytest.mark.parametrize("use_fmm", [True, False])
def test_off_surface_eval(ctx_factory, use_fmm, visualize=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

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
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    qbx = QBXLayerPotentialSource(
            pre_density_discr,
            4*target_order,
            qbx_order,
            fmm_order=fmm_order,
            )

    from pytential.target import PointsTarget
    fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)
    targets = PointsTarget(fplot.points)

    places = GeometryCollection((qbx, targets))
    density_discr = places.get_discretization(places.auto_source.geometry)

    from sumpy.kernel import LaplaceKernel
    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=-2)

    sigma = density_discr.zeros(actx) + 1
    fld_in_vol = bind(places, op)(actx, sigma=sigma)
    fld_in_vol_exact = -1

    err = actx.np.fabs(fld_in_vol - fld_in_vol_exact)
    linf_err = actx.to_numpy(err).max()
    print("l_inf error:", linf_err)

    if visualize:
        fplot.show_scalar_in_matplotlib(actx.to_numpy(fld_in_vol))
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
    actx = PyOpenCLArrayContext(queue)

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
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    direct_qbx = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order, qbx_order,
            fmm_order=False,
            target_association_tolerance=0.05,
            )
    fmm_qbx = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order, qbx_order,
            fmm_order=qbx_order + 3,
            _expansions_in_tree_have_extent=True,
            target_association_tolerance=0.05,
            )

    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)
    from pytential.target import PointsTarget
    ptarget = PointsTarget(fplot.points)
    from sumpy.kernel import LaplaceKernel

    places = GeometryCollection({
        "direct_qbx": direct_qbx,
        "fmm_qbx": fmm_qbx,
        "target": ptarget})

    direct_density_discr = places.get_discretization("direct_qbx")
    fmm_density_discr = places.get_discretization("fmm_qbx")

    from pytential.qbx import QBXTargetAssociationFailedException
    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=None)
    try:
        direct_sigma = direct_density_discr.zeros(actx) + 1
        direct_fld_in_vol = bind(places, op,
                auto_where=("direct_qbx", "target"))(
                        actx, sigma=direct_sigma)
    except QBXTargetAssociationFailedException as e:
        fplot.show_scalar_in_matplotlib(
            actx.to_numpy(actx.thaw(e.failed_target_flags)))
        import matplotlib.pyplot as pt
        pt.show()
        raise

    fmm_sigma = fmm_density_discr.zeros(actx) + 1
    fmm_fld_in_vol = bind(places, op,
            auto_where=("fmm_qbx", "target"))(
                    actx, sigma=fmm_sigma)

    err = actx.np.fabs(fmm_fld_in_vol - direct_fld_in_vol)
    linf_err = actx.to_numpy(err).max()
    print("l_inf error:", linf_err)

    if do_plot:
        #fplot.show_scalar_in_mayavi(0.1*.get(queue))
        fplot.write_vtk_file("potential.vts", [
            ("fmm_fld_in_vol", actx.to_numpy(fmm_fld_in_vol)),
            ("direct_fld_in_vol", actx.to_numpy(direct_fld_in_vol))
            ])

    assert linf_err < 1e-3

# }}}


# {{{ unregularized tests

def test_unregularized_with_ones_kernel(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nelements = 10
    order = 8

    mesh = make_curve_mesh(partial(ellipse, 1),
            np.linspace(0, 1, nelements+1),
            order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    from pytential.unregularized import UnregularizedLayerPotentialSource
    lpot_source = UnregularizedLayerPotentialSource(discr)
    from pytential.target import PointsTarget
    targets = PointsTarget(np.zeros((2, 1), dtype=float))

    places = GeometryCollection({
        sym.DEFAULT_SOURCE: lpot_source,
        sym.DEFAULT_TARGET: lpot_source,
        "target_non_self": targets})

    from sumpy.kernel import one_kernel_2d
    sigma_sym = sym.var("sigma")
    op = sym.IntG(one_kernel_2d, sigma_sym, qbx_forced_limit=None)

    sigma = discr.zeros(actx) + 1

    result_self = bind(places, op,
            auto_where=places.auto_where)(
                    actx, sigma=sigma)
    result_nonself = bind(places, op,
            auto_where=(places.auto_source, "target_non_self"))(
                    actx, sigma=sigma)

    from meshmode.dof_array import flatten
    assert np.allclose(actx.to_numpy(flatten(result_self)), 2 * np.pi)
    assert np.allclose(actx.to_numpy(result_nonself), 2 * np.pi)


def test_unregularized_off_surface_fmm_vs_direct(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nelements = 300
    target_order = 8
    fmm_order = 4

    # {{{ geometry

    mesh = make_curve_mesh(WobblyCircle.random(8, seed=30),
                np.linspace(0, 1, nelements+1),
                target_order)

    from pytential.unregularized import UnregularizedLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    density_discr = Discretization(
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    direct = UnregularizedLayerPotentialSource(
            density_discr,
            fmm_order=False,
            )
    fmm = direct.copy(
            fmm_level_to_order=lambda kernel, kernel_args, tree, level: fmm_order)

    sigma = density_discr.zeros(actx) + 1

    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=100)
    from pytential.target import PointsTarget
    ptarget = PointsTarget(fplot.points)

    from pytential import GeometryCollection
    places = GeometryCollection({
        "unregularized_direct": direct,
        "unregularized_fmm": fmm,
        "targets": ptarget})

    # }}}

    # {{{ check

    from sumpy.kernel import LaplaceKernel
    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=None)

    direct_fld_in_vol = bind(places, op,
            auto_where=("unregularized_direct", "targets"))(
                    actx, sigma=sigma)
    fmm_fld_in_vol = bind(places, op,
            auto_where=("unregularized_fmm", "targets"))(actx, sigma=sigma)

    err = actx.np.fabs(fmm_fld_in_vol - direct_fld_in_vol)
    linf_err = actx.to_numpy(err).max()
    print("l_inf error:", linf_err)

    assert linf_err < 5e-3

    # }}}

# }}}


# {{{ test 3D jump relations

@pytest.mark.parametrize("relation", ["sp", "nxcurls", "div_s"])
def test_3d_jump_relations(ctx_factory, relation, visualize=False):
    # logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

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
                actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(3))

        from pytential.qbx import QBXLayerPotentialSource
        qbx = QBXLayerPotentialSource(
                pre_discr, fine_order=4*target_order,
                qbx_order=qbx_order,
                fmm_order=qbx_order + 5,
                fmm_backend="fmmlib"
                )

        places = GeometryCollection(qbx)
        density_discr = places.get_discretization(places.auto_source.geometry)

        from sumpy.kernel import LaplaceKernel
        knl = LaplaceKernel(3)

        def nxcurlS(qbx_forced_limit):

            return sym.n_cross(sym.curl(sym.S(
                knl,
                sym.cse(sym.tangential_to_xyz(density_sym), "jxyz"),
                qbx_forced_limit=qbx_forced_limit)))

        from meshmode.dof_array import thaw
        x, y, z = thaw(actx, density_discr.nodes())
        m = actx.np

        if relation == "nxcurls":
            density_sym = sym.make_sym_vector("density", 2)

            jump_identity_sym = (
                    nxcurlS(+1)
                    - (nxcurlS("avg") + 0.5*sym.tangential_to_xyz(density_sym)))

            # The tangential coordinate system is element-local, so we can't just
            # conjure up some globally smooth functions, interpret their values
            # in the tangential coordinate system, and be done. Instead, generate
            # an XYZ function and project it.
            density = bind(places,
                    sym.xyz_to_tangential(sym.make_sym_vector("jxyz", 3)))(
                            actx,
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

        bound_jump_identity = bind(places, jump_identity_sym)
        jump_identity = bound_jump_identity(actx, density=density)

        h_max = bind(places, sym.h_max(qbx.ambient_dim))(actx)
        err = (
                norm(density_discr, jump_identity, np.inf)
                / norm(density_discr, density, np.inf))
        print("ERROR", h_max, err)

        eoc_rec.add_data_point(h_max, err)

        # {{{ visualization

        if visualize and relation == "nxcurls":
            nxcurlS_ext = bind(places, nxcurlS(+1))(actx, density=density)
            nxcurlS_avg = bind(places, nxcurlS("avg"))(actx, density=density)
            jtxyz = bind(places, sym.tangential_to_xyz(density_sym))(
                    actx, density=density)

            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(actx, qbx.density_discr, target_order+3)

            bdry_normals = bind(places, sym.normal(3))(actx)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % nel_factor, [
                ("jt", jtxyz),
                ("nxcurlS_ext", nxcurlS_ext),
                ("nxcurlS_avg", nxcurlS_avg),
                ("bdry_normals", bdry_normals),
                ])

        if visualize and relation == "sp":
            op = sym.Sp(knl, density_sym, qbx_forced_limit=+1)
            sp_ext = bind(places, op)(actx, density=density)
            op = sym.Sp(knl, density_sym, qbx_forced_limit="avg")
            sp_avg = bind(places, op)(actx, density=density)

            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(actx, qbx.density_discr, target_order+3)

            bdry_normals = bind(places,
                    sym.normal(3))(actx).as_vector(dtype=object)

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
