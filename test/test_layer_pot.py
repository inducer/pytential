from __future__ import annotations

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

import pytest
from functools import partial

import numpy as np

from arraycontext import flatten
from pytential import bind, sym, norm
from pytential import GeometryCollection
import meshmode.mesh.generation as mgen
from sumpy.visualization import FieldPlotter

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import logging
logger = logging.getLogger(__name__)

from pytential.utils import (  # noqa: F401
        pytest_teardown_function as teardown_function)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ geometry test

def test_geometry(actx_factory):
    actx = actx_factory()

    nelements = 30
    order = 5

    mesh = mgen.make_curve_mesh(partial(mgen.ellipse, 1),
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
def test_off_surface_eval(actx_factory, use_fmm, visualize=False):
    logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    nelements = 30
    target_order = 8
    qbx_order = 3
    if use_fmm:
        fmm_order = qbx_order
    else:
        fmm_order = False

    mesh = mgen.make_curve_mesh(partial(mgen.ellipse, 3),
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
    targets = PointsTarget(actx.freeze(actx.from_numpy(fplot.points)))

    places = GeometryCollection((qbx, targets))
    density_discr = places.get_discretization(places.auto_source.geometry)

    from sumpy.kernel import LaplaceKernel
    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=-2)

    sigma = density_discr.zeros(actx) + 1
    fld_in_vol = bind(places, op)(actx, sigma=sigma)
    fld_in_vol_exact = -1

    linf_err = actx.to_numpy(
            actx.np.linalg.norm(fld_in_vol - fld_in_vol_exact, ord=np.inf)
            )
    logger.info("l_inf error: %.12e", linf_err)

    if visualize:
        fplot.show_scalar_in_matplotlib(actx.to_numpy(fld_in_vol))
        import matplotlib.pyplot as pt
        pt.colorbar()
        pt.show()

    assert linf_err < 2e-3

# }}}


# {{{ test off-surface eval vs direct

def test_off_surface_eval_vs_direct(actx_factory, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    nelements = 300
    target_order = 8
    qbx_order = 3

    mesh = mgen.make_curve_mesh(mgen.WobblyCircle.random(8, seed=30),
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
    ptarget = PointsTarget(actx.freeze(actx.from_numpy(fplot.points)))
    from sumpy.kernel import LaplaceKernel

    places = GeometryCollection({
        "direct_qbx": direct_qbx,
        "fmm_qbx": fmm_qbx,
        "target": ptarget,
        }, auto_where=("fmm_qbx", "target"))

    direct_density_discr = places.get_discretization("direct_qbx")
    fmm_density_discr = places.get_discretization("fmm_qbx")

    from pytential.qbx import QBXTargetAssociationFailedError
    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=None)
    try:
        direct_sigma = direct_density_discr.zeros(actx) + 1
        direct_fld_in_vol = bind(places, op,
                auto_where=("direct_qbx", "target"))(
                        actx, sigma=direct_sigma)
    except QBXTargetAssociationFailedError as e:
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
        fplot.write_vtk_file("potential.vts", [
            ("fmm_fld_in_vol", actx.to_numpy(fmm_fld_in_vol)),
            ("direct_fld_in_vol", actx.to_numpy(direct_fld_in_vol))
            ])

    assert linf_err < 1e-3

# }}}


# {{{

def test_single_plus_double_with_single_fmm(actx_factory, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    nelements = 300
    target_order = 8
    qbx_order = 3

    mesh = mgen.make_curve_mesh(mgen.WobblyCircle.random(8, seed=30),
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
    ptarget = PointsTarget(actx.freeze(actx.from_numpy(fplot.points)))
    from sumpy.kernel import LaplaceKernel

    places = GeometryCollection({
        "direct_qbx": direct_qbx,
        "fmm_qbx": fmm_qbx,
        "target": ptarget,
        }, auto_where=("fmm_qbx", "target"))

    direct_density_discr = places.get_discretization("direct_qbx")
    fmm_density_discr = places.get_discretization("fmm_qbx")

    knl = LaplaceKernel(2)
    from pytential.qbx import QBXTargetAssociationFailedError
    op_d = sym.D(knl, sym.var("sigma"), qbx_forced_limit=None)
    op_s = sym.S(knl, sym.var("sigma"), qbx_forced_limit=None)
    op = op_d + op_s * 0.5
    try:
        direct_sigma = direct_density_discr.zeros(actx) + 1
        direct_fld_in_vol = bind(places, op,
                auto_where=("direct_qbx", "target"))(
                        actx, sigma=direct_sigma)
    except QBXTargetAssociationFailedError as e:
        fplot.show_scalar_in_matplotlib(
            actx.to_numpy(actx.thaw(e.failed_target_flags)))
        import matplotlib.pyplot as pt
        pt.show()
        raise

    fmm_sigma = fmm_density_discr.zeros(actx) + 1
    fmm_bound_op = bind(places, op, auto_where=("fmm_qbx", "target"))
    fmm_fld_in_vol = fmm_bound_op(actx, sigma=fmm_sigma)

    err = actx.np.fabs(fmm_fld_in_vol - direct_fld_in_vol)
    linf_err = actx.to_numpy(err).max()
    print("l_inf error:", linf_err)

    if do_plot:
        fplot.write_vtk_file("potential.vts", [
            ("fmm_fld_in_vol", actx.to_numpy(fmm_fld_in_vol)),
            ("direct_fld_in_vol", actx.to_numpy(direct_fld_in_vol))
            ])

    assert linf_err < 1e-3

    # check that using one FMM works
    op = op_d.copy(
        source_kernels=(*op_d.source_kernels, knl),
        densities=(*op_d.densities, 0.5 * sym.var("sigma")))
    single_fmm_bound_op = bind(places, op, auto_where=("fmm_qbx", "target"))
    print(single_fmm_bound_op.code)
    single_fmm_fld_in_vol = fmm_bound_op(actx, sigma=fmm_sigma)

    err = actx.np.fabs(fmm_fld_in_vol - single_fmm_fld_in_vol)
    linf_err = actx.to_numpy(err).max()
    print("l_inf error:", linf_err)

    assert linf_err < 1e-15


# }}}

# {{{ unregularized tests

def test_unregularized_with_ones_kernel(actx_factory):
    actx = actx_factory()

    nelements = 10
    order = 8

    mesh = mgen.make_curve_mesh(partial(mgen.ellipse, 1),
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
    targets = PointsTarget(actx.from_numpy(np.zeros((2, 1), dtype=np.float64)))

    places = GeometryCollection(
        {"source": lpot_source,
         "target": lpot_source,
         "target_non_self": targets},
        auto_where=("source", "target"),
    )

    from sumpy.kernel import one_kernel_2d
    sigma_sym = sym.var("sigma")
    op = sym.int_g_vec(one_kernel_2d, sigma_sym, qbx_forced_limit=None)

    sigma = discr.zeros(actx) + 1

    result_self = bind(places, op,
            auto_where=places.auto_where)(
                    actx, sigma=sigma)
    result_nonself = bind(places, op,
            auto_where=(places.auto_source, "target_non_self"))(
                    actx, sigma=sigma)

    assert np.allclose(actx.to_numpy(flatten(result_self, actx)), 2 * np.pi)
    assert np.allclose(actx.to_numpy(result_nonself), 2 * np.pi)


def test_unregularized_off_surface_fmm_vs_direct(actx_factory):
    actx = actx_factory()

    nelements = 300
    target_order = 8
    fmm_order = 4

    # {{{ geometry

    mesh = mgen.make_curve_mesh(mgen.WobblyCircle.random(8, seed=30),
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
    ptarget = PointsTarget(actx.from_numpy(fplot.points))

    from pytential import GeometryCollection
    places = GeometryCollection({
        "unregularized_direct": direct,
        "unregularized_fmm": fmm,
        "targets": ptarget,
        }, auto_where=("unregularized_fmm", "targets"))

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
def test_3d_jump_relations(actx_factory, relation, visualize=False):
    pytest.importorskip("pyfmmlib")
    actx = actx_factory()

    if relation == "div_s":
        target_order = 3
    else:
        target_order = 4

    qbx_order = target_order

    if relation == "sp":
        resolutions = [10, 14, 18]
    else:
        resolutions = [6, 10, 14]

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for nel_factor in resolutions:
        from meshmode.mesh.generation import generate_torus
        mesh = generate_torus(
                5, 2, n_major=2*nel_factor, n_minor=nel_factor,
                order=target_order,
                )

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
        pre_density_discr = Discretization(
                actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))

        from pytential.qbx import QBXLayerPotentialSource
        qbx = QBXLayerPotentialSource(
                pre_density_discr,
                fine_order=5*target_order,
                qbx_order=qbx_order,
                fmm_order=qbx_order + 5,
                fmm_backend="fmmlib"
                )

        places = GeometryCollection(qbx)
        density_discr = places.get_discretization(places.auto_source.geometry)

        from sumpy.kernel import LaplaceKernel
        knl = LaplaceKernel(places.ambient_dim)

        def nxcurlS(knl, density_sym, qbx_forced_limit):
            tangent_sigma_sym = sym.cse(sym.tangential_to_xyz(density_sym), "jxyz")
            return sym.n_cross(sym.curl(
                sym.S(knl, tangent_sigma_sym, qbx_forced_limit=qbx_forced_limit)
                ))

        x, y, z = actx.thaw(density_discr.nodes())
        if relation == "nxcurls":
            density_sym = sym.make_sym_vector("density", 2)
            jump_identity_sym = (
                    nxcurlS(knl, density_sym, +1)
                    - nxcurlS(knl, density_sym, "avg")
                    - 0.5*sym.tangential_to_xyz(density_sym)
                    )

            from pytools import obj_array

            # The tangential coordinate system is element-local, so we can't just
            # conjure up some globally smooth functions, interpret their values
            # in the tangential coordinate system, and be done. Instead, generate
            # an XYZ function and project it.
            jxyz = obj_array.new_1d([
                actx.np.cos(0.5*x) * actx.np.cos(0.5*y) * actx.np.cos(0.5*z),
                actx.np.sin(0.5*x) * actx.np.cos(0.5*y) * actx.np.sin(0.5*z),
                actx.np.sin(0.5*x) * actx.np.cos(0.5*y) * actx.np.cos(0.5*z),
                ])
            density = bind(
                    places, sym.xyz_to_tangential(sym.make_sym_vector("jxyz", 3))
                    )(actx, jxyz=jxyz)

        elif relation == "sp":
            density_sym = sym.var("density")
            jump_identity_sym = (
                    0.5 * density_sym
                    + sym.Sp(knl, density_sym, qbx_forced_limit=+1)
                    - sym.Sp(knl, density_sym, qbx_forced_limit="avg")
                    )

            density = actx.np.cos(2*x) * actx.np.cos(2*y) * actx.np.cos(z)

        elif relation == "div_s":
            density_sym = sym.var("density")
            sigma_sym = sym.normal(places.ambient_dim).as_vector() * density_sym
            jump_identity_sym = (
                    sym.div(sym.S(knl, sigma_sym, qbx_forced_limit="avg"))
                    + sym.D(knl, density_sym, qbx_forced_limit="avg"))

            density = actx.np.cos(2*x) * actx.np.cos(2*y) * actx.np.cos(z)

        else:
            raise ValueError(f"unexpected value of 'relation': '{relation}'")

        bound_jump_identity = bind(places, jump_identity_sym)
        jump_identity = bound_jump_identity(actx, density=density)

        h_max = actx.to_numpy(
                bind(places, sym.h_max(places.ambient_dim))(actx)
                )
        err = actx.to_numpy(
                norm(density_discr, jump_identity, np.inf)
                / norm(density_discr, density, np.inf))
        eoc_rec.add_data_point(h_max, err)

        logging.info("error: nel %d h_max %.5e %.5e", nel_factor, h_max, err)

        # {{{ visualization

        if not visualize:
            continue

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, density_discr, target_order)
        normals = bind(
                places, sym.normal(places.ambient_dim).as_vector()
                )(actx)
        error = actx.np.log10(actx.np.abs(jump_identity) + 1.0e-15)

        if relation == "nxcurls":
            nxcurlS_ext = bind(
                places, nxcurlS(knl, density_sym, +1))(actx, density=density)
            nxcurlS_avg = bind(
                places, nxcurlS(knl, density_sym, "avg"))(actx, density=density)
            jtxyz = bind(
                    places, sym.tangential_to_xyz(density_sym)
                    )(actx, density=density)

            vis.write_vtk_file(f"source-nxcurls-{nel_factor:03d}.vtu", [
                ("jt", jtxyz),
                ("nxcurlS_ext", nxcurlS_ext),
                ("nxcurlS_avg", nxcurlS_avg),
                ("bdry_normals", normals),
                ("error", error),
                ])

        elif relation == "sp":
            op = sym.Sp(knl, density_sym, qbx_forced_limit=+1)
            sp_ext = bind(places, op)(actx, density=density)
            op = sym.Sp(knl, density_sym, qbx_forced_limit="avg")
            sp_avg = bind(places, op)(actx, density=density)

            vis.write_vtk_file(f"source-sp-{nel_factor:03d}.vtu", [
                ("density", density),
                ("sp_ext", sp_ext),
                ("sp_avg", sp_avg),
                ("bdry_normals", normals),
                ("error", error),
                ])

        elif relation == "div_s":
            vis.write_vtk_file(f"source-div-{nel_factor:03d}.vtu", [
                ("density", density),
                ("bdry_normals", normals),
                ("error", error),
                ])

        # }}}

    logger.info("\n%s", str(eoc_rec))
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
