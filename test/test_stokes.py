__copyright__ = "Copyright (C) 2017 Natalie Beams"

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

from functools import partial
import pytest

import numpy as np

from arraycontext import flatten
from pytential import GeometryCollection, bind, sym
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
from pytools.obj_array import make_obj_array

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import extra_int_eq_data as eid
import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


def discr_rel_error(actx, discr, x, xref, p=None):
    from pytential import norm
    ref_norm = actx.to_numpy(norm(discr, xref, p=p))
    if ref_norm < 1.0e-14:
        ref_norm = 1

    diff_norm = actx.to_numpy(norm(discr, x - xref, p=p))
    return diff_norm / ref_norm


def dof_array_rel_error(actx, x, xref, p=None):
    ref_norm = actx.to_numpy(actx.np.linalg.norm(xref, ord=p))
    if ref_norm < 1.0e-14:
        ref_norm = 1

    diff_norm = actx.to_numpy(actx.np.linalg.norm(x - xref, ord=p))
    return diff_norm / ref_norm


# {{{ test_exterior_stokes

def run_exterior_stokes(actx_factory, *,
        ambient_dim, target_order, qbx_order, resolution,
        fmm_order=False,    # FIXME: FMM is slower than direct evaluation
        source_ovsmp=None,
        radius=1.5,
        mu=1.0,
        visualize=False,

        _target_association_tolerance=0.05,
        _expansions_in_tree_have_extent=True):
    actx = actx_factory()

    # {{{ geometry

    if source_ovsmp is None:
        source_ovsmp = 4 if ambient_dim == 2 else 8

    places = {}

    if ambient_dim == 2:
        from meshmode.mesh.generation import make_curve_mesh, ellipse
        mesh = make_curve_mesh(
                lambda t: radius * ellipse(1.0, t),
                np.linspace(0.0, 1.0, resolution + 1),
                target_order)
    elif ambient_dim == 3:
        from meshmode.mesh.generation import generate_sphere
        mesh = generate_sphere(radius, target_order + 1,
                uniform_refinement_rounds=resolution)
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    pre_density_discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(pre_density_discr,
            fine_order=source_ovsmp * target_order,
            qbx_order=qbx_order,
            fmm_order=fmm_order,
            target_association_tolerance=_target_association_tolerance,
            _expansions_in_tree_have_extent=_expansions_in_tree_have_extent)
    places["source"] = qbx

    from extra_int_eq_data import make_source_and_target_points
    point_source, point_target = make_source_and_target_points(
            actx,
            side=+1,
            inner_radius=0.5 * radius,
            outer_radius=2.0 * radius,
            ambient_dim=ambient_dim,
            )
    places["point_source"] = point_source
    places["point_target"] = point_target

    if visualize:
        from sumpy.visualization import make_field_plotter_from_bbox
        from meshmode.mesh.processing import find_bounding_box
        fplot = make_field_plotter_from_bbox(
                find_bounding_box(mesh),
                h=0.1, extend_factor=1.0)
        mask = np.linalg.norm(fplot.points, ord=2, axis=0) > (radius + 0.25)

        from pytential.target import PointsTarget
        plot_target = PointsTarget(fplot.points[:, mask].copy())
        places["plot_target"] = plot_target

        del mask

    places = GeometryCollection(places, auto_where="source")

    density_discr = places.get_discretization("source")
    logger.info("ndofs:     %d", density_discr.ndofs)
    logger.info("nelements: %d", density_discr.mesh.nelements)

    # }}}

    # {{{ symbolic

    sym_normal = sym.make_sym_vector("normal", ambient_dim)
    sym_mu = sym.var("mu")

    if ambient_dim == 2:
        from pytential.symbolic.stokes import HsiaoKressExteriorStokesOperator
        sym_omega = sym.make_sym_vector("omega", ambient_dim)
        op = HsiaoKressExteriorStokesOperator(omega=sym_omega)
    elif ambient_dim == 3:
        from pytential.symbolic.stokes import HebekerExteriorStokesOperator
        op = HebekerExteriorStokesOperator()
    else:
        raise AssertionError()

    sym_sigma = op.get_density_var("sigma")
    sym_bc = op.get_density_var("bc")

    sym_op = op.operator(sym_sigma, normal=sym_normal, mu=sym_mu)
    sym_rhs = op.prepare_rhs(sym_bc, mu=mu)

    sym_velocity = op.velocity(sym_sigma, normal=sym_normal, mu=sym_mu)

    sym_source_pot = op.stokeslet.apply(sym_sigma, sym_mu, qbx_forced_limit=None)

    # }}}

    # {{{ boundary conditions

    normal = bind(places, sym.normal(ambient_dim).as_vector())(actx)

    np.random.seed(42)
    charges = make_obj_array([
        actx.from_numpy(np.random.randn(point_source.ndofs))
        for _ in range(ambient_dim)
        ])

    if ambient_dim == 2:
        total_charge = make_obj_array([
            actx.to_numpy(actx.np.sum(c)) for c in charges
            ])
        omega = bind(places, total_charge * sym.Ones())(actx)

    if ambient_dim == 2:
        bc_context = {"mu": mu, "omega": omega}
        op_context = {"mu": mu, "omega": omega, "normal": normal}
    else:
        bc_context = {}
        op_context = {"mu": mu, "normal": normal}

    bc = bind(places, sym_source_pot,
            auto_where=("point_source", "source"))(actx, sigma=charges, mu=mu)

    rhs = bind(places, sym_rhs)(actx, bc=bc, **bc_context)
    bound_op = bind(places, sym_op)

    # }}}

    # {{{ solve

    from pytential.solve import gmres
    gmres_tol = 1.0e-9
    result = gmres(
            bound_op.scipy_op(actx, "sigma", np.float64, **op_context),
            rhs,
            x0=rhs,
            tol=gmres_tol,
            progress=visualize,
            stall_iterations=0,
            hard_failure=True)

    sigma = result.solution

    # }}}

    # {{{ check velocity at "point_target"

    velocity = bind(places, sym_velocity,
            auto_where=("source", "point_target"))(actx, sigma=sigma, **op_context)
    ref_velocity = bind(places, sym_source_pot,
            auto_where=("point_source", "point_target"))(actx, sigma=charges, mu=mu)

    v_error = [
            dof_array_rel_error(actx, u, uref)
            for u, uref in zip(velocity, ref_velocity)]
    h_max = actx.to_numpy(
            bind(places, sym.h_max(ambient_dim))(actx)
            )

    logger.info("resolution %4d h_max %.5e error " + ("%.5e " * ambient_dim),
            resolution, h_max, *v_error)

    # }}}}

    # {{{ visualize

    if not visualize:
        return h_max, v_error

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, density_discr, target_order)

    filename = "stokes_solution_{}d_{}_ovsmp_{}.vtu".format(
            ambient_dim, resolution, source_ovsmp)

    vis.write_vtk_file(filename, [
        ("density", sigma),
        ("bc", bc),
        ("rhs", rhs),
        ], overwrite=True)

    # }}}

    return h_max, v_error


@pytest.mark.parametrize("ambient_dim", [
    2,
    pytest.param(3, marks=pytest.mark.slowtest)
    ])
def test_exterior_stokes(actx_factory, ambient_dim, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)

    from pytools.convergence import EOCRecorder
    eocs = [EOCRecorder() for _ in range(ambient_dim)]

    target_order = 5
    source_ovsmp = 4
    qbx_order = 3

    if ambient_dim == 2:
        resolutions = [20, 35, 50]
    elif ambient_dim == 3:
        resolutions = [0, 1, 2]
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    for resolution in resolutions:
        h_max, errors = run_exterior_stokes(actx_factory,
                ambient_dim=ambient_dim,
                target_order=target_order,
                qbx_order=qbx_order,
                source_ovsmp=source_ovsmp,
                resolution=resolution,
                visualize=visualize)

        for eoc, e in zip(eocs, errors):
            eoc.add_data_point(h_max, e)

    for eoc in eocs:
        print(eoc.pretty_print(
            abscissa_format="%.8e",
            error_format="%.8e",
            eoc_format="%.2f"))

    for eoc in eocs:
        # This convergence data is not as clean as it could be. See
        # https://github.com/inducer/pytential/pull/32
        # for some discussion.
        order = min(target_order, qbx_order)
        assert eoc.order_estimate() > order - 0.5

# }}}


# {{{ test Stokeslet identity

def run_stokes_identity(actx_factory, case, identity, resolution, visualize=False):
    actx = actx_factory()

    qbx = case.get_layer_potential(actx, resolution, case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    density_discr = places.get_discretization(case.name)
    logger.info("ndofs:     %d", density_discr.ndofs)
    logger.info("nelements: %d", density_discr.mesh.nelements)

    # {{{ evaluate

    result = bind(places, identity.apply_operator())(actx)
    ref_result = bind(places, identity.ref_result())(actx)

    h_min = actx.to_numpy(
            bind(places, sym.h_min(places.ambient_dim))(actx)
            )
    h_max = actx.to_numpy(
            bind(places, sym.h_max(places.ambient_dim))(actx)
            )
    error = [
            discr_rel_error(actx, density_discr, x, xref, p=np.inf)
            for x, xref in zip(result, ref_result)]
    logger.info("resolution %4d h_min %.5e h_max %.5e error "
            + ("%.5e " * places.ambient_dim),
            resolution, h_min, h_max, *error)

    # }}}

    if visualize:
        filename = "stokes_{}_{}_resolution_{}".format(
                type(identity).__name__.lower(), case.name, resolution)

        if places.ambient_dim == 2:
            result = actx.to_numpy(flatten(result, actx))

            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.gca()

            ax.plot(result[0], "o-")
            ax.plot(ref_result[0], "k--")
            ax.grid()
            fig.savefig(f"{filename}_x")
            fig.clf()

            ax = fig.gca()
            ax.plot(result[1], "o-")
            ax.plot(ref_result[1], "k--")
            ax.grid()
            fig.savefig(f"{filename}_y")
            plt.close(fig)
        else:
            from meshmode.discretization.visualization import make_visualizer
            vis = make_visualizer(actx, density_discr,
                    vis_order=case.target_order,
                    force_equidistant=True)

            ref_error = actx.np.abs(result - ref_result) + 1.0e-16

            from pytential.symbolic.primitives import _scaled_max_curvature
            scaled_kappa = bind(places,
                    _scaled_max_curvature(places.ambient_dim),
                    auto_where=case.name)(actx)
            kappa = bind(places,
                    sym.mean_curvature(places.ambient_dim),
                    auto_where=case.name)(actx)

            vis.write_vtk_file(f"{filename}.vtu", [
                ("result", result),
                ("ref", ref_result),
                ("error", ref_error),
                ("log_error", actx.np.log10(ref_error)),
                ("kappa", kappa - 1.0),
                ("scaled_kappa", scaled_kappa),
                ], use_high_order=True, overwrite=True)

    return h_max, error


class StokesletIdentity:
    """[Pozrikidis1992] Problem 3.1.1"""

    def __init__(self, ambient_dim):
        from pytential.symbolic.stokes import StokesletWrapper
        self.ambient_dim = ambient_dim
        self.stokeslet = StokesletWrapper(self.ambient_dim)

    def apply_operator(self):
        sym_density = sym.normal(self.ambient_dim).as_vector()
        return self.stokeslet.apply(
                sym_density,
                mu_sym=1, qbx_forced_limit=+1)

    def ref_result(self):
        return make_obj_array([1.0e-15 * sym.Ones()] * self.ambient_dim)


@pytest.mark.parametrize("cls", [
    partial(eid.StarfishTestCase, resolutions=[16, 32, 64, 96, 128]),
    partial(eid.SpheroidTestCase, resolutions=[0, 1, 2]),
    ])
def test_stokeslet_identity(actx_factory, cls, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)

    source_ovsmp = 4 if cls.func.ambient_dim == 2 else 8
    case = cls(fmm_backend=None,
            target_order=5, qbx_order=3, source_ovsmp=source_ovsmp)
    identity = StokesletIdentity(case.ambient_dim)
    logger.info("\n%s", str(case))

    from pytools.convergence import EOCRecorder
    eocs = [EOCRecorder() for _ in range(case.ambient_dim)]

    for resolution in case.resolutions:
        h_max, errors = run_stokes_identity(
                actx_factory, case, identity,
                resolution=resolution,
                visualize=visualize)

        for eoc, e in zip(eocs, errors):
            eoc.add_data_point(h_max, e)

    for eoc in eocs:
        print(eoc.pretty_print(
            abscissa_format="%.8e",
            error_format="%.8e",
            eoc_format="%.2f"))

    for eoc in eocs:
        order = min(case.target_order, case.qbx_order)
        assert eoc.order_estimate() > order - 0.5

# }}}


# {{{ test Stresslet identity

class StressletIdentity:
    """[Pozrikidis1992] Equation 3.2.7"""

    def __init__(self, ambient_dim):
        from pytential.symbolic.stokes import StokesletWrapper
        self.ambient_dim = ambient_dim
        self.stokeslet = StokesletWrapper(self.ambient_dim)

    def apply_operator(self):
        sym_density = sym.normal(self.ambient_dim).as_vector()
        return self.stokeslet.apply_stress(
                sym_density, sym_density,
                mu_sym=1, qbx_forced_limit="avg")

    def ref_result(self):
        return -0.5 * sym.normal(self.ambient_dim).as_vector()


@pytest.mark.parametrize("cls", [
    partial(eid.StarfishTestCase, resolutions=[16, 32, 64, 96, 128]),
    partial(eid.SpheroidTestCase, resolutions=[0, 1, 2]),
    ])
def test_stresslet_identity(actx_factory, cls, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)

    source_ovsmp = 4 if cls.func.ambient_dim == 2 else 8
    case = cls(fmm_backend=None,
            target_order=5, qbx_order=3, source_ovsmp=source_ovsmp)
    identity = StressletIdentity(case.ambient_dim)
    logger.info("\n%s", str(case))

    from pytools.convergence import EOCRecorder
    eocs = [EOCRecorder() for _ in range(case.ambient_dim)]

    for resolution in case.resolutions:
        h_max, errors = run_stokes_identity(
                actx_factory, case, identity,
                resolution=resolution,
                visualize=visualize)

        for eoc, e in zip(eocs, errors):
            eoc.add_data_point(h_max, e)

    for eoc in eocs:
        print(eoc.pretty_print(
            abscissa_format="%.8e",
            error_format="%.8e",
            eoc_format="%.2f"))

    for eoc in eocs:
        order = min(case.target_order, case.qbx_order)
        assert eoc.order_estimate() > order - 1.0

# }}}


# You can test individual routines by typing
# $ python test_stokes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
