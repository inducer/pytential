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

import pytest

import numpy as np

from pytential import GeometryCollection, bind, sym
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
from pytools.obj_array import make_obj_array

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    "pyopencl-deprecated",
    ])


# {{{ test_exterior_stokes

def run_exterior_stokes(actx_factory, *,
        ambient_dim, target_order, qbx_order, resolution,
        fmm_order=None,    # FIXME: FMM is slower than direct evaluation
        source_ovsmp=None,
        radius=1.5,
        mu=1.0,
        nu=0.4,
        visualize=False,
        method="naive",

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
        from meshmode.mesh.generation import generate_icosphere
        mesh = generate_icosphere(radius, target_order + 1,
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

    if nu == 0.5:
        sym_nu = 0.5
    else:
        sym_nu = sym.var("nu")

    if ambient_dim == 2:
        from pytential.symbolic.stokes import HsiaoKressExteriorStokesOperator
        sym_omega = sym.make_sym_vector("omega", ambient_dim)
        op = HsiaoKressExteriorStokesOperator(omega=sym_omega, method=method,
                mu_sym=sym_mu, nu_sym=sym_nu)
    elif ambient_dim == 3:
        from pytential.symbolic.stokes import HebekerExteriorStokesOperator
        op = HebekerExteriorStokesOperator(method=method,
                mu_sym=sym_mu, nu_sym=sym_nu)
    else:
        raise AssertionError()

    sym_sigma = op.get_density_var("sigma")
    sym_bc = op.get_density_var("bc")

    sym_op = op.operator(sym_sigma, normal=sym_normal)
    sym_rhs = op.prepare_rhs(sym_bc)

    sym_velocity = op.velocity(sym_sigma, normal=sym_normal)

    if ambient_dim == 3:
        sym_source_pot = op.stokeslet.apply(sym_sigma, qbx_forced_limit=None)
    else:
        # Use the naive method here as biharmonic requires source derivatives
        # of point_source
        from pytential.symbolic.stokes import StokesletWrapper
        sym_source_pot = StokesletWrapper(ambient_dim, mu_sym=sym_mu,
            nu_sym=sym_nu, method="naive").apply(sym_sigma, qbx_forced_limit=None)

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
    direct_context = {"mu": mu}

    if sym_nu != 0.5:
        bc_context["nu"] = nu
        op_context["nu"] = nu
        direct_context["nu"] = nu

    bc_op = bind(places, sym_source_pot,
            auto_where=("point_source", "source"))
    bc = bc_op(actx, sigma=charges, **direct_context)

    rhs = bind(places, sym_rhs)(actx, bc=bc, **bc_context)
    bound_op = bind(places, sym_op)

    # }}}

    fmm_timing_data = {}
    bound_op.eval({"sigma": rhs, **op_context}, array_context=actx,
            timing_data=fmm_timing_data)

    def print_timing_data(timings, name):
        result = {k: 0 for k in list(timings.values())[0].keys()}
        total = 0
        for k, timing in timings.items():
            for k, v in timing.items():
                result[k] += v["wall_elapsed"]
                total += v["wall_elapsed"]
        result["total"] = total
        print(f"{name}={result}")

    # print_timing_data(fmm_timing_data, method)

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

    def rnorm2(x, y):
        y_norm = actx.np.linalg.norm(y.dot(y), ord=2)
        if y_norm < 1.0e-14:
            y_norm = 1.0

        d = x - y
        return actx.to_numpy(actx.np.linalg.norm(d.dot(d), ord=2) / y_norm)

    ps_velocity = bind(places, sym_velocity,
            auto_where=("source", "point_target"))(actx, sigma=sigma, **op_context)
    ex_velocity = bind(places, sym_source_pot,
            auto_where=("point_source", "point_target"))(actx, sigma=charges,
                    **direct_context)

    v_error = rnorm2(ps_velocity, ex_velocity)
    h_max = actx.to_numpy(
            bind(places, sym.h_max(ambient_dim))(actx)
            )

    logger.info("resolution %4d h_max %.5e error %.5e",
            resolution, h_max, v_error)

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


@pytest.mark.parametrize("ambient_dim, method, nu", [
    (2, "naive", 0.5),
    (2, "biharmonic", 0.5),
    pytest.param(3, "naive", 0.5, marks=pytest.mark.slowtest),
    (3, "biharmonic", 0.5),
    (3, "laplace", 0.5),

    (2, "biharmonic", 0.4),
    (3, "biharmonic", 0.4),
    (3, "laplace", 0.4),
    ])
def test_exterior_stokes(actx_factory, ambient_dim, method, nu, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    target_order = 3
    qbx_order = 3

    if ambient_dim == 2:
        fmm_order = 10
        resolutions = [20, 35, 50]
    elif ambient_dim == 3:
        fmm_order = 6
        resolutions = [0, 1, 2]
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    for resolution in resolutions:
        h_max, err = run_exterior_stokes(actx_factory,
                ambient_dim=ambient_dim,
                target_order=target_order,
                fmm_order=fmm_order,
                qbx_order=qbx_order,
                resolution=resolution,
                visualize=visualize,
                nu=nu,
                method=method)

        eoc.add_data_point(h_max, err)
        print(eoc)

    # This convergence data is not as clean as it could be. See
    # https://github.com/inducer/pytential/pull/32
    # for some discussion.
    assert eoc.order_estimate() > target_order - 0.5

# }}}


# You can test individual routines by typing
# $ python test_stokes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import pyopencl as cl
        context = cl._csc()
        queue = cl.CommandQueue(context)
        actx_factory = lambda : PyOpenCLArrayContext(queue)
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
