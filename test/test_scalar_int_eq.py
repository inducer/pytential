__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
import numpy.linalg as la

from arraycontext import flatten
from meshmode.discretization.visualization import make_visualizer

from sumpy.kernel import LaplaceKernel, HelmholtzKernel, BiharmonicKernel

from pytential import bind, sym
from pytential import GeometryCollection
from pytools.obj_array import flat_obj_array

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import extra_int_eq_data as inteq
import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test backend

def run_int_eq_test(actx,
        case, resolution, visualize=False, check_spectrum=False):
    refiner_extra_kwargs = {}
    if case.use_refinement:
        if case.knl_class == HelmholtzKernel and \
                getattr(case, "refine_on_helmholtz_k", True):
            k = case.knl_concrete_kwargs["k"]
            refiner_extra_kwargs["kernel_length_scale"] = 5 / k

        if hasattr(case, "scaled_max_curvature_threshold"):
            refiner_extra_kwargs["scaled_max_curvature_threshold"] = \
                    case.scaled_max_curvature_threshold

        if hasattr(case, "expansion_disturbance_tolerance"):
            refiner_extra_kwargs["expansion_disturbance_tolerance"] = \
                    case.expansion_disturbance_tolerance

        if hasattr(case, "refinement_maxiter"):
            refiner_extra_kwargs["maxiter"] = case.refinement_maxiter

        # refiner_extra_kwargs["visualize"] = True

    # {{{ construct geometries

    qbx = case.get_layer_potential(actx, resolution, case.target_order)
    point_source, point_target = inteq.make_source_and_target_points(actx,
            case.side, case.inner_radius, case.outer_radius, qbx.ambient_dim)

    places = {
            case.name: qbx,
            "point_source": point_source,
            "point_target": point_target
            }

    # plotting grid points
    ambient_dim = qbx.ambient_dim
    if visualize:
        vis_grid_spacing = getattr(case, "vis_grid_spacing",
                (0.1, 0.1, 0.1)[:ambient_dim]
                )
        vis_extend_factor = getattr(case, "vis_extend_factor", 0.2)

        from sumpy.visualization import make_field_plotter_from_bbox
        from meshmode.mesh.processing import find_bounding_box
        fplot = make_field_plotter_from_bbox(
                find_bounding_box(qbx.density_discr.mesh),
                h=vis_grid_spacing,
                extend_factor=vis_extend_factor)

        from pytential.target import PointsTarget
        plot_targets = PointsTarget(fplot.points)

        places.update({
            "qbx_target_tol": qbx.copy(target_association_tolerance=0.15),
            "plot_targets": plot_targets
            })

    places = GeometryCollection(places, auto_where=case.name)
    if case.use_refinement:
        from pytential.qbx.refinement import refine_geometry_collection
        places = refine_geometry_collection(places, **refiner_extra_kwargs)

    dd = sym.as_dofdesc(case.name).to_stage1()
    density_discr = places.get_discretization(dd.geometry)

    logger.info("nelements:     %d", density_discr.mesh.nelements)
    logger.info("ndofs:         %d", density_discr.ndofs)

    if case.use_refinement:
        logger.info("%d elements before refinement",
                qbx.density_discr.mesh.nelements)

        discr = places.get_discretization(dd.geometry, sym.QBX_SOURCE_STAGE1)
        logger.info("%d stage-1 elements after refinement",
                discr.mesh.nelements)

        discr = places.get_discretization(dd.geometry, sym.QBX_SOURCE_STAGE2)
        logger.info("%d stage-2 elements after refinement",
                discr.mesh.nelements)

        discr = places.get_discretization(dd.geometry, sym.QBX_SOURCE_QUAD_STAGE2)
        logger.info("quad stage-2 elements have %d nodes",
                discr.groups[0].nunit_dofs)

    # }}}

    # {{{ plot geometry

    if visualize and ambient_dim == 2:
        try:
            import matplotlib.pyplot as pt
        except ImportError:
            visualize = False

    if visualize:
        normals = bind(places, sym.normal(ambient_dim).as_vector())(actx)

        # show geometry, centers, normals
        if ambient_dim == 2:
            nodes = actx.to_numpy(
                    flatten(density_discr.nodes(), actx)
                    ).reshape(ambient_dim, -1)
            normals = actx.to_numpy(
                    flatten(normals, actx)
                    ).reshape(ambient_dim, -1)

            pt.plot(nodes[0], nodes[1], "x-")
            pt.quiver(nodes[0], nodes[1], normals[0], normals[1])
            pt.gca().set_aspect("equal")
            pt.savefig(f"pre-solve-source-{resolution}", dpi=300)
        elif ambient_dim == 3:
            bdry_vis = make_visualizer(actx, density_discr, case.target_order + 3)
            bdry_vis.write_vtk_file(f"pre-solve-source-{resolution}.vtu", [
                ("normals", normals),
                ])
        else:
            raise ValueError("invalid mesh dim")

    # }}}

    # {{{ set up operator

    knl = case.knl_class(ambient_dim)
    op = case.get_operator(ambient_dim)
    if knl.is_complex_valued:
        dtype = np.complex128
    else:
        dtype = np.float64

    sym_u = op.get_density_var("u")
    sym_bc = op.get_density_var("bc")
    sym_charges = sym.var("charges")

    sym_op_u = op.operator(sym_u)

    # }}}

    # {{{ set up test data

    np.random.seed(22)
    source_charges = np.random.randn(point_source.ndofs)
    source_charges[-1] = -np.sum(source_charges[:-1])
    source_charges = source_charges.astype(dtype)
    assert np.sum(source_charges) < 1.0e-15

    source_charges_dev = actx.from_numpy(source_charges)

    # }}}

    # {{{ establish BCs

    pot_src = sym.int_g_vec(
        # FIXME: qbx_forced_limit--really?
        knl, sym_charges, qbx_forced_limit=None, **case.knl_sym_kwargs)

    test_direct = bind(places,
            pot_src,
            auto_where=("point_source", "point_target"))(
                    actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

    if case.bc_type == "dirichlet":
        bc = bind(places,
                pot_src,
                auto_where=("point_source", case.name))(
                        actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

    elif case.bc_type == "neumann":
        bc = bind(places,
                sym.normal_derivative(ambient_dim, pot_src, dofdesc=case.name),
                auto_where=("point_source", case.name))(
                        actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

    elif case.bc_type == "clamped_plate":
        bc_u = bind(places,
                pot_src,
                auto_where=("point_source", case.name))(
                        actx, charges=source_charges_dev, **case.knl_concrete_kwargs)
        bc_du = bind(places,
                sym.normal_derivative(ambient_dim, pot_src, dofdesc=case.name),
                auto_where=("point_source", case.name))(
                        actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

        bc = flat_obj_array(bc_u, bc_du)
    else:
        raise ValueError(f"unknown bc_type: '{case.bc_type}'")

    # }}}

    # {{{ solve

    bound_op = bind(places, sym_op_u)
    rhs = bind(places, op.prepare_rhs(sym_bc))(actx, bc=bc)

    from pytential.qbx import QBXTargetAssociationFailedException
    try:
        from pytential.solve import gmres
        gmres_result = gmres(
                bound_op.scipy_op(actx, "u", dtype, **case.knl_concrete_kwargs),
                rhs,
                tol=case.gmres_tol,
                progress=True,
                hard_failure=True,
                stall_iterations=50, no_progress_factor=1.05)
    except QBXTargetAssociationFailedException as e:
        bdry_vis = make_visualizer(actx, density_discr, case.target_order + 3)

        bdry_vis.write_vtk_file(f"failed-targets-solve-{resolution}.vtu", [
            ("failed_targets", actx.thaw(e.failed_target_flags)),
            ])
        raise

    logger.info("gmres state: %s", gmres_result.state)
    weighted_u = gmres_result.solution

    # }}}

    # {{{ build matrix for spectrum check

    if check_spectrum:
        from sumpy.tools import build_matrix
        mat = build_matrix(
                bound_op.scipy_op(
                    actx, arg_name="u", dtype=dtype, **case.knl_concrete_kwargs))
        w, v = la.eig(mat)

        if visualize:
            pt.imshow(np.log10(1.0e-20 + np.abs(mat)))
            pt.colorbar()
            pt.show()

    # }}}

    # {{{ error computation

    if case.side != "scat":
        test_via_bdry = bind(places,
                op.representation(sym_u),
                auto_where=(case.name, "point_target")
                )(actx, u=weighted_u, **case.knl_concrete_kwargs)

        err = test_via_bdry - test_direct

        err = actx.to_numpy(flatten(err, actx))
        test_direct = actx.to_numpy(flatten(test_direct, actx))
        test_via_bdry = actx.to_numpy(flatten(test_via_bdry, actx))

        # {{{ remove effect of net source charge

        if (case.knl_class == LaplaceKernel
                and case.bc_type == "neumann"
                and case.side == -1):
            # remove constant offset in interior Laplace Neumann error
            tgt_ones = np.ones_like(test_direct)
            tgt_ones = tgt_ones / la.norm(tgt_ones)
            err = err - np.vdot(tgt_ones, err) * tgt_ones

        # }}}

        rel_err_2 = la.norm(err) / la.norm(test_direct)
        rel_err_inf = la.norm(err, np.inf) / la.norm(test_direct, np.inf)

        logger.info("rel_err_2: %.5e rel_err_inf: %.5e", rel_err_2, rel_err_inf)
    else:
        rel_err_2 = None
        rel_err_inf = None

    # }}}

    # {{{ test gradient

    if case.check_gradient and case.side != "scat":
        sym_grad_op = op.representation(sym_u,
                map_potentials=lambda p: sym.grad(ambient_dim, p),
                qbx_forced_limit=None)

        grad_from_src = bind(places,
                sym_grad_op,
                auto_where=(case.name, "point_target"))(
                        actx, u=weighted_u, **case.knl_concrete_kwargs)
        grad_ref = bind(places,
                sym.grad(ambient_dim, pot_src),
                auto_where=("point_source", "point_target"))(
                        actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

        grad_err = grad_from_src - grad_ref
        grad_ref = actx.to_numpy(flatten(grad_ref[0], actx))
        grad_err = actx.to_numpy(flatten(grad_err[0], actx))

        rel_grad_err_inf = la.norm(grad_err, np.inf) / la.norm(grad_ref, np.inf)
        logger.info("rel_grad_err_inf: %.5e", rel_grad_err_inf)
    else:
        rel_grad_err_inf = None

    # }}}

    # {{{ test tangential derivative

    if case.check_tangential_deriv and case.side != "scat":
        sym_tang_deriv_op = op.representation(sym_u,
                map_potentials=lambda p: sym.tangential_derivative(ambient_dim, p),
                qbx_forced_limit=case.side).as_scalar()

        tang_deriv_from_src = bind(places,
                sym_tang_deriv_op,
                auto_where=case.name)(
                        actx, u=weighted_u, **case.knl_concrete_kwargs)
        tang_deriv_ref = bind(places,
                sym.tangential_derivative(ambient_dim, pot_src).as_scalar(),
                auto_where=("point_source", case.name))(
                        actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

        tang_deriv_from_src = actx.to_numpy(flatten(tang_deriv_from_src, actx))
        tang_deriv_ref = actx.to_numpy(flatten(tang_deriv_ref, actx))
        td_err = tang_deriv_from_src - tang_deriv_ref

        if visualize:
            pt.plot(tang_deriv_ref.real, label="ref")
            pt.plot(tang_deriv_from_src.real, label="src")
            pt.legend()
            pt.savefig(f"tangential-derivative-{resolution}", dpi=300)

        rel_td_err_inf = la.norm(td_err, np.inf) / la.norm(tang_deriv_ref, np.inf)
        logger.info("rel_td_err_inf: %.5e" % rel_td_err_inf)
    else:
        rel_td_err_inf = None

    # }}}

    # {{{ any-D file plotting

    if visualize:
        sym_sqrt_j = sym.sqrt_jac_q_weight(ambient_dim)
        u = bind(places, sym_u / sym_sqrt_j)(actx, u=weighted_u)

        bdry_vis = make_visualizer(actx, density_discr, case.target_order + 3)
        bdry_vis.write_vtk_file(f"integral-equation-source-{resolution}.vtu", [
            ("u", u), ("bc", bc),
            ])

        try:
            solved_pot = bind(places,
                    op.representation(sym_u),
                    auto_where=("qbx_target_tol", "plot_targets"))(
                            actx, u=weighted_u, **case.knl_concrete_kwargs)
        except QBXTargetAssociationFailedException as e:
            fplot.write_vtk_file(f"failed-targets-plotter-{resolution}.vts", [
                ("failed_targets", actx.thaw(e.failed_target_flags))
                ])
            raise

        ones_density = density_discr.zeros(actx) + 1

        sym_indicator = -sym.D(LaplaceKernel(ambient_dim),
                op.get_density_var("sigma"),
                qbx_forced_limit=None)
        indicator = bind(places,
                sym_indicator,
                auto_where=("qbx_target_tol", "plot_targets"))(
                        actx, sigma=ones_density)

        true_pot = bind(places,
                pot_src,
                auto_where=("point_source", "plot_targets"))(
                        actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

        solved_pot = actx.to_numpy(solved_pot)
        true_pot = actx.to_numpy(true_pot)
        indicator = actx.to_numpy(indicator)

        if case.side == "scat":
            fplot.write_vtk_file(f"potential-{resolution}.vts", [
                ("pot_scattered", solved_pot),
                ("pot_incoming", -true_pot),
                ("indicator", indicator),
                ])
        else:
            fplot.write_vtk_file(f"potential-{resolution}.vts", [
                ("solved_pot", solved_pot),
                ("true_pot", true_pot),
                ("indicator", indicator),
                ])

    # }}}

    h_max = bind(places, sym.h_max(ambient_dim))(actx)
    return dict(
            h_max=actx.to_numpy(h_max),
            rel_err_2=rel_err_2,
            rel_err_inf=rel_err_inf,
            rel_td_err_inf=rel_td_err_inf,
            rel_grad_err_inf=rel_grad_err_inf,
            gmres_result=gmres_result)

# }}}


# {{{ test frontend

cases = [
        inteq.EllipseTestCase(
            knl_class_or_helmholtz_k=helmholtz_k,
            bc_type=bc_type,
            side=side)
        for helmholtz_k in [0, 1.2]
        for bc_type in ["dirichlet", "neumann"]
        for side in [-1, +1]
        ]


cases += [
        inteq.EllipseTestCase(
            knl_class_or_helmholtz_k=BiharmonicKernel,
            bc_type="clamped_plate", side=-1, fmm_backend=None),
        inteq.EllipseTestCase(
            knl_class_or_helmholtz_k=BiharmonicKernel,
            bc_type="clamped_plate", side=-1, fmm_backend="sumpy", fmm_order=15,
            gmres_tol=1e-9),
        inteq.EllipseTestCase(
            knl_class_or_helmholtz_k=BiharmonicKernel,
            bc_type="clamped_plate", side=-1, fmm_backend="sumpy", fmm_order=15,
            disable_fft=True),
        ]


# Sample test run:
# 'test_integral_equation(cl._csc, EllipseIntEqTestCase(LaplaceKernel, "dirichlet", +1), visualize=True)'  # noqa: E501

@pytest.mark.parametrize("case", cases)
def test_integral_equation(actx_factory, case, visualize=False):
    logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from pytools.convergence import EOCRecorder
    logger.info("\n%s", str(case))

    eoc_rec_target = EOCRecorder()
    eoc_rec_td = EOCRecorder()

    have_error_data = False
    for resolution in case.resolutions:
        result = run_int_eq_test(actx, case, resolution, visualize=visualize)

        if result["rel_err_2"] is not None:
            have_error_data = True
            eoc_rec_target.add_data_point(result["h_max"], result["rel_err_2"])

        if result["rel_td_err_inf"] is not None:
            eoc_rec_td.add_data_point(result["h_max"], result["rel_td_err_inf"])

    if case.bc_type == "dirichlet":
        tgt_order = case.qbx_order
    elif case.bc_type == "neumann":
        tgt_order = case.qbx_order - 1
    elif case.bc_type == "clamped_plate":
        tgt_order = case.qbx_order
    else:
        raise ValueError(f"unknown bc_type: '{case.bc_type}'")

    if have_error_data:
        logger.info("TARGET ERROR:")
        logger.info("\n%s", eoc_rec_target)
        assert eoc_rec_target.order_estimate() > tgt_order - 1.3

        if case.check_tangential_deriv:
            logger.info("TANGENTIAL DERIVATIVE ERROR:")
            logger.info("\n%s", eoc_rec_td)
            assert eoc_rec_td.order_estimate() > tgt_order - 2.3

# }}}


# You can test individual routines by typing
# $ python test_scalar_int_eq.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
