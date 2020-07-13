"""Calibrates a cost model and reports on the accuracy."""

import pyopencl as cl
import numpy as np
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw

from pytential import sym, bind
from pytools import one


# {{{ global params

TARGET_ORDER = 8
OVSMP_FACTOR = 5
TCF = 0.9
QBX_ORDER = 5
FMM_ORDER = 10
RUNS = 3

DEFAULT_LPOT_KWARGS = {
        "_box_extent_norm": "l2",
        "_from_sep_smaller_crit": "static_l2",
        }

PANELS_PER_ARM = 30
TRAINING_ARMS = (10, 15, 25)
TESTING_ARMS = (20,)


def starfish_lpot_source(actx, n_arms):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    from meshmode.mesh.generation import make_curve_mesh, NArmedStarfish

    mesh = make_curve_mesh(
            NArmedStarfish(n_arms, 0.8),
            np.linspace(0, 1, 1 + PANELS_PER_ARM * n_arms),
            TARGET_ORDER)

    pre_density_discr = Discretization(
            actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(TARGET_ORDER))

    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()
    lpot_kwargs.update(
            target_association_tolerance=0.025,
            _expansion_stick_out_factor=TCF,
            fmm_order=FMM_ORDER, qbx_order=QBX_ORDER,
            fmm_backend="fmmlib"
            )

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR * TARGET_ORDER,
            **lpot_kwargs)

    return lpot_source

# }}}


def training_geometries(actx):
    for n_arms in TRAINING_ARMS:
        yield starfish_lpot_source(actx, n_arms)


def test_geometries(actx):
    for n_arms in TESTING_ARMS:
        yield starfish_lpot_source(actx, n_arms)


def get_bound_op(places):
    from sumpy.kernel import LaplaceKernel
    op = sym.S(LaplaceKernel(places.ambient_dim),
            sym.var("sigma"),
            qbx_forced_limit=+1)

    return bind(places, op)


def get_test_density(actx, density_discr):
    nodes = thaw(actx, density_discr.nodes())
    sigma = actx.np.sin(10 * nodes[0])
    return sigma


def calibrate_cost_model(ctx):
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    from pytential.qbx.cost import CostModel, estimate_calibration_params
    cost_model = CostModel()

    model_results = []
    timing_results = []

    for lpot_source in training_geometries(actx):
        lpot_source = lpot_source.copy(cost_model=cost_model)

        from pytential import GeometryCollection
        places = GeometryCollection(lpot_source)
        density_discr = places.get_discretization(places.auto_source.geometry)

        bound_op = get_bound_op(places)
        sigma = get_test_density(actx, density_discr)

        cost_S = bound_op.get_modeled_cost(actx, sigma=sigma)

        # Warm-up run.
        bound_op.eval({"sigma": sigma}, array_context=actx)

        for _ in range(RUNS):
            timing_data = {}
            bound_op.eval({"sigma": sigma}, array_context=actx,
                    timing_data=timing_data)

            model_results.append(one(cost_S.values()))
            timing_results.append(one(timing_data.values()))

    calibration_params = (
            estimate_calibration_params(model_results, timing_results))

    return cost_model.with_calibration_params(calibration_params)


def test_cost_model(ctx, cost_model):
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    for lpot_source in test_geometries(actx):
        lpot_source = lpot_source.copy(cost_model=cost_model)

        from pytential import GeometryCollection
        places = GeometryCollection(lpot_source)
        density_discr = places.get_discretization(places.auto_source.geometry)

        bound_op = get_bound_op(places)
        sigma = get_test_density(actx, density_discr)

        cost_S = bound_op.get_modeled_cost(actx, sigma=sigma)
        model_result = (
                one(cost_S.values())
                .get_predicted_times(merge_close_lists=True))

        # Warm-up run.
        bound_op.eval({"sigma": sigma}, array_context=actx)

        temp_timing_results = []
        for _ in range(RUNS):
            timing_data = {}
            bound_op.eval({"sigma": sigma},
                    array_context=actx, timing_data=timing_data)
            temp_timing_results.append(one(timing_data.values()))

        timing_result = {}
        for param in model_result:
            timing_result[param] = (
                    sum(temp_timing_result[param]["process_elapsed"]
                        for temp_timing_result in temp_timing_results)) / RUNS

        from pytools import Table
        table = Table()
        table.add_row(["stage", "actual (s)", "predicted (s)"])
        for stage in model_result:
            row = [
                    stage,
                    "%.2f" % timing_result[stage],
                    "%.2f" % model_result[stage]]
            table.add_row(row)

        print(table)


def predict_cost(ctx):
    model = calibrate_cost_model(ctx)
    test_cost_model(ctx, model)


if __name__ == "__main__":
    predict_cost(cl.create_some_context(0))
