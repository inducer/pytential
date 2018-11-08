"""Calibrates a cost model and reports on the accuracy."""

import pyopencl as cl
import numpy as np

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


def starfish_lpot_source(queue, n_arms):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    from meshmode.mesh.generation import make_curve_mesh, NArmedStarfish

    mesh = make_curve_mesh(
            NArmedStarfish(n_arms, 0.8),
            np.linspace(0, 1, 1 + PANELS_PER_ARM * n_arms),
            TARGET_ORDER)

    pre_density_discr = Discretization(
            queue.context, mesh,
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

    lpot_source, _ = lpot_source.with_refinement()

    return lpot_source

# }}}


def training_geometries(queue):
    for n_arms in TRAINING_ARMS:
        yield starfish_lpot_source(queue, n_arms)


def test_geometries(queue):
    for n_arms in TESTING_ARMS:
        yield starfish_lpot_source(queue, n_arms)


def get_bound_op(lpot_source):
    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    return bind(lpot_source, op)


def get_test_density(queue, lpot_source):
    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    return sigma


def calibrate_cost_model(ctx):
    queue = cl.CommandQueue(ctx)

    from pytential.qbx.cost import CostModel, estimate_calibration_params

    perf_model = CostModel()

    model_results = []
    timing_results = []

    for lpot_source in training_geometries(queue):
        lpot_source = lpot_source.copy(cost_model=perf_model)
        bound_op = get_bound_op(lpot_source)
        sigma = get_test_density(queue, lpot_source)

        perf_S = bound_op.get_modeled_cost(queue, sigma=sigma)

        # Warm-up run.
        bound_op.eval(queue, {"sigma": sigma})

        for _ in range(RUNS):
            timing_data = {}
            bound_op.eval(queue, {"sigma": sigma}, timing_data=timing_data)

            model_results.append(one(perf_S.values()))
            timing_results.append(one(timing_data.values()))

    calibration_params = (
            estimate_calibration_params(model_results, timing_results))

    return perf_model.with_calibration_params(calibration_params)


def test_cost_model(ctx, perf_model):
    queue = cl.CommandQueue(ctx)

    for lpot_source in test_geometries(queue):
        lpot_source = lpot_source.copy(cost_model=perf_model)
        bound_op = get_bound_op(lpot_source)
        sigma = get_test_density(queue, lpot_source)

        perf_S = bound_op.get_modeled_cost(queue, sigma=sigma)
        model_result = (
                one(perf_S.values())
                .get_predicted_times(merge_close_lists=True))

        # Warm-up run.
        bound_op.eval(queue, {"sigma": sigma})

        temp_timing_results = []
        for _ in range(RUNS):
            timing_data = {}
            bound_op.eval(queue, {"sigma": sigma}, timing_data=timing_data)
            temp_timing_results.append(one(timing_data.values()))

        timing_result = {}
        for param in model_result:
            timing_result[param] = (
                    sum(temp_timing_result[param]["process_elapsed"]
                        for temp_timing_result in temp_timing_results)) / RUNS

        print("=" * 20)
        for stage in model_result:
            print("stage: ", stage)
            print("actual: ", timing_result[stage])
            print("predicted: ", model_result[stage])
        print("=" * 20)


def predict_cost(ctx):
    model = calibrate_cost_model(ctx)
    test_cost_model(ctx, model)


if __name__ == "__main__":
    predict_cost(cl.create_some_context(0))
