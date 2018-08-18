"""Trains a performance model and reports on the accuracy."""

import pyopencl as cl
import numpy as np  # noqa

from pytential import sym, bind
from pytools import one


# {{{ global params

TARGET_ORDER = 8
OVSMP_FACTOR = 5
TCF = 0.9
QBX_ORDER = 5
FMM_ORDER = 10
MESH_TOL = 1e-10
FORCE_STAGE2_UNIFORM_REFINEMENT_ROUNDS = 1
SCALED_MAX_CURVATURE_THRESHOLD = 0.8
MAX_LEAF_REFINE_WEIGHT = 512
RUNS = 3

DEFAULT_LPOT_KWARGS = {
        "_box_extent_norm": "l2",
        "_from_sep_smaller_crit": "static_l2",
        }

TRAINING_ARMS = (2, 3, 6)
TESTING_ARMS = (5,)


def urchin_lpot_source(queue, sph_harm_tuple,
        from_sep_smaller_threshold=None, use_tsqbx=True):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    sph_m, sph_n = sph_harm_tuple

    from meshmode.mesh.generation import generate_urchin as get_urchin
    mesh = get_urchin(
            order=target_order, m=sph_m, n=sph_n,
            est_rel_interp_tolerance=MESH_TOL)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    refiner_extra_kwargs = {
            #"visualize": True,
            "_force_stage2_uniform_refinement_rounds": (
                 FORCE_STAGE2_UNIFORM_REFINEMENT_ROUNDS),
            "_scaled_max_curvature_threshold": (
                 SCALED_MAX_CURVATURE_THRESHOLD),
            }

    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()
    lpot_kwargs.update(
            fmm_backend="fmmlib",
            _well_sep_is_n_away=2,
            _expansions_in_tree_have_extent=True,
            _expansion_stick_out_factor=TCF,
            _max_leaf_refine_weight=MAX_LEAF_REFINE_WEIGHT,
            target_association_tolerance=1e-3,
            fmm_order=FMM_ORDER, qbx_order=QBX_ORDER,
            _from_sep_smaller_min_nsources_cumul=from_sep_smaller_threshold,
            _use_tsqbx=use_tsqbx,
            )

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            **lpot_kwargs,)

    lpot_source, _ = lpot_source.with_refinement(**refiner_extra_kwargs)

    return lpot_source

# }}}


def training_geometries(queue):
    for n_arms in TRAINING_ARMS:
        yield urchin_lpot_source(queue, (n_arms // 2, n_arms), 100)


def test_geometries(queue):
    for n_arms in TESTING_ARMS:
        yield urchin_lpot_source(queue, (n_arms // 2, n_arms), 100)


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


def train_performance_model(ctx):
    queue = cl.CommandQueue(ctx)

    from pytential.qbx.performance import (
            PerformanceModel, estimate_calibration_params)

    perf_model = PerformanceModel()

    model_results = []
    timing_results = []

    for lpot_source in training_geometries(queue):
        lpot_source = lpot_source.copy(performance_model=perf_model)
        bound_op = get_bound_op(lpot_source)
        sigma = get_test_density(queue, lpot_source)

        perf_S = bound_op.get_modeled_performance(queue, sigma=sigma)

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


def test_performance_model(ctx, perf_model):
    queue = cl.CommandQueue(ctx)

    for lpot_source in test_geometries(queue):
        lpot_source = lpot_source.copy(performance_model=perf_model)
        bound_op = get_bound_op(lpot_source)
        sigma = get_test_density(queue, lpot_source)

        perf_S = bound_op.get_modeled_performance(queue, sigma=sigma)
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


def predict_performance(ctx):
    model = train_performance_model(ctx)
    test_performance_model(ctx, model)


if __name__ == "__main__":
    if 0:
        # Disabled - this is slow.
        predict_performance(cl.create_some_context(0))
