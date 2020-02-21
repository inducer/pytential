from __future__ import division, print_function

__copyright__ = """
    Copyright (C) 2018 Matt Wala
    Copyright (C) 2019 Hao Gao
"""

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

"""Calibrates a cost model and reports on the accuracy."""

import pyopencl as cl
import numpy as np

from pytential import sym, bind
from pytential.qbx.cost import QBXCostModel
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
    cost_model = QBXCostModel(queue)

    model_results = []
    timing_results = []

    for lpot_source in training_geometries(queue):
        lpot_source = lpot_source.copy(cost_model=cost_model)
        bound_op = get_bound_op(lpot_source)
        sigma = get_test_density(queue, lpot_source)

        modeled_cost, _ = bound_op.cost_per_stage(queue, "constant_one", sigma=sigma)

        # Warm-up run.
        bound_op.eval(queue, {"sigma": sigma})

        for _ in range(RUNS):
            timing_data = {}
            bound_op.eval(queue, {"sigma": sigma}, timing_data=timing_data)

            model_results.append(modeled_cost)
            timing_results.append(timing_data)

    calibration_params = cost_model.estimate_knl_specific_calibration_params(
        model_results, timing_results, time_field_name="process_elapsed"
    )

    return calibration_params


def test_cost_model(ctx, calibration_params):
    queue = cl.CommandQueue(ctx)
    cost_model = QBXCostModel(queue)

    for lpot_source in test_geometries(queue):
        lpot_source = lpot_source.copy(cost_model=cost_model)
        bound_op = get_bound_op(lpot_source)
        sigma = get_test_density(queue, lpot_source)

        cost_S, _ = bound_op.cost_per_stage(queue, calibration_params, sigma=sigma)
        model_result = one(cost_S.values())

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

        from pytools import Table
        table = Table()
        table.add_row(["stage", "actual (s)", "predicted (s)"])
        for stage in model_result:
            row = [
                    stage,
                    "%.2f" % timing_result[stage],
                    "%.2f" % model_result[stage]
            ]
            table.add_row(row)

        print(table)


def predict_cost(ctx):
    params = calibrate_cost_model(ctx)
    test_cost_model(ctx, params)


if __name__ == "__main__":
    predict_cost(cl.create_some_context(0))
