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

import numpy as np

import pyopencl as cl
from meshmode.array_context import PyOpenCLArrayContext
from pytools import one

from pytential import bind, sym
from pytential.qbx.cost import QBXCostModel


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
        InterpolatoryQuadratureSimplexGroupFactory,
    )
    from meshmode.mesh.generation import NArmedStarfish, make_curve_mesh

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
    nodes = actx.thaw(density_discr.nodes())
    sigma = actx.np.sin(10 * nodes[0])
    return sigma


def calibrate_cost_model(ctx):
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)
    cost_model = QBXCostModel()

    model_results = []
    timing_results = []

    for lpot_source in training_geometries(actx):
        lpot_source = lpot_source.copy(cost_model=cost_model)

        from pytential import GeometryCollection
        places = GeometryCollection(lpot_source)
        density_discr = places.get_discretization(places.auto_source.geometry)

        bound_op = get_bound_op(places)
        sigma = get_test_density(actx, density_discr)

        modeled_cost, _ = bound_op.cost_per_stage("constant_one", sigma=sigma)

        # Warm-up run.
        bound_op.eval({"sigma": sigma}, array_context=actx)

        for _ in range(RUNS):
            timing_data = {}
            bound_op.eval({"sigma": sigma}, array_context=actx)

            model_results.append(modeled_cost)
            timing_results.append(timing_data)

    calibration_params = cost_model.estimate_kernel_specific_calibration_params(
        model_results, timing_results, time_field_name="process_elapsed"
    )

    return calibration_params


def test_cost_model(ctx, calibration_params):
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)
    cost_model = QBXCostModel()

    for lpot_source in test_geometries(actx):
        lpot_source = lpot_source.copy(cost_model=cost_model)

        from pytential import GeometryCollection
        places = GeometryCollection(lpot_source)
        density_discr = places.get_discretization(places.auto_source.geometry)

        bound_op = get_bound_op(places)
        sigma = get_test_density(actx, density_discr)

        cost_S, _ = bound_op.cost_per_stage(calibration_params, sigma=sigma)
        model_result = one(cost_S.values())

        # Warm-up run.
        bound_op.eval({"sigma": sigma}, array_context=actx)

        temp_timing_results = []
        for _ in range(RUNS):
            timing_data = {}
            bound_op.eval({"sigma": sigma}, array_context=actx)
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
                    f"{timing_result[stage]:.2f}",
                    f"{model_result[stage]:.2f}",
            ]
            table.add_row(row)

        print(table)


def predict_cost(ctx):
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    params = calibrate_cost_model(ctx)
    test_cost_model(ctx, params)


if __name__ == "__main__":
    predict_cost(cl.create_some_context())
