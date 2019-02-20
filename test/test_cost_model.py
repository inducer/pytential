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

import pytest
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import numpy as np
import pyopencl as cl
from pytential.qbx import QBXLayerPotentialSource
from pytential.qbx.cost import (
    CLQBXCostModel, PythonQBXCostModel, pde_aware_translation_cost_model
)
import time

import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.mark.opencl
def test_compare_cl_and_py_cost_model(ctx_factory):
    nelements = 3600
    target_order = 16
    fmm_order = 5
    qbx_order = fmm_order

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # {{{ Construct geometry

    from meshmode.mesh.generation import make_curve_mesh, starfish
    mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelements), target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
    pre_density_discr = Discretization(
        ctx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(target_order)
    )

    qbx, _ = QBXLayerPotentialSource(
        pre_density_discr, 4 * target_order,
        qbx_order,
        fmm_order=fmm_order
    ).with_refinement()

    target_discrs_and_qbx_sides = tuple([(qbx.density_discr, 0)])
    geo_data_dev = qbx.qbx_fmm_geometry_data(target_discrs_and_qbx_sides)

    from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
    geo_data = ToHostTransferredGeoDataWrapper(queue, geo_data_dev)

    # }}}

    # {{{ Construct cost models

    cl_cost_model = CLQBXCostModel(queue, None)
    python_cost_model = PythonQBXCostModel(None)

    tree = geo_data.tree()
    xlat_cost = pde_aware_translation_cost_model(tree.targets.shape[0], tree.nlevels)

    constant_one_params = CLQBXCostModel.get_constantone_calibration_params()
    constant_one_params["p_qbx"] = 5
    for ilevel in range(tree.nlevels):
        constant_one_params["p_fmm_lev%d" % ilevel] = 10

    cl_cost_factors = cl_cost_model.qbx_cost_factors_for_kernels_from_model(
        tree.nlevels, xlat_cost, constant_one_params
    )

    python_cost_factors = python_cost_model.qbx_cost_factors_for_kernels_from_model(
        tree.nlevels, xlat_cost, constant_one_params
    )

    # }}}

    # {{{ Test process_form_qbxl

    cl_ndirect_sources_per_target_box = \
        cl_cost_model.get_ndirect_sources_per_target_box(geo_data_dev.traversal())

    queue.finish()
    start_time = time.time()

    cl_p2qbxl = cl_cost_model.process_form_qbxl(
        geo_data_dev, cl_cost_factors["p2qbxl_cost"],
        cl_ndirect_sources_per_target_box
    )

    queue.finish()
    logger.info("OpenCL time for process_form_qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    python_ndirect_sources_per_target_box = \
        python_cost_model.get_ndirect_sources_per_target_box(geo_data.traversal())

    start_time = time.time()

    python_p2qbxl = python_cost_model.process_form_qbxl(
        geo_data, python_cost_factors["p2qbxl_cost"],
        python_ndirect_sources_per_target_box
    )

    logger.info("Python time for process_form_qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_p2qbxl.get(), python_p2qbxl)

    # }}}

    # {{{ Test process_m2qbxl

    queue.finish()
    start_time = time.time()

    cl_m2qbxl = cl_cost_model.process_m2qbxl(
        geo_data_dev, cl_cost_factors["m2qbxl_cost"]
    )

    queue.finish()
    logger.info("OpenCL time for process_m2qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_m2qbxl = python_cost_model.process_m2qbxl(
        geo_data, python_cost_factors["m2qbxl_cost"]
    )

    logger.info("Python time for process_m2qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_m2qbxl.get(), python_m2qbxl)

    # }}}

    # {{{ Test process_l2qbxl

    queue.finish()
    start_time = time.time()

    cl_l2qbxl = cl_cost_model.process_l2qbxl(
        geo_data_dev, cl_cost_factors["l2qbxl_cost"]
    )

    queue.finish()
    logger.info("OpenCL time for process_l2qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_l2qbxl = python_cost_model.process_l2qbxl(
        geo_data, python_cost_factors["l2qbxl_cost"]
    )

    logger.info("Python time for process_l2qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_l2qbxl.get(), python_l2qbxl)

    # }}}

    # {{{ Test process_eval_qbxl

    queue.finish()
    start_time = time.time()

    cl_eval_qbxl = cl_cost_model.process_eval_qbxl(
        geo_data_dev, cl_cost_factors["qbxl2p_cost"]
    )

    queue.finish()
    logger.info("OpenCL time for process_eval_qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_eval_qbxl = python_cost_model.process_eval_qbxl(
        geo_data, python_cost_factors["qbxl2p_cost"]
    )

    logger.info("Python time for process_eval_qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(cl_eval_qbxl.get(), python_eval_qbxl)

    # }}}

    # {{{ Test eval_target_specific_qbxl

    queue.finish()
    start_time = time.time()

    cl_eval_target_specific_qbxl = cl_cost_model.process_eval_target_specific_qbxl(
        geo_data_dev, cl_cost_factors["p2p_tsqbx_cost"],
        cl_ndirect_sources_per_target_box
    )

    queue.finish()
    logger.info("OpenCL time for eval_target_specific_qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    start_time = time.time()

    python_eval_target_specific_qbxl = \
        python_cost_model.process_eval_target_specific_qbxl(
            geo_data, python_cost_factors["p2p_tsqbx_cost"],
            python_ndirect_sources_per_target_box
        )

    logger.info("Python time for eval_target_specific_qbxl: {0}".format(
        str(time.time() - start_time)
    ))

    assert np.array_equal(
        cl_eval_target_specific_qbxl.get(), python_eval_target_specific_qbxl
    )

    # }}}


if __name__ == "__main__":
    ctx_factory = cl.create_some_context
    test_compare_cl_and_py_cost_model(ctx_factory)

# vim: foldmethod=marker
