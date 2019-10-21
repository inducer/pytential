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

from boxtree.tools import ConstantOneExpansionWrangler
from pytential.qbx import QBXLayerPotentialSource
from sumpy.kernel import LaplaceKernel, HelmholtzKernel
from pytential import bind, sym, norm  # noqa
from pytools import one

from pytential.qbx.cost import (
    CLQBXCostModel, PythonQBXCostModel, pde_aware_translation_cost_model
)

import time

import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# {{{ Compare the time and result of OpenCL implementation and Python implementation

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

    cl_cost_model = CLQBXCostModel(queue)
    python_cost_model = PythonQBXCostModel()

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

# }}}


# {{{ global params

TARGET_ORDER = 8
OVSMP_FACTOR = 5
TCF = 0.9
QBX_ORDER = 5
FMM_ORDER = 10

DEFAULT_LPOT_KWARGS = {
        "_box_extent_norm": "l2",
        "_from_sep_smaller_crit": "static_l2",
        }


def get_lpot_source(queue, dim):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    if dim == 2:
        from meshmode.mesh.generation import starfish, make_curve_mesh
        mesh = make_curve_mesh(starfish, np.linspace(0, 1, 50), order=target_order)
    elif dim == 3:
        from meshmode.mesh.generation import generate_torus
        mesh = generate_torus(2, 1, order=target_order)
    else:
        raise ValueError("unsupported dimension: %d" % dim)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()
    lpot_kwargs.update(
            _expansion_stick_out_factor=TCF,
            fmm_order=FMM_ORDER,
            qbx_order=QBX_ORDER,
            fmm_backend="fmmlib",
            )

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            **lpot_kwargs)

    lpot_source, _ = lpot_source.with_refinement()

    return lpot_source


def get_density(queue, lpot_source):
    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    return cl.clmath.sin(10 * nodes[0])

# }}}


# {{{ test that timing data gathering can execute succesfully

def test_timing_data_gathering(ctx_getter):
    """Test that timing data gathering can execute succesfully."""

    pytest.importorskip("pyfmmlib")

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    lpot_source = get_lpot_source(queue, 2)
    sigma = get_density(queue, lpot_source)

    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    op_S = bind(lpot_source, sym_op_S)

    timing_data = {}
    op_S.eval(queue, dict(sigma=sigma), timing_data=timing_data)
    assert timing_data
    print(timing_data)

# }}}


# {{{ test cost model

@pytest.mark.parametrize("dim, use_target_specific_qbx, per_box", (
    (2, False, False),
    (3, False, False),
    (3, True, False),
    (2, False, True),
    (3, False, True),
    (3, True, True)))
def test_cost_model(ctx_getter, dim, use_target_specific_qbx, per_box):
    """Test that cost model gathering can execute successfully."""
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    lpot_source = (
            get_lpot_source(queue, dim)
            .copy(
                _use_target_specific_qbx=use_target_specific_qbx,
                cost_model=CLQBXCostModel(queue)))

    sigma = get_density(queue, lpot_source)

    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)

    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
    op_S = bind(lpot_source, sym_op_S)
    cost_S, _ = op_S.get_modeled_cost(
        queue, "constant_one", per_box=per_box, sigma=sigma
    )
    assert len(cost_S) == 1

    sym_op_S_plus_D = (
            sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
            + sym.D(k_sym, sigma_sym))
    op_S_plus_D = bind(lpot_source, sym_op_S_plus_D)
    cost_S_plus_D, _ = op_S_plus_D.get_modeled_cost(
        queue, "constant_one", per_box=per_box, sigma=sigma
    )
    assert len(cost_S_plus_D) == 2

# }}}


# {{{ test cost model metadata gathering

def test_cost_model_metadata_gathering(ctx_getter):
    """Test that the cost model correctly gathers metadata."""
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

    fmm_level_to_order = SimpleExpansionOrderFinder(tol=1e-5)

    lpot_source = get_lpot_source(queue, 2).copy(
            fmm_level_to_order=fmm_level_to_order)

    sigma = get_density(queue, lpot_source)

    sigma_sym = sym.var("sigma")
    k_sym = HelmholtzKernel(2, "k")
    k = 2

    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1, k=sym.var("k"))
    op_S = bind(lpot_source, sym_op_S)

    _, metadata = op_S.get_modeled_cost(
        queue, "constant_one", sigma=sigma, k=k, per_box=False, return_metadata=True)
    metadata = one(metadata.values())

    geo_data = lpot_source.qbx_fmm_geometry_data(
            target_discrs_and_qbx_sides=((lpot_source.density_discr, 1),))

    tree = geo_data.tree()

    assert metadata["p_qbx"] == QBX_ORDER
    assert metadata["nlevels"] == tree.nlevels
    assert metadata["nsources"] == tree.nsources
    assert metadata["ntargets"] == tree.ntargets
    assert metadata["ncenters"] == geo_data.ncenters

    for level in range(tree.nlevels):
        assert (
                metadata["p_fmm_lev%d" % level]
                == fmm_level_to_order(k_sym, {"k": 2}, tree, level))

# }}}


# {{{ constant one wrangler

class ConstantOneQBXExpansionWrangler(ConstantOneExpansionWrangler):

    def __init__(self, queue, geo_data, use_target_specific_qbx):
        from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
        geo_data = ToHostTransferredGeoDataWrapper(queue, geo_data)

        self.geo_data = geo_data
        self.trav = geo_data.traversal()
        self.using_tsqbx = (
                use_target_specific_qbx
                # None means use by default if possible
                or use_target_specific_qbx is None)

        ConstantOneExpansionWrangler.__init__(self, geo_data.tree())

    def _get_target_slice(self, ibox):
        non_qbx_box_target_lists = self.geo_data.non_qbx_box_target_lists()
        pstart = non_qbx_box_target_lists.box_target_starts[ibox]
        return slice(
                pstart, pstart
                + non_qbx_box_target_lists.box_target_counts_nonchild[ibox])

    def output_zeros(self):
        non_qbx_box_target_lists = self.geo_data.non_qbx_box_target_lists()
        return np.zeros(non_qbx_box_target_lists.nfiltered_targets)

    def full_output_zeros(self):
        from pytools.obj_array import make_obj_array
        return make_obj_array([np.zeros(self.tree.ntargets)])

    def qbx_local_expansion_zeros(self):
        return np.zeros(self.geo_data.ncenters)

    def reorder_potentials(self, potentials):
        raise NotImplementedError("reorder_potentials should not "
                "be called on a QBXExpansionWrangler")

    def form_global_qbx_locals(self, src_weights):
        local_exps = self.qbx_local_expansion_zeros()
        ops = 0

        if self.using_tsqbx:
            return local_exps, self.timing_future(ops)

        global_qbx_centers = self.geo_data.global_qbx_centers()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        for tgt_icenter in global_qbx_centers:
            itgt_box = qbx_center_to_target_box[tgt_icenter]

            start, end = (
                    self.trav.neighbor_source_boxes_starts[itgt_box:itgt_box + 2])

            src_sum = 0
            for src_ibox in self.trav.neighbor_source_boxes_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                ops += src_pslice.stop - src_pslice.start
                src_sum += np.sum(src_weights[src_pslice])

            local_exps[tgt_icenter] = src_sum

        return local_exps, self.timing_future(ops)

    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        local_exps = self.qbx_local_expansion_zeros()
        ops = 0

        global_qbx_centers = self.geo_data.global_qbx_centers()

        for isrc_level, ssn in enumerate(self.trav.from_sep_smaller_by_level):
            for tgt_icenter in global_qbx_centers:
                icontaining_tgt_box = (
                        self.geo_data
                        .qbx_center_to_target_box_source_level(isrc_level)
                        [tgt_icenter])

                if icontaining_tgt_box == -1:
                    continue

                start, stop = (
                        ssn.starts[icontaining_tgt_box],
                        ssn.starts[icontaining_tgt_box+1])

                for src_ibox in ssn.lists[start:stop]:
                    local_exps[tgt_icenter] += multipole_exps[src_ibox]
                    ops += 1

        return local_exps, self.timing_future(ops)

    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()
        ops = 0

        global_qbx_centers = self.geo_data.global_qbx_centers()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        for tgt_icenter in global_qbx_centers:
            isrc_box = qbx_center_to_target_box[tgt_icenter]
            src_ibox = self.trav.target_boxes[isrc_box]
            qbx_expansions[tgt_icenter] += local_exps[src_ibox]
            ops += 1

        return qbx_expansions, self.timing_future(ops)

    def eval_qbx_expansions(self, qbx_expansions):
        output = self.full_output_zeros()
        ops = 0

        global_qbx_centers = self.geo_data.global_qbx_centers()
        center_to_tree_targets = self.geo_data.center_to_tree_targets()

        for src_icenter in global_qbx_centers:
            start, end = (
                    center_to_tree_targets.starts[src_icenter:src_icenter+2])
            for icenter_tgt in range(start, end):
                center_itgt = center_to_tree_targets.lists[icenter_tgt]
                output[0][center_itgt] += qbx_expansions[src_icenter]
                ops += 1

        return output, self.timing_future(ops)

    def eval_target_specific_qbx_locals(self, src_weights):
        pot = self.full_output_zeros()
        ops = 0

        if not self.using_tsqbx:
            return pot, self.timing_future(ops)

        global_qbx_centers = self.geo_data.global_qbx_centers()
        center_to_tree_targets = self.geo_data.center_to_tree_targets()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        target_box_to_src_sum = {}
        target_box_to_nsrcs = {}

        for ictr in global_qbx_centers:
            tgt_ibox = qbx_center_to_target_box[ictr]

            isrc_box_start, isrc_box_end = (
                    self.trav.neighbor_source_boxes_starts[tgt_ibox:tgt_ibox+2])

            if tgt_ibox not in target_box_to_src_sum:
                nsrcs = 0
                src_sum = 0

                for isrc_box in range(isrc_box_start, isrc_box_end):
                    src_ibox = self.trav.neighbor_source_boxes_lists[isrc_box]

                    isrc_start = self.tree.box_source_starts[src_ibox]
                    isrc_end = (isrc_start
                            + self.tree.box_source_counts_nonchild[src_ibox])

                    src_sum += sum(src_weights[isrc_start:isrc_end])
                    nsrcs += isrc_end - isrc_start

                target_box_to_src_sum[tgt_ibox] = src_sum
                target_box_to_nsrcs[tgt_ibox] = nsrcs

            src_sum = target_box_to_src_sum[tgt_ibox]
            nsrcs = target_box_to_nsrcs[tgt_ibox]

            ictr_tgt_start, ictr_tgt_end = center_to_tree_targets.starts[ictr:ictr+2]

            for ictr_tgt in range(ictr_tgt_start, ictr_tgt_end):
                ctr_itgt = center_to_tree_targets.lists[ictr_tgt]
                pot[0][ctr_itgt] = src_sum

            ops += (ictr_tgt_end - ictr_tgt_start) * nsrcs

        return pot, self.timing_future(ops)

# }}}


# {{{ verify cost model

class OpCountingTranslationCostModel(object):
    """A translation cost model which assigns at cost of 1 to each operation."""

    def __init__(self, dim, nlevels):
        pass

    @staticmethod
    def direct():
        return 1

    p2qbxl = direct
    p2p_tsqbx = direct
    qbxl2p = direct

    @staticmethod
    def p2l(level):
        return 1

    l2p = p2l
    p2m = p2l
    m2p = p2l
    m2qbxl = p2l
    l2qbxl = p2l

    @staticmethod
    def m2m(src_level, tgt_level):
        return 1

    l2l = m2m
    m2l = m2m


@pytest.mark.parametrize("dim, off_surface, use_target_specific_qbx", (
        (2, False, False),
        (2, True,  False),
        (3, False, False),
        (3, False, True),
        (3, True,  False),
        (3, True,  True)))
def test_cost_model_correctness(ctx_getter, dim, off_surface,
        use_target_specific_qbx):
    """Check that computed cost matches that of a constant-one FMM."""
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    cost_model = CLQBXCostModel(
        queue, translation_cost_model_factory=OpCountingTranslationCostModel
    )

    lpot_source = get_lpot_source(queue, dim).copy(
            cost_model=cost_model,
            _use_target_specific_qbx=use_target_specific_qbx)

    # Construct targets.
    if off_surface:
        from pytential.target import PointsTarget
        from boxtree.tools import make_uniform_particle_array
        ntargets = 10 ** 3
        targets = PointsTarget(
                make_uniform_particle_array(queue, ntargets, dim, np.float))
        target_discrs_and_qbx_sides = ((targets, 0),)
        qbx_forced_limit = None
    else:
        targets = lpot_source.density_discr
        target_discrs_and_qbx_sides = ((targets, 1),)
        qbx_forced_limit = 1

    # Construct bound op, run cost model.
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=qbx_forced_limit)

    op_S = bind((lpot_source, targets), sym_op_S)
    sigma = get_density(queue, lpot_source)

    from pytools import one
    modeled_time, _ = op_S.get_modeled_cost(
        queue, "constant_one", per_box=False, sigma=sigma
    )
    modeled_time = one(modeled_time.values())

    # Run FMM with ConstantOneWrangler. This can't be done with pytential's
    # high-level interface, so call the FMM driver directly.
    from pytential.qbx.fmm import drive_fmm
    geo_data = lpot_source.qbx_fmm_geometry_data(
            target_discrs_and_qbx_sides=target_discrs_and_qbx_sides)

    wrangler = ConstantOneQBXExpansionWrangler(
            queue, geo_data, use_target_specific_qbx)
    nnodes = lpot_source.quad_stage2_density_discr.nnodes
    src_weights = np.ones(nnodes)

    timing_data = {}
    potential = drive_fmm(wrangler, src_weights, timing_data,
            traversal=wrangler.trav)[0][geo_data.ncenters:]

    # Check constant one wrangler for correctness.
    assert (potential == nnodes).all()

    # Check that the cost model matches the timing data returned by the
    # constant one wrangler.
    mismatches = []
    for stage in timing_data:
        if stage not in modeled_time:
            assert timing_data[stage]["ops_elapsed"] == 0
        else:
            if timing_data[stage]["ops_elapsed"] != modeled_time[stage]:
                mismatches.append(
                    (stage, timing_data[stage]["ops_elapsed"], modeled_time[stage]))

    assert not mismatches, "\n".join(str(s) for s in mismatches)

    # {{{ Test per-box cost

    total_cost = 0.0
    for stage in timing_data:
        total_cost += timing_data[stage]["ops_elapsed"]

    per_box_cost, _ = op_S.get_modeled_cost(
        queue, "constant_one", per_box=True, sigma=sigma
    )
    per_box_cost = one(per_box_cost.values())

    total_aggregate_cost = cost_model.aggregate(per_box_cost)
    assert total_cost == (
            total_aggregate_cost
            + modeled_time["coarsen_multipoles"]
            + modeled_time["refine_locals"]
    )

    # }}}

# }}}


# {{{ test order varying by level

def test_cost_model_order_varying_by_level(ctx_getter):
    """For FMM order varying by level, this checks to ensure that the costs are
    different. The varying-level case should have larger cost.
    """

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ constant level to order

    def level_to_order_constant(kernel, kernel_args, tree, level):
        return 1

    lpot_source = get_lpot_source(queue, 2).copy(
            cost_model=CLQBXCostModel(queue),
            fmm_level_to_order=level_to_order_constant)

    sigma_sym = sym.var("sigma")

    k_sym = LaplaceKernel(2)
    sym_op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    sigma = get_density(queue, lpot_source)

    cost_constant, metadata = bind(lpot_source, sym_op).get_modeled_cost(
        queue, "constant_one", per_box=False, sigma=sigma
    )

    cost_constant = one(cost_constant.values())
    metadata = one(metadata.values())

    # }}}

    # {{{ varying level to order

    def level_to_order_varying(kernel, kernel_args, tree, level):
        return metadata["nlevels"] - level

    lpot_source = get_lpot_source(queue, 2).copy(
            cost_model=CLQBXCostModel(queue),
            fmm_level_to_order=level_to_order_varying)

    sigma = get_density(queue, lpot_source)

    cost_varying, _ = bind(lpot_source, sym_op).get_modeled_cost(
        queue, "constant_one", per_box=False, sigma=sigma
    )

    cost_varying = one(cost_varying.values())

    # }}}

    assert sum(cost_varying.values()) > sum(cost_constant.values())

# }}}


# You can test individual routines by typing
# $ python test_cost_model.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])


# vim: foldmethod=marker
