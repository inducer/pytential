from __future__ import division, print_function

__copyright__ = "Copyright (C) 2018 Matt Wala"

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

import numpy as np
import numpy.linalg as la  # noqa

from boxtree.tools import ConstantOneExpansionWrangler
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from pytools import one
from sumpy.kernel import LaplaceKernel, HelmholtzKernel

from pytential import bind, sym, norm  # noqa
from pytential.qbx.cost import CostModel


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
            fmm_order=FMM_ORDER, qbx_order=QBX_ORDER
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


# {{{ test timing data gathering

def test_timing_data_gathering(ctx_getter):
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

@pytest.mark.parametrize("dim", (2, 3))
def test_cost_model(ctx_getter, dim):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    lpot_source = get_lpot_source(queue, dim)
    sigma = get_density(queue, lpot_source)

    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)

    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
    op_S = bind(lpot_source, sym_op_S)
    perf_S = op_S.get_modeled_cost(queue, sigma=sigma)
    assert len(perf_S) == 1

    sym_op_S_plus_D = (
            sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
            + sym.D(k_sym, sigma_sym))
    op_S_plus_D = bind(lpot_source, sym_op_S_plus_D)
    perf_S_plus_D = op_S_plus_D.get_modeled_cost(queue, sigma=sigma)
    assert len(perf_S_plus_D) == 2

# }}}


# {{{ test cost model parameter gathering

def test_cost_model_parameter_gathering(ctx_getter):
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

    perf_S = one(op_S.get_modeled_cost(queue, sigma=sigma, k=k).values())

    geo_data = lpot_source.qbx_fmm_geometry_data(
            target_discrs_and_qbx_sides=((lpot_source.density_discr, 1),))

    tree = geo_data.tree()

    assert perf_S.params["p_qbx"] == QBX_ORDER
    assert perf_S.params["nlevels"] == tree.nlevels
    assert perf_S.params["nsources"] == tree.nsources
    assert perf_S.params["ntargets"] == tree.ntargets
    assert perf_S.params["ncenters"] == geo_data.ncenters

    for level in range(tree.nlevels):
        assert (
                perf_S.params["p_fmm_lev%d" % level]
                == fmm_level_to_order(k_sym, {"k": 2}, tree, level))

# }}}


# {{{ constant one wrangler

class ConstantOneQBXExpansionWrangler(ConstantOneExpansionWrangler):

    def __init__(self, queue, geo_data, use_target_specific_qbx):
        from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
        geo_data = ToHostTransferredGeoDataWrapper(queue, geo_data)

        self.geo_data = geo_data
        self.trav = geo_data.traversal()
        self.use_target_specific_qbx = use_target_specific_qbx

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

        if self.use_target_specific_qbx:
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

        if not self.use_target_specific_qbx:
            return pot, self.timing_future(ops)

        global_qbx_centers = self.geo_data.global_qbx_centers()
        center_to_tree_targets = self.geo_data.center_to_tree_targets()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        for ictr in global_qbx_centers:
            tgt_ibox = qbx_center_to_target_box[ictr]

            ictr_tgt_start, ictr_tgt_end = center_to_tree_targets.starts[ictr:ictr+2]

            for ictr_tgt in range(ictr_tgt_start, ictr_tgt_end):
                ctr_itgt = center_to_tree_targets.lists[ictr_tgt]

                isrc_box_start, isrc_box_end = (
                        self.trav.neighbor_source_boxes_starts[tgt_ibox:tgt_ibox+2])

                for isrc_box in range(isrc_box_start, isrc_box_end):
                    src_ibox = self.trav.neighbor_source_boxes_lists[isrc_box]

                    isrc_start = self.tree.box_source_starts[src_ibox]
                    isrc_end = (isrc_start
                            + self.tree.box_source_counts_nonchild[src_ibox])

                    pot[0][ctr_itgt] += sum(src_weights[isrc_start:isrc_end])
                    ops += isrc_end - isrc_start

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
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    perf_model = (
            CostModel(
                translation_cost_model_factory=OpCountingTranslationCostModel))

    lpot_source = get_lpot_source(queue, dim).copy(
            cost_model=perf_model,
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
    perf_S = one(op_S.get_modeled_cost(queue, sigma=sigma).values())

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

    modeled_time = perf_S.get_predicted_times(merge_close_lists=True)

    # Check that the cost model matches the timing data returned by the
    # constant one wrangler.
    mismatches = []
    for stage in timing_data:
        if timing_data[stage]["ops_elapsed"] != modeled_time[stage]:
            mismatches.append(
                    (stage, timing_data[stage]["ops_elapsed"], modeled_time[stage]))

    assert not mismatches, "\n".join(str(s) for s in mismatches)

# }}}


# {{{ test order varying by level

CONSTANT_ONE_PARAMS = dict(
        c_l2l=1,
        c_l2p=1,
        c_l2qbxl=1,
        c_m2l=1,
        c_m2m=1,
        c_m2p=1,
        c_m2qbxl=1,
        c_p2l=1,
        c_p2m=1,
        c_p2p=1,
        c_p2qbxl=1,
        c_qbxl2p=1,
        c_p2p_tsqbx=1,
        )


def test_cost_model_order_varying_by_level(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ constant level to order

    def level_to_order_constant(kernel, kernel_args, tree, level):
        return 1

    lpot_source = get_lpot_source(queue, 2).copy(
            cost_model=CostModel(
                calibration_params=CONSTANT_ONE_PARAMS),
            fmm_level_to_order=level_to_order_constant)

    sigma_sym = sym.var("sigma")

    k_sym = LaplaceKernel(2)
    sym_op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    sigma = get_density(queue, lpot_source)

    perf_constant = one(
            bind(lpot_source, sym_op)
            .get_modeled_cost(queue, sigma=sigma).values())

    # }}}

    # {{{ varying level to order

    varying_order_params = perf_constant.params.copy()

    nlevels = perf_constant.params["nlevels"]
    for level in range(nlevels):
        varying_order_params["p_fmm_lev%d" % level] = nlevels - level

    perf_varying = perf_constant.with_params(varying_order_params)

    # }}}

    # This only checks to ensure that the costs are different. The varying-level
    # case should have larger cost.

    assert (
            sum(perf_varying.get_predicted_times().values())
            > sum(perf_constant.get_predicted_times().values()))

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
