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

from pytential import bind, sym, norm  # noqa


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

    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    op_S = bind(lpot_source, sym_op_S)

    timing_data = {}
    op_S.eval(queue, dict(sigma=sigma), timing_data=timing_data)
    assert timing_data
    print(timing_data)

# }}}


# {{{ test performance model

@pytest.mark.parametrize("dim", (2, 3))
def test_performance_model(ctx_getter, dim):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    lpot_source = get_lpot_source(queue, dim)
    sigma = get_density(queue, lpot_source)

    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)

    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
    op_S = bind(lpot_source, sym_op_S)
    perf_S = op_S.get_modeled_performance(queue, sigma=sigma)
    assert len(perf_S) == 1

    sym_op_S_plus_D = (
            sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
            + sym.D(k_sym, sigma_sym))
    op_S_plus_D = bind(lpot_source, sym_op_S_plus_D)
    perf_S_plus_D = op_S_plus_D.get_modeled_performance(queue, sigma=sigma)
    assert len(perf_S_plus_D) == 2

# }}}


# {{{ constant one wrangler

class ConstantOneQBXExpansionWrangler(ConstantOneExpansionWrangler):

    def __init__(self, queue, geo_data):
        from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
        geo_data = ToHostTransferredGeoDataWrapper(queue, geo_data)

        self.geo_data = geo_data
        self.trav = geo_data.traversal()

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

        global_qbx_centers = self.geo_data.global_qbx_centers()
        qbx_center_to_target_box = self.geo_data.qbx_center_to_target_box()

        for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
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

# }}}


# {{{ verify performance model

CONSTANT_ONE_PARAMS = dict(
        p_qbx=1,
        p_fmm=1,
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
        )


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("off_surface", (True, False))
def test_performance_model_correctness(ctx_getter, dim, off_surface):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from pytential.qbx.performance import PerformanceModel

    # We set uses_pde_expansions=False, so that a translation is modeled as
    # simply costing nsrc_coeffs * ntgt_coeffs. By adjusting the symbolic
    # parameters to equal 1 (done below), this provides a straightforward way
    # to obtain the raw operation count for each FMM stage.
    lpot_source = get_lpot_source(queue, dim).copy(
            performance_model=PerformanceModel(uses_pde_expansions=False))

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

    # Construct bound op, run performance model.
    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=qbx_forced_limit)

    op_S = bind((lpot_source, targets), sym_op_S)
    sigma = get_density(queue, lpot_source)

    from pytools import one
    perf_S = one(op_S.get_modeled_performance(queue, sigma=sigma).values())
    # Set all parameters equal to 1, to obtain raw op counts.
    perf_S = perf_S.with_params(CONSTANT_ONE_PARAMS)

    # Run FMM with ConstantOneWrangler. This can't be done with pytential's
    # high-level interface, so call the FMM driver directly.
    from pytential.qbx.fmm import drive_fmm
    geo_data = lpot_source.qbx_fmm_geometry_data(
            target_discrs_and_qbx_sides=target_discrs_and_qbx_sides)

    wrangler = ConstantOneQBXExpansionWrangler(queue, geo_data)
    nnodes = lpot_source.quad_stage2_density_discr.nnodes
    src_weights = np.ones(nnodes)

    timing_data = {}
    potential = drive_fmm(wrangler, src_weights, timing_data,
            traversal=wrangler.trav)[0][geo_data.ncenters:]

    # Check constant one wrangler for correctness.
    assert (potential == nnodes).all()

    modeled_time = perf_S.get_predicted_times(merge_close_lists=True)

    # Check that the performance model matches the timing data returned by the
    # constant one wrangler.
    mismatches = []
    for stage in timing_data:
        if timing_data[stage]["ops_elapsed"] != modeled_time[stage]:
            mismatches.append(
                    (stage, timing_data[stage]["ops_elapsed"], modeled_time[stage]))

    assert not mismatches, "\n".join(str(s) for s in mismatches)

# }}}


# You can test individual routines by typing
# $ python test_performance_model.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])


# vim: foldmethod=marker
