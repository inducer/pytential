from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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


from six.moves import range, zip
import numpy as np  # noqa
import pyopencl as cl  # noqa
import pyopencl.array  # noqa
from sumpy.fmm import (SumpyExpansionWranglerCodeContainer,
        SumpyExpansionWrangler, level_to_rscale, SumpyTimingFuture)

from pytools import memoize_method
from pytential.qbx.interactions import P2QBXLFromCSR, M2QBXL, L2QBXL, QBXL2P

from boxtree.fmm import TimingRecorder
from pytools import log_process, ProcessLogger

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: QBXSumpyExpansionWranglerCodeContainer

.. autoclass:: QBXExpansionWrangler

.. autofunction:: drive_fmm
"""


# {{{ sumpy expansion wrangler

class QBXSumpyExpansionWranglerCodeContainer(SumpyExpansionWranglerCodeContainer):
    def __init__(self, cl_context,
            multipole_expansion_factory, local_expansion_factory,
            qbx_local_expansion_factory, out_kernels):
        SumpyExpansionWranglerCodeContainer.__init__(self,
                cl_context, multipole_expansion_factory, local_expansion_factory,
                out_kernels)

        self.qbx_local_expansion_factory = qbx_local_expansion_factory

    @memoize_method
    def qbx_local_expansion(self, order):
        return self.qbx_local_expansion_factory(order, self.use_rscale)

    @memoize_method
    def p2qbxl(self, order):
        return P2QBXLFromCSR(self.cl_context,
                self.qbx_local_expansion(order))

    @memoize_method
    def m2qbxl(self, source_order, target_order):
        return M2QBXL(self.cl_context,
                self.multipole_expansion_factory(source_order),
                self.qbx_local_expansion_factory(target_order))

    @memoize_method
    def l2qbxl(self, source_order, target_order):
        return L2QBXL(self.cl_context,
                self.local_expansion_factory(source_order),
                self.qbx_local_expansion_factory(target_order))

    @memoize_method
    def qbxl2p(self, order):
        return QBXL2P(self.cl_context,
                self.qbx_local_expansion_factory(order),
                self.out_kernels)

    def get_wrangler(self, queue, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs={},
            kernel_extra_kwargs=None):
        return QBXExpansionWrangler(self, queue, geo_data,
                dtype,
                qbx_order, fmm_level_to_order,
                source_extra_kwargs,
                kernel_extra_kwargs)


class QBXExpansionWrangler(SumpyExpansionWrangler):
    """A specialized implementation of the
    :class:`boxtree.fmm.ExpansionWranglerInterface` for the QBX FMM.
    The conventional ('point') FMM is carried out on a filtered
    set of targets
    (see :meth:`pytential.discretization.qbx.geometry.\
QBXFMMGeometryData.non_qbx_box_target_lists`),
    and thus all *non-QBX* potential arrays handled by this wrangler don't
    include all targets in the tree, just the non-QBX ones.

    .. rubric:: QBX-specific methods

    .. automethod:: form_global_qbx_locals

    .. automethod:: translate_box_local_to_qbx_local

    .. automethod:: eval_qbx_expansions
    """

    def __init__(self, code_container, queue, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs, kernel_extra_kwargs):
        SumpyExpansionWrangler.__init__(self,
                code_container, queue, geo_data.tree(),
                dtype, fmm_level_to_order, source_extra_kwargs, kernel_extra_kwargs)

        self.qbx_order = qbx_order
        self.geo_data = geo_data

    # {{{ data vector utilities

    def output_zeros(self):
        """This ought to be called ``non_qbx_output_zeros``, but since
        it has to override the superclass's behavior to integrate seamlessly,
        it needs to be called just :meth:`output_zeros`.
        """

        nqbtl = self.geo_data.non_qbx_box_target_lists()

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                cl.array.zeros(
                    self.queue,
                    nqbtl.nfiltered_targets,
                    dtype=self.dtype)
                for k in self.code.out_kernels])

    def full_output_zeros(self):
        # The superclass generates a full field of zeros, for all
        # (not just non-QBX) targets.
        return SumpyExpansionWrangler.output_zeros(self)

    def qbx_local_expansion_zeros(self):
        order = self.qbx_order
        qbx_l_expn = self.code.qbx_local_expansion(order)

        return cl.array.zeros(
                    self.queue,
                    (self.geo_data.ncenters,
                        len(qbx_l_expn)),
                    dtype=self.dtype)

    def reorder_sources(self, source_array):
        return (source_array
                .with_queue(self.queue)
                [self.tree.user_source_ids]
                .with_queue(None))

    def reorder_potentials(self, potentials):
        raise NotImplementedError("reorder_potentials should not "
            "be called on a QBXExpansionWrangler")

        # Because this is a multi-stage, more complicated process that combines
        # potentials from non-QBX targets and QBX targets, reordering takes
        # place in multiple stages below.

    # }}}

    # {{{ source/target dispatch

    # box_source_list_kwargs inherited from superclass

    def box_target_list_kwargs(self):
        # This only covers the non-QBX targets.

        nqbtl = self.geo_data.non_qbx_box_target_lists()
        return dict(
                box_target_starts=nqbtl.box_target_starts,
                box_target_counts_nonchild=(
                    nqbtl.box_target_counts_nonchild),
                targets=nqbtl.targets)

    # }}}

    # {{{ qbx-related

    @log_process(logger)
    def form_global_qbx_locals(self, src_weights):
        local_exps = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return local_exps

        traversal = geo_data.traversal()

        starts = traversal.neighbor_source_boxes_starts
        lists = traversal.neighbor_source_boxes_lists

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        p2qbxl = self.code.p2qbxl(self.qbx_order)

        evt, (result,) = p2qbxl(
                self.queue,
                global_qbx_centers=geo_data.global_qbx_centers(),
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                qbx_centers=geo_data.centers(),
                qbx_expansion_radii=geo_data.expansion_radii(),

                source_box_starts=starts,
                source_box_lists=lists,
                strengths=src_weights,
                qbx_expansions=local_exps,

                **kwargs)

        assert local_exps is result
        result.add_event(evt)

        return (result, SumpyTimingFuture(self.queue, [evt]))

    @log_process(logger)
    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.ncenters == 0:
            return qbx_expansions

        traversal = geo_data.traversal()

        wait_for = multipole_exps.events

        events = []

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):
            m2qbxl = self.code.m2qbxl(
                    self.level_orders[isrc_level],
                    self.qbx_order)

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(multipole_exps, isrc_level)

            evt, (qbx_expansions_res,) = m2qbxl(self.queue,
                    qbx_center_to_target_box_source_level=(
                        geo_data.qbx_center_to_target_box_source_level(isrc_level)
                    ),

                    centers=self.tree.box_centers,
                    qbx_centers=geo_data.centers(),
                    qbx_expansion_radii=geo_data.expansion_radii(),

                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    qbx_expansions=qbx_expansions,

                    src_box_starts=ssn.starts,
                    src_box_lists=ssn.lists,

                    src_rscale=level_to_rscale(self.tree, isrc_level),

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            events.append(evt)

            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return (qbx_expansions, SumpyTimingFuture(self.queue, events))

    @log_process(logger)
    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.ncenters == 0:
            return qbx_expansions
        trav = geo_data.traversal()

        wait_for = local_exps.events

        events = []

        for isrc_level in range(geo_data.tree().nlevels):
            l2qbxl = self.code.l2qbxl(
                    self.level_orders[isrc_level],
                    self.qbx_order)

            target_level_start_ibox, target_locals_view = \
                    self.local_expansions_view(local_exps, isrc_level)

            evt, (qbx_expansions_res,) = l2qbxl(
                    self.queue,
                    qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                    target_boxes=trav.target_boxes,
                    target_base_ibox=target_level_start_ibox,

                    centers=self.tree.box_centers,
                    qbx_centers=geo_data.centers(),
                    qbx_expansion_radii=geo_data.expansion_radii(),

                    expansions=target_locals_view,
                    qbx_expansions=qbx_expansions,

                    src_rscale=level_to_rscale(self.tree, isrc_level),

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            events.append(evt)

            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return (qbx_expansions, SumpyTimingFuture(self.queue, events))

    @log_process(logger)
    def eval_qbx_expansions(self, qbx_expansions):
        pot = self.full_output_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return pot

        ctt = geo_data.center_to_tree_targets()

        qbxl2p = self.code.qbxl2p(self.qbx_order)

        evt, pot_res = qbxl2p(self.queue,
                qbx_centers=geo_data.centers(),
                qbx_expansion_radii=geo_data.expansion_radii(),

                global_qbx_centers=geo_data.global_qbx_centers(),

                center_to_targets_starts=ctt.starts,
                center_to_targets_lists=ctt.lists,

                targets=self.tree.targets,

                qbx_expansions=qbx_expansions,
                result=pot,

                **self.kernel_extra_kwargs.copy())

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return (pot, SumpyTimingFuture(self.queue, [evt]))

    # }}}

# }}}


# {{{ FMM top-level

def drive_fmm(expansion_wrangler, src_weights, timing_data=None):
    """Top-level driver routine for the QBX fast multipole calculation.

    :arg geo_data: A :class:`QBXFMMGeometryData` instance.
    :arg expansion_wrangler: An object exhibiting the
        :class:`ExpansionWranglerInterface`.
    :arg src_weights: Source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.
    :arg timing_data: Either *None* or a dictionary that collects
        timing data.

    Returns the potentials computed by *expansion_wrangler*.

    See also :func:`boxtree.fmm.drive_fmm`.
    """
    wrangler = expansion_wrangler

    geo_data = wrangler.geo_data
    traversal = geo_data.traversal()
    tree = traversal.tree
    recorder = TimingRecorder()

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    fmm_proc = ProcessLogger(logger, "qbx fmm")

    src_weights = wrangler.reorder_sources(src_weights)

    # {{{ construct local multipoles

    mpole_exps, timing_future = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weights)

    recorder.add("form_multipoles", timing_future)

    # }}}

    # {{{ propagate multipoles upward

    mpole_exps, timing_future = wrangler.coarsen_multipoles(
            traversal.level_start_source_parent_box_nrs,
            traversal.source_parent_boxes,
            mpole_exps)

    recorder.add("coarsen_multipoles", timing_future)

    # }}}

    # {{{ direct evaluation from neighbor source boxes ("list 1")

    non_qbx_potentials, timing_future = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights)

    recorder.add("eval_direct", timing_future)

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    local_exps, timing_future = wrangler.multipole_to_local(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            mpole_exps)

    recorder.add("multipole_to_local", timing_future)

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    mpole_result, timing_future = wrangler.eval_multipoles(
            traversal.target_boxes_sep_smaller_by_source_level,
            traversal.from_sep_smaller_by_level,
            mpole_exps)

    recorder.add("eval_multipoles", timing_future)

    non_qbx_potentials = non_qbx_potentials + mpole_result

    # assert that list 3 close has been merged into list 1
    assert traversal.from_sep_close_smaller_starts is None

    # }}}

    # {{{ form locals for separated bigger source boxes ("list 4")

    local_result, timing_future = wrangler.form_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            src_weights)

    recorder.add("form_locals", timing_future)

    local_exps = local_exps + local_result

    # assert that list 4 close has been merged into list 1
    assert traversal.from_sep_close_bigger_starts is None

    # }}}

    # {{{ propagate local_exps downward

    local_exps, timing_future = wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps)

    recorder.add("refine_locals", timing_future)

    # }}}

    # {{{ evaluate locals

    local_result, timing_future = wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            local_exps)

    recorder.add("eval_locals", timing_future)

    non_qbx_potentials = non_qbx_potentials + local_result

    # }}}

    # {{{ wrangle qbx expansions

    qbx_expansions, timing_future = wrangler.form_global_qbx_locals(src_weights)

    recorder.add("form_global_qbx_locals", timing_future)

    local_result, timing_future = (
            wrangler.translate_box_multipoles_to_qbx_local(mpole_exps))

    recorder.add("translate_box_multipoles_to_qbx_local", timing_future)

    qbx_expansions = qbx_expansions + local_result

    local_result, timing_future = (
            wrangler.translate_box_local_to_qbx_local(local_exps))

    recorder.add("translate_box_local_to_qbx_local", timing_future)

    qbx_expansions = qbx_expansions + local_result

    qbx_potentials, timing_future = wrangler.eval_qbx_expansions(qbx_expansions)

    recorder.add("eval_qbx_expansions", timing_future)

    # }}}

    # {{{ reorder potentials

    nqbtl = geo_data.non_qbx_box_target_lists()

    all_potentials_in_tree_order = wrangler.full_output_zeros()

    for ap_i, nqp_i in zip(all_potentials_in_tree_order, non_qbx_potentials):
        ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

    all_potentials_in_tree_order += qbx_potentials

    def reorder_and_finalize_potentials(x):
        # "finalize" gives host FMMs (like FMMlib) a chance to turn the
        # potential back into a CL array.
        return wrangler.finalize_potentials(x[tree.sorted_target_ids])

    from pytools.obj_array import with_object_array_or_scalar
    result = with_object_array_or_scalar(
            reorder_and_finalize_potentials, all_potentials_in_tree_order)

    # }}}

    fmm_proc.done()

    if timing_data is not None:
        timing_data.update(recorder.summarize())

    return result

# }}}


# vim: foldmethod=marker
