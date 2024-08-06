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

import pyopencl as cl
import pyopencl.array
from sumpy.fmm import (SumpyTreeIndependentDataForWrangler,
        SumpyExpansionWrangler, SumpyTimingFuture)

from pytools import memoize_method
from pytential.qbx.interactions import P2QBXLFromCSR, M2QBXL, L2QBXL, QBXL2P

from boxtree.timing import TimingRecorder
from pytools import log_process, ProcessLogger

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: QBXSumpyTreeIndependentDataForWrangler

.. autoclass:: QBXExpansionWrangler

.. autofunction:: drive_fmm
"""


# {{{ sumpy expansion wrangler

class QBXSumpyTreeIndependentDataForWrangler(SumpyTreeIndependentDataForWrangler):
    def __init__(self, cl_context,
            multipole_expansion_factory, local_expansion_factory,
            qbx_local_expansion_factory, target_kernels, source_kernels):
        super().__init__(
                cl_context, multipole_expansion_factory, local_expansion_factory,
                target_kernels=target_kernels, source_kernels=source_kernels)

        self.qbx_local_expansion_factory = qbx_local_expansion_factory

    @memoize_method
    def qbx_local_expansion(self, order):
        return self.qbx_local_expansion_factory(order, self.use_rscale)

    @memoize_method
    def p2qbxl(self, order):
        return P2QBXLFromCSR(self.cl_context,
                self.qbx_local_expansion(order), kernels=self.source_kernels)

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
                self.target_kernels)

    @property
    def wrangler_cls(self):
        return QBXExpansionWrangler


class QBXExpansionWrangler(SumpyExpansionWrangler):
    """A specialized implementation of the
    :class:`boxtree.fmm.ExpansionWranglerInterface` for the QBX FMM.
    The conventional ('point') FMM is carried out on a filtered
    set of targets
    (see :meth:`pytential.qbx.geometry.QBXFMMGeometryData.\
non_qbx_box_target_lists`),
    and thus all *non-QBX* potential arrays handled by this wrangler don't
    include all targets in the tree, just the non-QBX ones.

    .. rubric:: QBX-specific methods

    .. automethod:: form_global_qbx_locals

    .. automethod:: translate_box_local_to_qbx_local

    .. automethod:: eval_qbx_expansions
    """

    def __init__(self, tree_indep, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs, kernel_extra_kwargs,
            translation_classes_data=None,
            _use_target_specific_qbx=None):
        if _use_target_specific_qbx:
            raise ValueError("TSQBX is not implemented in sumpy")

        base_kernel = tree_indep.get_base_kernel()
        if translation_classes_data is None and base_kernel.is_translation_invariant:
            from pytential.qbx.fmm import translation_classes_builder
            traversal = geo_data.traversal()
            actx = geo_data._setup_actx

            translation_classes_data, _ = translation_classes_builder(actx)(
                actx.queue, traversal, traversal.tree, is_translation_per_level=True)

        super().__init__(
                tree_indep, traversal,
                dtype, fmm_level_to_order, source_extra_kwargs, kernel_extra_kwargs,
                translation_classes_data=translation_classes_data)

        self.qbx_order = qbx_order
        self.geo_data = geo_data
        self.using_tsqbx = False

    # {{{ data vector utilities

    def output_zeros(self, template_ary):
        """This ought to be called ``non_qbx_output_zeros``, but since
        it has to override the superclass's behavior to integrate seamlessly,
        it needs to be called just :meth:`output_zeros`.
        """

        nqbtl = self.geo_data.non_qbx_box_target_lists()

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                cl.array.zeros(
                    template_ary.queue,
                    nqbtl.nfiltered_targets,
                    dtype=self.dtype)
                for k in self.tree_indep.target_kernels])

    def full_output_zeros(self, template_ary):
        # The superclass generates a full field of zeros, for all
        # (not just non-QBX) targets.
        return super().output_zeros(template_ary)

    def qbx_local_expansion_zeros(self, template_ary):
        order = self.qbx_order
        qbx_l_expn = self.tree_indep.qbx_local_expansion(order)

        return cl.array.zeros(
                    template_ary.queue,
                    (self.geo_data.ncenters,
                        len(qbx_l_expn)),
                    dtype=self.dtype)

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

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
        return {
                "box_target_starts": nqbtl.box_target_starts,
                "box_target_counts_nonchild": (
                    nqbtl.box_target_counts_nonchild),
                "targets": nqbtl.targets}

    # }}}

    # {{{ qbx-related

    @log_process(logger)
    def form_global_qbx_locals(self, src_weight_vecs):
        queue = src_weight_vecs[0].queue

        local_exps = self.qbx_local_expansion_zeros(src_weight_vecs[0])
        events = []

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return (local_exps, SumpyTimingFuture(queue, events))

        traversal = geo_data.traversal()

        starts = traversal.neighbor_source_boxes_starts
        lists = traversal.neighbor_source_boxes_lists

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        p2qbxl = self.tree_indep.p2qbxl(self.qbx_order)

        evt, (result,) = p2qbxl(
                queue,
                global_qbx_centers=geo_data.global_qbx_centers(),
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                qbx_centers=geo_data.flat_centers(),
                qbx_expansion_radii=geo_data.flat_expansion_radii(),

                source_box_starts=starts,
                source_box_lists=lists,
                strengths=src_weight_vecs,
                qbx_expansions=local_exps,

                **kwargs)

        events.append(evt)
        assert local_exps is result
        result.add_event(evt)

        return (result, SumpyTimingFuture(queue, events))

    @log_process(logger)
    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        queue = multipole_exps.queue
        qbx_expansions = self.qbx_local_expansion_zeros(multipole_exps)
        events = []

        geo_data = self.geo_data
        if geo_data.ncenters == 0:
            return (qbx_expansions, SumpyTimingFuture(queue, events))

        traversal = geo_data.traversal()

        wait_for = multipole_exps.events

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):
            m2qbxl = self.tree_indep.m2qbxl(
                    self.level_orders[isrc_level],
                    self.qbx_order)

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(multipole_exps, isrc_level)

            evt, (qbx_expansions_res,) = m2qbxl(queue,
                    qbx_center_to_target_box_source_level=(
                        geo_data.qbx_center_to_target_box_source_level(isrc_level)
                    ),

                    centers=self.tree.box_centers,
                    qbx_centers=geo_data.flat_centers(),
                    qbx_expansion_radii=geo_data.flat_expansion_radii(),

                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    qbx_expansions=qbx_expansions,

                    src_box_starts=ssn.starts,
                    src_box_lists=ssn.lists,

                    src_rscale=self.level_to_rscale(isrc_level),

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            events.append(evt)
            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return (qbx_expansions, SumpyTimingFuture(queue, events))

    @log_process(logger)
    def translate_box_local_to_qbx_local(self, local_exps):
        queue = local_exps.queue
        qbx_expansions = self.qbx_local_expansion_zeros(local_exps)

        geo_data = self.geo_data
        events = []

        if geo_data.ncenters == 0:
            return (qbx_expansions, SumpyTimingFuture(queue, events))

        trav = geo_data.traversal()

        wait_for = local_exps.events

        for isrc_level in range(geo_data.tree().nlevels):
            l2qbxl = self.tree_indep.l2qbxl(
                    self.level_orders[isrc_level],
                    self.qbx_order)

            target_level_start_ibox, target_locals_view = \
                    self.local_expansions_view(local_exps, isrc_level)

            evt, (qbx_expansions_res,) = l2qbxl(
                    queue,
                    qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                    target_boxes=trav.target_boxes,
                    target_base_ibox=target_level_start_ibox,

                    centers=self.tree.box_centers,
                    qbx_centers=geo_data.flat_centers(),
                    qbx_expansion_radii=geo_data.flat_expansion_radii(),

                    expansions=target_locals_view,
                    qbx_expansions=qbx_expansions,

                    src_rscale=self.level_to_rscale(isrc_level),

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            events.append(evt)
            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return (qbx_expansions, SumpyTimingFuture(queue, events))

    @log_process(logger)
    def eval_qbx_expansions(self, qbx_expansions):
        queue = qbx_expansions.queue
        pot = self.full_output_zeros(qbx_expansions)

        geo_data = self.geo_data
        events = []

        if len(geo_data.global_qbx_centers()) == 0:
            return (pot, SumpyTimingFuture(queue, events))

        ctt = geo_data.center_to_tree_targets()

        qbxl2p = self.tree_indep.qbxl2p(self.qbx_order)

        _, pot_res = qbxl2p(queue,
                qbx_centers=geo_data.flat_centers(),
                qbx_expansion_radii=geo_data.flat_expansion_radii(),

                global_qbx_centers=geo_data.global_qbx_centers(),

                center_to_targets_starts=ctt.starts,
                center_to_targets_lists=ctt.lists,

                targets=self.tree.targets,

                qbx_expansions=qbx_expansions,
                result=pot,

                **self.kernel_extra_kwargs.copy())

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return (pot, SumpyTimingFuture(queue, events))

    @log_process(logger)
    def eval_target_specific_qbx_locals(self, src_weight_vecs):
        template_ary = src_weight_vecs[0]
        return (self.full_output_zeros(template_ary),
                SumpyTimingFuture(template_ary.queue, events=()))

    # }}}


def translation_classes_builder(actx):
    from pytools import memoize_in

    @memoize_in(actx, (QBXExpansionWrangler, translation_classes_builder))
    def make_container():
        from boxtree.translation_classes import TranslationClassesBuilder
        return TranslationClassesBuilder(actx.context)

    return make_container()

# }}}


# {{{ FMM top-level

def _reorder_and_finalize_potentials(
        wrangler, non_qbx_potentials, qbx_potentials, template_ary):
    nqbtl = wrangler.geo_data.non_qbx_box_target_lists()

    all_potentials_in_tree_order = wrangler.full_output_zeros(template_ary)

    for ap_i, nqp_i in zip(all_potentials_in_tree_order, non_qbx_potentials):
        ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

    all_potentials_in_tree_order += qbx_potentials

    def reorder_and_finalize_potentials(x):
        # "finalize" gives host FMMs (like FMMlib) a chance to turn the
        # potential back into a CL array.
        return wrangler.finalize_potentials(x[
            wrangler.geo_data.traversal().tree.sorted_target_ids], template_ary)

    from pytools.obj_array import obj_array_vectorize
    return obj_array_vectorize(
            reorder_and_finalize_potentials, all_potentials_in_tree_order)


def drive_fmm(expansion_wrangler, src_weight_vecs, timing_data=None,
        traversal=None):
    """Top-level driver routine for the QBX fast multipole calculation.

    :arg geo_data: A :class:`pytential.qbx.geometry.QBXFMMGeometryData` instance.
    :arg expansion_wrangler: An object exhibiting the
        :class:`boxtree.fmm.ExpansionWranglerInterface`.
    :arg src_weight_vecs: A sequence of source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.
    :arg timing_data: Either *None* or a dictionary that collects
        timing data.

    Returns the potentials computed by *expansion_wrangler*.

    See also :func:`boxtree.fmm.drive_fmm`.
    """
    wrangler = expansion_wrangler

    geo_data = wrangler.geo_data

    if traversal is None:
        traversal = geo_data.traversal()

    template_ary = src_weight_vecs[0]

    recorder = TimingRecorder()

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    fmm_proc = ProcessLogger(logger, "qbx fmm")

    src_weight_vecs = [wrangler.reorder_sources(weight)
        for weight in src_weight_vecs]

    # {{{ construct local multipoles

    mpole_exps, timing_future = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weight_vecs)

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
            src_weight_vecs)

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
            src_weight_vecs)

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

    # form_global_qbx_locals and eval_target_specific_qbx_locals are responsible
    # for the same interactions (directly evaluated portion of the potentials
    # via unified List 1).  Which one is used depends on the wrangler. If one of
    # them is unused the corresponding output entries will be zero.

    qbx_expansions, timing_future = wrangler.form_global_qbx_locals(src_weight_vecs)

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

    ts_result, timing_future = \
        wrangler.eval_target_specific_qbx_locals(src_weight_vecs)

    qbx_potentials = qbx_potentials + ts_result

    recorder.add("eval_target_specific_qbx_locals", timing_future)

    # }}}

    # {{{ reorder potentials

    result = _reorder_and_finalize_potentials(
        wrangler, non_qbx_potentials, qbx_potentials, template_ary)

    # }}}

    fmm_proc.done()

    if timing_data is not None:
        timing_data.update(recorder.summarize())
    return result

# }}}

# vim: foldmethod=marker
