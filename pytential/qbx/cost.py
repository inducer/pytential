from __future__ import division, absolute_import

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2018 Matt Wala
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

from collections import OrderedDict

import logging

import numpy as np
import pyopencl as cl
from six.moves import range
import sympy as sp

from pytools import log_process
from pymbolic import var

logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: CostModel
.. autoclass:: ParametrizedCosts

.. autoclass:: TranslationCostModel

.. autofunction:: pde_aware_translation_cost_model
.. autofunction:: taylor_translation_cost_model

.. autofunction:: estimate_calibration_params
"""


# {{{ translation cost model

class TranslationCostModel(object):
    """Provides modeled costs for individual translations or evaluations."""

    def __init__(self, ncoeffs_qbx, ncoeffs_fmm_by_level, uses_point_and_shoot):
        self.ncoeffs_qbx = ncoeffs_qbx
        self.ncoeffs_fmm_by_level = ncoeffs_fmm_by_level
        self.uses_point_and_shoot = uses_point_and_shoot

    @staticmethod
    def direct():
        return var("c_p2p")

    def p2qbxl(self):
        return var("c_p2qbxl") * self.ncoeffs_qbx

    def p2p_tsqbx(self):
        # This term should be linear in the QBX order, which is the
        # square root of the number of QBX coefficients.
        return var("c_p2p_tsqbx") * self.ncoeffs_qbx ** (1/2)

    def qbxl2p(self):
        return var("c_qbxl2p") * self.ncoeffs_qbx

    def p2l(self, level):
        return var("c_p2l") * self.ncoeffs_fmm_by_level[level]

    def l2p(self, level):
        return var("c_l2p") * self.ncoeffs_fmm_by_level[level]

    def p2m(self, level):
        return var("c_p2m") * self.ncoeffs_fmm_by_level[level]

    def m2p(self, level):
        return var("c_m2p") * self.ncoeffs_fmm_by_level[level]

    def m2m(self, src_level, tgt_level):
        return var("c_m2m") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def l2l(self, src_level, tgt_level):
        return var("c_l2l") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def m2l(self, src_level, tgt_level):
        return var("c_m2l") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def m2qbxl(self, level):
        return var("c_m2qbxl") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[level],
                self.ncoeffs_qbx)

    def l2qbxl(self, level):
        return var("c_l2qbxl") * self.e2e_cost(
                self.ncoeffs_fmm_by_level[level],
                self.ncoeffs_qbx)

    def e2e_cost(self, nsource_coeffs, ntarget_coeffs):
        if self.uses_point_and_shoot:
            return (
                    # Rotate the coordinate system to be z axis aligned.
                    nsource_coeffs ** (3 / 2)
                    # Translate the expansion along the z axis.
                    + nsource_coeffs ** (1 / 2) * ntarget_coeffs
                    # Rotate the coordinate system back.
                    + ntarget_coeffs ** (3 / 2))

        return nsource_coeffs * ntarget_coeffs

# }}}


# {{{ translation cost model factories

def pde_aware_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation operators that make use of the
    knowledge that the potential satisfies a PDE.
    """
    p_qbx = var("p_qbx")
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])

    uses_point_and_shoot = False

    ncoeffs_fmm = (p_fmm + 1) ** (dim - 1)
    ncoeffs_qbx = (p_qbx + 1) ** (dim - 1)

    if dim == 3:
        uses_point_and_shoot = True

    return TranslationCostModel(
            ncoeffs_qbx=ncoeffs_qbx,
            ncoeffs_fmm_by_level=ncoeffs_fmm,
            uses_point_and_shoot=uses_point_and_shoot)


def taylor_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation based on Taylor expansions
    in Cartesian coordinates.
    """
    p_qbx = var("p_qbx")
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])

    ncoeffs_fmm = (p_fmm + 1) ** dim
    ncoeffs_qbx = (p_qbx + 1) ** dim

    return TranslationCostModel(
            ncoeffs_qbx=ncoeffs_qbx,
            ncoeffs_fmm_by_level=ncoeffs_fmm,
            uses_point_and_shoot=False)

# }}}


# {{{ parameterized costs returned by cost model

class ParametrizedCosts(object):
    """A container for data returned by the cost model.

    This holds both symbolic costs as well as parameter values. To obtain a
    prediction of the running time, use :meth:`get_predicted_times`.

    .. attribute:: raw_costs

        A dictionary mapping algorithmic stage names to symbolic costs.

    .. attribute:: params

        A dictionary mapping names of symbolic parameters to values.  Parameters
        appear in *raw_costs* and may include values such as QBX or FMM order
        as well as calibration constants.

    .. automethod:: copy
    .. automethod:: with_params
    .. automethod:: get_predicted_times
    """

    def __init__(self, raw_costs, params):
        self.raw_costs = OrderedDict(raw_costs)
        self.params = params

    def with_params(self, params):
        """Return a copy of *self* with parameters updated to include *params*."""
        new_params = self.params.copy()
        new_params.update(params)
        return type(self)(
                raw_costs=self.raw_costs.copy(),
                params=new_params)

    def copy(self):
        return self.with_params({})

    def __str__(self):
        return "".join([
                type(self).__name__,
                "(raw_costs=",
                str(self.raw_costs),
                ", params=",
                str(self.params),
                ")"])

    def __repr__(self):
        return "".join([
                type(self).__name__,
                "(raw_costs=",
                repr(self.raw_costs),
                ", params=",
                repr(self.params),
                ")"])

    def get_predicted_times(self, merge_close_lists=False):
        """Return a dictionary mapping stage names to predicted time in seconds.

        :arg merge_close_lists: If *True*, the returned estimate combines
            the cost of "close" lists (Lists 1, 3 close, and 4 close). If
            *False*, the time of each "close" list is reported separately.
        """
        from pymbolic import evaluate
        from functools import partial

        get_time = partial(evaluate, context=self.params)

        result = OrderedDict()

        for name, val in self.raw_costs.items():
            if merge_close_lists:
                for suffix in ("_list1", "_list3", "_list4"):
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break

            result[name] = get_time(val) + result.get(name, 0)

        return result

# }}}


# {{{ cost model

class CostModel(object):
    """
    .. automethod:: with_calibration_params
    .. automethod:: __call__

    The cost model relies on a translation cost model. See
    :class:`TranslationCostModel` for the translation cost model interface.
    """

    def __init__(self,
            translation_cost_model_factory=pde_aware_translation_cost_model,
            calibration_params=None):
        """
        :arg translation_cost_model_factory: A callable which, given arguments
            (*dim*, *nlevels*), returns a translation cost model.
        """
        self.translation_cost_model_factory = translation_cost_model_factory
        if calibration_params is None:
            calibration_params = dict()
        self.calibration_params = calibration_params

    def with_calibration_params(self, calibration_params):
        """Return a copy of *self* with a new set of calibration parameters."""
        return type(self)(
                translation_cost_model_factory=self.translation_cost_model_factory,
                calibration_params=calibration_params)

    # {{{ form multipoles

    def process_form_multipoles(self, xlat_cost, traversal, tree):
        result = 0

        for level in range(tree.nlevels):
            src_count = 0
            start, stop = traversal.level_start_source_box_nrs[level:level + 2]
            for src_ibox in traversal.source_boxes[start:stop]:
                nsrcs = tree.box_source_counts_nonchild[src_ibox]
                src_count += nsrcs
            result += src_count * xlat_cost.p2m(level)

        return dict(form_multipoles=result)

    # }}}

    # {{{ propagate multipoles upward

    def process_coarsen_multipoles(self, xlat_cost, traversal, tree):
        result = 0

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            cost = xlat_cost.m2m(source_level, target_level)

            nmultipoles = 0
            start, stop = traversal.level_start_source_parent_box_nrs[
                            target_level:target_level+2]
            for ibox in traversal.source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        nmultipoles += 1

            result += cost * nmultipoles

        return dict(coarsen_multipoles=result)

    # }}}

    # {{{ collect direct interaction data

    @staticmethod
    def _collect_direction_interaction_data(traversal, tree):
        ntarget_boxes = len(traversal.target_boxes)

        # target box index -> nsources
        nlist1_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)
        nlist3close_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)
        nlist4close_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)

        for itgt_box in range(ntarget_boxes):
            nlist1_srcs = 0
            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nlist1_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist1_srcs_by_itgt_box[itgt_box] = nlist1_srcs

            nlist3close_srcs = 0
            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                        traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nlist3close_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist3close_srcs_by_itgt_box[itgt_box] = nlist3close_srcs

            nlist4close_srcs = 0
            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                        traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nlist4close_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist4close_srcs_by_itgt_box[itgt_box] = nlist4close_srcs

        result = {}
        result["nlist1_srcs_by_itgt_box"] = nlist1_srcs_by_itgt_box
        result["nlist3close_srcs_by_itgt_box"] = nlist3close_srcs_by_itgt_box
        result["nlist4close_srcs_by_itgt_box"] = nlist4close_srcs_by_itgt_box

        return result

    # }}}

    # {{{ direct evaluation to point targets (lists 1, 3 close, 4 close)

    def process_direct(self, xlat_cost, traversal, direct_interaction_data,
            box_target_counts_nonchild):
        nlist1_srcs_by_itgt_box = (
                direct_interaction_data["nlist1_srcs_by_itgt_box"])
        nlist3close_srcs_by_itgt_box = (
                direct_interaction_data["nlist3close_srcs_by_itgt_box"])
        nlist4close_srcs_by_itgt_box = (
                direct_interaction_data["nlist4close_srcs_by_itgt_box"])

        # list -> number of source-target interactions
        npart_direct_list1 = 0
        npart_direct_list3 = 0
        npart_direct_list4 = 0

        for itgt_box, tgt_ibox in enumerate(traversal.target_boxes):
            ntargets = box_target_counts_nonchild[tgt_ibox]

            npart_direct_list1 += ntargets * nlist1_srcs_by_itgt_box[itgt_box]
            npart_direct_list3 += ntargets * nlist3close_srcs_by_itgt_box[itgt_box]
            npart_direct_list4 += ntargets * nlist4close_srcs_by_itgt_box[itgt_box]

        result = {}
        result["eval_direct_list1"] = npart_direct_list1 * xlat_cost.direct()
        result["eval_direct_list3"] = npart_direct_list3 * xlat_cost.direct()
        result["eval_direct_list4"] = npart_direct_list4 * xlat_cost.direct()

        return result

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    def process_list2(self, xlat_cost, traversal, tree):
        nm2l_by_level = np.zeros(tree.nlevels, dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(traversal.target_or_target_parent_boxes):
            start, end = traversal.from_sep_siblings_starts[itgt_box:itgt_box+2]

            level = tree.box_levels[tgt_ibox]
            nm2l_by_level[level] += end-start

        result = sum(
                cost * xlat_cost.m2l(ilevel, ilevel)
                for ilevel, cost in enumerate(nm2l_by_level))

        return dict(multipole_to_local=result)

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    def process_list3(self, xlat_cost, traversal, tree, box_target_counts_nonchild):
        nmp_eval_by_source_level = np.zeros(tree.nlevels, dtype=np.intp)

        assert tree.nlevels == len(traversal.from_sep_smaller_by_level)

        for ilevel, sep_smaller_list in enumerate(
                traversal.from_sep_smaller_by_level):
            for itgt_box, tgt_ibox in enumerate(
                        traversal.target_boxes_sep_smaller_by_source_level[ilevel]):
                ntargets = box_target_counts_nonchild[tgt_ibox]
                start, end = sep_smaller_list.starts[itgt_box:itgt_box+2]
                nmp_eval_by_source_level[ilevel] += ntargets * (end-start)

        result = sum(
                cost * xlat_cost.m2p(ilevel)
                for ilevel, cost in enumerate(nmp_eval_by_source_level))

        return dict(eval_multipoles=result)

    # }}}

    # {{{ form locals for separated bigger source boxes ("list 4")

    def process_list4(self, xlat_cost, traversal, tree):
        nform_local_by_source_level = np.zeros(tree.nlevels, dtype=np.intp)

        for itgt_box in range(len(traversal.target_or_target_parent_boxes)):
            start, end = traversal.from_sep_bigger_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.from_sep_bigger_lists[start:end]:
                nsources = tree.box_source_counts_nonchild[src_ibox]
                level = tree.box_levels[src_ibox]
                nform_local_by_source_level[level] += nsources

        result = sum(
                cost * xlat_cost.p2l(ilevel)
                for ilevel, cost in enumerate(nform_local_by_source_level))

        return dict(form_locals=result)

    # }}}

    # {{{ propogate locals downward

    def process_refine_locals(self, xlat_cost, traversal, tree):
        result = 0

        for target_lev in range(1, tree.nlevels):
            start, stop = traversal.level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            source_lev = target_lev - 1
            result += (stop-start) * xlat_cost.l2l(source_lev, target_lev)

        return dict(refine_locals=result)

    # }}}

    # {{{ evaluate local expansions at non-qbx targets

    def process_eval_locals(self, xlat_cost, traversal, tree, nqbtl):
        ntargets_by_level = np.zeros(tree.nlevels, dtype=np.intp)

        for target_lev in range(tree.nlevels):
            start, stop = traversal.level_start_target_box_nrs[
                    target_lev:target_lev+2]
            for tgt_ibox in traversal.target_boxes[start:stop]:
                ntargets_by_level[target_lev] += (
                        nqbtl.box_target_counts_nonchild[tgt_ibox])

        result = sum(
                cost * xlat_cost.l2p(ilevel)
                for ilevel, cost in enumerate(ntargets_by_level))

        return dict(eval_locals=result)

    # }}}

    # {{{ collect data about direct interactions with qbx centers

    @staticmethod
    def _collect_qbxl_direct_interaction_data(direct_interaction_data,
            global_qbx_centers, qbx_center_to_target_box, center_to_targets_starts):
        nlist1_srcs_by_itgt_box = (
                direct_interaction_data["nlist1_srcs_by_itgt_box"])
        nlist3close_srcs_by_itgt_box = (
                direct_interaction_data["nlist3close_srcs_by_itgt_box"])
        nlist4close_srcs_by_itgt_box = (
                direct_interaction_data["nlist4close_srcs_by_itgt_box"])

        # center -> nsources
        np2qbxl_list1_by_center = np.zeros(len(global_qbx_centers), dtype=np.intp)
        np2qbxl_list3_by_center = np.zeros(len(global_qbx_centers), dtype=np.intp)
        np2qbxl_list4_by_center = np.zeros(len(global_qbx_centers), dtype=np.intp)

        # center -> number of associated targets
        nqbxl2p_by_center = np.zeros(len(global_qbx_centers), dtype=np.intp)

        for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
            start, end = center_to_targets_starts[tgt_icenter:tgt_icenter+2]
            nqbxl2p_by_center[itgt_center] = end - start

            itgt_box = qbx_center_to_target_box[tgt_icenter]
            np2qbxl_list1_by_center[itgt_center] = (
                    nlist1_srcs_by_itgt_box[itgt_box])
            np2qbxl_list3_by_center[itgt_center] = (
                    nlist3close_srcs_by_itgt_box[itgt_box])
            np2qbxl_list4_by_center[itgt_center] = (
                    nlist4close_srcs_by_itgt_box[itgt_box])

        result = {}
        result["np2qbxl_list1_by_center"] = np2qbxl_list1_by_center
        result["np2qbxl_list3_by_center"] = np2qbxl_list3_by_center
        result["np2qbxl_list4_by_center"] = np2qbxl_list4_by_center
        result["nqbxl2p_by_center"] = nqbxl2p_by_center

        return result

    # }}}

    # {{{ eval target specific qbx expansions

    def process_eval_target_specific_qbxl(self, xlat_cost, direct_interaction_data,
            global_qbx_centers, qbx_center_to_target_box, center_to_targets_starts):

        counts = self._collect_qbxl_direct_interaction_data(
                direct_interaction_data, global_qbx_centers,
                qbx_center_to_target_box, center_to_targets_starts)

        result = {}
        result["eval_target_specific_qbx_locals_list1"] = (
                sum(counts["np2qbxl_list1_by_center"] * counts["nqbxl2p_by_center"])
                * xlat_cost.p2p_tsqbx())
        result["eval_target_specific_qbx_locals_list3"] = (
                sum(counts["np2qbxl_list3_by_center"] * counts["nqbxl2p_by_center"])
                * xlat_cost.p2p_tsqbx())
        result["eval_target_specific_qbx_locals_list4"] = (
                sum(counts["np2qbxl_list4_by_center"] * counts["nqbxl2p_by_center"])
                * xlat_cost.p2p_tsqbx())

        return result

    # }}}

    # {{{ form global qbx locals

    def process_form_qbxl(self, xlat_cost, direct_interaction_data,
            global_qbx_centers, qbx_center_to_target_box, center_to_targets_starts):

        counts = self._collect_qbxl_direct_interaction_data(
                direct_interaction_data, global_qbx_centers,
                qbx_center_to_target_box, center_to_targets_starts)

        result = {}
        result["form_global_qbx_locals_list1"] = (
                sum(counts["np2qbxl_list1_by_center"]) * xlat_cost.p2qbxl())
        result["form_global_qbx_locals_list3"] = (
                sum(counts["np2qbxl_list3_by_center"]) * xlat_cost.p2qbxl())
        result["form_global_qbx_locals_list4"] = (
                sum(counts["np2qbxl_list4_by_center"]) * xlat_cost.p2qbxl())

        return result

    # }}}

    # {{{ translate from list 3 multipoles to qbx local expansions

    def process_m2qbxl(self, xlat_cost, traversal, tree, global_qbx_centers,
            qbx_center_to_target_box_source_level):
        nm2qbxl_by_source_level = np.zeros(tree.nlevels, dtype=np.intp)

        assert tree.nlevels == len(traversal.from_sep_smaller_by_level)

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):
            for tgt_icenter in global_qbx_centers:
                icontaining_tgt_box = qbx_center_to_target_box_source_level[
                        isrc_level][tgt_icenter]

                if icontaining_tgt_box == -1:
                    continue

                start, stop = (
                        ssn.starts[icontaining_tgt_box],
                        ssn.starts[icontaining_tgt_box+1])

                nm2qbxl_by_source_level[isrc_level] += stop-start

        result = sum(
                cost * xlat_cost.m2qbxl(ilevel)
                for ilevel, cost in enumerate(nm2qbxl_by_source_level))

        return dict(translate_box_multipoles_to_qbx_local=result)

    # }}}

    # {{{ translate from box locals to qbx local expansions

    def process_l2qbxl(self, xlat_cost, traversal, tree, global_qbx_centers,
            qbx_center_to_target_box):
        nl2qbxl_by_level = np.zeros(tree.nlevels, dtype=np.intp)

        for tgt_icenter in global_qbx_centers:
            itgt_box = qbx_center_to_target_box[tgt_icenter]
            tgt_ibox = traversal.target_boxes[itgt_box]
            level = tree.box_levels[tgt_ibox]
            nl2qbxl_by_level[level] += 1

        result = sum(
                cost * xlat_cost.l2qbxl(ilevel)
                for ilevel, cost in enumerate(nl2qbxl_by_level))

        return dict(translate_box_local_to_qbx_local=result)

    # }}}

    # {{{ evaluate qbx local expansions

    def process_eval_qbxl(self, xlat_cost, global_qbx_centers,
            center_to_targets_starts):
        result = 0

        for src_icenter in global_qbx_centers:
            start, end = center_to_targets_starts[src_icenter:src_icenter+2]
            result += (end - start)

        result *= xlat_cost.qbxl2p()

        return dict(eval_qbx_expansions=result)

    # }}}

    @log_process(logger, "model cost")
    def __call__(self, geo_data, kernel, kernel_arguments):
        """Analyze the given geometry and return cost data.

        :returns: An instance of :class:`ParametrizedCosts`.
        """
        # FIXME: This should suport target filtering.

        result = OrderedDict()

        lpot_source = geo_data.lpot_source

        use_tsqbx = lpot_source._use_target_specific_qbx

        with cl.CommandQueue(geo_data.cl_context) as queue:
            tree = geo_data.tree().get(queue)
            traversal = geo_data.traversal(merge_close_lists=False).get(queue)
            nqbtl = geo_data.non_qbx_box_target_lists().get(queue)

        box_target_counts_nonchild = nqbtl.box_target_counts_nonchild

        params = dict(
                nlevels=tree.nlevels,
                nboxes=tree.nboxes,
                nsources=tree.nsources,
                ntargets=tree.ntargets,
                ncenters=geo_data.ncenters,
                p_qbx=lpot_source.qbx_order,
                )

        for ilevel in range(tree.nlevels):
            params["p_fmm_lev%d" % ilevel] = (
                    lpot_source.fmm_level_to_order(
                        kernel.get_base_kernel(), kernel_arguments, tree, ilevel))

        params.update(self.calibration_params)

        xlat_cost = (
                self.translation_cost_model_factory(tree.dimensions, tree.nlevels))

        # {{{ construct local multipoles

        result.update(self.process_form_multipoles(xlat_cost, traversal, tree))

        # }}}

        # {{{ propagate multipoles upward

        result.update(self.process_coarsen_multipoles(xlat_cost, traversal, tree))

        # }}}

        direct_interaction_data = (
                self._collect_direction_interaction_data(traversal, tree))

        # {{{ direct evaluation to point targets (lists 1, 3 close, 4 close)

        result.update(self.process_direct(
                xlat_cost, traversal, direct_interaction_data,
                box_target_counts_nonchild))

        # }}}

        # {{{ translate separated siblings' ("list 2") mpoles to local

        result.update(self.process_list2(xlat_cost, traversal, tree))

        # }}}

        # {{{ evaluate sep. smaller mpoles ("list 3") at particles

        result.update(self.process_list3(
                xlat_cost, traversal, tree, box_target_counts_nonchild))

        # }}}

        # {{{ form locals for separated bigger source boxes ("list 4")

        result.update(self.process_list4(xlat_cost, traversal, tree))

        # }}}

        # {{{ propagate local_exps downward

        result.update(self.process_refine_locals(xlat_cost, traversal, tree))

        # }}}

        # {{{ evaluate locals

        result.update(self.process_eval_locals(xlat_cost, traversal, tree, nqbtl))

        # }}}

        global_qbx_centers = geo_data.global_qbx_centers()

        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        center_to_targets_starts = geo_data.center_to_tree_targets().starts
        qbx_center_to_target_box_source_level = np.empty(
                (tree.nlevels,), dtype=object)

        for src_level in range(tree.nlevels):
            qbx_center_to_target_box_source_level[src_level] = (
                    geo_data.qbx_center_to_target_box_source_level(src_level))

        with cl.CommandQueue(geo_data.cl_context) as queue:
            global_qbx_centers = global_qbx_centers.get(
                    queue=queue)
            qbx_center_to_target_box = qbx_center_to_target_box.get(
                    queue=queue)
            center_to_targets_starts = center_to_targets_starts.get(
                    queue=queue)
            for src_level in range(tree.nlevels):
                qbx_center_to_target_box_source_level[src_level] = (
                        qbx_center_to_target_box_source_level[src_level]
                        .get(queue=queue))

        # {{{ form global qbx locals or evaluate target specific qbx expansions

        if use_tsqbx:
            result.update(self.process_eval_target_specific_qbxl(
                    xlat_cost, direct_interaction_data, global_qbx_centers,
                    qbx_center_to_target_box, center_to_targets_starts))
        else:
            result.update(self.process_form_qbxl(
                    xlat_cost, direct_interaction_data, global_qbx_centers,
                    qbx_center_to_target_box, center_to_targets_starts))

        # }}}

        # {{{ translate from list 3 multipoles to qbx local expansions

        result.update(self.process_m2qbxl(
                xlat_cost, traversal, tree, global_qbx_centers,
                qbx_center_to_target_box_source_level))

        # }}}

        # {{{ translate from box local expansions to qbx local expansions

        result.update(self.process_l2qbxl(
                xlat_cost, traversal, tree, global_qbx_centers,
                qbx_center_to_target_box))

        # }}}

        # {{{ evaluate qbx local expansions

        result.update(self.process_eval_qbxl(
                xlat_cost, global_qbx_centers, center_to_targets_starts))

        # }}}

        return ParametrizedCosts(result, params)

# }}}


# {{{ calibrate cost model

def _collect(expr, variables):
    """Collect terms with respect to a list of variables.

    This applies :func:`sympy.simplify.collect` to the a :mod:`pymbolic` expression
    with respect to the iterable of names in *variables*.

    Returns a dictionary mapping variable names to terms.
    """
    from pymbolic.interop.sympy import PymbolicToSympyMapper, SympyToPymbolicMapper
    p2s = PymbolicToSympyMapper()
    s2p = SympyToPymbolicMapper()

    from sympy.simplify import collect
    sympy_variables = [sp.var(v) for v in variables]
    collect_result = collect(p2s(expr), sympy_variables, evaluate=False)

    result = {}
    for v in variables:
        try:
            result[v] = s2p(collect_result[sp.var(v)])
        except KeyError:
            continue

    return result


_FMM_STAGE_TO_CALIBRATION_PARAMETER = {
        "form_multipoles": "c_p2m",
        "coarsen_multipoles": "c_m2m",
        "eval_direct": "c_p2p",
        "multipole_to_local": "c_m2l",
        "eval_multipoles": "c_m2p",
        "form_locals": "c_p2l",
        "refine_locals": "c_l2l",
        "eval_locals": "c_l2p",
        "form_global_qbx_locals": "c_p2qbxl",
        "translate_box_multipoles_to_qbx_local": "c_m2qbxl",
        "translate_box_local_to_qbx_local": "c_l2qbxl",
        "eval_qbx_expansions": "c_qbxl2p",
        "eval_target_specific_qbx_locals": "c_p2p_tsqbx",
        }


def estimate_calibration_params(model_results, timing_results):
    """Given a set of model results and matching timing results, estimate the best
    calibration parameters for the model.
    """

    params = set(_FMM_STAGE_TO_CALIBRATION_PARAMETER.values())

    nresults = len(model_results)

    if nresults != len(timing_results):
        raise ValueError("must have same number of model and timing results")

    uncalibrated_times = {}
    actual_times = {}

    for param in params:
        uncalibrated_times[param] = np.zeros(nresults)
        actual_times[param] = np.zeros(nresults)

    from pymbolic import evaluate

    for i, model_result in enumerate(model_results):
        context = model_result.params.copy()
        for param in params:
            context[param] = var(param)

        # Represents the total modeled cost, but leaves the calibration
        # parameters symbolic.
        total_modeled_cost = evaluate(
                sum(model_result.raw_costs.values()),
                context=context)

        collected_times = _collect(total_modeled_cost, params)

        for param, time in collected_times.items():
            uncalibrated_times[param][i] = time

    for i, timing_result in enumerate(timing_results):
        for param, time in timing_result.items():
            calibration_param = (
                    _FMM_STAGE_TO_CALIBRATION_PARAMETER[param])
            actual_times[calibration_param][i] = time["process_elapsed"]

    result = {}

    for param in params:
        uncalibrated = uncalibrated_times[param]
        actual = actual_times[param]

        if np.allclose(uncalibrated, 0):
            result[param] = float("NaN")
            continue

        result[param] = (
                actual.dot(uncalibrated) / uncalibrated.dot(uncalibrated))

    return result

# }}}

# vim: foldmethod=marker
