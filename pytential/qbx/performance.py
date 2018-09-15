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


from six.moves import range
import numpy as np  # noqa
import pyopencl as cl  # noqa
import pyopencl.array  # noqa
import sympy as sp


from pymbolic import var
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: PerformanceModel
.. autoclass:: ParametrizedCosts

.. autofunction:: estimate_calibration_params
"""


# {{{ translation cost model

class TranslationCostModel(object):
    """Provides modeled costs for individual translations or evaluations."""

    def __init__(self, p_qbx, p_fmm, ncoeffs_qbx, ncoeffs_fmm,
                 translation_source_power, translation_target_power,
                 translation_max_power):
        self.p_qbx = p_qbx
        self.p_fmm = p_fmm
        self.ncoeffs_qbx = ncoeffs_qbx
        self.ncoeffs_fmm = ncoeffs_fmm
        self.translation_source_power = translation_source_power
        self.translation_target_power = translation_target_power
        self.translation_max_power = translation_max_power

    def direct(self):
        return var("c_p2p")

    def p2qbxl(self):
        return var("c_p2qbxl") * self.ncoeffs_qbx

    def qbxl2p(self):
        return var("c_qbxl2p") * self.ncoeffs_qbx

    def p2l(self):
        return var("c_p2l") * self.ncoeffs_fmm

    def l2p(self):
        return var("c_l2p") * self.ncoeffs_fmm

    def p2m(self):
        return var("c_p2m") * self.ncoeffs_fmm

    def m2p(self):
        return var("c_m2p") * self.ncoeffs_fmm

    def m2m(self):
        return var("c_m2m") * self.e2e_cost(self.p_fmm, self.p_fmm)

    def l2l(self):
        return var("c_l2l") * self.e2e_cost(self.p_fmm, self.p_fmm)

    def m2l(self):
        return var("c_m2l") * self.e2e_cost(self.p_fmm, self.p_fmm)

    def m2qbxl(self):
        return var("c_m2qbxl") * self.e2e_cost(self.p_fmm, self.p_qbx)

    def l2qbxl(self):
        return var("c_l2qbxl") * self.e2e_cost(self.p_fmm, self.p_qbx)

    def e2e_cost(self, p_source, p_target):
        from pymbolic.primitives import Max
        return (
                p_source ** self.translation_source_power
                * p_target ** self.translation_target_power
                * Max((p_source, p_target)) ** self.translation_max_power)

# }}}


# {{{ parameterized costs returned by performance model

class ParametrizedCosts(object):
    """A container for data returned by the performance model.

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


# {{{ performance model

class PerformanceModel(object):
    """
    .. automethod:: with_calibration_params
    .. automethod:: __call__
    """

    def __init__(self,
            uses_pde_expansions=True,
            translation_source_power=None,
            translation_target_power=None,
            translation_max_power=None,
            summarize_parallel=None,
            calibration_params=None):
        """
        :arg uses_pde_expansions: A :class:`bool` indicating whether the FMM
            uses translation operators that make use of the knowledge that the
            potential satisfies a PDE.

        :arg summarize_parallel: a function of two arguments
            *(parallel_array, sym_multipliers)* used to model the cost after
            taking into account parallelization. *parallel_array* represents a
            partitioning of the work into elementary (typically box-based) tasks,
            each with a given number of operations. *sym_multipliers* is a symbolic
            value representing time per modeled operation. By default, all tasks
            are summed into one number encompassing the total cost.
        """
        self.uses_pde_expansions = uses_pde_expansions
        self.translation_source_power = translation_source_power
        self.translation_target_power = translation_target_power
        self.translation_max_power = translation_max_power
        if summarize_parallel is None:
            summarize_parallel = self.summarize_parallel_default
        self.summarize_parallel = summarize_parallel
        if calibration_params is None:
            calibration_params = dict()
        self.calibration_params = calibration_params

    def with_calibration_params(self, calibration_params):
        """Return a copy of *self* with a new set of calibration parameters."""
        return type(self)(
                uses_pde_expansions=self.uses_pde_expansions,
                translation_source_power=self.translation_source_power,
                translation_target_power=self.translation_target_power,
                translation_max_power=self.translation_max_power,
                summarize_parallel=self.summarize_parallel,
                calibration_params=calibration_params)

    @staticmethod
    def summarize_parallel_default(parallel_array, sym_multipliers):
        return np.sum(parallel_array) * sym_multipliers

    # {{{ propagate multipoles upward

    def process_coarsen_multipoles(self, xlat_cost, tree, traversal):
        nmultipoles = 0

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            start, stop = traversal.level_start_source_parent_box_nrs[
                            target_level:target_level+2]
            for ibox in traversal.source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        nmultipoles += 1

        return dict(coarsen_multipoles=(
                self.summarize_parallel(nmultipoles, xlat_cost.m2m())))

    # }}}

    # {{{ direct evaluation to point targets (lists 1, 3 close, 4 close)

    def process_direct(self, xlat_cost, traversal, tree, box_target_counts_nonchild):
        # box -> nsources * ntargets
        npart_direct_list1 = np.zeros(len(traversal.target_boxes), dtype=np.intp)
        npart_direct_list3 = np.zeros(len(traversal.target_boxes), dtype=np.intp)
        npart_direct_list4 = np.zeros(len(traversal.target_boxes), dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(traversal.target_boxes):
            ntargets = box_target_counts_nonchild[tgt_ibox]

            npart_direct_list1_srcs = 0
            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nsources = tree.box_source_counts_nonchild[src_ibox]

                npart_direct_list1_srcs += nsources

            npart_direct_list1[itgt_box] = ntargets * npart_direct_list1_srcs

            npart_direct_list3_srcs = 0

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                        traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nsources = tree.box_source_counts_nonchild[src_ibox]

                    npart_direct_list3_srcs += nsources

            npart_direct_list3[itgt_box] = ntargets * npart_direct_list3_srcs

            npart_direct_list4_srcs = 0

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                        traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nsources = tree.box_source_counts_nonchild[src_ibox]

                    npart_direct_list4_srcs += nsources

            npart_direct_list4[itgt_box] = ntargets * npart_direct_list4_srcs

        result = {}
        result["eval_direct_list1"] = (
                self.summarize_parallel(npart_direct_list1, xlat_cost.direct()))
        result["eval_direct_list3"] = (
                self.summarize_parallel(npart_direct_list3, xlat_cost.direct()))
        result["eval_direct_list4"] = (
                self.summarize_parallel(npart_direct_list4, xlat_cost.direct()))

        return result

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    def process_list2(self, xlat_cost, traversal):
        nm2l = np.zeros(len(traversal.target_or_target_parent_boxes), dtype=np.intp)

        for itgt_box in range(len(traversal.target_or_target_parent_boxes)):
            start, end = traversal.from_sep_siblings_starts[itgt_box:itgt_box+2]

            nm2l[itgt_box] += end-start

        return dict(multipole_to_local=(
                self.summarize_parallel(nm2l, xlat_cost.m2l())))

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    def process_list3(self, xlat_cost, traversal, tree, box_target_counts_nonchild):
        nmp_eval = np.zeros(
                (tree.nlevels, len(traversal.target_boxes)),
                dtype=np.intp)

        assert tree.nlevels == len(traversal.from_sep_smaller_by_level)

        for ilevel, sep_smaller_list in enumerate(
                traversal.from_sep_smaller_by_level):
            for itgt_box, tgt_ibox in enumerate(
                        traversal.target_boxes_sep_smaller_by_source_level[ilevel]):
                ntargets = box_target_counts_nonchild[tgt_ibox]
                start, end = sep_smaller_list.starts[itgt_box:itgt_box+2]
                nmp_eval[ilevel, sep_smaller_list.nonempty_indices[itgt_box]] = (
                        ntargets * (end-start)
                )

        return dict(eval_multipoles=(
                self.summarize_parallel(nmp_eval, xlat_cost.m2p())))

    # }}}

    # {{{ form locals for separated bigger source boxes ("list 4")

    def process_list4(self, xlat_cost, traversal, tree):
        nform_local = np.zeros(
                len(traversal.target_or_target_parent_boxes),
                dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(traversal.target_or_target_parent_boxes):
            start, end = traversal.from_sep_bigger_starts[itgt_box:itgt_box+2]

            nform_local_box = 0
            for src_ibox in traversal.from_sep_bigger_lists[start:end]:
                nsources = tree.box_source_counts_nonchild[src_ibox]

                nform_local_box += nsources

            nform_local[itgt_box] = nform_local_box

        return dict(form_locals=(
                self.summarize_parallel(nform_local, xlat_cost.p2l())))

    # }}}

    # {{{ form global qbx locals

    def process_form_qbxl(self, xlat_cost, traversal, tree, global_qbx_centers,
            qbx_center_to_target_box):

        # center -> nsources
        np2qbxl_list1 = np.zeros(len(global_qbx_centers), dtype=np.intp)
        np2qbxl_list3 = np.zeros(len(global_qbx_centers), dtype=np.intp)
        np2qbxl_list4 = np.zeros(len(global_qbx_centers), dtype=np.intp)

        for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
            itgt_box = qbx_center_to_target_box[tgt_icenter]

            np2qbxl_list1_srcs = 0
            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nsources = tree.box_source_counts_nonchild[src_ibox]

                np2qbxl_list1_srcs += nsources

            np2qbxl_list1[itgt_center] = np2qbxl_list1_srcs

            np2qbxl_list3_srcs = 0

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                        traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nsources = tree.box_source_counts_nonchild[src_ibox]

                    np2qbxl_list3_srcs += nsources

            np2qbxl_list3[itgt_center] = np2qbxl_list3_srcs

            np2qbxl_list4_srcs = 0

            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                        traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nsources = tree.box_source_counts_nonchild[src_ibox]

                    np2qbxl_list4_srcs += nsources

            np2qbxl_list4[itgt_center] = np2qbxl_list4_srcs

        result = {}
        result["form_global_qbx_locals_list1"] = (
                self.summarize_parallel(np2qbxl_list1, xlat_cost.p2qbxl()))
        result["form_global_qbx_locals_list3"] = (
                self.summarize_parallel(np2qbxl_list3, xlat_cost.p2qbxl()))
        result["form_global_qbx_locals_list4"] = (
                self.summarize_parallel(np2qbxl_list4, xlat_cost.p2qbxl()))

        return result

    # }}}

    # {{{ translate from list 3 multipoles to qbx local expansions

    def process_m2qbxl(self, xlat_cost, traversal, tree, global_qbx_centers,
            qbx_center_to_target_box_source_level):
        nm2qbxl = np.zeros(
                (tree.nlevels, len(global_qbx_centers)),
                dtype=np.intp)

        assert tree.nlevels == len(traversal.from_sep_smaller_by_level)

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):

            for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
                icontaining_tgt_box = qbx_center_to_target_box_source_level[
                    isrc_level][tgt_icenter]

                if icontaining_tgt_box == -1:
                    continue

                start, stop = (
                        ssn.starts[icontaining_tgt_box],
                        ssn.starts[icontaining_tgt_box+1])

                nm2qbxl[isrc_level, itgt_center] += stop-start

        return dict(translate_box_multipoles_to_qbx_local=(
                self.summarize_parallel(nm2qbxl, xlat_cost.m2qbxl())))

    # }}}

    # {{{ evaluate qbx local expansions

    def process_eval_qbxl(self, xlat_cost, global_qbx_centers,
            center_to_targets_starts):
        nqbx_eval = np.zeros(len(global_qbx_centers), dtype=np.intp)

        for isrc_center, src_icenter in enumerate(global_qbx_centers):
            start, end = center_to_targets_starts[src_icenter:src_icenter+2]
            nqbx_eval[isrc_center] += end-start

        return dict(eval_qbx_expansions=(
                self.summarize_parallel(nqbx_eval, xlat_cost.qbxl2p())))

    # }}}

    # {{{ set up translation cost model

    def get_translation_cost_model(self, d):
        from pymbolic import var
        p_qbx = var("p_qbx")
        p_fmm = var("p_fmm")

        if self.uses_pde_expansions:
            ncoeffs_fmm = p_fmm ** (d-1)
            ncoeffs_qbx = p_qbx ** (d-1)

            if d == 2:
                default_translation_source_power = 1
                default_translation_target_power = 1
                default_translation_max_power = 0

            elif d == 3:
                # Based on a reading of FMMlib, i.e. a point-and-shoot FMM.
                default_translation_source_power = 0
                default_translation_target_power = 0
                default_translation_max_power = 3

            else:
                raise ValueError("Don't know how to estimate expansion complexities "
                        "for dimension %d" % d)

        else:
            ncoeffs_fmm = p_fmm ** d
            ncoeffs_qbx = p_qbx ** d
            default_translation_source_power = d
            default_translation_target_power = d
            default_translation_max_power = 0

        translation_source_power = (
                default_translation_source_power
                if self.translation_source_power is None
                else self.translation_source_power)

        translation_target_power = (
                default_translation_target_power
                if self.translation_target_power is None
                else self.translation_target_power)

        translation_max_power = (
                default_translation_max_power
                if self.translation_max_power is None
                else self.translation_max_power)

        return TranslationCostModel(
                p_qbx=p_qbx,
                p_fmm=p_fmm,
                ncoeffs_qbx=ncoeffs_qbx,
                ncoeffs_fmm=ncoeffs_fmm,
                translation_source_power=translation_source_power,
                translation_target_power=translation_target_power,
                translation_max_power=translation_max_power)

    # }}}

    def __call__(self, geo_data):
        """Analyze the given geometry and return performance data.

        :returns: An instance of :class:`ParametrizedCosts`.
        """
        # FIXME: This should suport target filtering.

        result = OrderedDict()

        lpot_source = geo_data.lpot_source

        nqbtl = geo_data.non_qbx_box_target_lists()

        with cl.CommandQueue(geo_data.cl_context) as queue:
            tree = geo_data.tree().get(queue=queue)
            traversal = geo_data.traversal(merge_close_lists=False).get(queue=queue)
            box_target_counts_nonchild = (
                    nqbtl.box_target_counts_nonchild.get(queue=queue))

        params = dict(
                nlevels=tree.nlevels,
                nboxes=tree.nboxes,
                nsources=tree.nsources,
                ntargets=tree.ntargets,
                ncenters=geo_data.ncenters,
                p_qbx=lpot_source.qbx_order,
                # FIXME: Assumes this is a constant
                p_fmm=lpot_source.fmm_level_to_order(None, None, None, None),
                )

        params.update(self.calibration_params)

        xlat_cost = self.get_translation_cost_model(tree.dimensions)

        # {{{ construct local multipoles

        result["form_multipoles"] = tree.nsources * xlat_cost.p2m()

        # }}}

        # {{{ propagate multipoles upward

        result.update(self.process_coarsen_multipoles(xlat_cost, tree, traversal))

        # }}}

        # {{{ direct evaluation to point targets (lists 1, 3 close, 4 close)

        result.update(self.process_direct(
                xlat_cost, traversal, tree, box_target_counts_nonchild))

        # }}}

        # {{{ translate separated siblings' ("list 2") mpoles to local

        result.update(self.process_list2(xlat_cost, traversal))

        # }}}

        # {{{ evaluate sep. smaller mpoles ("list 3") at particles

        result.update(self.process_list3(
                xlat_cost, traversal, tree, box_target_counts_nonchild))

        # }}}

        # {{{ form locals for separated bigger source boxes ("list 4")

        result.update(self.process_list4(xlat_cost, traversal, tree))

        # }}}

        # {{{ propagate local_exps downward

        result["refine_locals"] = (
                # Don't count the root box.
                max(traversal.ntarget_or_target_parent_boxes - 1, 0)
                * xlat_cost.l2l())

        # }}}

        # {{{ evaluate locals

        result["eval_locals"] = nqbtl.nfiltered_targets * xlat_cost.l2p()

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

        # {{{ form global qbx locals

        result.update(self.process_form_qbxl(
                xlat_cost, traversal, tree, global_qbx_centers,
                qbx_center_to_target_box))

        # }}}

        # {{{ translate from list 3 multipoles to qbx local expansions

        result.update(self.process_m2qbxl(
                xlat_cost, traversal, tree, global_qbx_centers,
                qbx_center_to_target_box_source_level))

        # }}}

        # {{{ translate from box local expansions to qbx local expansions

        result["translate_box_local_to_qbx_local"] = (
                len(global_qbx_centers) * xlat_cost.l2qbxl())

        # }}}

        # {{{ evaluate qbx local expansions

        result.update(self.process_eval_qbxl(
                xlat_cost, global_qbx_centers, center_to_targets_starts))

        # }}}

        return ParametrizedCosts(result, params)

# }}}


# {{{ calibrate performance model

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
