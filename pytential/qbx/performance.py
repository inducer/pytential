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


import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: PerformanceModel
.. autofunction:: assemble_performance_data
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
        return 1

    def p2qbxl(self):
        return self.ncoeffs_qbx

    qbxl2p = p2qbxl

    def p2l(self):
        return self.ncoeffs_fmm

    l2p = p2l
    p2m = p2l
    m2p = p2l

    def m2m(self):
        return self.e2e_cost(self.p_fmm, self.p_fmm)

    l2l = m2m
    m2l = m2m

    def m2qbxl(self):
        return self.e2e_cost(self.p_fmm, self.p_qbx)

    l2qbxl = m2qbxl

    def e2e_cost(self, p_source, p_target):
        from pymbolic.primitives import Max
        return (
                p_source ** self.translation_source_power
                * p_target ** self.translation_target_power
                * Max((p_source, p_target)) ** self.translation_max_power)

# }}}


# {{{ performance model

class PerformanceModel(object):

    def __init__(self,
            uses_pde_expansions=True,
            translation_source_power=None,
            translation_target_power=None,
            translation_max_power=None,
            summarize_parallel=None,
            merge_close_lists=True):
        """
        :arg uses_pde_expansions: A :class:`bool` indicating whether the FMM
            uses translation operators that make use of the knowledge that the
            potential satisfies a PDE.
        :arg summarize_parallel: a function of two arguments
            *(parallel_array, sym_multipliers)* used to process an array of
            workloads of 'parallelizable units'. By default, all workloads are
            summed into one number encompassing the total workload.
        :arg merge_close_lists: A :class:`bool` indicating whether or not all
            boxes requiring direct evaluation should be merged into a single
            interaction list. If *False*, *part_direct* and *p2qbxl* will be
            suffixed with the originating list as follows:

            * *_neighbor* (List 1)
            * *_sep_smaller* (List 3 close)
            * *_sep_bigger* (List 4 close).
        """
        self.uses_pde_expansions = uses_pde_expansions
        self.translation_source_power = translation_source_power
        self.translation_target_power = translation_target_power
        self.translation_max_power = translation_max_power
        if summarize_parallel is None:
            summarize_parallel = self.summarize_parallel_default
        self.summarize_parallel = summarize_parallel
        self.merge_close_lists = merge_close_lists

    @staticmethod
    def summarize_parallel_default(parallel_array, sym_multipliers):
        return np.sum(parallel_array) * sym_multipliers

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

            if self.merge_close_lists:
                continue

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
        if self.merge_close_lists:
            result["part_direct"] = (
                    self.summarize_parallel(npart_direct_list1, xlat_cost.direct()))
        else:
            result["part_direct_neighbor"] = (
                    self.summarize_parallel(npart_direct_list1, xlat_cost.direct()))
            result["part_direct_sep_smaller"] = (
                    self.summarize_parallel(npart_direct_list3, xlat_cost.direct()))
            result["part_direct_sep_bigger"] = (
                    self.summarize_parallel(npart_direct_list4, xlat_cost.direct()))

        return result

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    def process_list2(self, xlat_cost, traversal):
        nm2l = np.zeros(len(traversal.target_or_target_parent_boxes), dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(traversal.target_or_target_parent_boxes):
            start, end = traversal.from_sep_siblings_starts[itgt_box:itgt_box+2]

            nm2l[itgt_box] += end-start

        return dict(m2l=self.summarize_parallel(nm2l, xlat_cost.m2l()))

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

        return dict(
                mp_eval=self.summarize_parallel(nmp_eval, xlat_cost.m2p()))

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

        return dict(form_local=(
                self.summarize_parallel(nform_local, xlat_cost.p2l())))

    # }}}

    # {{{ form global qbx locals

    def process_form_qbxl(self, xlat_cost, traversal, tree, global_qbx_centers,
            qbx_center_to_target_box, ncenters):
        result = {}

        result["ncenters"] = ncenters

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

            if self.merge_close_lists:
                continue

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

        if self.merge_close_lists:
            result["p2qbxl"] = (
                    self.summarize_parallel(np2qbxl_list1, xlat_cost.p2qbxl()))
        else:
            result["p2qbxl_neighbor"] = (
                    self.summarize_parallel(np2qbxl_list1, xlat_cost.p2qbxl()))
            result["p2qbxl_sep_smaller"] = (
                    self.summarize_parallel(np2qbxl_list3, xlat_cost.p2qbxl()))
            result["p2qbxl_sep_bigger"] = (
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

        return dict(m2qbxl=self.summarize_parallel(nm2qbxl, xlat_cost.m2qbxl()))

    # }}}

    # {{{ evaluate qbx local expansions

    def process_eval_qbxl(self, xlat_cost, global_qbx_centers,
            center_to_targets_starts):
        nqbx_eval = np.zeros(len(global_qbx_centers), dtype=np.intp)

        for isrc_center, src_icenter in enumerate(global_qbx_centers):
            start, end = center_to_targets_starts[src_icenter:src_icenter+2]
            nqbx_eval[isrc_center] += end-start

        return dict(qbxl2p=self.summarize_parallel(nqbx_eval, xlat_cost.qbxl2p()))

    # }}}

    def __call__(self, geo_data):
        # FIXME: This should suport target filtering.

        from collections import OrderedDict
        result = OrderedDict()

        nqbtl = geo_data.non_qbx_box_target_lists()

        with cl.CommandQueue(geo_data.cl_context) as queue:
            tree = geo_data.tree().get(queue=queue)
            traversal = geo_data.traversal(self.merge_close_lists).get(queue=queue)
            box_target_counts_nonchild = (
                    nqbtl.box_target_counts_nonchild.get(queue=queue))

        # {{{ set up translation cost model

        from pymbolic import var
        p_qbx = var("p_qbx")
        p_fmm = var("p_fmm")

        d = tree.dimensions
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

        xlat_cost = TranslationCostModel(
                p_qbx=p_qbx,
                p_fmm=p_fmm,
                ncoeffs_qbx=ncoeffs_qbx,
                ncoeffs_fmm=ncoeffs_fmm,
                translation_source_power=translation_source_power,
                translation_target_power=translation_target_power,
                translation_max_power=translation_max_power)

        # }}}

        result.update(
                nlevels=tree.nlevels,
                nboxes=tree.nboxes,
                nsources=tree.nsources,
                ntargets=tree.ntargets)

        # {{{ construct local multipoles

        result["form_mp"] = tree.nsources * xlat_cost.p2m()

        # }}}

        # {{{ propagate multipoles upward

        result["prop_upward"] = tree.nboxes * xlat_cost.m2m()

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

        result["prop_downward"] = tree.nboxes * xlat_cost.l2l()

        # }}}

        # {{{ evaluate locals

        result["eval_part"] = tree.ntargets * xlat_cost.l2p()

        # }}}

        global_qbx_centers = geo_data.global_qbx_centers()

        # If merge_close_lists is False above, then this builds another traversal
        # (which is OK).
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        center_to_targets_starts = geo_data.center_to_tree_targets().starts
        qbx_center_to_target_box_source_level = np.empty(
            (tree.nlevels,), dtype=object
        )

        for src_level in range(tree.nlevels):
            qbx_center_to_target_box_source_level[src_level] = (
                geo_data.qbx_center_to_target_box_source_level(src_level)
            )

        with cl.CommandQueue(geo_data.cl_context) as queue:
            global_qbx_centers = global_qbx_centers.get(
                    queue=queue)
            qbx_center_to_target_box = qbx_center_to_target_box.get(
                    queue=queue)
            center_to_targets_starts = center_to_targets_starts.get(
                    queue=queue)
            for src_level in range(tree.nlevels):
                qbx_center_to_target_box_source_level[src_level] = (
                    qbx_center_to_target_box_source_level[src_level].get(queue=queue)
                )

        # {{{ form global qbx locals

        result.update(self.process_form_qbxl(
                xlat_cost, traversal, tree, global_qbx_centers,
                qbx_center_to_target_box, geo_data.ncenters))

        # }}}

        # {{{ translate from list 3 multipoles to qbx local expansions

        result.update(self.process_m2qbxl(
                xlat_cost, traversal, tree, global_qbx_centers,
                qbx_center_to_target_box_source_level))

        # }}}

        # {{{ translate from box local expansions to qbx local expansions

        result["l2qbxl"] = geo_data.ncenters * xlat_cost.l2qbxl()

        # }}}

        # {{{ evaluate qbx local expansions

        result.update(self.process_eval_qbxl(
                xlat_cost, global_qbx_centers, center_to_targets_starts))

        # }}}

        return result

# }}}


# {{{ assemble_performance_data

def assemble_performance_data(geo_data, uses_pde_expansions,
        translation_source_power=None, translation_target_power=None,
        translation_max_power=None,
        summarize_parallel=None, merge_close_lists=True):
    """
    :arg uses_pde_expansions: A :class:`bool` indicating whether the FMM
        uses translation operators that make use of the knowledge that the
        potential satisfies a PDE.
    :arg summarize_parallel: a function of two arguments
        *(parallel_array, sym_multipliers)* used to process an array of
        workloads of 'parallelizable units'. By default, all workloads are
        summed into one number encompassing the total workload.
    :arg merge_close_lists: A :class:`bool` indicating whether or not all
        boxes requiring direct evaluation should be merged into a single
        interaction list. If *False*, *part_direct* and *p2qbxl* will be
        suffixed with the originating list as follows:

        * *_neighbor* (List 1)
        * *_sep_smaller* (List 3 close)
        * *_sep_bigger* (List 4 close).
    """

    return PerformanceModel(
            uses_pde_expansions,
            translation_source_power,
            translation_target_power,
            translation_max_power,
            summarize_parallel,
            merge_close_lists)(geo_data)

# }}}

# vim: foldmethod=marker
