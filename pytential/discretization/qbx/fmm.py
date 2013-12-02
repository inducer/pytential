from __future__ import division

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


import numpy as np  # noqa
import pyopencl as cl  # noqa
import pyopencl.array  # noqa
from sumpy.fmm import SumpyExpansionWranglerCodeContainer, SumpyExpansionWrangler


import logging
logger = logging.getLogger(__name__)

__doc__ = """
.. autoclass:: QBXExpansionWranglerCodeContainer

.. autoclass:: QBXExpansionWrangler

.. autofunction:: drive_fmm
"""


# {{{ expansion wrangler

class QBXExpansionWranglerCodeContainer(SumpyExpansionWranglerCodeContainer):
    def __init__(self, cl_context,
            multipole_expansion, local_expansion, qbx_local_expansion, out_kernels):
        SumpyExpansionWranglerCodeContainer.__init__(self,
                cl_context, multipole_expansion, local_expansion, out_kernels)

        self.qbx_local_expansion = qbx_local_expansion

        from pytential.discretization.qbx.interactions import (
                P2QBXLFromCSR, M2QBXL, L2QBXL, QBXL2P)

        self.p2qbxl = P2QBXLFromCSR(cl_context, qbx_local_expansion)
        self.m2qbxl = M2QBXL(cl_context,
                multipole_expansion, qbx_local_expansion)
        self.l2qbxl = L2QBXL(cl_context,
                local_expansion, qbx_local_expansion)
        self.qbxl2p = QBXL2P(cl_context, qbx_local_expansion, out_kernels)

    def get_wrangler(self, queue, geo_data, dtype, extra_kwargs={}):
        return QBXExpansionWrangler(self, queue, geo_data,
                dtype, extra_kwargs)


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

    def __init__(self, code_container, queue, geo_data,
            dtype, extra_kwargs):
        SumpyExpansionWrangler.__init__(self,
                code_container, queue, geo_data.tree(),
                dtype, extra_kwargs)
        self.geo_data = geo_data

    # {{{ data vector utilities

    def potential_zeros(self):
        """This ought to be called ``non_qbx_potential_zeros``, but since
        it has to override the superclass's behavior to integrate seamlessly,
        it needs to be called just :meth:`potential_zeros`.
        """

        nqbtl = self.geo_data.non_qbx_box_target_lists()

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                cl.array.zeros(
                    self.queue,
                    nqbtl.nfiltered_targets,
                    dtype=self.dtype)
                for k in self.code.out_kernels])

    def full_potential_zeros(self):
        # The superclass generates a full field of zeros, for all
        # (not just non-QBX) targets.
        return SumpyExpansionWrangler.potential_zeros(self)

    def qbx_local_expansion_zeros(self):
        return cl.array.zeros(
                    self.queue,
                    (self.geo_data.center_info().ncenters,
                        len(self.code.qbx_local_expansion)),
                    dtype=self.dtype)

    def reorder_src_weights(self, src_weights):
        return src_weights.with_queue(self.queue)[self.tree.user_point_source_ids]

    def reorder_potentials(self, potentials):
        raise NotImplementedError("reorder_potentials should not "
            "be called on a QBXExpansionWrangler")

        # Because this is a multi-stage, more complicated process that combines
        # potentials from non-QBX targets and QBX targets, reordering takes
        # place in multiple stages below.

    # }}}

    # {{{ source/target dispatch

    def box_source_list_kwargs(self):
        return dict(
                box_source_starts=self.tree.box_point_source_starts,
                box_source_counts_nonchild=
                self.tree.box_point_source_counts_nonchild,
                sources=self.tree.point_sources)

    def box_target_list_kwargs(self):
        # This only covers the non-QBX targets.

        nqbtl = self.geo_data.non_qbx_box_target_lists()
        return dict(
                box_target_starts=nqbtl.box_target_starts,
                box_target_counts_nonchild=
                nqbtl.box_target_counts_nonchild,
                targets=nqbtl.targets)

    # }}}

    # {{{ qbx-related

    def form_global_qbx_locals(self, starts, lists, src_weights):
        local_exps = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return local_exps

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        evt, (result,) = self.code.p2qbxl(self.queue,
                global_qbx_centers=geo_data.global_qbx_centers(),
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                qbx_centers=geo_data.center_info().centers,

                source_box_starts=starts,
                source_box_lists=lists,
                strengths=src_weights,
                qbx_expansions=local_exps,

                **kwargs)

        assert local_exps is result

        return result

    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.center_info().ncenters == 0:
            return qbx_expansions

        traversal = geo_data.traversal()

        evt, (qbx_expansions_res,) = self.code.m2qbxl(self.queue,
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),

                centers=self.tree.box_centers,
                qbx_centers=geo_data.center_info().centers,

                src_expansions=multipole_exps,
                qbx_expansions=qbx_expansions,

                src_box_starts=traversal.sep_smaller_starts,
                src_box_lists=traversal.sep_smaller_lists,

                **self.extra_kwargs)

        assert qbx_expansions_res is qbx_expansions
        return qbx_expansions

    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.center_info().ncenters == 0:
            return qbx_expansions

        evt, (qbx_expansions_res,) = self.code.l2qbxl(self.queue,
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                target_boxes=geo_data.traversal().target_boxes,

                centers=self.tree.box_centers,
                qbx_centers=geo_data.center_info().centers,

                expansions=local_exps,
                qbx_expansions=qbx_expansions,

                **self.extra_kwargs)

        assert qbx_expansions_res is qbx_expansions
        return qbx_expansions

    def eval_qbx_expansions(self, qbx_expansions):
        pot = self.full_potential_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return pot

        ctt = geo_data.center_to_tree_targets()

        evt, pot_res = self.code.qbxl2p(self.queue,
                qbx_centers=geo_data.center_info().centers,
                global_qbx_centers=geo_data.global_qbx_centers(),

                center_to_targets_starts=ctt.starts,
                center_to_targets_lists=ctt.lists,

                targets=self.tree.targets,

                qbx_expansions=qbx_expansions,
                result=pot,

                **self.extra_kwargs.copy())

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot

    # }}}

# }}}


# {{{ FMM top-level

def drive_fmm(expansion_wrangler, src_weights):
    """Top-level driver routine for a fast multipole calculation.

    In part, this is intended as a template for custom FMMs, in the sense that
    you may copy and paste its
    `source code <https://github.com/inducer/boxtree/blob/master/boxtree/fmm.py>`_
    as a starting point.

    Nonetheless, many common applications (such as point-to-point FMMs) can be
    covered by supplying the right *expansion_wrangler* to this routine.

    :arg geo_data: A :class:`QBXFMMGeometryData` instance.
    :arg expansion_wrangler: An object exhibiting the
        :class:`ExpansionWranglerInterface`.
    :arg src_weights: Source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.

    Returns the potentials computed by *expansion_wrangler*.

    See also :func:`boxtree.fmm.drive_fmm`.
    """
    wrangler = expansion_wrangler

    geo_data = wrangler.geo_data
    traversal = geo_data.traversal()
    tree = traversal.tree

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    logger.debug("start qbx fmm")

    logger.debug("reorder source weights")

    src_weights = wrangler.reorder_src_weights(src_weights)

    # {{{ construct local multipoles

    logger.debug("construct local multipoles")

    mpole_exps = wrangler.form_multipoles(
            traversal.source_boxes,
            src_weights)

    # }}}

    # {{{ propagate multipoles upward

    logger.debug("propagate multipoles upward")

    for lev in xrange(tree.nlevels-1, -1, -1):
        start_parent_box, end_parent_box = \
                traversal.level_start_source_parent_box_nrs[lev:lev+2]
        wrangler.coarsen_multipoles(
                traversal.source_parent_boxes[start_parent_box:end_parent_box],
                mpole_exps)

    # }}}

    # {{{ direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")

    non_qbx_potentials = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights)

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")

    local_exps = wrangler.multipole_to_local(
            traversal.target_or_target_parent_boxes,
            traversal.sep_siblings_starts,
            traversal.sep_siblings_lists,
            mpole_exps)

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    non_qbx_potentials = non_qbx_potentials + wrangler.eval_multipoles(
            traversal.target_boxes,
            traversal.sep_smaller_starts,
            traversal.sep_smaller_lists,
            mpole_exps)

    # assert that list 3 close has been merged into list 1
    assert traversal.sep_close_smaller_starts is None

    # }}}

    # {{{ form locals for separated bigger mpoles ("list 4")

    logger.debug("form locals for separated bigger mpoles ('list 4 far')")

    local_exps = local_exps + wrangler.form_locals(
            traversal.target_or_target_parent_boxes,
            traversal.sep_bigger_starts,
            traversal.sep_bigger_lists,
            src_weights)

    # assert that list 4 close has been merged into list 1
    assert traversal.sep_close_bigger_starts is None

    # }}}

    # {{{ propagate local_exps downward

    logger.debug("propagate local_exps downward")

    for lev in xrange(1, tree.nlevels):
        start_box, end_box = \
                traversal.level_start_target_or_target_parent_box_nrs[lev:lev+2]
        wrangler.refine_locals(
                traversal.target_or_target_parent_boxes[start_box:end_box],
                local_exps)

    # }}}

    # {{{ evaluate locals

    logger.debug("evaluate locals")

    non_qbx_potentials = non_qbx_potentials + wrangler.eval_locals(
            traversal.target_boxes,
            local_exps)

    # }}}

    # {{{ wrangle qbx expansions

    logger.debug("form global qbx expansions from list 1")
    qbx_expansions = wrangler.form_global_qbx_locals(
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights)

    logger.debug("translate from list 3 multipoles to qbx local expansions")
    qbx_expansions = qbx_expansions + \
            wrangler.translate_box_multipoles_to_qbx_local(mpole_exps)

    logger.debug("translate from box local expansions to contained "
            "qbx local expansions")
    qbx_expansions = qbx_expansions + \
            wrangler.translate_box_local_to_qbx_local(local_exps)

    logger.debug("evaluate qbx local expansions")
    qbx_potentials = wrangler.eval_qbx_expansions(
            qbx_expansions)

    # }}}

    # {{{ reorder potentials

    logger.debug("reorder potentials")

    nqbtl = geo_data.non_qbx_box_target_lists()

    from pytools.obj_array import make_obj_array
    all_potentials_in_tree_order = make_obj_array([
            cl.array.zeros(
                wrangler.queue,
                tree.ntargets,
                dtype=wrangler.dtype)
            for k in wrangler.code.out_kernels])

    for ap_i, nqp_i in zip(all_potentials_in_tree_order, non_qbx_potentials):
        ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

    all_potentials_in_tree_order += qbx_potentials

    def reorder_potentials(x):
        return x[tree.sorted_target_ids]

    from pytools.obj_array import with_object_array_or_scalar
    result = with_object_array_or_scalar(
            reorder_potentials, all_potentials_in_tree_order)

    # }}}

    logger.debug("qbx fmm complete")

    return result

# }}}


# vim: foldmethod=marker
