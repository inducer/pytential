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
from sumpy.fmm import SumpyExpansionWranglerCodeContainer, SumpyExpansionWrangler

from pytools import memoize_method
from pytential.qbx.interactions import P2QBXLFromCSR, M2QBXL, L2QBXL, QBXL2P

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: QBXExpansionWranglerCodeContainer
.. autoclass:: QBMXExpansionWranglerCodeContainer

.. autoclass:: QBXExpansionWrangler
.. autoclass:: QBMXExpansionWrangler

.. autofunction:: drive_qbx_fmm
.. autofunction:: drive_qbmx_fmm
"""


# {{{ expansion wrangler

class QBXExpansionWranglerCodeContainer(SumpyExpansionWranglerCodeContainer):
    def __init__(self, cl_context,
            multipole_expansion_factory, local_expansion_factory,
            qbx_local_expansion_factory, out_kernels):
        SumpyExpansionWranglerCodeContainer.__init__(self,
                cl_context, multipole_expansion_factory, local_expansion_factory,
                out_kernels)

        self.qbx_local_expansion_factory = qbx_local_expansion_factory

    @memoize_method
    def qbx_local_expansion(self, order):
        return self.qbx_local_expansion_factory(order)

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
        order = self.qbx_order
        qbx_l_expn = self.code.qbx_local_expansion(order)

        return cl.array.zeros(
                    self.queue,
                    (self.geo_data.center_info().ncenters,
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

    def box_source_list_kwargs(self):
        return dict(
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=(
                    self.tree.box_source_counts_nonchild),
                sources=self.tree.sources)

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

    def form_global_qbx_locals(self, starts, lists, src_weights):
        local_exps = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return local_exps

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        p2qbxl = self.code.p2qbxl(self.qbx_order)

        evt, (result,) = p2qbxl(
                self.queue,
                global_qbx_centers=geo_data.global_qbx_centers(),
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                qbx_centers=geo_data.center_info().centers,

                source_box_starts=starts,
                source_box_lists=lists,
                strengths=src_weights,
                qbx_expansions=local_exps,

                **kwargs)

        assert local_exps is result
        result.add_event(evt)

        return result

    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.center_info().ncenters == 0:
            return qbx_expansions

        traversal = geo_data.traversal()

        wait_for = multipole_exps.events

        for isrc_level, ssn in enumerate(traversal.sep_smaller_by_level):
            m2qbxl = self.code.m2qbxl(
                    self.level_orders[isrc_level],
                    self.qbx_order)

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(multipole_exps, isrc_level)

            evt, (qbx_expansions_res,) = m2qbxl(self.queue,
                    qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),

                    centers=self.tree.box_centers,
                    qbx_centers=geo_data.center_info().centers,

                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    qbx_expansions=qbx_expansions,

                    src_box_starts=ssn.starts,
                    src_box_lists=ssn.lists,

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return qbx_expansions

    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.center_info().ncenters == 0:
            return qbx_expansions
        trav = geo_data.traversal()

        wait_for = local_exps.events

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
                    qbx_centers=geo_data.center_info().centers,

                    expansions=target_locals_view,
                    qbx_expansions=qbx_expansions,

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return qbx_expansions

    def eval_qbx_expansions(self, qbx_expansions):
        pot = self.full_potential_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return pot

        ctt = geo_data.center_to_tree_targets()

        qbxl2p = self.code.qbxl2p(self.qbx_order)

        evt, pot_res = qbxl2p(self.queue,
                qbx_centers=geo_data.center_info().centers,
                global_qbx_centers=geo_data.global_qbx_centers(),

                center_to_targets_starts=ctt.starts,
                center_to_targets_lists=ctt.lists,

                targets=self.tree.targets,

                qbx_expansions=qbx_expansions,
                result=pot,

                **self.kernel_extra_kwargs.copy())

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot

    # }}}


class QBMXExpansionWranglerCodeContainer(SumpyExpansionWranglerCodeContainer):
    def __init__(self, cl_context,
            multipole_expansion_factory, local_expansion_factory,
            qbx_local_expansion_factory, out_kernels):
        SumpyExpansionWranglerCodeContainer.__init__(self,
                cl_context, multipole_expansion_factory, local_expansion_factory,
                out_kernels)

        self.qbx_local_expansion_factory = qbx_local_expansion_factory

    @memoize_method
    def qbx_multiple_expansion(self, order):
        return self.qbx_multipole_expansion_factory(order)

    @memoize_method
    def p2qbxm(self, order):
        return P2QBXLFromCSR(self.cl_context,
                self.qbx_local_expansion(order))

    @memoize_method
    def qbxm2m(self, source_order, target_order):
        return M2QBXL(self.cl_context,
                self.multipole_expansion_factory(source_order),
                self.qbx_local_expansion_factory(target_order))

    def get_wrangler(self, queue, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs={},
            kernel_extra_kwargs=None):
        return QBXExpansionWrangler(self, queue, geo_data,
                dtype,
                qbx_order, fmm_level_to_order,
                source_extra_kwargs,
                kernel_extra_kwargs)


class QBMXExpansionWrangler(SumpyExpansionWrangler):
    """A specialized implementation of the
    :class:`boxtree.fmm.ExpansionWranglerInterface` for the QBX FMM.
    The conventional ('point') FMM is carried out on a filtered
    set of targets
    (see :meth:`pytential.discretization.qbx.geometry.\
QBXFMMGeometryData.non_qbx_box_target_lists`),
    and thus all *non-QBX* potential arrays handled by this wrangler don't
    include all targets in the tree, just the non-QBX ones.

    .. rubric:: QBMX-specific methods

    .. automethod:: form_global_qbx_multipoles

    .. automethod:: translate_qbx_multipole_to_box_multipole
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
        order = self.qbx_order
        qbx_l_expn = self.code.qbx_local_expansion(order)

        return cl.array.zeros(
                    self.queue,
                    (self.geo_data.center_info().ncenters,
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

    def box_source_list_kwargs(self):
        return dict(
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=(
                    self.tree.box_source_counts_nonchild),
                sources=self.tree.sources)

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

    def form_global_qbx_locals(self, starts, lists, src_weights):
        local_exps = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return local_exps

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        p2qbxl = self.code.p2qbxl(self.qbx_order)

        evt, (result,) = p2qbxl(
                self.queue,
                global_qbx_centers=geo_data.global_qbx_centers(),
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                qbx_centers=geo_data.center_info().centers,

                source_box_starts=starts,
                source_box_lists=lists,
                strengths=src_weights,
                qbx_expansions=local_exps,

                **kwargs)

        assert local_exps is result
        result.add_event(evt)

        return result

    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.center_info().ncenters == 0:
            return qbx_expansions

        traversal = geo_data.traversal()

        wait_for = multipole_exps.events

        for isrc_level, ssn in enumerate(traversal.sep_smaller_by_level):
            m2qbxl = self.code.m2qbxl(
                    self.level_orders[isrc_level],
                    self.qbx_order)

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(multipole_exps, isrc_level)

            evt, (qbx_expansions_res,) = m2qbxl(self.queue,
                    qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),

                    centers=self.tree.box_centers,
                    qbx_centers=geo_data.center_info().centers,

                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    qbx_expansions=qbx_expansions,

                    src_box_starts=ssn.starts,
                    src_box_lists=ssn.lists,

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return qbx_expansions

    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.center_info().ncenters == 0:
            return qbx_expansions
        trav = geo_data.traversal()

        wait_for = local_exps.events

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
                    qbx_centers=geo_data.center_info().centers,

                    expansions=target_locals_view,
                    qbx_expansions=qbx_expansions,

                    wait_for=wait_for,

                    **self.kernel_extra_kwargs)

            wait_for = [evt]
            assert qbx_expansions_res is qbx_expansions

        qbx_expansions.add_event(evt)

        return qbx_expansions

    def eval_qbx_expansions(self, qbx_expansions):
        pot = self.full_potential_zeros()

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return pot

        ctt = geo_data.center_to_tree_targets()

        qbxl2p = self.code.qbxl2p(self.qbx_order)

        evt, pot_res = qbxl2p(self.queue,
                qbx_centers=geo_data.center_info().centers,
                global_qbx_centers=geo_data.global_qbx_centers(),

                center_to_targets_starts=ctt.starts,
                center_to_targets_lists=ctt.lists,

                targets=self.tree.targets,

                qbx_expansions=qbx_expansions,
                result=pot,

                **self.kernel_extra_kwargs.copy())

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot

    # }}}

# }}}


# {{{ FMM top-level

def drive_qbx_fmm(expansion_wrangler, src_weights):
    """Top-level driver routine for the QBX fast multipole calculation.

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
    src_weights = wrangler.reorder_sources(src_weights)

    # {{{ construct local multipoles

    logger.debug("construct local multipoles")
    mpole_exps = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weights)

    # }}}

    # {{{ propagate multipoles upward

    logger.debug("propagate multipoles upward")
    wrangler.coarsen_multipoles(
            traversal.level_start_source_parent_box_nrs,
            traversal.source_parent_boxes,
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
            traversal.level_start_target_or_target_parent_box_nrs,
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
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            traversal.sep_smaller_by_level,
            mpole_exps)

    # assert that list 3 close has been merged into list 1
    assert traversal.sep_close_smaller_starts is None

    # }}}

    # {{{ form locals for separated bigger mpoles ("list 4")

    logger.debug("form locals for separated bigger mpoles ('list 4 far')")

    local_exps = local_exps + wrangler.form_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.sep_bigger_starts,
            traversal.sep_bigger_lists,
            src_weights)

    # assert that list 4 close has been merged into list 1
    assert traversal.sep_close_bigger_starts is None

    # }}}

    # {{{ propagate local_exps downward

    logger.debug("propagate local_exps downward")
    wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps)

    # }}}

    # {{{ evaluate locals

    logger.debug("evaluate locals")
    non_qbx_potentials = non_qbx_potentials + wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
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


def write_performance_model(outf, geo_data):
    from pymbolic import var
    p_fmm = var("p_fmm")
    p_qbx = var("p_qbx")

    nqbtl = geo_data.non_qbx_box_target_lists()

    with cl.CommandQueue(geo_data.cl_context) as queue:
        tree = geo_data.tree().get(queue=queue)
        traversal = geo_data.traversal().get(queue=queue)
        box_target_counts_nonchild = (
                nqbtl.box_target_counts_nonchild.get(queue=queue))

    outf.write("# ------------------------------\n")

    outf.write("nlevels = {cost}\n"
            .format(cost=tree.nlevels))
    outf.write("nboxes = {cost}\n"
            .format(cost=tree.nboxes))
    outf.write("nsources = {cost}\n"
            .format(cost=tree.nsources))
    outf.write("ntargets = {cost}\n"
            .format(cost=tree.ntargets))

    # {{{ construct local multipoles

    outf.write("form_mp = {cost}\n"
            .format(cost=tree.nsources*p_fmm))

    # }}}

    # {{{ propagate multipoles upward

    outf.write("prop_upward = {cost}\n"
            .format(cost=tree.nboxes*p_fmm**2))

    # }}}

    # {{{ direct evaluation from neighbor source boxes ("list 1")

    npart_direct = 0
    for itgt_box, tgt_ibox in enumerate(traversal.target_boxes):
        ntargets = box_target_counts_nonchild[tgt_ibox]

        start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
        for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
            nsources = tree.box_source_counts_nonchild[src_ibox]

            npart_direct += ntargets * nsources

    outf.write("part_direct = {cost}\n"
            .format(cost=npart_direct))

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    nm2l = 0
    for itgt_box, tgt_ibox in enumerate(traversal.target_or_target_parent_boxes):
        start, end = traversal.sep_siblings_starts[itgt_box:itgt_box+2]

        nm2l += (end-start)

    outf.write("m2l = {cost}\n"
            .format(cost=nm2l * p_fmm**2))

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    nmp_eval = 0

    for itgt_box, tgt_ibox in enumerate(traversal.target_boxes):
        ntargets = box_target_counts_nonchild[tgt_ibox]

        start, end = traversal.sep_smaller_starts[itgt_box:itgt_box+2]

        nmp_eval += ntargets * (end-start)

    outf.write("mp_eval = {cost}\n"
            .format(cost=nmp_eval * p_fmm))

    # }}}

    # {{{ form locals for separated bigger mpoles ("list 4")

    nform_local = 0

    for itgt_box, tgt_ibox in enumerate(traversal.target_or_target_parent_boxes):
        start, end = traversal.sep_bigger_starts[itgt_box:itgt_box+2]

        for src_ibox in traversal.sep_bigger_lists[start:end]:
            nsources = tree.box_source_counts_nonchild[src_ibox]

            nform_local += nsources

    outf.write("form_local = {cost}\n"
            .format(cost=nform_local * p_fmm))

    # }}}

    # {{{ propagate local_exps downward

    outf.write("prop_downward = {cost}\n"
            .format(cost=tree.nboxes*p_fmm**2))

    # }}}

    # {{{ evaluate locals

    outf.write("eval_part = {cost}\n"
            .format(cost=tree.ntargets*p_fmm))

    # }}}

    # {{{ form global qbx locals

    nqbxl_direct = 0

    global_qbx_centers = geo_data.global_qbx_centers()
    qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
    center_to_targets_starts = geo_data.center_to_tree_targets().starts

    ncenters = geo_data.center_info().ncenters

    outf.write("ncenters = {cost}\n"
            .format(cost=ncenters))

    with cl.CommandQueue(geo_data.cl_context) as queue:
        global_qbx_centers = global_qbx_centers.get(queue=queue)
        qbx_center_to_target_box = qbx_center_to_target_box.get(queue=queue)
        center_to_targets_starts = center_to_targets_starts.get(queue=queue)

    for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
        itgt_box = qbx_center_to_target_box[tgt_icenter]
        tgt_ibox = traversal.target_or_target_parent_boxes[itgt_box]

        start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
        for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
            nsources = tree.box_source_counts_nonchild[src_ibox]

            nqbxl_direct += nsources

    outf.write("qbxl_direct = {cost}\n"
            .format(cost=nqbxl_direct * p_qbx))

    # }}}

    # {{{ translate from list 3 multipoles to qbx local expansions

    nqbx_m2l = 0

    for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
        itgt_box = qbx_center_to_target_box[tgt_icenter]

        start, end = traversal.sep_smaller_starts[itgt_box:itgt_box+2]
        nqbx_m2l += end - start

    outf.write("qbx_m2l = {cost}\n"
            .format(cost=nqbx_m2l * p_fmm * p_qbx))

    # }}}

    # {{{ translate from box local expansions to qbx local expansions

    outf.write("qbx_l2l = {cost}\n"
            .format(cost=ncenters * p_fmm * p_qbx))

    # }}}

    # {{{ evaluate qbx local expansions

    nqbx_eval = 0

    for iglobal_center in range(ncenters):
        src_icenter = global_qbx_centers[iglobal_center]

        start, end = center_to_targets_starts[src_icenter:src_icenter+2]
        nqbx_eval += end-start

    outf.write("qbx_eval = {cost}\n"
            .format(cost=nqbx_eval * p_qbx))

    # }}}


# vim: foldmethod=marker
