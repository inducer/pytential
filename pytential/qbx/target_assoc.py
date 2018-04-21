# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2016 Matt Wala
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


import numpy as np
from pytools import memoize_method

import pyopencl as cl
import pyopencl.array # noqa

from boxtree.tools import DeviceDataRecord
from boxtree.area_query import AreaQueryElementwiseTemplate
from boxtree.tools import InlineBinarySearch
from cgen import Enum
from pytential.qbx.utils import (
    QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS, TreeWranglerBase,
    TreeCodeContainerMixin)


unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

from pytools import log_process

import logging
logger = logging.getLogger(__name__)

__doc__ = """
The goal of target association is to:
   * decide which targets require QBX,
   * decide which centers to use for targets that require QBX,
   * if no good centers are available for a target that requires QBX,
     flag the appropriate panels for refinement.

Requesting a target side
^^^^^^^^^^^^^^^^^^^^^^^^

A target may further specify how it should be treated by target association.

.. _qbx-side-request-table:

.. table:: Values for target side requests

   ===== ==============================================
   Value Meaning
   ===== ==============================================
   0     Volume target. If near a QBX center,
         the value from the QBX expansion is returned,
         otherwise the volume potential is returned.

   -1    Surface target. Return interior limit from
         interior-side QBX expansion.

   +1    Surface target. Return exterior limit from
         exterior-side QBX expansion.

   -2    Volume target. If within an *interior* QBX disk,
         the value from the QBX expansion is returned,
         otherwise the volume potential is returned.

   +2    Volume target. If within an *exterior* QBX disk,
         the value from the QBX expansion is returned,
         otherwise the volume potential is returned.
   ===== ==============================================

Return values
^^^^^^^^^^^^^

.. autoclass:: QBXTargetAssociation

.. autoclass:: QBXTargetAssociationFailedException

Target association driver
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TargetAssociationCodeContainer

.. autoclass:: TargetAssociationWrangler

.. autofunction:: associate_targets_to_qbx_centers
"""


#
# HOW DOES TARGET ASSOCIATION WORK?
#
#
# The flow chart of what happens to target t is shown below. Pass names are in
# parentheses:
#
# START HERE
# |
# v
# +-----------------------+ No +-----------------------------------+
# |(QBXTargetMarker)      |--->|Mark t as not requiring QBX.       |
# |Is t close to a source?|    +-----------------------------------+
# +-----------------------+
# | Yes
# v
# +-----------------------+ No +-----------------------------------+
# |(QBXCenterFinder)      |--->|(QBXFailedTargetAssociationRefiner)|
# |Is there a valid center|    |Mark panels close to t for         |
# |close to t?            |    |refinement.                        |
# +-----------------------+    +-----------------------------------+
# | Yes
# v
# +-----------------------+
# |Associate t with the   |
# |best available center. |
# +-----------------------+
#


# {{{ kernels

class target_status_enum(Enum):  # noqa
    c_name = "TargetStatus"
    dtype = np.int32
    c_value_prefix = ""

    UNMARKED = 0
    MARKED_QBX_CENTER_PENDING = 1
    MARKED_QBX_CENTER_FOUND = 2


class target_flag_enum(Enum):  # noqa
    c_name = "TargetFlag"
    dtype = np.int32
    c_value_prefix = ""

    INTERIOR_OR_EXTERIOR_VOLUME_TARGET = 0
    INTERIOR_SURFACE_TARGET = -1
    EXTERIOR_SURFACE_TARGET = +1
    INTERIOR_VOLUME_TARGET = -2
    EXTERIOR_VOLUME_TARGET = +2


def _generate_enum_code(enum):
    return "\n".join(enum.generate())


TARGET_ASSOC_DEFINES = "".join([
    _generate_enum_code(target_status_enum),
    _generate_enum_code(target_flag_enum),
])


QBX_TARGET_MARKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_source_starts,
        particle_id_t *box_to_source_lists,
        particle_id_t source_offset,
        particle_id_t target_offset,
        particle_id_t *sorted_target_ids,
        coord_t *tunnel_radius_by_source,
        coord_t *box_to_search_dist,

        /* output */
        int *target_status,
        int *found_target_close_to_panel,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
    """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""//CL//
        coord_vec_t tgt_coords;
        ${load_particle("INDEX_FOR_TARGET_PARTICLE(i)", "tgt_coords")}
        {
            ${find_guiding_box("tgt_coords", 0, "my_box")}
            ${load_center("ball_center", "my_box", declare=False)}
            ${ball_radius} = box_to_search_dist[my_box];
        }
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""//CL//
        for (particle_id_t source_idx = box_to_source_starts[${leaf_box_id}];
             source_idx < box_to_source_starts[${leaf_box_id} + 1];
             ++source_idx)
        {
            particle_id_t source = box_to_source_lists[source_idx];
            coord_vec_t source_coords;
            ${load_particle("INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

            if (distance(source_coords, tgt_coords)
                    <= tunnel_radius_by_source[source])
            {
                target_status[i] = MARKED_QBX_CENTER_PENDING;
                *found_target_close_to_panel = 1;
            }
        }
    """,
    name="mark_targets",
    preamble=TARGET_ASSOC_DEFINES)


QBX_CENTER_FINDER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_center_starts,
        particle_id_t *box_to_center_lists,
        particle_id_t center_offset,
        particle_id_t target_offset,
        particle_id_t *sorted_target_ids,
        coord_t *expansion_radii_by_center_with_tolerance,
        coord_t *box_to_search_dist,
        int *target_flags,

        /* input/output */
        int *target_status,

        /* output */
        int *target_to_center,
        coord_t *min_dist_to_center,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
    """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""//CL//
        coord_vec_t tgt_coords;
        ${load_particle("INDEX_FOR_TARGET_PARTICLE(i)", "tgt_coords")}
        {
            ${find_guiding_box("tgt_coords", 0, "my_box")}
            ${load_center("ball_center", "my_box", declare=False)}
            ${ball_radius} = box_to_search_dist[my_box];
        }
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""//CL//
        if (target_status[i] == MARKED_QBX_CENTER_PENDING
                // Found one in a prior leaf, but there may well be another
                // that's closer.
                || target_status[i] == MARKED_QBX_CENTER_FOUND)
        {
            for (particle_id_t center_idx = box_to_center_starts[${leaf_box_id}];
                 center_idx < box_to_center_starts[${leaf_box_id} + 1];
                 ++center_idx)
            {
                particle_id_t center = box_to_center_lists[center_idx];

                int center_side = SIDE_FOR_CENTER_PARTICLE(center);

                // Sign of side should match requested target sign.
                if (center_side * target_flags[i] < 0)
                {
                    continue;
                }

                coord_vec_t center_coords;
                ${load_particle(
                    "INDEX_FOR_CENTER_PARTICLE(center)", "center_coords")}
                coord_t my_dist_to_center = distance(tgt_coords, center_coords);

                if (my_dist_to_center
                        <= expansion_radii_by_center_with_tolerance[center]
                    && my_dist_to_center < min_dist_to_center[i])
                {
                    target_status[i] = MARKED_QBX_CENTER_FOUND;
                    min_dist_to_center[i] = my_dist_to_center;
                    target_to_center[i] = center;
                }
            }
        }
    """,
    name="find_centers",
    preamble=TARGET_ASSOC_DEFINES)


QBX_FAILED_TARGET_ASSOCIATION_REFINER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_source_starts,
        particle_id_t *box_to_source_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t source_offset,
        particle_id_t target_offset,
        int npanels,
        particle_id_t *sorted_target_ids,
        coord_t *tunnel_radius_by_source,
        int *target_status,
        coord_t *box_to_search_dist,

        /* output */
        int *refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
    """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""//CL//
        coord_vec_t target_coords;
        ${load_particle("INDEX_FOR_TARGET_PARTICLE(i)", "target_coords")}
        {
            ${find_guiding_box("target_coords", 0, "my_box")}
            ${load_center("ball_center", "my_box", declare=False)}
            ${ball_radius} = box_to_search_dist[my_box];
        }
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""//CL//
        for (particle_id_t source_idx = box_to_source_starts[${leaf_box_id}];
             source_idx < box_to_source_starts[${leaf_box_id} + 1];
             ++source_idx)
        {
            particle_id_t source = box_to_source_lists[source_idx];

            coord_vec_t source_coords;
            ${load_particle("INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

            bool is_close =
                distance(target_coords, source_coords)
                <= tunnel_radius_by_source[source];

            if (is_close && target_status[i] == MARKED_QBX_CENTER_PENDING)
            {
                particle_id_t panel = bsearch(
                    panel_to_source_starts, npanels + 1, source);
                refine_flags[panel] = 1;
                *found_panel_to_refine = 1;
            }
        }
    """,
    name="refine_panels",
    preamble=TARGET_ASSOC_DEFINES + str(InlineBinarySearch("particle_id_t")))

# }}}


# {{{ target associator

class QBXTargetAssociationFailedException(Exception):
    """
    .. attribute:: refine_flags
    .. attribute:: failed_target_flags
    """
    def __init__(self, refine_flags, failed_target_flags, message):
        self.refine_flags = refine_flags
        self.failed_target_flags = failed_target_flags
        self.message = message

    def __str__(self):
        return (
                self.message
                + " You may examine this exception object's 'failed_target_flags' "
                "attribute as per-node data on the target geometry to determine "
                "which targets were not associated.")

    def __repr__(self):
        return "<%s>" % type(self).__name__


class QBXTargetAssociation(DeviceDataRecord):
    """
    .. attribute:: target_to_center
    """
    pass


class TargetAssociationCodeContainer(TreeCodeContainerMixin):

    def __init__(self, cl_context, tree_code_container):
        self.cl_context = cl_context
        self.tree_code_container = tree_code_container

    @memoize_method
    def target_marker(self, dimensions, coord_dtype, box_id_dtype,
            peer_list_idx_dtype, particle_id_dtype, max_levels):
        return QBX_TARGET_MARKER.generate(
                self.cl_context,
                dimensions,
                coord_dtype,
                box_id_dtype,
                peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def center_finder(self, dimensions, coord_dtype, box_id_dtype,
            peer_list_idx_dtype, particle_id_dtype, max_levels):
        return QBX_CENTER_FINDER.generate(
                self.cl_context,
                dimensions,
                coord_dtype,
                box_id_dtype,
                peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def refiner_for_failed_target_association(self, dimensions, coord_dtype,
            box_id_dtype, peer_list_idx_dtype, particle_id_dtype, max_levels):
        return QBX_FAILED_TARGET_ASSOCIATION_REFINER.generate(
                self.cl_context,
                dimensions,
                coord_dtype,
                box_id_dtype,
                peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def space_invader_query(self):
        from boxtree.area_query import SpaceInvaderQueryBuilder
        return SpaceInvaderQueryBuilder(self.cl_context)

    def get_wrangler(self, queue):
        return TargetAssociationWrangler(self, queue)


class TargetAssociationWrangler(TreeWranglerBase):

    def __init__(self, code_container, queue):
        self.code_container = code_container
        self.queue = queue

    @log_process(logger)
    def mark_targets(self, tree, peer_lists, lpot_source, target_status,
                     debug, wait_for=None):
        # Round up level count--this gets included in the kernel as
        # a stack bound. Rounding avoids too many kernel versions.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.code_container.target_marker(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        found_target_close_to_panel = cl.array.zeros(self.queue, 1, np.int32)
        found_target_close_to_panel.finish()

        # Perform a space invader query over the sources.
        source_slice = tree.sorted_target_ids[tree.qbx_user_source_slice]
        sources = [
                axis.with_queue(self.queue)[source_slice] for axis in tree.sources]
        tunnel_radius_by_source = (
                lpot_source._close_target_tunnel_radius("nsources")
                .with_queue(self.queue))

        # Target-marking algorithm (TGTMARK):
        #
        # (1) Use a space invader query to tag each leaf box that intersects with the
        # "near-source-detection tunnel" with the distance to the closest source.
        #
        # (2) Do an area query around all targets with the radius resulting
        # from the space invader query, enumerate sources in that vicinity.
        # If a source is found whose distance to the target is less than the
        # source's tunnel radius, mark that target as pending.
        # (or below: mark the source for refinement)

        # Note that this comment is referred to below by "TGTMARK". If you
        # remove this comment or change the algorithm here, make sure that
        # the reference below is still accurate.

        # Trade off for space-invaders vs directly tagging targets in
        # endangered boxes:
        #
        # (-) More complicated
        # (-) More actual work
        # (+) Taking the point of view of the targets could potentially lead to
        # more parallelism, if you think of the targets as unbounded while the
        # sources are fixed (which sort of makes sense, given that the number
        # of targets per box is not bounded).

        box_to_search_dist, evt = self.code_container.space_invader_query()(
                self.queue,
                tree,
                sources,
                tunnel_radius_by_source,
                peer_lists,
                wait_for=wait_for)
        wait_for = [evt]

        tunnel_radius_by_source = lpot_source._close_target_tunnel_radius("nsources")

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_source_starts,
                tree.box_to_qbx_source_lists,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_target_slice.start,
                tree.sorted_target_ids,
                tunnel_radius_by_source,
                box_to_search_dist,
                target_status,
                found_target_close_to_panel,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=self.queue,
            wait_for=wait_for)

        if debug:
            target_status.finish()
            # Marked target = 1, 0 otherwise
            marked_target_count = cl.array.sum(target_status).get()
            logger.debug("target association: {}/{} targets marked close to panels"
                         .format(marked_target_count, tree.nqbxtargets))

        cl.wait_for_events([evt])

        return (found_target_close_to_panel == 1).all().get()

    @log_process(logger)
    def try_find_centers(self, tree, peer_lists, lpot_source,
                         target_status, target_flags, target_assoc,
                         target_association_tolerance, debug, wait_for=None):
        # Round up level count--this gets included in the kernel as
        # a stack bound. Rounding avoids too many kernel versions.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.code_container.center_finder(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        if debug:
            target_status.finish()
            marked_target_count = int(cl.array.sum(target_status).get())

        # Perform a space invader query over the centers.
        center_slice = (
                tree.sorted_target_ids[tree.qbx_user_center_slice]
                .with_queue(self.queue))
        centers = [
                axis.with_queue(self.queue)[center_slice] for axis in tree.sources]
        expansion_radii_by_center = \
                lpot_source._expansion_radii("ncenters").with_queue(self.queue)
        expansion_radii_by_center_with_tolerance = \
                expansion_radii_by_center * (1 + target_association_tolerance)

        # Idea:
        #
        # (1) Tag leaf boxes around centers with max distance to usable center.
        # (2) Area query from targets with those radii to find closest eligible
        # center.

        box_to_search_dist, evt = self.code_container.space_invader_query()(
                self.queue,
                tree,
                centers,
                expansion_radii_by_center_with_tolerance,
                peer_lists,
                wait_for=wait_for)
        wait_for = [evt]

        min_dist_to_center = cl.array.empty(
                self.queue, tree.nqbxtargets, tree.coord_dtype)
        min_dist_to_center.fill(np.inf)

        wait_for.extend(min_dist_to_center.events)

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_center_starts,
                tree.box_to_qbx_center_lists,
                tree.qbx_user_center_slice.start,
                tree.qbx_user_target_slice.start,
                tree.sorted_target_ids,
                expansion_radii_by_center_with_tolerance,
                box_to_search_dist,
                target_flags,
                target_status,
                target_assoc.target_to_center,
                min_dist_to_center,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=self.queue,
            wait_for=wait_for)

        if debug:
            target_status.finish()
            # Associated target = 2, marked target = 1
            ntargets_associated = (
                int(cl.array.sum(target_status).get()) - marked_target_count)
            assert ntargets_associated >= 0
            logger.debug("target association: {} targets were assigned centers"
                         .format(ntargets_associated))

        cl.wait_for_events([evt])

    @log_process(logger)
    def mark_panels_for_refinement(self, tree, peer_lists, lpot_source,
                                   target_status, refine_flags, debug,
                                   wait_for=None):
        # Round up level count--this gets included in the kernel as
        # a stack bound. Rounding avoids too many kernel versions.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.code_container.refiner_for_failed_target_association(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        found_panel_to_refine = cl.array.zeros(self.queue, 1, np.int32)
        found_panel_to_refine.finish()

        # Perform a space invader query over the sources.
        source_slice = tree.user_source_ids[tree.qbx_user_source_slice]
        sources = [
                axis.with_queue(self.queue)[source_slice] for axis in tree.sources]
        tunnel_radius_by_source = (
                lpot_source._close_target_tunnel_radius("nsources")
                .with_queue(self.queue))

        # See (TGTMARK) above for algorithm.

        box_to_search_dist, evt = self.code_container.space_invader_query()(
                self.queue,
                tree,
                sources,
                tunnel_radius_by_source,
                peer_lists,
                wait_for=wait_for)
        wait_for = [evt]

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_source_starts,
                tree.box_to_qbx_source_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_target_slice.start,
                tree.nqbxpanels,
                tree.sorted_target_ids,
                lpot_source._close_target_tunnel_radius("nsources"),
                target_status,
                box_to_search_dist,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=self.queue,
            wait_for=wait_for)

        if debug:
            refine_flags.finish()
            # Marked panel = 1, 0 otherwise
            marked_panel_count = cl.array.sum(refine_flags).get()
            logger.debug("target association: {} panels flagged for refinement"
                         .format(marked_panel_count))

        cl.wait_for_events([evt])

        return (found_panel_to_refine == 1).all().get()

    def make_target_flags(self, target_discrs_and_qbx_sides):
        ntargets = sum(discr.nnodes for discr, _ in target_discrs_and_qbx_sides)
        target_flags = cl.array.empty(self.queue, ntargets, dtype=np.int32)
        offset = 0

        for discr, flags in target_discrs_and_qbx_sides:
            if np.isscalar(flags):
                target_flags[offset:offset + discr.nnodes].fill(flags)
            else:
                assert len(flags) == discr.nnodes
                target_flags[offset:offset + discr.nnodes] = flags
            offset += discr.nnodes

        target_flags.finish()
        return target_flags

    def make_default_target_association(self, ntargets):
        target_to_center = cl.array.empty(self.queue, ntargets, dtype=np.int32)
        target_to_center.fill(-1)
        target_to_center.finish()

        return QBXTargetAssociation(target_to_center=target_to_center)


def associate_targets_to_qbx_centers(lpot_source, wrangler,
        target_discrs_and_qbx_sides, target_association_tolerance,
        debug=True, wait_for=None):
    """
    Associate targets to centers in a layer potential source.

    :arg lpot_source: An instance of :class:`QBXLayerPotentialSource`

    :arg wrangler: An instance of :class:`TargetAssociationWrangler`

    :arg target_discrs_and_qbx_sides:

        a list of tuples ``(discr, sides)``, where
        *discr* is a
        :class:`pytential.discretization.Discretization`
        or a
        :class:`pytential.discretization.target.TargetBase` instance, and
        *sides* is either a :class:`int` or
        an array of (:class:`numpy.int8`) side requests for each
        target.

        The side request can take on the values in :ref:`qbx-side-request-table`.

    :raises QBXTargetAssociationFailedException:
        when target association failed to find a center for a target.
        The returned exception object contains suggested refine flags.

    :returns: A :class:`QBXTargetAssociation`.
    """

    tree = wrangler.build_tree(lpot_source,
            [discr for discr, _ in target_discrs_and_qbx_sides])

    peer_lists = wrangler.find_peer_lists(tree)

    target_status = cl.array.zeros(wrangler.queue, tree.nqbxtargets, dtype=np.int32)
    target_status.finish()

    have_close_targets = wrangler.mark_targets(tree, peer_lists,
           lpot_source, target_status, debug)

    target_assoc = wrangler.make_default_target_association(tree.nqbxtargets)

    if not have_close_targets:
        return target_assoc.with_queue(None)

    target_flags = wrangler.make_target_flags(target_discrs_and_qbx_sides)

    wrangler.try_find_centers(tree, peer_lists, lpot_source, target_status,
            target_flags, target_assoc, target_association_tolerance, debug)

    center_not_found = (
        target_status == target_status_enum.MARKED_QBX_CENTER_PENDING)

    if center_not_found.any().get():
        surface_target = (
            (target_flags == target_flag_enum.INTERIOR_SURFACE_TARGET)
            | (target_flags == target_flag_enum.EXTERIOR_SURFACE_TARGET))

        if (center_not_found & surface_target).any().get():
            fail_msg = "An on-surface target was not assigned a QBX center."
        else:
            fail_msg = "Some targets were not assigned a QBX center."

        fail_msg += (
            " Make sure to check the values you are passing "
            "for qbx_forced_limit on your symbolic layer potential "
            "operators. Those (or their default values) may "
            "constrain center choice to on-or-near surface "
            "sides of the geometry in a way that causes this issue.")

        fail_msg += (
            " As a last resort, you can try increasing "
            "the 'target_association_tolerance' parameter, but "
            "this could also cause an invalid center assignment.")

        refine_flags = cl.array.zeros(
                wrangler.queue, tree.nqbxpanels, dtype=np.int32)
        have_panel_to_refine = wrangler.mark_panels_for_refinement(
                tree, peer_lists, lpot_source, target_status, refine_flags, debug)

        assert have_panel_to_refine
        raise QBXTargetAssociationFailedException(
                refine_flags=refine_flags.with_queue(None),
                failed_target_flags=center_not_found.with_queue(None),
                message=fail_msg)

    return target_assoc.with_queue(None)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
