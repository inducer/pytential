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
from pytential.qbx.utils import (
    QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS)


unwrap_args = AreaQueryElementwiseTemplate.unwrap_args


import logging
logger = logging.getLogger(__name__)

#
# TODO:
# - Documentation
#
#==================
# HOW DOES TARGET ASSOCIATION WORK?
#
# The goal of the target association is to:
#   a) decide which targets require QBX, and
#   b) decide which centers to use for targets that require QBX, and
#   c) if no good centers are available for a target that requires QBX,
#      flag the appropriate panels for refinement.
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


# {{{ kernels

TARGET_ASSOC_DEFINES = r"""
enum TargetStatus
{
    UNMARKED,
    MARKED_QBX_CENTER_PENDING,
    MARKED_QBX_CENTER_FOUND
};

enum TargetFlag
{
    INTERIOR_OR_EXTERIOR_VOLUME_TARGET = 0,
    INTERIOR_SURFACE_TARGET = -1,
    EXTERIOR_SURFACE_TARGET = +1,
    INTERIOR_VOLUME_TARGET  = -2,
    EXTERIOR_VOLUME_TARGET  = +2
};
"""


class target_status_enum(object):  # noqa
    # NOTE: Must match "enum TargetStatus" above
    UNMARKED = 0
    MARKED_QBX_CENTER_PENDING = 1
    MARKED_QBX_CENTER_FOUND = 2


class target_flag_enum(object):  # noqa
    # NOTE: Must match "enum TargetFlag" above
    INTERIOR_OR_EXTERIOR_VOLUME_TARGET = 0
    INTERIOR_SURFACE_TARGET = -1
    EXTERIOR_SURFACE_TARGET = +1
    INTERIOR_VOLUME_TARGET = -2
    EXTERIOR_VOLUME_TARGET = +2


QBX_TARGET_MARKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_source_starts,
        particle_id_t *box_to_source_lists,
        particle_id_t source_offset,
        particle_id_t target_offset,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
        coord_t *box_to_search_dist,

        /* output */
        int *target_status,
        int *found_target_close_to_panel,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
    """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        coord_vec_t tgt_coords;
        ${load_particle("INDEX_FOR_TARGET_PARTICLE(i)", "tgt_coords")}
        {
            ${find_guiding_box("tgt_coords", 0, "my_box")}
            ${load_center("ball_center", "my_box", declare=False)}
            ${ball_radius} = box_to_search_dist[my_box];
        }
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        for (particle_id_t source_idx = box_to_source_starts[${leaf_box_id}];
             source_idx < box_to_source_starts[${leaf_box_id} + 1];
             ++source_idx)
        {
            particle_id_t source = box_to_source_lists[source_idx];
            coord_vec_t source_coords;
            ${load_particle("INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

            if (distance(source_coords, tgt_coords) <= panel_sizes[source] / 2)
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
        coord_t *panel_sizes,
        coord_t *box_to_search_dist,
        coord_t stick_out_factor,
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
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        coord_vec_t tgt_coords;
        ${load_particle("INDEX_FOR_TARGET_PARTICLE(i)", "tgt_coords")}
        {
            ${find_guiding_box("tgt_coords", 0, "my_box")}
            ${load_center("ball_center", "my_box", declare=False)}
            ${ball_radius} = box_to_search_dist[my_box];
        }
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
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
            ${load_particle("INDEX_FOR_CENTER_PARTICLE(center)", "center_coords")}
            coord_t my_dist_to_center = distance(tgt_coords, center_coords);

            if (my_dist_to_center
                    <= (panel_sizes[center] / 2) * (1 + stick_out_factor)
                && my_dist_to_center < min_dist_to_center[i])
            {
                target_status[i] = MARKED_QBX_CENTER_FOUND;
                min_dist_to_center[i] = my_dist_to_center;
                target_to_center[i] = center;
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
        coord_t *panel_sizes,
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
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        coord_vec_t target_coords;
        ${load_particle("INDEX_FOR_TARGET_PARTICLE(i)", "target_coords")}
        {
            ${find_guiding_box("target_coords", 0, "my_box")}
            ${load_center("ball_center", "my_box", declare=False)}
            ${ball_radius} = box_to_search_dist[my_box];
        }
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        for (particle_id_t source_idx = box_to_source_starts[${leaf_box_id}];
             source_idx < box_to_source_starts[${leaf_box_id} + 1];
             ++source_idx)
        {
            particle_id_t source = box_to_source_lists[source_idx];

            coord_vec_t source_coords;
            ${load_particle("INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

            bool is_close =
                distance(target_coords, source_coords)
                <= panel_sizes[source] / 2;

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
    def __init__(self, refine_flags, failed_target_flags):
        self.refine_flags = refine_flags
        self.failed_target_flags = failed_target_flags

    def __repr__(self):
        return "<%s>" % type(self).__name__


class QBXTargetAssociation(DeviceDataRecord):
    """
    .. attribute:: target_to_center
    """
    pass


class QBXTargetAssociator(object):

    def __init__(self, cl_context):
        from boxtree.tree_build import TreeBuilder
        self.tree_builder = TreeBuilder(cl_context)
        self.cl_context = cl_context
        from boxtree.area_query import PeerListFinder, SpaceInvaderQueryBuilder
        self.peer_list_finder = PeerListFinder(cl_context)
        self.space_invader_query = SpaceInvaderQueryBuilder(cl_context)

    # {{{ kernel generation

    @memoize_method
    def get_qbx_target_marker(self,
                              dimensions,
                              coord_dtype,
                              box_id_dtype,
                              peer_list_idx_dtype,
                              particle_id_dtype,
                              max_levels):
        return QBX_TARGET_MARKER.generate(
            self.cl_context,
            dimensions,
            coord_dtype,
            box_id_dtype,
            peer_list_idx_dtype,
            max_levels,
            extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def get_qbx_center_finder(self,
                              dimensions,
                              coord_dtype,
                              box_id_dtype,
                              peer_list_idx_dtype,
                              particle_id_dtype,
                              max_levels):
        return QBX_CENTER_FINDER.generate(
            self.cl_context,
            dimensions,
            coord_dtype,
            box_id_dtype,
            peer_list_idx_dtype,
            max_levels,
            extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def get_qbx_failed_target_association_refiner(self, dimensions, coord_dtype,
                                                 box_id_dtype, peer_list_idx_dtype,
                                                 particle_id_dtype, max_levels):
        return QBX_FAILED_TARGET_ASSOCIATION_REFINER.generate(
            self.cl_context,
            dimensions,
            coord_dtype,
            box_id_dtype,
            peer_list_idx_dtype,
            max_levels,
            extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    # }}}

    def mark_targets(self, queue, tree, peer_lists, lpot_source, target_status,
                     debug, wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_qbx_target_marker(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        found_target_close_to_panel = cl.array.zeros(queue, 1, np.int32)
        found_target_close_to_panel.finish()

        # Perform a space invader query over the sources.
        source_slice = tree.user_source_ids[tree.qbx_user_source_slice]
        sources = [axis.with_queue(queue)[source_slice] for axis in tree.sources]
        panel_sizes = lpot_source.panel_sizes("nsources").with_queue(queue)

        box_to_search_dist, evt = self.space_invader_query(
                queue,
                tree,
                sources,
                panel_sizes / 2,
                peer_lists,
                wait_for=wait_for)
        wait_for = [evt]

        logger.info("target association: marking targets close to panels")

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_source_starts,
                tree.box_to_qbx_source_lists,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_target_slice.start,
                tree.sorted_target_ids,
                panel_sizes,
                box_to_search_dist,
                target_status,
                found_target_close_to_panel,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=queue,
            wait_for=wait_for)

        if debug:
            target_status.finish()
            # Marked target = 1, 0 otherwise
            marked_target_count = cl.array.sum(target_status).get()
            logger.debug("target association: {}/{} targets marked close to panels"
                         .format(marked_target_count, tree.nqbxtargets))

        cl.wait_for_events([evt])

        logger.info("target association: done marking targets close to panels")

        return (found_target_close_to_panel == 1).all().get()

    def try_find_centers(self, queue, tree, peer_lists, lpot_source,
                         target_status, target_flags, target_assoc,
                         stick_out_factor, debug, wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_qbx_center_finder(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        if debug:
            target_status.finish()
            marked_target_count = int(cl.array.sum(target_status).get())

        # Perform a space invader query over the centers.
        center_slice = \
                tree.sorted_target_ids[tree.qbx_user_center_slice].with_queue(queue)
        centers = [axis.with_queue(queue)[center_slice] for axis in tree.sources]
        panel_sizes = lpot_source.panel_sizes("ncenters").with_queue(queue)

        box_to_search_dist, evt = self.space_invader_query(
                queue,
                tree,
                centers,
                panel_sizes * ((1 + stick_out_factor) / 2),
                peer_lists,
                wait_for=wait_for)
        wait_for = [evt]

        min_dist_to_center = cl.array.empty(
                queue, tree.nqbxtargets, tree.coord_dtype)
        min_dist_to_center.fill(np.inf)

        wait_for.extend(min_dist_to_center.events)

        logger.info("target association: finding centers for targets")

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_center_starts,
                tree.box_to_qbx_center_lists,
                tree.qbx_user_center_slice.start,
                tree.qbx_user_target_slice.start,
                tree.sorted_target_ids,
                panel_sizes,
                box_to_search_dist,
                stick_out_factor,
                target_flags,
                target_status,
                target_assoc.target_to_center,
                min_dist_to_center,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=queue,
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
        logger.info("target association: done finding centers for targets")
        return

    def mark_panels_for_refinement(self, queue, tree, peer_lists, lpot_source,
                                   target_status, refine_flags, debug,
                                   wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_qbx_failed_target_association_refiner(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        found_panel_to_refine = cl.array.zeros(queue, 1, np.int32)
        found_panel_to_refine.finish()

        # Perform a space invader query over the sources.
        source_slice = tree.user_source_ids[tree.qbx_user_source_slice]
        sources = [axis.with_queue(queue)[source_slice] for axis in tree.sources]
        panel_sizes = lpot_source.panel_sizes("nsources").with_queue(queue)

        box_to_search_dist, evt = self.space_invader_query(
                queue,
                tree,
                sources,
                panel_sizes / 2,
                peer_lists,
                wait_for=wait_for)
        wait_for = [evt]

        logger.info("target association: marking panels for refinement")

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
                lpot_source.panel_sizes("nsources"),
                target_status,
                box_to_search_dist,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=queue,
            wait_for=wait_for)

        if debug:
            refine_flags.finish()
            # Marked panel = 1, 0 otherwise
            marked_panel_count = cl.array.sum(refine_flags).get()
            logger.debug("target association: {} panels flagged for refinement"
                         .format(marked_panel_count))

        cl.wait_for_events([evt])

        logger.info("target association: done marking panels for refinement")

        return (found_panel_to_refine == 1).all().get()

    def make_target_flags(self, queue, target_discrs_and_qbx_sides):
        ntargets = sum(discr.nnodes for discr, _ in target_discrs_and_qbx_sides)
        target_flags = cl.array.empty(queue, ntargets, dtype=np.int32)
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

    def make_default_target_association(self, queue, ntargets):
        target_to_center = cl.array.empty(queue, ntargets, dtype=np.int32)
        target_to_center.fill(-1)
        target_to_center.finish()

        return QBXTargetAssociation(target_to_center=target_to_center)

    def __call__(self, lpot_source, target_discrs_and_qbx_sides,
                 stick_out_factor=1e-10, debug=True, wait_for=None):
        """
        Entry point for calling the target associator.

        :arg lpot_source: An instance of :class:`NewQBXLayerPotentialSource`

        :arg target_discrs_and_qbx_sides:

            a list of tuples ``(discr, sides)``, where
            *discr* is a
            :class:`pytential.discretization.Discretization`
            or a
            :class:`pytential.discretization.target.TargetBase` instance, and
            *sides* is either a :class:`int` or
            an array of (:class:`numpy.int8`) side requests for each
            target.

            The side request can take the following values for each target:

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

        :raises QBXTargetAssociationFailedException:
            when target association failed to find a center for a target.
            The returned exception object contains suggested refine flags.

        :returns:
        """

        with cl.CommandQueue(self.cl_context) as queue:
            from pytential.qbx.utils import build_tree_with_qbx_metadata

            tree = build_tree_with_qbx_metadata(
                    queue,
                    self.tree_builder,
                    lpot_source,
                    [discr for discr, _ in target_discrs_and_qbx_sides])

            peer_lists, evt = self.peer_list_finder(queue, tree, wait_for)
            wait_for = [evt]

            target_status = cl.array.zeros(queue, tree.nqbxtargets, dtype=np.int32)
            target_status.finish()

            have_close_targets = self.mark_targets(queue, tree, peer_lists,
                                                   lpot_source, target_status,
                                                   debug)

            target_assoc = self.make_default_target_association(
                queue, tree.nqbxtargets)

            if not have_close_targets:
                return target_assoc.with_queue(None)

            target_flags = self.make_target_flags(queue, target_discrs_and_qbx_sides)

            self.try_find_centers(queue, tree, peer_lists, lpot_source,
                                  target_status, target_flags, target_assoc,
                                  stick_out_factor, debug)

            center_not_found = (
                target_status == target_status_enum.MARKED_QBX_CENTER_PENDING)

            if center_not_found.any().get():
                surface_target = (
                    (target_flags == target_flag_enum.INTERIOR_SURFACE_TARGET)
                    | (target_flags == target_flag_enum.EXTERIOR_SURFACE_TARGET))

                if (center_not_found & surface_target).any().get():
                    logger.warning("An on-surface target was not "
                            "assigned a center. As a remedy you can try increasing "
                            "the \"stick_out_factor\" parameter, but this could "
                            "also cause an invalid center assignment.")

                refine_flags = cl.array.zeros(queue, tree.nqbxpanels, dtype=np.int32)
                have_panel_to_refine = self.mark_panels_for_refinement(queue,
                                                tree, peer_lists,
                                                lpot_source, target_status,
                                                refine_flags, debug)
                assert have_panel_to_refine
                raise QBXTargetAssociationFailedException(
                        refine_flags=refine_flags.with_queue(None),
                        failed_target_flags=center_not_found.with_queue(None))

            return target_assoc.with_queue(None)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
