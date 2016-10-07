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

import logging
logger = logging.getLogger(__name__)

# HOW DOES TARGET ASSOCIATION WORK?
#
# The goal of the target association is to:
#   a) decide which targets require QBX, and
#   b) decide which centers to use for targets that require QBX, and
#   c) if no good centers are available for a target, decide which panels to
#      refine.
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
#define EPSILON .05

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

MARKED_QBX_CENTER_PENDING = 1


from boxtree.area_query import AreaQueryElementwiseTemplate
from boxtree.tools import InlineBinarySearch
from pytential.qbx.utils import QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS


QBX_TARGET_MARKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_target_starts,
        particle_id_t *box_to_target_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t source_offset,
        particle_id_t target_offset,
        int npanels,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,

        /* output */
        int *target_status,
        int *found_target_close_to_panel,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
    """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        ${load_particle("INDEX_FOR_SOURCE_PARTICLE(i)", ball_center)}
        particle_id_t my_panel = bsearch(panel_to_source_starts, npanels + 1, i);
        ${ball_radius} = panel_sizes[my_panel] / 2;
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        for (particle_id_t target_idx = box_to_target_starts[${leaf_box_id}];
             target_idx < box_to_target_starts[${leaf_box_id} + 1];
             ++target_idx)
        {
            particle_id_t target = box_to_target_lists[target_idx];
            coord_vec_t target_coords;
            ${load_particle("INDEX_FOR_TARGET_PARTICLE(target)", "target_coords")}

            if (distance(target_coords, ${ball_center}) <= ${ball_radius})
            {
                target_status[target] = MARKED_QBX_CENTER_PENDING;
                *found_target_close_to_panel = 1;
            }
        }
    """,
    name="mark_targets",
    preamble=TARGET_ASSOC_DEFINES + str(InlineBinarySearch("particle_id_t")))


QBX_CENTER_FINDER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_target_starts,
        particle_id_t *box_to_target_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t target_offset,
        int npanels,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
        int *target_flags,

        /* input/output */
        int *target_status,

        /* output */
        int *target_to_center,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
    """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        ${load_particle("INDEX_FOR_CENTER_PARTICLE(i)", ball_center)}
        particle_id_t my_panel = bsearch(
            panel_to_source_starts, npanels + 1, SOURCE_FOR_CENTER_PARTICLE(i));
        ${ball_radius} = panel_sizes[my_panel] / 2;
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        int my_side = SIDE_FOR_CENTER_PARTICLE(i);

        for (particle_id_t target_idx = box_to_target_starts[${leaf_box_id}];
             target_idx < box_to_target_starts[${leaf_box_id} + 1];
             ++target_idx)
        {
            particle_id_t target = box_to_target_lists[target_idx];
            coord_vec_t target_coords;
            ${load_particle("INDEX_FOR_TARGET_PARTICLE(target)", "target_coords")}

            coord_t my_dist_to_target = distance(target_coords, ${ball_center});

            if (/* Sign of side should match requested target sign. */
                my_side * target_flags[target] < 0
                /* Target should be covered by center. */
                || my_dist_to_target <= ${ball_radius} * (1 + EPSILON))
            {
                continue;
            }

            target_status[target] = MARKED_QBX_CENTER_FOUND;

            int curr_center, curr_center_updated;
            do
            {
                /* Read closest center. */
                curr_center = target_to_center[target];

                /* Check if I am closer than recorded closest center. */
                if (curr_center != -1)
                {
                    coord_vec_t curr_center_coords;
                    ${load_particle(
                         "INDEX_FOR_CENTER_PARTICLE(curr_center)",
                         "curr_center_coords")}

                    if (distance(target_coords, curr_center_coords)
                            <= my_dist_to_target)
                    {
                        /* The current center is closer, don't update. */
                        break;
                    }
                }

                /* Try to update the memory location. */
                curr_center_updated = atomic_cmpxchg(
                    (volatile __global int *) &target_to_center[target],
                    curr_center, i);
            } while (curr_center != curr_center_updated);
        }
    """,
    name="find_centers",
    preamble=TARGET_ASSOC_DEFINES + str(InlineBinarySearch("particle_id_t")))


QBX_FAILED_TARGET_ASSOCIATION_REFINER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_target_starts,
        particle_id_t *box_to_target_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t source_offset,
        particle_id_t target_offset,
        int npanels,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
        int *target_status,

        /* output */
        int *refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
    """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        ${load_particle("INDEX_FOR_SOURCE_PARTICLE(i)", ball_center)}
        particle_id_t my_panel = bsearch(panel_to_source_starts, npanels + 1, i);
        ${ball_radius} = panel_sizes[my_panel] / 2;
    """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        for (particle_id_t target_idx = box_to_target_starts[${leaf_box_id}];
             target_idx < box_to_target_starts[${leaf_box_id} + 1];
             ++target_idx)
        {
            particle_id_t target = box_to_target_lists[target_idx];
            coord_vec_t target_coords;
            ${load_particle("INDEX_FOR_TARGET_PARTICLE(target)", "target_coords")}

            bool is_close = distance(target_coords, ${ball_center}) <= ${ball_radius};

            if (is_close && target_status[target] == MARKED_QBX_CENTER_PENDING)
            {
                refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
            }
        }
    """,
    name="refine_panels",
    preamble=TARGET_ASSOC_DEFINES + str(InlineBinarySearch("particle_id_t")))

# }}}


# {{{ target associator

class TargetAssociationFailedException(Exception):
    """
    .. attribute:: refine_flags
    """
    pass


class QBXTargetAssociator(object):

    def __init__(self, cl_context):
        from pytential.qbx.utils import TreeWithQBXMetadataBuilder
        self.tree_builder = TreeWithQBXMetadataBuilder(cl_context)
        self.cl_context = cl_context
        from boxtree.area_query import PeerListFinder
        self.peer_list_finder = PeerListFinder(cl_context)

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

    def mark_targets(self, queue, tree, peer_lists, lpot_source, target_status, debug,
                     wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_qbx_target_marker(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        from boxtree.area_query import AreaQueryElementwiseTemplate
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        found_target_close_to_panel = cl.array.zeros(queue, 1, np.int32)
        found_target_close_to_panel.finish()

        logger.info("target association: marking targets close to panels")

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_target_starts,
                tree.box_to_qbx_target_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_target_slice.start,
                tree.nqbxpanels,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("nelements"),
                target_status,
                found_target_close_to_panel,
                *tree.sources),
            range=slice(tree.nqbxsources),
            queue=queue,
            wait_for=wait_for)

        if debug:
            target_status.finish()
            # Marked target = 1, 0 otherwise
            marked_target_count = cl.array.sum(target_status).get()
            logger.debug("target association: {} targets marked close to panels"
                         .format(marked_target_count))

        cl.wait_for_events([evt])

        logger.info("target association: done marking targets close to panels")

        return (found_target_close_to_panel.get() == 1).all()

    def try_find_centers(self, queue, tree, peer_lists, lpot_source,
                         target_status, target_flags, target_to_center, debug,
                         wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_qbx_center_finder(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        from boxtree.area_query import AreaQueryElementwiseTemplate
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        if debug:
            target_status.finish()
            marked_target_count = int(cl.array.sum(target_status).get())

        logger.info("target association: finding centers for targets")

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_target_starts,
                tree.box_to_qbx_target_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.qbx_user_target_slice.start,
                tree.nqbxpanels,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("nelements"),
                target_flags,
                target_status,
                target_to_center,
                *tree.sources),
            range=slice(tree.nqbxcenters),
            queue=queue,
            wait_for=wait_for)

        if debug:
            target_status.finish()
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

        from boxtree.area_query import AreaQueryElementwiseTemplate
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        found_panel_to_refine = cl.array.zeros(queue, 1, np.int32)
        found_panel_to_refine.finish()

        logger.info("target association: marking panels for refinement")

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_target_starts,
                tree.box_to_qbx_target_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_target_slice.start,
                tree.nqbxpanels,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("nelements"),
                target_status,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxsources),
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

        return (found_panel_to_refine.get() == 1).all()

    def make_target_flags(self, queue, target_discrs):
        ntargets = sum(discr.nnodes for discr, _ in target_discrs)
        target_flags = cl.array.empty(queue, ntargets, dtype=np.int32)
        offset = 0
        for discr, flags, in target_discrs:
            if np.isscalar(flags):
                target_flags[offset:offset + discr.nnodes].fill(flags)
            else:
                assert len(flags) == discr.nnodes
                target_flags[offset:offset + discr.nnodes] = flags
            offset += discr.nnodes
        target_flags.finish()
        return target_flags

    def __call__(self, lpot_source, target_discrs, debug=True, wait_for=None):
        with cl.CommandQueue(self.cl_context) as queue:
            tree = self.tree_builder(queue, lpot_source, [discr for discr, _ in target_discrs])
            peer_lists, evt = self.peer_list_finder(queue, tree, wait_for)
            wait_for = [evt]

            # Get target flags array.
            target_status = cl.array.zeros(queue, tree.nqbxtargets, dtype=np.int32)
            target_status.finish()

            have_close_targets = self.mark_targets(queue, tree,
                                                   peer_lists,
                                                   lpot_source,
                                                   target_status, debug)

            target_to_center = cl.array.empty(queue, tree.ntargets, dtype=np.int32)
            target_to_center.fill(-1)

            if not have_close_targets:
                return target_to_center.with_queue(None)

            target_flags = self.make_target_flags(queue, target_discrs)

            self.try_find_centers(queue, tree, peer_lists, lpot_source,
                                  target_status, target_flags, target_to_center,
                                  debug)

            if True: #(target_status == MARKED_QBX_CENTER_PENDING).any().get():
                refine_flags = cl.array.zeros(queue, tree.nqbxpanels, dtype=np.int32)
                have_panel_to_refine = self.mark_panels_for_refinement(queue,
                                                tree, peer_lists,
                                                lpot_source, target_status,
                                                refine_flags, debug)
                assert have_panel_to_refine
                raise TargetAssociationFailedException(refine_flags.with_queue(None))

            return target_to_center.with_queue(None)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
