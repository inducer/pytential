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
from boxtree.tools import DeviceDataRecord
from boxtree.area_query import AreaQueryElementwiseTemplate
from boxtree.tools import InlineBinarySearch

from cgen import Enum

from arraycontext import PyOpenCLArrayContext, flatten
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
     flag the appropriate elements for refinement.

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
# |Is there a valid center|    |Mark elements close to t for       |
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
        int *found_target_close_to_element,

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
                atomic_or(found_target_close_to_element, 1);
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
        int *target_to_center_plus,
        int *target_to_center_minus,
        coord_t *min_dist_to_center_plus,
        coord_t *min_dist_to_center_minus,
        coord_t *min_rel_dist_to_center_plus,
        coord_t *min_rel_dist_to_center_minus,

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

                // Relative distance is weighted by disk radius, or,
                // equivalently, by the expanded radius (constant times disk radius).
                coord_t my_rel_dist_to_center = my_dist_to_center / (
                    expansion_radii_by_center_with_tolerance[center]);

                if (my_rel_dist_to_center > 1)
                {
                    continue;
                }

                // The idea is to use relative distance to determine the closest
                // disk center on each side, and then pick the one with smaller
                // absolute distance.
                //
                // Specifically, the following code does two things:
                //
                // 1. Find the center with minimal relative distance on either side
                // 2. Find the side with minimal Euclidean distance from one of the
                //    chosen centers in step 1.
                //
                // min_dist_to_center_plus and min_dist_to_center_minus
                // hold the absolute distances from minimum
                // relative distance centers.
                //
                // Refer to:
                // - https://gitlab.tiker.net/inducer/pytential/issues/132
                // - https://gitlab.tiker.net/inducer/pytential/merge_requests/181
                if (center_side > 0)
                {
                    if (my_rel_dist_to_center < min_rel_dist_to_center_plus[i])
                    {
                        target_status[i] = MARKED_QBX_CENTER_FOUND;
                        min_rel_dist_to_center_plus[i] = my_rel_dist_to_center;
                        min_dist_to_center_plus[i] = my_dist_to_center;
                        target_to_center_plus[i] = center;
                    }
                }
                else
                {
                    if (my_rel_dist_to_center < min_rel_dist_to_center_minus[i])
                    {
                        target_status[i] = MARKED_QBX_CENTER_FOUND;
                        min_rel_dist_to_center_minus[i] = my_rel_dist_to_center;
                        min_dist_to_center_minus[i] = my_dist_to_center;
                        target_to_center_minus[i] = center;
                    }
                }
            }

            // If the "winner" on either side is updated,
            // bind to the newly found center if it is closer than the
            // currently found one.

            if (min_dist_to_center_plus[i] < min_dist_to_center_minus[i])
            {
                target_to_center[i] = target_to_center_plus[i];
            }
            else
            {
                // This also includes the case where both distances are INFINITY,
                // thus target_to_center_plus and target_to_center_minus are
                // required to be initialized with the same default center
                // -1 as make_default_target_association
                target_to_center[i] = target_to_center_minus[i];
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
        particle_id_t *element_to_source_starts,
        particle_id_t source_offset,
        particle_id_t target_offset,
        int nelements,
        particle_id_t *sorted_target_ids,
        coord_t *tunnel_radius_by_source,
        int *target_status,
        coord_t *box_to_search_dist,

        /* output */
        int *refine_flags,
        int *found_element_to_refine,

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
                particle_id_t element = bsearch(
                    element_to_source_starts, nelements + 1, source);
                atomic_or(&refine_flags[element], 1);
                atomic_or(found_element_to_refine, 1);
            }
        }
    """,
    name="refine_elements",
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

    def __init__(self, actx: PyOpenCLArrayContext, tree_code_container):
        self.array_context = actx
        self.tree_code_container = tree_code_container

    @property
    def cl_context(self):
        return self.array_context.context

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

    def get_wrangler(self, actx: PyOpenCLArrayContext):
        return TargetAssociationWrangler(actx, code_container=self)


class TargetAssociationWrangler(TreeWranglerBase):

    @log_process(logger)
    def mark_targets(self, places, dofdesc,
            tree, peer_lists, target_status,
            debug, wait_for=None):
        from pytential import bind, sym
        ambient_dim = places.ambient_dim
        actx = self.array_context

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

        found_target_close_to_element = actx.zeros(1, np.int32)
        found_target_close_to_element.finish()

        # Perform a space invader query over the sources.
        source_slice = tree.sorted_target_ids[tree.qbx_user_source_slice]
        sources = [actx.thaw(axis)[source_slice] for axis in tree.sources]

        tunnel_radius_by_source = flatten(
                bind(
                    places,
                    sym._close_target_tunnel_radii(ambient_dim, dofdesc=dofdesc),
                    )(actx),
                actx)

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
                found_target_close_to_element,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=self.queue,
            wait_for=wait_for)

        if debug:
            target_status.finish()
            # Marked target = 1, 0 otherwise
            marked_target_count = actx.to_numpy(actx.np.sum(target_status)).item()
            logger.debug(
                    "target association: %d / %d targets marked close to elements",
                    marked_target_count, tree.nqbxtargets)

        import pyopencl as cl
        cl.wait_for_events([evt])

        return actx.to_numpy(actx.np.all(found_target_close_to_element == 1))

    @log_process(logger)
    def find_centers(self, places, dofdesc,
            tree, peer_lists, target_status, target_flags, target_assoc,
            target_association_tolerance,
            debug, wait_for=None):
        from pytential import bind, sym
        ambient_dim = places.ambient_dim
        actx = self.array_context

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
            marked_target_count = actx.to_numpy(actx.np.sum(target_status)).item()

        # Perform a space invader query over the centers.
        center_slice = actx.thaw(tree.sorted_target_ids[tree.qbx_user_center_slice])
        centers = [actx.thaw(axis)[center_slice] for axis in tree.sources]
        expansion_radii_by_center = bind(places,
                sym.expansion_radii(ambient_dim,
                    granularity=sym.GRANULARITY_CENTER,
                    dofdesc=dofdesc)
                )(actx)
        expansion_radii_by_center_with_tolerance = flatten(
                expansion_radii_by_center * (1 + target_association_tolerance),
                actx)

        # Idea:
        #
        # (1) Tag leaf boxes around centers with max distance to usable center.
        # (2) Area query from targets with those radii to find closest eligible
        # center in terms of relative distance.

        box_to_search_dist, evt = self.code_container.space_invader_query()(
                self.queue,
                tree,
                centers,
                expansion_radii_by_center_with_tolerance,
                peer_lists,
                wait_for=wait_for)
        wait_for = [evt]

        def make_target_field(fill_val, dtype=tree.coord_dtype):
            arr = actx.empty(tree.nqbxtargets, dtype)
            arr.fill(fill_val)
            wait_for.extend(arr.events)
            return arr

        target_to_center_plus = make_target_field(-1, np.int32)
        target_to_center_minus = make_target_field(-1, np.int32)

        min_dist_to_center_plus = make_target_field(np.inf)
        min_dist_to_center_minus = make_target_field(np.inf)

        min_rel_dist_to_center_plus = make_target_field(np.inf)
        min_rel_dist_to_center_minus = make_target_field(np.inf)

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
                target_to_center_plus,
                target_to_center_minus,
                min_dist_to_center_plus,
                min_dist_to_center_minus,
                min_rel_dist_to_center_plus,
                min_rel_dist_to_center_minus,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=self.queue,
            wait_for=wait_for)

        if debug:
            target_status.finish()
            # Associated target = 2, marked target = 1
            ntargets_associated = (
                actx.to_numpy(actx.np.sum(target_status)).item()
                - marked_target_count)
            assert ntargets_associated >= 0
            logger.debug("target association: %d targets were assigned centers",
                    ntargets_associated)

        import pyopencl as cl
        cl.wait_for_events([evt])

    @log_process(logger)
    def mark_elements_for_refinement(self, places, dofdesc,
            tree, peer_lists, target_status, refine_flags,
            debug, wait_for=None):
        from pytential import bind, sym
        ambient_dim = places.ambient_dim
        actx = self.array_context

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

        found_element_to_refine = actx.zeros(1, np.int32)
        found_element_to_refine.finish()

        # Perform a space invader query over the sources.
        source_slice = tree.user_source_ids[tree.qbx_user_source_slice]
        sources = [actx.thaw(axis)[source_slice] for axis in tree.sources]

        tunnel_radius_by_source = flatten(
                bind(
                    places,
                    sym._close_target_tunnel_radii(ambient_dim, dofdesc=dofdesc),
                    )(self.array_context),
                self.array_context)

        # see (TGTMARK) above for algorithm.

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
                tree.qbx_element_to_source_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_target_slice.start,
                tree.nqbxelements,
                tree.sorted_target_ids,
                tunnel_radius_by_source,
                target_status,
                box_to_search_dist,
                refine_flags,
                found_element_to_refine,
                *tree.sources),
            range=slice(tree.nqbxtargets),
            queue=self.queue,
            wait_for=wait_for)

        if debug:
            refine_flags.finish()
            # Marked element = 1, 0 otherwise
            marked_element_count = actx.to_numpy(actx.np.sum(refine_flags)).item()
            logger.debug("target association: %d elements flagged for refinement",
                         marked_element_count)

        import pyopencl as cl
        cl.wait_for_events([evt])

        return actx.to_numpy(actx.np.all(found_element_to_refine == 1))

    def make_target_flags(self, target_discrs_and_qbx_sides):
        actx = self.array_context

        ntargets = sum(discr.ndofs for discr, _ in target_discrs_and_qbx_sides)
        target_flags = actx.empty(ntargets, dtype=np.int32)

        offset = 0
        for discr, flags in target_discrs_and_qbx_sides:
            if np.isscalar(flags):
                target_flags[offset:offset + discr.ndofs].fill(flags)
            else:
                assert len(flags) == discr.ndofs
                target_flags[offset:offset + discr.ndofs] = flags
            offset += discr.ndofs

        target_flags.finish()
        return target_flags

    def make_default_target_association(self, ntargets):
        target_to_center = self.array_context.empty(ntargets, dtype=np.int32)
        target_to_center.fill(-1)
        target_to_center.finish()

        return QBXTargetAssociation(target_to_center=target_to_center)


def associate_targets_to_qbx_centers(places, geometry, wrangler,
        target_discrs_and_qbx_sides, target_association_tolerance,
        debug=True, wait_for=None):
    """
    Associate targets to centers in a layer potential source.

    :arg places: A :class:`~pytential.GeometryCollection`.
    :arg geometry: Name of the source geometry in *places* for which to
        associate targets.
    :arg wrangler: An instance of :class:`TargetAssociationWrangler`
    :arg target_discrs_and_qbx_sides:
        a list of tuples ``(discr, sides)``, where *discr* is a
        :class:`meshmode.discretization.Discretization`
        or a
        :class:`pytential.target.TargetBase` instance, and
        *sides* is either a :class:`int` or an array of (:class:`numpy.int8`)
        side requests for each target.

        The side request can take on the values in :ref:`qbx-side-request-table`.

    :raises pytential.qbx.QBXTargetAssociationFailedException:
        when target association failed to find a center for a target.
        The returned exception object contains suggested refine flags.

    :returns: A :class:`QBXTargetAssociation`.
    """
    actx = wrangler.array_context

    from pytential import sym
    dofdesc = sym.as_dofdesc(geometry).to_stage1()

    tree = wrangler.build_tree(places,
            sources_list=[dofdesc.geometry],
            targets_list=[discr for discr, _ in target_discrs_and_qbx_sides])

    peer_lists = wrangler.find_peer_lists(tree)

    target_status = actx.zeros(tree.nqbxtargets, dtype=np.int32)
    target_status.finish()

    have_close_targets = wrangler.mark_targets(places, dofdesc,
            tree, peer_lists, target_status,
            debug)

    target_assoc = wrangler.make_default_target_association(tree.nqbxtargets)
    if not have_close_targets:
        return target_assoc.with_queue(None)

    target_flags = wrangler.make_target_flags(target_discrs_and_qbx_sides)

    wrangler.find_centers(places, dofdesc,
            tree, peer_lists, target_status,
            target_flags, target_assoc, target_association_tolerance,
            debug)

    center_not_found = (
        target_status == target_status_enum.MARKED_QBX_CENTER_PENDING)

    if actx.to_numpy(actx.np.any(center_not_found)):
        surface_target = (
            (target_flags == target_flag_enum.INTERIOR_SURFACE_TARGET)
            | (target_flags == target_flag_enum.EXTERIOR_SURFACE_TARGET))

        if actx.to_numpy(actx.np.any(center_not_found & surface_target)):
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

        refine_flags = actx.zeros(tree.nqbxelements, dtype=np.int32)
        have_element_to_refine = wrangler.mark_elements_for_refinement(
                places, dofdesc,
                tree, peer_lists, target_status, refine_flags,
                debug)

        assert have_element_to_refine
        raise QBXTargetAssociationFailedException(
                refine_flags=actx.freeze(refine_flags),
                failed_target_flags=actx.freeze(center_not_found),
                message=fail_msg)

    return target_assoc.with_queue(None)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
