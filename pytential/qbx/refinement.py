# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2016, 2017 Matt Wala
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


import loopy as lp
import numpy as np
import pyopencl as cl

from pytools import memoize_method
from boxtree.area_query import AreaQueryElementwiseTemplate
from boxtree.tools import InlineBinarySearch
from pytential.qbx.utils import (
        QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS, TreeWranglerBase)

import logging
logger = logging.getLogger(__name__)


# max_levels granularity for the stack used by the tree descent code in the
# area query kernel.
MAX_LEVELS_INCREMENT = 10


__doc__ = """
The refiner takes a layer potential source and refines it until it satisfies
three global QBX refinement criteria:

   * *Condition 1* (Expansion disk undisturbed by sources)
      A center must be closest to its own source.

   * *Condition 2* (Sufficient quadrature sampling from all source panels)
      The quadrature contribution from each panel is as accurate
      as from the center's own source panel.

   * *Condition 3* (Panel size bounded based on kernel length scale)
      The panel size is bounded by a kernel length scale. This
      applies only to Helmholtz kernels.

Warnings emitted by refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RefinerNotConvergedWarning

Helper functions
^^^^^^^^^^^^^^^^

.. autofunction:: make_empty_refine_flags

Refiner driver
^^^^^^^^^^^^^^

.. autoclass:: RefinerCodeContainer

.. autoclass:: RefinerWrangler

.. autofunction:: refine_for_global_qbx
"""

# {{{ kernels

# Refinement checker for Condition 1.
EXPANSION_DISK_UNDISTURBED_BY_SOURCES_CHECKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_source_starts,
        particle_id_t *box_to_source_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t *panel_to_center_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t *sorted_target_ids,
        coord_t *center_danger_zone_radii_by_panel,
        int npanels,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        /* Find the panel associated with this center. */
        particle_id_t my_panel = bsearch(panel_to_center_starts, npanels + 1, i);

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(i)", ball_center)}
        ${ball_radius} = center_danger_zone_radii_by_panel[my_panel];
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        /* Check that each source in the leaf box is sufficiently far from the
           center; if not, mark the panel for refinement. */

        for (particle_id_t source_idx = box_to_source_starts[${leaf_box_id}];
             source_idx < box_to_source_starts[${leaf_box_id} + 1];
             ++source_idx)
        {
            particle_id_t source = box_to_source_lists[source_idx];
            particle_id_t panel = bsearch(
                panel_to_source_starts, npanels + 1, source);

            if (my_panel == panel)
            {
                continue;
            }

            coord_vec_t source_coords;
            ${load_particle("INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

            bool is_close = (
                distance(${ball_center}, source_coords)
                <= center_danger_zone_radii_by_panel[my_panel]);

            if (is_close)
            {
                panel_refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="check_center_closest_to_orig_panel",
    preamble=str(InlineBinarySearch("particle_id_t")))


# Refinement checker for Condition 2.
SUFFICIENT_SOURCE_QUADRATURE_RESOLUTION_CHECKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_center_starts,
        particle_id_t *box_to_center_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t *sorted_target_ids,
        coord_t *source_danger_zone_radii_by_panel,
        int npanels,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        /* Find the panel associated with this source. */
        particle_id_t my_panel = bsearch(panel_to_source_starts, npanels + 1, i);

        ${load_particle("INDEX_FOR_SOURCE_PARTICLE(i)", ball_center)}
        ${ball_radius} = source_danger_zone_radii_by_panel[my_panel];
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        /* Check that each center in the leaf box is sufficiently far from the
           panel; if not, mark the panel for refinement. */

        for (particle_id_t center_idx = box_to_center_starts[${leaf_box_id}];
             center_idx < box_to_center_starts[${leaf_box_id} + 1];
             ++center_idx)
        {
            particle_id_t center = box_to_center_lists[center_idx];

            coord_vec_t center_coords;
            ${load_particle(
                "INDEX_FOR_CENTER_PARTICLE(center)", "center_coords")}

            bool is_close = (
                distance(${ball_center}, center_coords)
                <= source_danger_zone_radii_by_panel[my_panel]);

            if (is_close)
            {
                panel_refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="check_source_quadrature_resolution",
    preamble=str(InlineBinarySearch("particle_id_t")))

# }}}


# {{{ code container

class RefinerCodeContainer(object):

    def __init__(self, cl_context):
        self.cl_context = cl_context

    @memoize_method
    def expansion_disk_undisturbed_by_sources_checker(
            self, dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
            particle_id_dtype, max_levels):
        return EXPANSION_DISK_UNDISTURBED_BY_SOURCES_CHECKER.generate(
                self.cl_context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def sufficient_source_quadrature_resolution_checker(
            self, dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
            particle_id_dtype, max_levels):
        return SUFFICIENT_SOURCE_QUADRATURE_RESOLUTION_CHECKER.generate(
                self.cl_context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def kernel_length_scale_to_panel_size_ratio_checker(self):
        knl = lp.make_kernel(
            "{[panel]: 0<=panel<npanels}",
            """
            for panel
                <> oversize = panel_sizes[panel] > kernel_length_scale
                if oversize
                    refine_flags[panel] = 1
                    refine_flags_updated = 1 {id=write_refine_flags_updated}
                end
            end
            """,
            options="return_dict",
            silenced_warnings="write_race(write_refine_flags_updated)",
            name="refine_kernel_length_scale_to_panel_size_ratio")
        knl = lp.split_iname(knl, "panel", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    @memoize_method
    def tree_builder(self):
        from boxtree.tree_build import TreeBuilder
        return TreeBuilder(self.cl_context)

    @memoize_method
    def peer_list_finder(self):
        from boxtree.area_query import PeerListFinder
        return PeerListFinder(self.cl_context)

    def get_wrangler(self, queue):
        """
        :arg queue:
        """
        return RefinerWrangler(self, queue)

# }}}


# {{{ wrangler

class RefinerWrangler(TreeWranglerBase):

    def __init__(self, code_container, queue):
        self.code_container = code_container
        self.queue = queue

    # {{{ check subroutines for conditions 1-3

    def check_expansion_disks_undisturbed_by_sources(self,
            lpot_source, tree, peer_lists, refine_flags,
            debug, wait_for=None):

        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = MAX_LEVELS_INCREMENT * div_ceil(
                tree.nlevels, MAX_LEVELS_INCREMENT)

        knl = self.code_container.expansion_disk_undisturbed_by_sources_checker(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        logger.info("refiner: checking center is closest to orig panel")

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        found_panel_to_refine = cl.array.zeros(self.queue, 1, np.int32)
        found_panel_to_refine.finish()
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        # FIXME: This shouldn't be by panel
        center_danger_zone_radii_by_panel = \
                lpot_source._expansion_radii("npanels")

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_source_starts,
                tree.box_to_qbx_source_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_panel_to_center_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.sorted_target_ids,
                center_danger_zone_radii_by_panel,
                tree.nqbxpanels,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxcenters),
            queue=self.queue,
            wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking center is closest to orig panel")

        return found_panel_to_refine.get()[0] == 1

    def check_sufficient_source_quadrature_resolution(
            self, lpot_source, tree, peer_lists, refine_flags, debug,
            wait_for=None):

        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = MAX_LEVELS_INCREMENT * div_ceil(
                tree.nlevels, MAX_LEVELS_INCREMENT)

        knl = self.code_container.sufficient_source_quadrature_resolution_checker(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)
        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        logger.info("refiner: checking sufficient quadrature resolution")

        found_panel_to_refine = cl.array.zeros(self.queue, 1, np.int32)
        found_panel_to_refine.finish()

        source_danger_zone_radii_by_panel = (
                lpot_source._fine_panel_sizes("npanels")
                .with_queue(self.queue) / 4)
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_center_starts,
                tree.box_to_qbx_center_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.sorted_target_ids,
                source_danger_zone_radii_by_panel,
                tree.nqbxpanels,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxsources),
            queue=self.queue,
            wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking sufficient quadrature resolution")

        return found_panel_to_refine.get()[0] == 1

    def check_kernel_length_scale_to_panel_size_ratio(self, lpot_source,
            kernel_length_scale, refine_flags, debug, wait_for=None):
        knl = self.code_container.kernel_length_scale_to_panel_size_ratio_checker()

        logger.info("refiner: checking kernel length scale to panel size ratio")

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        evt, out = knl(self.queue,
                       panel_sizes=lpot_source._panel_sizes("npanels"),
                       refine_flags=refine_flags,
                       refine_flags_updated=np.array(0),
                       kernel_length_scale=np.array(kernel_length_scale),
                       wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking kernel length scale to panel size ratio")

        return (out["refine_flags_updated"].get() == 1).all()

    # }}}

    def refine(self, density_discr, refiner, refine_flags, factory, debug):
        """
        Refine the underlying mesh and discretization.
        """
        if isinstance(refine_flags, cl.array.Array):
            refine_flags = refine_flags.get(self.queue)
        refine_flags = refine_flags.astype(np.bool)

        logger.info("refiner: calling meshmode")

        refiner.refine(refine_flags)
        from meshmode.discretization.connection import make_refinement_connection
        conn = make_refinement_connection(refiner, density_discr, factory)

        logger.info("refiner: done calling meshmode")

        return conn

# }}}


class RefinerNotConvergedWarning(UserWarning):
    pass


def make_empty_refine_flags(queue, lpot_source, use_base_fine_discr=False):
    """Return an array on the device suitable for use as element refine flags.

    :arg queue: An instance of :class:`pyopencl.CommandQueue`.
    :arg lpot_source: An instance of :class:`QBXLayerPotentialSource`.

    :returns: A :class:`pyopencl.array.Array` suitable for use as refine flags,
        initialized to zero.
    """
    discr = (lpot_source.base_fine_density_discr
            if use_base_fine_discr
            else lpot_source.density_discr)
    result = cl.array.zeros(queue, discr.mesh.nelements, np.int32)
    result.finish()
    return result


# {{{ main entry point

def refine_for_global_qbx(lpot_source, wrangler,
        group_factory, kernel_length_scale=None,
        # FIXME: Set debug=False once everything works.
        refine_flags=None, debug=True, maxiter=50):
    """
    Entry point for calling the refiner.

    :arg lpot_source: An instance of :class:`QBXLayerPotentialSource`.

    :arg wrangler: An instance of :class:`RefinerWrangler`.

    :arg group_factory: An instance of
        :class:`meshmode.mesh.discretization.ElementGroupFactory`. Used for
        discretizing the coarse refined mesh.

    :arg kernel_length_scale: The kernel length scale, or *None* if not
        applicable. All panels are refined to below this size.

    :arg refine_flags: A :class:`pyopencl.array.Array` indicating which
        panels should get refined initially, or `None` if no initial
        refinement should be done. Should have size equal to the number of
        panels. See also :func:`make_empty_refine_flags()`.

    :arg maxiter: The maximum number of refiner iterations.

    :returns: A tuple ``(lpot_source, *conn*)`` where ``lpot_source`` is the
        refined layer potential source, and ``conn`` is a
        :class:`meshmode.discretization.connection.DiscretizationConnection`
        going from the original mesh to the refined mesh.
    """

    # TODO: Stop doing redundant checks by avoiding panels which no longer need
    # refinement.

    from meshmode.mesh.refinement import Refiner
    from meshmode.discretization.connection import (
            ChainedDiscretizationConnection, make_same_mesh_connection)

    refiner = Refiner(lpot_source.density_discr.mesh)
    connections = []

    # Do initial refinement.
    if refine_flags is not None:
        conn = wrangler.refine(
                lpot_source.density_discr, refiner, refine_flags, group_factory,
                debug)
        connections.append(conn)
        lpot_source = lpot_source.copy(density_discr=conn.to_discr)

    # {{{ first stage refinement

    must_refine = True
    niter = 0

    while must_refine:
        must_refine = False
        niter += 1

        if niter > maxiter:
            from warnings import warn
            warn(
                    "Max iteration count reached in QBX layer potential source"
                    " refiner.",
                    RefinerNotConvergedWarning)
            break

        # Build tree and auxiliary data.
        # FIXME: The tree should not have to be rebuilt at each iteration.
        tree = wrangler.build_tree(lpot_source)
        peer_lists = wrangler.find_peer_lists(tree)
        refine_flags = make_empty_refine_flags(wrangler.queue, lpot_source)

        # Check condition 1.
        must_refine |= wrangler.check_expansion_disks_undisturbed_by_sources(
                lpot_source, tree, peer_lists, refine_flags, debug)

        # Check condition 3.
        if kernel_length_scale is not None:
            must_refine |= (
                    wrangler.check_kernel_length_scale_to_panel_size_ratio(
                        lpot_source, kernel_length_scale, refine_flags, debug))

        if must_refine:
            conn = wrangler.refine(
                    lpot_source.density_discr, refiner, refine_flags,
                    group_factory, debug)
            connections.append(conn)
            lpot_source = lpot_source.copy(density_discr=conn.to_discr)

        del tree
        del refine_flags
        del peer_lists

    # }}}

    # {{{ second stage refinement

    must_refine = True
    niter = 0
    fine_connections = []

    base_fine_density_discr = lpot_source.density_discr

    while must_refine:
        must_refine = False
        niter += 1

        if niter > maxiter:
            from warnings import warn
            warn(
                    "Max iteration count reached in QBX layer potential source"
                    " refiner.",
                    RefinerNotConvergedWarning)
            break

        # Build tree and auxiliary data.
        # FIXME: The tree should not have to be rebuilt at each iteration.
        tree = wrangler.build_tree(lpot_source, use_base_fine_discr=True)
        peer_lists = wrangler.find_peer_lists(tree)
        refine_flags = make_empty_refine_flags(
                wrangler.queue, lpot_source, use_base_fine_discr=True)

        must_refine |= wrangler.check_sufficient_source_quadrature_resolution(
                lpot_source, tree, peer_lists, refine_flags, debug)

        if must_refine:
            conn = wrangler.refine(
                    base_fine_density_discr,
                    refiner, refine_flags, group_factory, debug)
            base_fine_density_discr = conn.to_discr
            fine_connections.append(conn)
            lpot_source = lpot_source.copy(
                    base_resampler=ChainedDiscretizationConnection(
                        fine_connections))

        del tree
        del refine_flags
        del peer_lists

    # }}}

    lpot_source = lpot_source.copy(debug=debug, _refined_for_global_qbx=True)

    if len(connections) == 0:
        # FIXME: This is inefficient
        connection = make_same_mesh_connection(
                lpot_source.density_discr,
                lpot_source.density_discr)
    else:
        connection = ChainedDiscretizationConnection(connections)

    return lpot_source, connection

# }}}


# vim: foldmethod=marker:filetype=pyopencl
