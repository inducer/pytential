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
from pyopencl.elementwise import ElementwiseTemplate
from boxtree.tools import InlineBinarySearch
from pytential.qbx.utils import QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS

unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

import logging
logger = logging.getLogger(__name__)


MAX_LEVELS_INCREMENT = 10


__doc__ = """
Refinement
^^^^^^^^^^

The refiner takes a layer potential source and ensures that it satisfies three
QBX refinement criteria:

   * *Condition 1* (Expansion disk undisturbed by sources)
      A center must be closest to its own source.

   * *Condition 2* (Sufficient quadrature sampling from all source panels)
      The quadrature contribution from each panel is as accurate
      as from the center's own source panel.

   * *Condition 3* (Panel size bounded based on wavelength)
      (Helmholtz only) The panel size is bounded with respect to the wavelength.

.. autoclass:: RefinerCodeContainer
.. autoclass:: RefinerWrangler

.. automethod:: make_empty_refine_flags
.. automethod:: refine_for_global_qbx
"""

# {{{ kernels

TUNNEL_QUERY_DISTANCE_FINDER_TEMPLATE = ElementwiseTemplate(
    arguments=r"""//CL:mako//
        /* input */
        particle_id_t source_offset,
        particle_id_t panel_offset,
        int npanels,
        particle_id_t *panel_to_source_starts,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,

        /* output */
        float *tunnel_query_dists,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    operation=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""//CL:mako//
        /* Find my panel. */
        particle_id_t panel = bsearch(panel_to_source_starts, npanels + 1, i);

        /* Compute dist(tunnel region, panel center) */

        coord_vec_t center_of_mass;
        ${load_particle("INDEX_FOR_PANEL_PARTICLE(panel)", "center_of_mass")}

        coord_vec_t center;
        ${load_particle("INDEX_FOR_SOURCE_PARTICLE(i)", "center")}

        coord_t panel_size = panel_sizes[panel];

        coord_t max_dist = 0;

        %for ax in AXIS_NAMES[:dimensions]:
        {
            max_dist = fmax(max_dist,
                distance(center_of_mass.${ax}, center.${ax} + panel_size / 2));
            max_dist = fmax(max_dist,
                distance(center_of_mass.${ax}, center.${ax} - panel_size / 2));
        }
        %endfor

        // The atomic max operation supports only integer types.
        // However, max_dist is of a floating point type.
        // For comparison purposes we reinterpret the bits of max_dist
        // as an integer. The comparison result is the same as for positive
        // IEEE floating point numbers, so long as the float/int endianness
        // matches (fingers crossed).
        atomic_max(
            (volatile __global int *)
                &tunnel_query_dists[panel],
            as_int((float) max_dist));
        """,
    name="find_tunnel_query_distance",
    preamble=str(InlineBinarySearch("particle_id_t")))


# Refinement checker for Condition 1.
CENTER_IS_CLOSEST_TO_ORIG_PANEL_CHECKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_panel_starts,
        particle_id_t *box_to_panel_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t *panel_to_center_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
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
        particle_id_t my_panel = bsearch(panel_to_center_starts, npanels + 1, i);

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(i)", ball_center)}
        ${ball_radius} = panel_sizes[my_panel] / 2;
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        for (particle_id_t panel_idx = box_to_panel_starts[${leaf_box_id}];
             panel_idx < box_to_panel_starts[${leaf_box_id} + 1];
             ++panel_idx)
        {
            particle_id_t panel = box_to_panel_lists[panel_idx];

            // Skip self.
            if (my_panel == panel)
            {
                continue;
            }

            bool is_close = false;

            for (particle_id_t source = panel_to_source_starts[panel];
                 source < panel_to_source_starts[panel + 1];
                 ++source)
            {
                coord_vec_t source_coords;

                ${load_particle(
                    "INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

                is_close |= (
                    distance(${ball_center}, source_coords)
                    <= panel_sizes[my_panel] / 2);
            }

            if (is_close)
            {
                panel_refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="refine_center_closest_to_orig_panel",
    preamble=str(InlineBinarySearch("particle_id_t")))


# {{{ refiner code container

class RefinerCodeContainer(object):

    def __init__(self, cl_context):
        self.cl_context = cl_context

    @memoize_method
    def tunnel_query_distance_finder(
            self, dimensions, coord_dtype, particle_id_dtype):
        from pyopencl.tools import dtype_to_ctype
        from boxtree.tools import AXIS_NAMES
        logger.info("refiner: building tunnel query distance finder kernel")

        knl = TUNNEL_QUERY_DISTANCE_FINDER_TEMPLATE.build(
                self.cl_context,
                type_aliases=(
                    ("particle_id_t", particle_id_dtype),
                    ("coord_t", coord_dtype),
                    ),
                var_values=(
                    ("dimensions", dimensions),
                    ("AXIS_NAMES", AXIS_NAMES),
                    ("coord_dtype", coord_dtype),
                    ("dtype_to_ctype", dtype_to_ctype),
                    ("vec_types", tuple(cl.array.vec.types.items())),
                    ))

        logger.info("refiner: done building tunnel query distance finder kernel")
        return knl

    @memoize_method
    def center_is_closest_to_orig_panel_checker(
            self, dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
            particle_id_dtype, max_levels):
        return CENTER_IS_CLOSEST_TO_ORIG_PANEL_CHECKER.generate(self.cl_context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def helmholtz_k_to_panel_size_ratio_checker(self):
        knl = lp.make_kernel(
            "{[panel]: 0<=panel<npanels}",
            """
            for panel
                <> oversize = panel_sizes[panel] * helmholtz_k > 5
                if oversize
                    refine_flags[panel] = 1
                    refine_flags_updated = 1 {id=write_refine_flags_updated}
                end
            end
            """,
            options="return_dict",
            silenced_warnings="write_race(write_refine_flags_updated)",
            name="refine_helmholtz_k_to_panel_size_ratio")
        knl = lp.split_iname(knl, "panel", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    @memoize_method
    def tree_builder(self):
        from pytential.qbx.utils import TreeWithQBXMetadataBuilder
        return TreeWithQBXMetadataBuilder(self.cl_context)

    @memoize_method
    def peer_list_finder(self):
        from boxtree.area_query import PeerListFinder
        return PeerListFinder(self.cl_context)

    def get_wrangler(self, queue, refiner):
        """
        :arg queue:
        :arg refiner:
        """
        return RefinerWrangler(self, queue, refiner)

# }}}


# {{{ refiner wrangler

class RefinerWrangler(object):

    def __init__(self, code_container, queue, refiner):
        self.code_container = code_container
        self.queue = queue
        self.refiner = refiner

    def check_center_is_closest_to_orig_panel(self,
            lpot_source, tree, peer_lists, refine_flags,
            debug, wait_for=None):

        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = MAX_LEVELS_INCREMENT * div_ceil(
                tree.nlevels, MAX_LEVELS_INCREMENT)

        knl = self.code_container.center_is_closest_to_orig_panel_checker(
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

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_panel_starts,
                tree.box_to_qbx_panel_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_panel_to_center_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("npanels"),
                tree.nqbxpanels,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxcenters),
            queue=self.queue)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking center is closest to orig panel")

        return found_panel_to_refine.get()[0] == 1

    def check_sufficient_quadrature_resolution(self):
        return True

    def check_helmholtz_k_to_panel_size_ratio(self, lpot_source,
                helmholtz_k, refine_flags, debug, wait_for=None):
        knl = self.code_container.helmholtz_k_to_panel_size_ratio_checker()

        logger.info("refiner: checking helmholtz k to panel size ratio")

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        evt, out = knl(self.queue,
                       panel_sizes=lpot_source.panel_sizes("npanels"),
                       refine_flags=refine_flags,
                       refine_flags_updated=np.array(0),
                       helmholtz_k=np.array(helmholtz_k),
                       wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking helmholtz k to panel size ratio")

        return (out["refine_flags_updated"].get() == 1).all()

    # }}}

    # {{{

    def get_tunnel_query_dists(self, queue, tree, lpot_source):
        """
        Compute radii for the tubular neighborhood around each panel center of mass.
        """
        nqbxpanels = lpot_source.density_discr.mesh.nelements
        # atomic_max only works on float32
        tq_dists = cl.array.zeros(queue, nqbxpanels, np.float32)
        tq_dists.finish()

        knl = self.get_tunnel_query_distance_finder(tree.dimensions,
                tree.coord_dtype, tree.particle_id_dtype)

        evt = knl(tree.qbx_user_source_slice.start,
                  tree.qbx_user_panel_slice.start,
                  nqbxpanels,
                  tree.qbx_panel_to_source_starts,
                  tree.sorted_target_ids,
                  lpot_source.panel_sizes("npanels"),
                  tq_dists,
                  *tree.sources,
                  queue=queue,
                  range=slice(tree.nqbxsources))

        cl.wait_for_events([evt])

        if tree.coord_dtype != tq_dists.dtype:
            tq_dists = tq_dists.astype(tree.coord_dtype)

        return tq_dists, evt

    # }}}

    def build_tree(self, lpot_source):
        tb = self.code_container.tree_builder()
        return tb(self.queue, lpot_source)

    def find_peer_lists(self, tree):
        plf = self.code_container.peer_list_finder()
        peer_lists, evt = plf(self.queue, tree)
        cl.wait_for_events([evt])
        return peer_lists

    def refine(self, lpot_source, refine_flags, factory, debug):
        """
        Refine the underlying mesh and discretization.
        """
        if isinstance(refine_flags, cl.array.Array):
            refine_flags = refine_flags.get(self.queue)
        refine_flags = refine_flags.astype(np.bool)

        logger.info("refiner: calling meshmode")

        self.refiner.refine(refine_flags)
        from meshmode.discretization.connection import make_refinement_connection

        conn = make_refinement_connection(
                self.refiner, lpot_source.density_discr,
                factory)

        logger.info("refiner: done calling meshmode")

        new_lpot_source = lpot_source.copy(
                density_discr=conn.to_discr)

        return new_lpot_source, conn

# }}}


def make_empty_refine_flags(queue, lpot_source):
    """Return an array on the device suitable for use as element refine flags.

    :arg queue: An instance of :class:`pyopencl.CommandQueue`.
    :arg lpot_source: An instance of :class:`QBXLayerPotentialSource`.

    :returns: A :class:`pyopencl.array.Array` suitable for use as refine flags,
        initialized to zero.
    """
    result = cl.array.zeros(
            queue, lpot_source.density_discr.mesh.nelements, np.int32)
    result.finish()
    return result


def refine_for_global_qbx(lpot_source, code_container,
        group_factory, helmholtz_k=None,
        # FIXME: Set debug=False once everything works.
        refine_flags=None, debug=True, maxiter=50):
    """
    Entry point for calling the refiner.

    :arg lpot_source: An instance of :class:`QBXLayerPotentialSource`.

    :arg code_container: An instance of :class:`RefinerCodeContainer`.

    :arg group_factory: An instance of
        :class:`meshmode.mesh.discretization.ElementGroupFactory`. Used for
        discretizing the refined mesh.

    :arg helmholtz_k: The Helmholtz parameter, or `None` if not applicable.

    :arg refine_flags: A :class:`pyopencl.array.Array` indicating which
        panels should get refined initially, or `None` if no initial
        refinement should be done. Should have size equal to the number of
        panels. See also :func:`make_empty_refine_flags()`.

    :arg maxiter: The maximum number of refiner iterations.

    :returns: A tuple ``(lpot_source, conns)`` where ``lpot_source`` is the
        refined layer potential source, and ``conns`` is a list of
        :class:`meshmode.discretization.connection.DiscretizationConnection`
        objects going from the original mesh to the refined mesh.
    """

    # Algorithm:
    #
    # 1. Do initial refinement, if requested.
    #
    # 2. While not converged:
    #        Refine until each center is closest to its own
    #        source and panel sizes bounded by wavelength.
    #
    # 3. While not converged:
    #        Refine fine density discretization until
    #        sufficient quadrature resolution from all panels achieved.
    #
    # TODO: Stop doing redundant checks by avoiding panels which no longer need
    # refinement.

    from meshmode.mesh.refinement import Refiner
    refiner = Refiner(lpot_source.density_discr.mesh)
    connections = []

    with cl.CommandQueue(lpot_source.cl_context) as queue:
        wrangler = code_container.get_wrangler(queue, refiner)

        # Do initial refinement.
        if refine_flags is not None:
            lpot_source, conn = wrangler.refine(lpot_source, refine_flags,
                    group_factory, debug)
            connections.append(conn)

        # {{{ first stage refinement

        must_refine = True
        niter = 0

        while must_refine:
            must_refine = False
            niter += 1

            if niter > maxiter:
                logger.warning(
                    "Max iteration count reached in QBX layer potential source"
                    " refiner (first stage).")
                break

            # Build tree and auxiliary data.
            # FIXME: The tree should not have to be rebuilt at each iteration.
            tree = wrangler.build_tree(lpot_source)
            peer_lists = wrangler.find_peer_lists(tree)
            refine_flags = make_empty_refine_flags(queue, lpot_source)

            # Check condition 1.
            must_refine |= wrangler.check_center_is_closest_to_orig_panel(
                    lpot_source, tree, peer_lists, refine_flags, debug)

            # Check condition 3.
            if helmholtz_k is not None:
                must_refine |= wrangler.check_helmholtz_k_to_panel_size_ratio(
                        lpot_source, helmholtz_k, refine_flags, debug)

            if must_refine:
                lpot_source, conn = wrangler.refine(lpot_source, refine_flags,
                        group_factory, debug)
                connections.append(conn)

            del tree
            del refine_flags
            del peer_lists

        # }}}

        # {{{ second stage refinement

        # }}}

    lpot_source = lpot_source.copy(debug=debug, refined_for_global_qbx=True)

    return lpot_source, connections


# vim: foldmethod=marker:filetype=pyopencl
