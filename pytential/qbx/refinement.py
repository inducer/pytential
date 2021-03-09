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
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import flatten, DOFArray
import numpy as np
import pyopencl as cl

from pytools import memoize_method
from boxtree.area_query import AreaQueryElementwiseTemplate
from boxtree.tools import InlineBinarySearch
from pytential.qbx.utils import (
        QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS, TreeWranglerBase,
        TreeCodeContainerMixin)

from pytools import ProcessLogger, log_process

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

.. autofunction:: refine_geometry_collection
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
        coord_t *center_danger_zone_radii,
        coord_t expansion_disturbance_tolerance,
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
        particle_id_t icenter = i;

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(icenter)", ball_center)}
        ${ball_radius} = (1-expansion_disturbance_tolerance)
                    * center_danger_zone_radii[icenter];
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        /* Check that each source in the leaf box is sufficiently far from the
           center; if not, mark the panel for refinement. */

        for (particle_id_t source_idx = box_to_source_starts[${leaf_box_id}];
             source_idx < box_to_source_starts[${leaf_box_id} + 1];
             ++source_idx)
        {
            particle_id_t source = box_to_source_lists[source_idx];
            particle_id_t source_panel = bsearch(
                panel_to_source_starts, npanels + 1, source);

            /* Find the panel associated with this center. */
            particle_id_t center_panel = bsearch(panel_to_center_starts, npanels + 1,
                icenter);

            coord_vec_t source_coords;
            ${load_particle("INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

            bool is_close = (
                distance(${ball_center}, source_coords)
                <= (1-expansion_disturbance_tolerance)
                        * center_danger_zone_radii[icenter]);

            if (is_close)
            {
                atomic_or(&panel_refine_flags[center_panel], 1);
                atomic_or(found_panel_to_refine, 1);
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
                atomic_or(&panel_refine_flags[my_panel], 1);
                atomic_or(found_panel_to_refine, 1);
                break;
            }
        }
        """,
    name="check_source_quadrature_resolution",
    preamble=str(InlineBinarySearch("particle_id_t")))

# }}}


# {{{ code container

class RefinerCodeContainer(TreeCodeContainerMixin):

    def __init__(self, actx: PyOpenCLArrayContext, tree_code_container):
        self.array_context = actx
        self.tree_code_container = tree_code_container

    @memoize_method
    def expansion_disk_undisturbed_by_sources_checker(
            self, dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
            particle_id_dtype, max_levels):
        return EXPANSION_DISK_UNDISTURBED_BY_SOURCES_CHECKER.generate(
                self.array_context.context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def sufficient_source_quadrature_resolution_checker(
            self, dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
            particle_id_dtype, max_levels):
        return SUFFICIENT_SOURCE_QUADRATURE_RESOLUTION_CHECKER.generate(
                self.array_context.context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def element_prop_threshold_checker(self):
        knl = lp.make_kernel(
            "{[ielement]: 0<=ielement<nelements}",
            """
            for ielement
                <> over_threshold = element_property[ielement] > threshold
                if over_threshold
                    refine_flags[ielement] = 1
                    refine_flags_updated = 1 {id=write_refine_flags_updated, atomic}
                end
            end
            """,
            [
                lp.GlobalArg("refine_flags_updated", shape=(), for_atomic=True),
                "..."
                ],
            options="return_dict",
            silenced_warnings="write_race(write_refine_flags_updated)",
            name="refine_kernel_length_scale_to_quad_resolution_ratio",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.split_iname(knl, "ielement", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    def get_wrangler(self):
        """
        :arg queue:
        """
        return RefinerWrangler(self.array_context, self)

# }}}


# {{{ wrangler

class RefinerWrangler(TreeWranglerBase):
    # {{{ check subroutines for conditions 1-3

    @log_process(logger)
    def check_expansion_disks_undisturbed_by_sources(self,
            stage1_density_discr, tree, peer_lists,
            expansion_disturbance_tolerance,
            refine_flags,
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

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        found_panel_to_refine = cl.array.zeros(self.queue, 1, np.int32)
        found_panel_to_refine.finish()
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        from pytential import bind, sym
        center_danger_zone_radii = flatten(
            bind(stage1_density_discr,
                sym.expansion_radii(stage1_density_discr.ambient_dim,
                    granularity=sym.GRANULARITY_CENTER))(self.array_context))

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
                center_danger_zone_radii,
                expansion_disturbance_tolerance,
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

        return found_panel_to_refine.get()[0] == 1

    @log_process(logger)
    def check_sufficient_source_quadrature_resolution(self,
            stage2_density_discr, tree, peer_lists, refine_flags,
            debug, wait_for=None):

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

        found_panel_to_refine = cl.array.zeros(self.queue, 1, np.int32)
        found_panel_to_refine.finish()

        from pytential import bind, sym
        dd = sym.as_dofdesc(sym.GRANULARITY_ELEMENT).to_stage2()
        source_danger_zone_radii_by_panel = flatten(
                bind(stage2_density_discr,
                    sym._source_danger_zone_radii(
                        stage2_density_discr.ambient_dim, dofdesc=dd))
                (self.array_context))
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

        return found_panel_to_refine.get()[0] == 1

    def check_element_prop_threshold(self, element_property, threshold, refine_flags,
            debug, wait_for=None):
        knl = self.code_container.element_prop_threshold_checker()

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        from pytential.utils import flatten_if_needed
        element_property = flatten_if_needed(
                self.array_context, element_property)

        evt, out = knl(self.queue,
                       element_property=element_property,
                       refine_flags=refine_flags,
                       refine_flags_updated=np.array(0),
                       threshold=np.array(threshold),
                       wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        return (out["refine_flags_updated"] == 1).all()

    # }}}

    def refine(self, density_discr, refiner, refine_flags, factory, debug):
        """
        Refine the underlying mesh and discretization.
        """
        if isinstance(refine_flags, cl.array.Array):
            refine_flags = refine_flags.get(self.queue)
        refine_flags = refine_flags.astype(bool)

        with ProcessLogger(logger, "refine mesh"):
            refiner.refine(refine_flags)
            from meshmode.discretization.connection import (
                    make_refinement_connection)
            conn = make_refinement_connection(
                    self.array_context, refiner, density_discr, factory)

        return conn

# }}}


# {{{ stage1/stage2 refinement

class RefinerNotConvergedWarning(UserWarning):
    pass


def make_empty_refine_flags(queue, density_discr):
    """Return an array on the device suitable for use as element refine flags.

    :arg queue: An instance of :class:`pyopencl.CommandQueue`.
    :arg lpot_source: An instance of :class:`pytential.qbx.QBXLayerPotentialSource`.

    :returns: A :class:`pyopencl.array.Array` suitable for use as refine flags,
        initialized to zero.
    """
    result = cl.array.zeros(queue, density_discr.mesh.nelements, np.int32)
    result.finish()
    return result


def _warn_max_iterations(violated_criteria, expansion_disturbance_tolerance):
    from warnings import warn
    warn(
            "QBX layer potential source refiner did not terminate "
            "after %d iterations (the maximum). "
            "You may call 'refine_geometry_collection()' manually "
            "and pass 'visualize=True' to see what area of the geometry is "
            "causing trouble. If the issue is disturbance of expansion disks, "
            "you may pass a slightly increased value (currently: %g) for "
            "'expansion_disturbance_tolerance'. As a last resort, "
            "you may use Python's warning filtering mechanism to "
            "not treat this warning as an error. The criteria triggering "
            "refinement in each iteration were: %s. " % (
                len(violated_criteria),
                expansion_disturbance_tolerance,
                ", ".join(
                    "%d: %s" % (i+1, vc_text)
                    for i, vc_text in enumerate(violated_criteria))),
            RefinerNotConvergedWarning)


def _visualize_refinement(actx: PyOpenCLArrayContext, discr,
        niter, stage_nr, stage_name, flags, visualize=False):
    if not visualize:
        return

    if stage_nr not in (1, 2):
        raise ValueError("unexpected stage number")

    flags = flags.get()
    logger.info("for stage %s: splitting %d/%d stage-%d elements",
            stage_name, np.sum(flags), discr.mesh.nelements, stage_nr)

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, 3)

    assert len(flags) == discr.mesh.nelements

    flags = flags.astype(bool)
    nodes_flags_template = discr.zeros(actx)
    nodes_flags = []
    for grp in discr.groups:
        meg = grp.mesh_el_group
        nodes_flags_grp = actx.to_numpy(nodes_flags_template[grp.index])
        nodes_flags_grp[
                flags[meg.element_nr_base:meg.nelements+meg.element_nr_base]] = 1
        nodes_flags.append(actx.from_numpy(nodes_flags_grp))

    nodes_flags = DOFArray(actx, tuple(nodes_flags))

    vis_data = [
        ("refine_flags", nodes_flags),
        ]

    if 0:
        from pytential import sym, bind
        bdry_normals = bind(discr, sym.normal(discr.ambient_dim))(
                actx).as_vector(dtype=object)
        vis_data.append(("bdry_normals", bdry_normals),)

    vis.write_vtk_file(f"refinement-{stage_name}-{niter:03d}.vtu", vis_data)


def _make_quad_stage2_discr(lpot_source, stage2_density_discr):
    from meshmode.discretization.poly_element import (
            OrderAndTypeBasedGroupFactory,
            QuadratureSimplexElementGroup,
            GaussLegendreTensorProductElementGroup)

    return stage2_density_discr.copy(
            group_factory=OrderAndTypeBasedGroupFactory(
                lpot_source.fine_order,
                simplex_group_class=QuadratureSimplexElementGroup,
                tensor_product_group_class=GaussLegendreTensorProductElementGroup),
            )


def _make_temporary_collection(lpot_source,
        stage1_density_discr=None,
        stage2_density_discr=None):
    from pytential import sym
    from pytential import GeometryCollection

    name = "_tmp_refine_source"
    places = GeometryCollection(lpot_source, auto_where=name)

    if stage1_density_discr is not None:
        places._add_discr_to_cache(stage1_density_discr,
                name, sym.QBX_SOURCE_STAGE1)

    if stage2_density_discr is not None:
        quad_stage2_density_discr = \
                _make_quad_stage2_discr(lpot_source, stage2_density_discr)

        places._add_discr_to_cache(stage2_density_discr,
                name, sym.QBX_SOURCE_STAGE2)
        places._add_discr_to_cache(quad_stage2_density_discr,
                name, sym.QBX_SOURCE_QUAD_STAGE2)

    return places


def _refine_qbx_stage1(lpot_source, density_discr,
        wrangler, group_factory,
        kernel_length_scale=None,
        scaled_max_curvature_threshold=None,
        expansion_disturbance_tolerance=None,
        maxiter=None, debug=None, visualize=False):
    from pytential import bind, sym
    from meshmode.discretization.connection import ChainedDiscretizationConnection
    if lpot_source._disable_refinement:
        return density_discr, ChainedDiscretizationConnection([],
                from_discr=density_discr)

    from meshmode.mesh.refinement import RefinerWithoutAdjacency
    refiner = RefinerWithoutAdjacency(density_discr.mesh)

    # TODO: Stop doing redundant checks by avoiding panels which no longer need
    # refinement.

    connections = []
    violated_criteria = []
    iter_violated_criteria = ["start"]
    niter = 0

    actx = wrangler.array_context

    stage1_density_discr = density_discr
    while iter_violated_criteria:
        iter_violated_criteria = []
        niter += 1

        if niter > maxiter:
            _warn_max_iterations(
                    violated_criteria, expansion_disturbance_tolerance)
            break

        refine_flags = make_empty_refine_flags(
                wrangler.queue, stage1_density_discr)

        if kernel_length_scale is not None:
            with ProcessLogger(logger,
                    "checking kernel length scale to panel size ratio"):

                quad_resolution = bind(stage1_density_discr,
                        sym._quad_resolution(stage1_density_discr.ambient_dim,
                            dofdesc=sym.GRANULARITY_ELEMENT))(actx)

                violates_kernel_length_scale = \
                        wrangler.check_element_prop_threshold(
                                element_property=quad_resolution,
                                threshold=kernel_length_scale,
                                refine_flags=refine_flags, debug=debug)

                if violates_kernel_length_scale:
                    iter_violated_criteria.append("kernel length scale")
                    _visualize_refinement(actx, stage1_density_discr,
                            niter, 1, "kernel-length-scale", refine_flags,
                            visualize=visualize)

        if scaled_max_curvature_threshold is not None:
            with ProcessLogger(logger,
                    "checking scaled max curvature threshold"):
                scaled_max_curv = bind(stage1_density_discr,
                    sym.ElementwiseMax(sym._scaled_max_curvature(
                        stage1_density_discr.ambient_dim),
                        dofdesc=sym.GRANULARITY_ELEMENT))(actx)

                violates_scaled_max_curv = \
                        wrangler.check_element_prop_threshold(
                                element_property=scaled_max_curv,
                                threshold=scaled_max_curvature_threshold,
                                refine_flags=refine_flags, debug=debug)

                if violates_scaled_max_curv:
                    iter_violated_criteria.append("curvature")
                    _visualize_refinement(actx, stage1_density_discr,
                            niter, 1, "curvature", refine_flags,
                            visualize=visualize)

        if not iter_violated_criteria:
            # Only start building trees once the simple length-based criteria
            # are happy.
            places = _make_temporary_collection(lpot_source,
                    stage1_density_discr=stage1_density_discr)

            # Build tree and auxiliary data.
            # FIXME: The tree should not have to be rebuilt at each iteration.
            tree = wrangler.build_tree(places,
                    sources_list=[places.auto_source.geometry])
            peer_lists = wrangler.find_peer_lists(tree)

            has_disturbed_expansions = \
                    wrangler.check_expansion_disks_undisturbed_by_sources(
                            stage1_density_discr, tree, peer_lists,
                            expansion_disturbance_tolerance,
                            refine_flags, debug)
            if has_disturbed_expansions:
                iter_violated_criteria.append("disturbed expansions")
                _visualize_refinement(actx, stage1_density_discr,
                        niter, 1, "disturbed-expansions", refine_flags,
                        visualize=visualize)

            del tree
            del peer_lists

        if iter_violated_criteria:
            violated_criteria.append(" and ".join(iter_violated_criteria))

            conn = wrangler.refine(
                    stage1_density_discr, refiner, refine_flags,
                    group_factory, debug)
            stage1_density_discr = conn.to_discr
            connections.append(conn)

        del refine_flags

    conn = ChainedDiscretizationConnection(connections,
            from_discr=density_discr)

    return stage1_density_discr, conn


def _refine_qbx_stage2(lpot_source, stage1_density_discr,
        wrangler, group_factory,
        expansion_disturbance_tolerance=None,
        force_stage2_uniform_refinement_rounds=None,
        maxiter=None, debug=None, visualize=False):
    from meshmode.discretization.connection import ChainedDiscretizationConnection
    if lpot_source._disable_refinement:
        return stage1_density_discr, ChainedDiscretizationConnection([],
                from_discr=stage1_density_discr)

    from meshmode.mesh.refinement import RefinerWithoutAdjacency
    refiner = RefinerWithoutAdjacency(stage1_density_discr.mesh)

    # TODO: Stop doing redundant checks by avoiding panels which no longer need
    # refinement.

    connections = []
    violated_criteria = []
    iter_violated_criteria = ["start"]
    niter = 0

    stage2_density_discr = stage1_density_discr
    while iter_violated_criteria:
        iter_violated_criteria = []
        niter += 1

        if niter > maxiter:
            _warn_max_iterations(
                    violated_criteria, expansion_disturbance_tolerance)
            break

        places = _make_temporary_collection(lpot_source,
                stage1_density_discr=stage1_density_discr,
                stage2_density_discr=stage2_density_discr)

        # Build tree and auxiliary data.
        # FIXME: The tree should not have to be rebuilt at each iteration.
        tree = wrangler.build_tree(places,
                sources_list=[places.auto_source.geometry],
                use_stage2_discr=True)
        peer_lists = wrangler.find_peer_lists(tree)
        refine_flags = make_empty_refine_flags(
                wrangler.queue, stage2_density_discr)

        has_insufficient_quad_resolution = \
                wrangler.check_sufficient_source_quadrature_resolution(
                        stage2_density_discr, tree, peer_lists, refine_flags,
                        debug)
        if has_insufficient_quad_resolution:
            iter_violated_criteria.append("insufficient quadrature resolution")
            _visualize_refinement(wrangler.array_context, stage2_density_discr,
                    niter, 2, "quad-resolution", refine_flags,
                    visualize=visualize)

        if iter_violated_criteria:
            violated_criteria.append(" and ".join(iter_violated_criteria))

            conn = wrangler.refine(
                    stage2_density_discr,
                    refiner, refine_flags, group_factory, debug)
            stage2_density_discr = conn.to_discr
            connections.append(conn)

        del tree
        del refine_flags
        del peer_lists

    for _ in range(force_stage2_uniform_refinement_rounds):
        conn = wrangler.refine(
                stage2_density_discr,
                refiner,
                np.ones(stage2_density_discr.mesh.nelements, dtype=bool),
                group_factory, debug)
        stage2_density_discr = conn.to_discr
        connections.append(conn)

    conn = ChainedDiscretizationConnection(connections,
            from_discr=stage1_density_discr)

    return stage2_density_discr, conn


def _refine_qbx_quad_stage2(lpot_source, stage2_density_discr):
    from meshmode.discretization.connection import make_same_mesh_connection
    discr = _make_quad_stage2_discr(lpot_source, stage2_density_discr)
    conn = make_same_mesh_connection(
            lpot_source._setup_actx, discr, stage2_density_discr)

    return discr, conn

# }}}


# {{{ _refine_for_global_qbx

def _refine_for_global_qbx(places, dofdesc, wrangler,
        group_factory=None,
        kernel_length_scale=None,
        force_stage2_uniform_refinement_rounds=None,
        scaled_max_curvature_threshold=None,
        expansion_disturbance_tolerance=None,
        maxiter=None, debug=None, visualize=False,
        _copy_collection=False):
    """Entry point for calling the refiner. Once the refinement is complete,
    the refined discretizations can be obtained from *places* by calling
    :meth:`~pytential.GeometryCollection.get_discretization`.

    :returns: a new version of the :class:`pytential.GeometryCollection`
        *places* with (what)?
        Depending on *_copy_collection*, *places* is updated in-place
        or copied.
    """

    from pytential import sym
    dofdesc = sym.as_dofdesc(dofdesc)

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = places.get_geometry(dofdesc.geometry)
    if not isinstance(lpot_source, QBXLayerPotentialSource):
        raise ValueError(f"'{dofdesc.geometry}' is not a QBXLayerPotentialSource")

    # {{{

    if maxiter is None:
        maxiter = 10

    if debug is None:
        # FIXME: Set debug=False by default once everything works.
        debug = lpot_source.debug

    if expansion_disturbance_tolerance is None:
        expansion_disturbance_tolerance = 0.025

    if force_stage2_uniform_refinement_rounds is None:
        force_stage2_uniform_refinement_rounds = 0

    if group_factory is None:
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        group_factory = InterpolatoryQuadratureSimplexGroupFactory(
                lpot_source.density_discr.groups[0].order)

    # }}}

    # {{{

    # FIXME: would be nice if this was an IntFlag or something ordered
    stage_index_map = {
            sym.QBX_SOURCE_STAGE1: 1,
            sym.QBX_SOURCE_STAGE2: 2,
            sym.QBX_SOURCE_QUAD_STAGE2: 3
            }
    if dofdesc.discr_stage not in stage_index_map:
        raise ValueError(f"unknown discr stage: {dofdesc.discr_stage}")

    stage_index = stage_index_map[dofdesc.discr_stage]
    geometry = dofdesc.geometry

    def add_to_cache(refine_discr, refine_conn, from_ds, to_ds):
        places._add_discr_to_cache(refine_discr, geometry, to_ds)
        places._add_conn_to_cache(refine_conn, geometry, from_ds, to_ds)

    def get_from_cache(from_ds, to_ds):
        discr = places._get_discr_from_cache(geometry, to_ds)
        conn = places._get_conn_from_cache(geometry, from_ds, to_ds)
        return discr, conn

    if _copy_collection:
        places = places.copy()

    # }}}

    # {{{

    discr = lpot_source.density_discr
    if stage_index >= 1:
        ds = (None, sym.QBX_SOURCE_STAGE1)
        try:
            discr, conn = get_from_cache(*ds)
        except KeyError:
            discr, conn = _refine_qbx_stage1(
                    lpot_source, discr, wrangler, group_factory,
                    kernel_length_scale=kernel_length_scale,
                    scaled_max_curvature_threshold=(
                        scaled_max_curvature_threshold),
                    expansion_disturbance_tolerance=(
                        expansion_disturbance_tolerance),
                    maxiter=maxiter, debug=debug, visualize=visualize)
            add_to_cache(discr, conn, *ds)

    if stage_index >= 2:
        ds = (sym.QBX_SOURCE_STAGE1, sym.QBX_SOURCE_STAGE2)
        try:
            discr, conn = get_from_cache(*ds)
        except KeyError:
            discr, conn = _refine_qbx_stage2(
                    lpot_source, discr, wrangler, group_factory,
                    expansion_disturbance_tolerance=(
                        expansion_disturbance_tolerance),
                    force_stage2_uniform_refinement_rounds=(
                        force_stage2_uniform_refinement_rounds),
                    maxiter=maxiter, debug=debug, visualize=visualize)
            add_to_cache(discr, conn, *ds)

    if stage_index >= 3:
        ds = (sym.QBX_SOURCE_STAGE2, sym.QBX_SOURCE_QUAD_STAGE2)
        try:
            discr, conn = get_from_cache(*ds)
        except KeyError:
            discr, conn = _refine_qbx_quad_stage2(lpot_source, discr)
            add_to_cache(discr, conn, *ds)

    # }}}

    return places

# }}}


# {{{ refine_geometry_collection

def refine_geometry_collection(places,
        group_factory=None,
        refine_discr_stage=None,
        kernel_length_scale=None,
        force_stage2_uniform_refinement_rounds=None,
        scaled_max_curvature_threshold=None,
        expansion_disturbance_tolerance=None,
        maxiter=None,
        debug=None, visualize=False):
    """Entry point for refining all the
    :class:`~pytential.qbx.QBXLayerPotentialSource` in the given collection.
    The :class:`~pytential.GeometryCollection` performs
    on-demand refinement, but this function can be used to tweak the
    parameters.

    :arg places: A :class:`~pytential.GeometryCollection`.
    :arg refine_discr_stage: Defines up to which stage the refinement should
        be performed. One of
        :class:`~pytential.symbolic.primitives.QBX_SOURCE_STAGE1`,
        :class:`~pytential.symbolic.primitives.QBX_SOURCE_STAGE2` or
        :class:`~pytential.symbolic.primitives.QBX_SOURCE_QUAD_STAGE2`.
    :arg group_factory: An instance of
        :class:`meshmode.discretization.poly_element.ElementGroupFactory`. Used for
        discretizing the coarse refined mesh.

    :arg kernel_length_scale: The kernel length scale, or *None* if not
        applicable. All panels are refined to below this size.
    :arg maxiter: The maximum number of refiner iterations.
    """

    from pytential import sym
    if refine_discr_stage is None:
        if force_stage2_uniform_refinement_rounds is not None:
            refine_discr_stage = sym.QBX_SOURCE_STAGE2
        else:
            refine_discr_stage = sym.QBX_SOURCE_STAGE1

    from pytential.qbx import QBXLayerPotentialSource
    places = places.copy()
    for geometry in places.places:
        dofdesc = sym.as_dofdesc(geometry).copy(
                discr_stage=refine_discr_stage)
        lpot_source = places.get_geometry(dofdesc.geometry)
        if not isinstance(lpot_source, QBXLayerPotentialSource):
            continue

        _refine_for_global_qbx(places, dofdesc,
                lpot_source.refiner_code_container.get_wrangler(),
                group_factory=group_factory,
                kernel_length_scale=kernel_length_scale,
                scaled_max_curvature_threshold=scaled_max_curvature_threshold,
                expansion_disturbance_tolerance=expansion_disturbance_tolerance,
                force_stage2_uniform_refinement_rounds=(
                    force_stage2_uniform_refinement_rounds),
                maxiter=maxiter, debug=debug, visualize=visualize,
                _copy_collection=False)

    return places

# }}}

# vim: foldmethod=marker:filetype=pyopencl
