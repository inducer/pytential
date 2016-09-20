# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
from six.moves import range, zip

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
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


import loopy as lp
import numpy as np
from pytools import memoize_method
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory

import pyopencl as cl

import logging
logger = logging.getLogger(__name__)


# {{{ layer potential source

class NewQBXLayerPotentialSource(object):
    """A source discretization for a QBX layer potential.

    .. attribute :: density_discr
    .. attribute :: qbx_order
    .. attribute :: fmm_order
    .. attribute :: cl_context
    .. automethod :: centers
    .. automethod :: panel_sizes
    .. automethod :: weights_and_area_elements

    See :ref:`qbxguts` for some information on the inner workings of this.
    """
    def __init__(self, density_discr, fine_order,
            qbx_order=None, fmm_order=None,
            qbx_level_to_order=None, fmm_level_to_order=None,
            # FIXME set debug=False once everything works
            real_dtype=np.float64, debug=True,
            performance_data_file=None):
        """
        :arg fine_order: The total degree to which the (upsampled)
            underlying quadrature is exact.
        :arg fmm_order: `False` for direct calculation. ``None`` will set
            a reasonable(-ish?) default.
        """

        self.fine_density_discr = Discretization(
                density_discr.cl_context, density_discr.mesh,
                QuadratureSimplexGroupFactory(fine_order), real_dtype)

        from meshmode.discretization.connection import make_same_mesh_connection
        self.resampler = make_same_mesh_connection(
                self.fine_density_discr, density_discr)

        if fmm_level_to_order is None:
            if fmm_order is None and qbx_order is not None:
                fmm_order = qbx_order + 1

        if qbx_order is not None and qbx_level_to_order is not None:
            raise TypeError("may not specify both qbx_order an qbx_level_to_order")
        if fmm_order is not None and fmm_level_to_order is not None:
            raise TypeError("may not specify both fmm_order an fmm_level_to_order")

        if fmm_level_to_order is None:
            if fmm_order is False:
                fmm_level_to_order = False
            else:
                def fmm_level_to_order(level):
                    return fmm_order

        if qbx_level_to_order is None:
            def qbx_level_to_order(level):
                return qbx_order

        self.fine_order = fine_order
        self.qbx_level_to_order = qbx_level_to_order
        self.density_discr = density_discr
        self.fmm_level_to_order = fmm_level_to_order
        self.debug = debug
        self.performance_data_file = performance_data_file

    def el_view(self, discr, group_nr, global_array):
        """Return a view of *global_array* of shape
        ``(..., discr.groups[group_nr].nelements)``
        where *global_array* is of shape ``(..., nelements)``,
        where *nelements* is the global (per-discretization) node count.
        """

        group = discr.groups[group_nr]
        el_nr_base = sum(group.nelements for group in discr.groups[:group_nr])

        return global_array[
            ..., el_nr_base:el_nr_base + group.nelements] \
            .reshape(
                global_array.shape[:-1]
                + (group.nelements,))

    @property
    def ambient_dim(self):
        return self.density_discr.ambient_dim

    @property
    def cl_context(self):
        return self.density_discr.cl_context

    @property
    def real_dtype(self):
        return self.density_discr.real_dtype

    @property
    def complex_dtype(self):
        return self.density_discr.complex_dtype

    @memoize_method
    def panel_centers_of_mass(self):
        knl = lp.make_kernel(
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nunit_nodes}""",
            """
                panels[dim, k] = sum(i, nodes[dim, k, i])/nunit_nodes
                """,
            default_offset=lp.auto, name="find_panel_centers_of_mass")

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.split_iname(knl, "k", 128, inner_tag="l.0", outer_tag="g.0")
        knl = lp.tag_inames(knl, dict(dim="ilp"))

        with cl.CommandQueue(self.cl_context) as queue:
            mesh = self.density_discr.mesh
            panels = cl.array.empty(queue, (mesh.ambient_dim, mesh.nelements),
                                    dtype=self.density_discr.real_dtype)
            for group_nr, group in enumerate(self.density_discr.groups):
                _, (result,) = knl(queue,
                    nelements=group.nelements,
                    nunit_nodes=group.nunit_nodes,
                    nodes=group.view(self.density_discr.nodes()),
                    panels=self.el_view(self.density_discr, group_nr, panels))
            panels.finish()
            panels = panels.with_queue(None)
            return tuple(panels[d, :] for d in range(mesh.ambient_dim))

    @memoize_method
    def panel_sizes(self, last_dim_length="nnodes"):
        assert last_dim_length in ["nnodes", "nelements"]
        # To get the panel size this does the equivalent of ∫ 1 ds.
        # FIXME: Kernel optimizations

        discr = self.density_discr

        if last_dim_length == "nnodes":
            knl = lp.make_kernel(
                "{[i,j,k]: 0<=i<nelements and 0<=j,k<nunit_nodes}",
                "panel_sizes[i,j] = sum(k, ds[i,k])",
                name="compute_size")
            knl = lp.tag_inames(knl, dict(i="g.0", j="l.0"))

            def panel_size_view(discr, group_nr):
                return discr.groups[group_nr].view

        elif last_dim_length == "nelements":
            knl = lp.make_kernel(
                "{[i,j]: 0<=i<nelements and 0<=j<nunit_nodes}",
                "panel_sizes[i] = sum(j, ds[i,j])",
                name="compute_size")
            from functools import partial

            def panel_size_view(discr, group_nr):
                return partial(self.el_view, discr, group_nr)

        with cl.CommandQueue(self.cl_context) as queue:
            from pytential import bind, sym
            ds = bind(discr, sym.area_element() * sym.QWeight())(queue)
            panel_sizes = cl.array.empty(
                queue, discr.nnodes
                if last_dim_length == "nnodes"
                else discr.mesh.nelements, discr.real_dtype)
            for group_nr, group in enumerate(discr.groups):
                _, (result,) = knl(queue,
                    nelements=group.nelements,
                    nunit_nodes=group.nunit_nodes,
                    ds=group.view(ds),
                    panel_sizes=panel_size_view(
                        discr, group_nr)(panel_sizes))
            panel_sizes.finish()
            return panel_sizes.with_queue(None)

    @memoize_method
    def centers(self, sign):
        from pytential import sym, bind
        with cl.CommandQueue(self.cl_context) as queue:
            nodes = bind(self.density_discr, sym.Nodes())(queue)
            normals = bind(self.density_discr, sym.normal())(queue)
            panel_sizes = self.panel_sizes().with_queue(queue)
            return (nodes + normals * sign * panel_sizes / 2).as_vector(np.object)

    @memoize_method
    def weights_and_area_elements(self):
        import pytential.symbolic.primitives as p
        from pytential.symbolic.execution import bind
        with cl.CommandQueue(self.cl_context) as queue:
            # fine_density_discr is not guaranteed to be usable for
            # interpolation/differentiation. Use density_discr to find
            # area element instead, then upsample that.

            area_element = self.resampler(queue,
                    bind(self.density_discr,
                        p.area_element())(queue))

            qweight = bind(self.fine_density_discr, p.QWeight())(queue)

            return (area_element.with_queue(queue)*qweight).with_queue(None)

# }}}

from pyopencl.elementwise import ElementwiseTemplate
from boxtree.area_query import AreaQueryElementwiseTemplate
from boxtree.tools import InlineBinarySearch


REFINER_C_MACROS = r"""//CL:mako//
// A note on node numberings: sources, centers, and panels each
// have their own numbering starting at 0. These macros convert
// the per-class numbering into the internal tree particle number.
#define INDEX_FOR_CENTER_PARTICLE(i) (sorted_target_ids[center_offset + i])
#define INDEX_FOR_PANEL_PARTICLE(i) (sorted_target_ids[panel_offset + i])
#define INDEX_FOR_SOURCE_PARTICLE(i) (sorted_target_ids[source_offset + i])

## Convert to dict first, as this may be passed as a tuple-of-tuples.
<% vec_types_dict = dict(vec_types) %>
typedef ${dtype_to_ctype(vec_types_dict[coord_dtype, dimensions])} coord_vec_t;
"""


REFINER_MAKO_DEFS = r"""//CL:mako//
<%def name="load_particle(particle, coords)">
    <% zerovect = ["0"] * 2 ** (dimensions - 1).bit_length() %>
    /* Zero initialize, to allow for use in distance computations. */
    ${coords} = (coord_vec_t) (${", ".join(zerovect)});

    %for ax in AXIS_NAMES[:dimensions]:
        ${coords}.${ax} = sources_${ax}[${particle}];
    %endfor
</%def>
"""


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
            coord_t *sources_${ax},
        %endfor
        """,
    operation=REFINER_MAKO_DEFS + REFINER_C_MACROS + r"""//CL:mako//
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


# Implements "Algorithm for triggering refinement based on Condition 1"
CENTER_IS_CLOSEST_TO_ORIG_PANEL_REFINER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_panel_starts,
        particle_id_t *box_to_panel_lists,
        /* XXX: starts dtype */
        particle_id_t *panel_to_source_starts,
        particle_id_t *panel_to_center_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
        int npanels,
        coord_t r_max,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *sources_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=REFINER_MAKO_DEFS + REFINER_C_MACROS + r"""
        particle_id_t my_panel = bsearch(panel_to_center_starts, npanels + 1, i);

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(i)", ball_center)}
        ${ball_radius} = r_max + panel_sizes[my_panel] / 2;
        """,
    leaf_found_op=REFINER_MAKO_DEFS + r"""
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
    name="refine_center_to_own_panel",
    preamble=str(InlineBinarySearch("particle_id_t")))


# Implements "Algorithm for triggering refinement based on Condition 2"
CENTER_IS_FAR_FROM_NONNEIGHBOR_PANEL_REFINER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_panel_starts,
        particle_id_t *box_to_panel_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t *panel_to_center_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t panel_offset,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
        /* XXX: data type... */
        particle_id_t *panel_adjacency_starts,
        particle_id_t *panel_adjacency_lists,
        int npanels,
        coord_t *tunnel_query_dists,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *sources_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=REFINER_MAKO_DEFS + REFINER_C_MACROS + r"""
        particle_id_t my_panel = bsearch(panel_to_center_starts, npanels + 1, i);
        coord_vec_t my_center_coords;

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(i)", "my_center_coords")}
        ${load_particle("INDEX_FOR_PANEL_PARTICLE(my_panel)", ball_center)}
        ${ball_radius} = tunnel_query_dists[my_panel];
        """,
    leaf_found_op=REFINER_MAKO_DEFS + r"""
        for (particle_id_t panel_idx = box_to_panel_starts[${leaf_box_id}];
             panel_idx < box_to_panel_starts[${leaf_box_id} + 1];
             ++panel_idx)
        {
            particle_id_t panel = box_to_panel_lists[panel_idx];

            bool is_self_or_adjacent = (my_panel == panel);

            for (particle_id_t adj_panel_idx = panel_adjacency_starts[my_panel];
                 adj_panel_idx < panel_adjacency_starts[my_panel + 1];
                 ++adj_panel_idx)
            {
                is_self_or_adjacent |= (
                    panel_adjacency_lists[adj_panel_idx] == panel);
            }

            // Skip self and adjacent panels.
            if (is_self_or_adjacent)
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
                    distance(my_center_coords, source_coords)
                    <= panel_sizes[panel] / 2);
            }

            if (is_close)
            {
                panel_refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="refine_center_to_other_panel",
    preamble=str(InlineBinarySearch("particle_id_t")))


from boxtree.tree import Tree


class TreeWithQBXMetadata(Tree):
    """
    .. attribute:: box_to_qbx_panel_starts
    .. attribute:: box_to_qbx_panel_lists
    .. attribute:: qbx_panel_to_source_starts
    .. attribute:: qbx_panel_to_center_starts
    .. attribute:: qbx_user_source_range
    .. attribute:: qbx_user_center_range
    .. attribute:: qbx_user_panel_range
    XXX
    """
    pass


class QBXLayerPotentialSourceRefiner(object):

    def __init__(self, context):
        self.context = context
        from boxtree.tree_build import TreeBuilder
        self.tree_builder = TreeBuilder(self.context)
        from boxtree.area_query import PeerListFinder
        self.peer_list_finder = PeerListFinder(self.context)

    # {{{ tree creation

    def create_tree(self, queue, lpot_source):
        # The ordering of particles is as follows:
        # - sources go first
        # - then centers
        # - then panels (=centers of mass)

        sources = lpot_source.density_discr.nodes()
        centers = self.get_interleaved_centers(queue, lpot_source)
        centers_of_mass = lpot_source.panel_centers_of_mass()

        particles = tuple(
                cl.array.concatenate(dim_coords, queue=queue)
                for dim_coords in zip(sources, centers, centers_of_mass))

        nparticles = len(particles[0])
        npanels = len(centers_of_mass[0])
        nsources = len(sources[0])
        ncenters = len(centers[0])
        assert 2 * nsources == ncenters

        qbx_user_source_range = range(0, nsources)
        nsourcescenters = 3 * nsources
        qbx_user_center_range = range(nsources, nsourcescenters)
        qbx_user_panel_range = range(nsourcescenters, nsourcescenters + npanels)

        # Build tree with sources, centers, and centers of mass. Split boxes
        # only because of sources.
        refine_weights = cl.array.zeros(queue, nparticles, np.int32)
        refine_weights[:nsources].fill(1)
        MAX_REFINE_WEIGHT = 30

        refine_weights.finish()

        tree, evt = self.tree_builder(queue, particles,
                                      max_leaf_refine_weight=MAX_REFINE_WEIGHT,
                                      refine_weights=refine_weights)

        # Compute box => panel relation
        qbx_panel_flags = refine_weights
        qbx_panel_flags.fill(0)
        qbx_panel_flags[3 * nsources:].fill(1)
        qbx_panel_flags.finish()

        from boxtree.tree import filter_target_lists_in_user_order
        box_to_qbx_panel = \
                filter_target_lists_in_user_order(queue, tree, qbx_panel_flags).\
                with_queue(queue)
        # Fix up offset.
        box_to_qbx_panel.target_lists -= 3 * nsources

        qbx_panel_to_source_starts = cl.array.empty(
            queue, npanels + 1, dtype=tree.particle_id_dtype)

        # Compute panel => source relation
        el_offset = 0
        for group in lpot_source.density_discr.groups:
            qbx_panel_to_source_starts[el_offset:el_offset + group.nelements] = \
                    cl.array.arange(queue, group.node_nr_base,
                                    group.node_nr_base + group.nnodes,
                                    group.nunit_nodes,
                                    dtype=tree.particle_id_dtype)
            el_offset += group.nelements
        qbx_panel_to_source_starts[-1] = nsources

        # Compute panel => center relation
        qbx_panel_to_center_starts = 2 * qbx_panel_to_source_starts

        # Transfer all tree attributes.
        tree_attrs = {}
        for attr_name in tree.__class__.fields:
            try:
                tree_attrs[attr_name] = getattr(tree, attr_name)
            except AttributeError:
                pass

        logger.info("refiner: done building tree")

        # XXX: evt management
        return TreeWithQBXMetadata(
            box_to_qbx_panel_starts=box_to_qbx_panel.target_starts,
            box_to_qbx_panel_lists=box_to_qbx_panel.target_lists,
            qbx_panel_to_source_starts=qbx_panel_to_source_starts,
            qbx_panel_to_center_starts=qbx_panel_to_center_starts,
            qbx_user_source_range=qbx_user_source_range,
            qbx_user_panel_range=qbx_user_panel_range,
            qbx_user_center_range=qbx_user_center_range,
            nqbxpanels=npanels,
            nqbxsources=nsources,
            nqbxcenters=ncenters,
            **tree_attrs).with_queue(None)

    # }}}

    # {{{ kernels

    @memoize_method
    def get_tunnel_query_distance_finder(self, dimensions, coord_dtype,
                                         particle_id_dtype):
        from pyopencl.tools import dtype_to_ctype
        from boxtree.tools import AXIS_NAMES
        logger.info("refiner: building tunnel query distance finder kernel")

        knl = TUNNEL_QUERY_DISTANCE_FINDER_TEMPLATE.build(
                self.context,
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
    def get_center_is_closest_to_orig_panel_refiner(self, dimensions,
                                                    coord_dtype, box_id_dtype,
                                                    peer_list_idx_dtype,
                                                    particle_id_dtype,
                                                    max_levels):
        return CENTER_IS_CLOSEST_TO_ORIG_PANEL_REFINER.generate(self.context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def get_center_is_far_from_nonneighbor_panel_refiner(self, dimensions,
                                                         coord_dtype,
                                                         box_id_dtype,
                                                         peer_list_idx_dtype,
                                                         particle_id_dtype,
                                                         max_levels):
        return CENTER_IS_FAR_FROM_NONNEIGHBOR_PANEL_REFINER.generate(self.context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def get_2_to_1_panel_ratio_refiner(self):
        knl = lp.make_kernel([
            "{[panel]: 0<=panel<npanels}",
            "{[ineighbor]: neighbor_start<=ineighbor<neighbor_stop}"
            ],
            """
            for panel
                <> neighbor_start = panel_adjacency_starts[panel]
                <> neighbor_stop = panel_adjacency_starts[panel + 1]
                for ineighbor
                    <> neighbor = panel_adjacency_lists[ineighbor]
                    <> oversize = (refine_flags_prev[panel] == 0
                           and (
                               (panel_sizes[panel] > 2 * panel_sizes[neighbor]) or
                               (panel_sizes[panel] > panel_sizes[neighbor] and
                                   refine_flags_prev[neighbor] == 1)))
                    refine_flags[panel] = 1 {if=oversize}
                    refine_flags_updated = 1 {if=oversize}
                end
            end
            """, [
            lp.GlobalArg("panel_adjacency_lists", shape=None),
            "..."
            ],
            options="return_dict")
        knl = lp.split_iname(knl, "panel", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    @memoize_method
    def get_helmholtz_k_to_panel_ratio_refiner(self):
        knl = lp.make_kernel(
            "{[panel]: 0<=panel<npanels}",
            """
            for panel
                <> oversize = panel_sizes[panel] > 5 * omega
                refine_flags[panel] = 1 {if=oversize}
                refine_flags_updated = 1 {if=oversize}
            end
            """,
            options="return_dict")
        knl = lp.split_iname(knl, "panel", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    @memoize_method
    def get_interleaver_kernel(self):
        knl = lp.make_kernel(
            "{[i]: 0<=i<srclen}",
            """
            dst[2*i] = src1[i]
            dst[2*i + 1] = src2[i]
            """, [
                lp.GlobalArg("dst", shape=None),
                "..."
            ])
        knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    # }}}

    # {{{ refinement triggering

    def refinement_check_center_is_closest_to_orig_panel(self, queue, tree,
                lpot_source, peer_lists, tq_dists, refine_flags, wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_center_is_closest_to_orig_panel_refiner(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        found_panel_to_refine = cl.array.zeros(queue, 1, np.int32)
        found_panel_to_refine.finish()

        r_max = cl.array.max(tq_dists).get()

        from boxtree.area_query import AreaQueryElementwiseTemplate
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_panel_starts,
                tree.box_to_qbx_panel_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_panel_to_center_starts,
                tree.qbx_user_source_range.start,
                tree.qbx_user_center_range.start,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("nelements"),
                tree.nqbxpanels,
                r_max,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=range(tree.nqbxcenters),
            queue=queue)

        cl.wait_for_events([evt])

        return found_panel_to_refine.get()[0] == 1

    def refinement_check_center_is_far_from_nonneighbor_panels(self, queue,
                tree, lpot_source, peer_lists, tq_dists, refine_flags,
                wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_center_is_far_from_nonneighbor_panel_refiner(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        found_panel_to_refine = cl.array.zeros(queue, 1, np.int32)
        found_panel_to_refine.finish()

        from boxtree.area_query import AreaQueryElementwiseTemplate
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        adjacency = self.get_adjacency_on_device(queue, lpot_source)

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_panel_starts,
                tree.box_to_qbx_panel_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_panel_to_center_starts,
                tree.qbx_user_source_range.start,
                tree.qbx_user_center_range.start,
                tree.qbx_user_panel_range.start,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("nelements"),
                adjacency.adjacency_starts,
                adjacency.adjacency_lists,
                tree.nqbxpanels,
                tq_dists,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=range(tree.nqbxcenters),
            queue=queue)

        cl.wait_for_events([evt])

        return found_panel_to_refine.get()[0] == 1

    def refinement_check_helmholtz_k_panel_ratio(self, queue, lpot_source,
                                                 helmholtz_k, refine_flags):
        knl = self.get_helmholtz_k_to_panel_ratio_refiner()

        evt, out = knl(queue,
                       panel_sizes=lpot_source.panel_sizes("nelements"),
                       refine_flags=refine_flags,
                       refine_flags_updated=np.array(0))

        cl.wait_for_events([evt])

        return (out["refine_flags_updated"].get() == 1).all()

    def refinement_check_2_to_1_panel_size_ratio(self, queue, lpot_source,
                refine_flags, wait_for):
        knl = self.get_2_to_1_panel_ratio_refiner()
        adjacency = self.get_adjacency_on_device(queue, lpot_source)

        done_checking = False
        refine_flags_updated = False

        while not done_checking:
            evt, out = knl(queue,
                           npanels=lpot_source.density_discr.mesh.nelements,
                           panel_sizes=lpot_source.panel_sizes("nelements"),
                           refine_flags=refine_flags,
                           # It's safe to pass this here, as the resulting data
                           # race won't affect the final result of the
                           # computation.
                           refine_flags_prev=refine_flags,
                           refine_flags_updated=np.array(0),
                           panel_adjacency_starts=adjacency.adjacency_starts,
                           panel_adjacency_lists=adjacency.adjacency_lists,
                           wait_for=wait_for)

            cl.wait_for_events([evt])

            if (out["refine_flags_updated"].get() == 1).all():
                refine_flags_updated = True
                done_checking = False
            else:
                done_checking = True

        return refine_flags_updated

    # }}}

    # {{{ other utilities

    def get_tunnel_query_dists(self, queue, tree, lpot_source):
        nqbxpanels = lpot_source.density_discr.mesh.nelements
        # atomic_max only works on float32
        tq_dists = cl.array.zeros(queue, nqbxpanels, np.float32)
        tq_dists.finish()

        knl = self.get_tunnel_query_distance_finder(tree.dimensions,
                tree.coord_dtype, tree.particle_id_dtype)

        evt = knl(tree.qbx_user_source_range.start,
                  tree.qbx_user_panel_range.start,
                  nqbxpanels,
                  tree.qbx_panel_to_source_starts,
                  tree.sorted_target_ids,
                  lpot_source.panel_sizes("nelements"),
                  tq_dists,
                  *tree.sources,
                  queue=queue,
                  range=tree.qbx_user_source_range)

        if tree.coord_dtype != tq_dists.dtype:
            tq_dists = tq_dists.astype(tree.coord_dtype)

        return tq_dists, evt

    def get_adjacency_on_device(self, queue, lpot_source):
        """
        Take adjacency information from the mesh and place it onto the device.
        """
        from boxtree.tools import DeviceDataRecord
        adjacency = lpot_source.density_discr.mesh.nodal_adjacency
        adjacency_starts = cl.array.to_device(queue, adjacency.neighbors_starts)
        adjacency_lists = cl.array.to_device(queue, adjacency.neighbors)

        return DeviceDataRecord(
            adjacency_starts=adjacency_starts,
            adjacency_lists=adjacency_lists)

    def get_refine_flags(self, queue, lpot_source):
        """
        Return an array on the device suitable for use as element refine flags.
        """
        result = cl.array.zeros(
            queue, lpot_source.density_discr.mesh.nelements, np.int32)
        return result, result.events[0]

    def get_interleaved_centers(self, queue, lpot_source):
        """
        Return an array of shape (dim, ncenters) in which interior centers are placed
        next to corresponding exterior centers.
        """
        knl = self.get_interleaver_kernel()
        int_centers = lpot_source.centers(-1)
        ext_centers = lpot_source.centers(+1)

        result = []
        wait_for = []

        for int_axis, ext_axis in zip(int_centers, ext_centers):
            axis = cl.array.empty(queue, len(int_axis) * 2, int_axis.dtype)
            evt, _ = knl(queue, src1=int_axis, src2=ext_axis, dst=axis)
            result.append(axis)
            wait_for.append(evt)

        cl.wait_for_events(wait_for)

        return result

    # }}}

    def refine(self, queue, lpot_source, refine_flags, refiner, order):
        if isinstance(refine_flags, cl.array.Array):
            refine_flags = refine_flags.get(queue)
        refine_flags = refine_flags.astype(np.bool)

        logger.info("refiner: calling meshmode")

        refiner.refine(refine_flags)
        from meshmode.discretization.connection import make_refinement_connection

        conn = make_refinement_connection(
                refiner, lpot_source.density_discr,
                QuadratureSimplexGroupFactory(order))

        logger.info("refiner: done calling meshmode")

        new_density_discr = conn.to_discr

        new_lpot_source = NewQBXLayerPotentialSource(
            new_density_discr, lpot_source.fine_order,
            qbx_level_to_order=lpot_source.qbx_level_to_order,
            fmm_level_to_order=lpot_source.fmm_level_to_order,
            # FIXME set debug=False once everything works
            real_dtype=lpot_source.real_dtype, debug=True)

        return new_lpot_source, conn

    def plot_discr(self, lpot_source):
        with cl.CommandQueue(self.context) as queue:
            tree = self.create_tree(queue, lpot_source).get(queue=queue)
            from boxtree.visualization import TreePlotter
            import matplotlib.pyplot as plt
            tp = TreePlotter(tree)
            tp.draw_tree()
            sources = (tree.sources[0], tree.sources[1])
            sti = tree.sorted_target_ids
            plt.plot(sources[0][sti[tree.qbx_user_source_range]],
                     sources[1][sti[tree.qbx_user_source_range]],
                     lw=0, marker=".", label="sources")
            plt.plot(sources[0][sti[tree.qbx_user_center_range]],
                     sources[1][sti[tree.qbx_user_center_range]],
                     lw=0, marker=".", label="centers")
            plt.axis("equal")
            plt.legend()
            plt.show()

    def __call__(self, lpot_source, order, helmholtz_k=None, maxiter=50):
        from meshmode.mesh.refinement import Refiner
        refiner = Refiner(lpot_source.density_discr.mesh)
        connections = []

        with cl.CommandQueue(self.context) as queue:
            done_refining = False
            niter = 0

            while not done_refining:
                niter += 1

                if niter > maxiter:
                    logger.warning(
                        "Max iteration count reached in QBX layer potential source"
                        " refiner.")
                    break

                # Build tree and auxiliary data.
                tree = self.create_tree(queue, lpot_source)
                wait_for = []

                peer_lists, evt = self.peer_list_finder(queue, tree, wait_for)
                wait_for = [evt]

                refine_flags, evt = self.get_refine_flags(queue, lpot_source)
                wait_for.append(evt)

                tq_dists, evt = self.get_tunnel_query_dists(queue, tree, lpot_source)
                wait_for.append(evt)

                # Run refinement checkers.
                must_refine = False

                must_refine |= \
                        self.refinement_check_center_is_closest_to_orig_panel(
                            queue, tree, lpot_source, peer_lists, tq_dists,
                            refine_flags, wait_for)

                must_refine |= \
                        self.refinement_check_center_is_far_from_nonneighbor_panels(
                            queue, tree, lpot_source, peer_lists, tq_dists,
                            refine_flags, wait_for)

                must_refine |= \
                        self.refinement_check_2_to_1_panel_size_ratio(
                            queue, lpot_source, refine_flags, wait_for)

                if helmholtz_k:
                    must_refine |= \
                            self.refinement_check_helmholtz_k_to_panel_size_ratio(
                                queue, lpot_source, helmholtz_k, refine_flags,
                                wait_for)

                if must_refine:
                    lpot_source, conn = self.refine(
                            queue, lpot_source, refine_flags, refiner, order)
                    connections.append(conn)

                del tree
                del peer_lists
                del tq_dists
                del refine_flags
                done_refining = not must_refine

        return lpot_source, connections

# vim: foldmethod=marker:filetype=pyopencl
