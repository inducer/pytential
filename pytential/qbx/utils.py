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


import loopy as lp
import numpy as np
from boxtree.tree import Tree
import pyopencl as cl
import pyopencl.array # noqa
from pytools import memoize_method

import logging
logger = logging.getLogger(__name__)


# {{{ c and mako snippets

QBX_TREE_C_PREAMBLE = r"""//CL:mako//
// A note on node numberings: sources, centers, and panels each
// have their own numbering starting at 0. These macros convert
// the per-class numbering into the internal tree particle number.
#define INDEX_FOR_CENTER_PARTICLE(i) (sorted_target_ids[center_offset + i])
#define INDEX_FOR_PANEL_PARTICLE(i) (sorted_target_ids[panel_offset + i])
#define INDEX_FOR_SOURCE_PARTICLE(i) (sorted_target_ids[source_offset + i])
#define INDEX_FOR_TARGET_PARTICLE(i) (sorted_target_ids[target_offset + i])

## Convert to dict first, as this may be passed as a tuple-of-tuples.
<% vec_types_dict = dict(vec_types) %>
typedef ${dtype_to_ctype(vec_types_dict[coord_dtype, dimensions])} coord_vec_t;
"""


QBX_TREE_MAKO_DEFS = r"""//CL:mako//
<%def name="load_particle(particle, coords)">
    <% zerovect = ["0"] * 2 ** (dimensions - 1).bit_length() %>
    /* Zero initialize, to allow for use in distance computations. */
    ${coords} = (coord_vec_t) (${", ".join(zerovect)});

    %for ax in AXIS_NAMES[:dimensions]:
        ${coords}.${ax} = particles_${ax}[${particle}];
    %endfor
</%def>
"""

# }}}


# {{{ tree creation

class TreeWithQBXMetadataBuilder(object):

    class TreeWithQBXMetadata(Tree):
        """
        .. attribute:: nqbxpanels
        .. attribuet:: nqbxsources
        .. attribute:: nqbxcenters
        .. attribute:: nqbxtargets

        .. attribute:: box_to_qbx_panel_starts
        .. attribute:: box_to_qbx_panel_lists

        .. attribute:: box_to_qbx_target_starts
        .. attribute:: box_to_qbx_target_lists

        .. attribute:: qbx_panel_to_source_starts
        .. attribute:: qbx_panel_to_center_starts

        .. attribute:: qbx_user_source_slice
        .. attribute:: qbx_user_center_slice
        .. attribute:: qbx_user_panel_slice
        .. attribute:: qbx_user_target_slice
        """
        pass

    def __init__(self, context):
        self.context = context
        from boxtree.tree_build import TreeBuilder
        self.tree_builder = TreeBuilder(self.context)

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

    def __call__(self, queue, lpot_source, targets_list=()):
        """
        Return a :class:`TreeWithQBXMetadata` built from the given layer
        potential source.

        :arg queue: An instance of :class:`pyopencl.CommandQueue`

        :arg lpot_source: An instance of
            :class:`pytential.qbx.NewQBXLayerPotentialSource`.

        :arg targets_list: A list of :class:`pytential.target.TargetBase`
        """
        # The ordering of particles is as follows:
        # - sources go first
        # - then centers
        # - then panels (=centers of mass)
        # - then targets

        logger.info("start building tree with qbx metadata")

        sources = lpot_source.density_discr.nodes()
        centers = self.get_interleaved_centers(queue, lpot_source)
        centers_of_mass = lpot_source.panel_centers_of_mass()
        targets = (tgt.nodes() for tgt in targets_list)

        particles = tuple(
                cl.array.concatenate(dim_coords, queue=queue)
                for dim_coords in zip(sources, centers, centers_of_mass, *targets))

        # Counts
        nparticles = len(particles[0])
        npanels = len(centers_of_mass[0])
        nsources = len(sources[0])
        ncenters = len(centers[0])
        # Each source gets an interior / exterior center.
        assert 2 * nsources == ncenters
        ntargets = sum(tgt.nnodes for tgt in targets_list)

        # Slices
        qbx_user_source_slice = slice(0, nsources)
        panel_slice_start = 3 * nsources
        qbx_user_center_slice = slice(nsources, panel_slice_start)
        target_slice_start = panel_slice_start + npanels
        qbx_user_panel_slice = slice(panel_slice_start, panel_slice_start + npanels)
        qbx_user_target_slice = slice(target_slice_start,
                                      target_slice_start + ntargets)

        # Build tree with sources, centers, and centers of mass. Split boxes
        # only because of sources.
        refine_weights = cl.array.zeros(queue, nparticles, np.int32)
        refine_weights[:nsources].fill(1)
        MAX_REFINE_WEIGHT = 128

        refine_weights.finish()

        tree, evt = self.tree_builder(queue, particles,
                                      max_leaf_refine_weight=MAX_REFINE_WEIGHT,
                                      refine_weights=refine_weights)

        # Compute box => panel relation
        qbx_panel_flags = refine_weights
        del refine_weights
        qbx_panel_flags.fill(0)
        qbx_panel_flags[qbx_user_panel_slice].fill(1)
        qbx_panel_flags.finish()

        from boxtree.tree import filter_target_lists_in_user_order
        box_to_qbx_panel = (
                filter_target_lists_in_user_order(queue, tree, qbx_panel_flags)
                .with_queue(queue))
        # Fix up offset.
        box_to_qbx_panel.target_lists -= panel_slice_start

        # Compute box => target relation
        qbx_target_flags = qbx_panel_flags
        del qbx_panel_flags
        qbx_target_flags.fill(0)
        qbx_target_flags[qbx_user_target_slice].fill(1)
        qbx_target_flags.finish()

        box_to_qbx_target = (
                filter_target_lists_in_user_order(queue, tree, qbx_target_flags)
                .with_queue(queue))
        # Fix up offset.
        box_to_qbx_target.target_lists -= target_slice_start

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

        logger.info("done building tree with qbx metadata")

        return self.TreeWithQBXMetadata(
            box_to_qbx_panel_starts=box_to_qbx_panel.target_starts,
            box_to_qbx_panel_lists=box_to_qbx_panel.target_lists,
            box_to_qbx_target_starts=box_to_qbx_target.target_starts,
            box_to_qbx_target_lists=box_to_qbx_target.target_lists,
            qbx_panel_to_source_starts=qbx_panel_to_source_starts,
            qbx_panel_to_center_starts=qbx_panel_to_center_starts,
            qbx_user_source_slice=qbx_user_source_slice,
            qbx_user_panel_slice=qbx_user_panel_slice,
            qbx_user_center_slice=qbx_user_center_slice,
            qbx_user_target_slice=qbx_user_target_slice,
            nqbxpanels=npanels,
            nqbxsources=nsources,
            nqbxcenters=ncenters,
            nqbxtargets=ntargets,
            **tree_attrs).with_queue(None)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
