# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
from six.moves import range, zip

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


# {{{ tree creation

class TreeWithQBXMetadataBuilder(object):

    class TreeWithQBXMetadata(Tree):
        """
        .. attribute:: nqbxpanels
        .. attribuet:: nsources
        .. attribute:: ncenters

        .. attribute:: box_to_qbx_panel_starts
        .. attribute:: box_to_qbx_panel_lists

        .. attribute:: qbx_panel_to_source_starts
        .. attribute:: qbx_panel_to_center_starts

        .. attribute:: qbx_user_source_range
        .. attribute:: qbx_user_center_range
        .. attribute:: qbx_user_panel_range
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

    def __call__(self, queue, lpot_source):
        """
        Return a :class:`TreeWithQBXMetadata` built from the given layer
        potential source.

        :arg queue: An instance of :class:`pyopencl.CommandQueue`

        :arg lpot_source: An instance of
            :class:`pytential.qbx.NewQBXLayerPotentialSource`.
        """
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
        # Each source gets an interior / exterior center.
        assert 2 * nsources == ncenters

        qbx_user_source_range = range(0, nsources)
        nsourcescenters = 3 * nsources
        qbx_user_center_range = range(nsources, nsourcescenters)
        qbx_user_panel_range = range(nsourcescenters, nsourcescenters + npanels)

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
        qbx_panel_flags.fill(0)
        qbx_panel_flags[3 * nsources:].fill(1)
        qbx_panel_flags.finish()

        from boxtree.tree import filter_target_lists_in_user_order
        box_to_qbx_panel = (
                filter_target_lists_in_user_order(queue, tree, qbx_panel_flags)
                .with_queue(queue))
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

        return self.TreeWithQBXMetadata(
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

# vim: foldmethod=marker:filetype=pyopencl
