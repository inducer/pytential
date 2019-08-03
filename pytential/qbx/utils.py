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
from boxtree.tree import Tree
import pyopencl as cl
import pyopencl.array # noqa
from pytools import memoize_method

import logging
logger = logging.getLogger(__name__)

from pytools import log_process


# {{{ c and mako snippets

QBX_TREE_C_PREAMBLE = r"""//CL:mako//
// A note on node numberings: sources, centers, and panels each
// have their own numbering starting at 0. These macros convert
// the per-class numbering into the internal tree particle number.
#define INDEX_FOR_CENTER_PARTICLE(i) (sorted_target_ids[center_offset + i])
#define INDEX_FOR_SOURCE_PARTICLE(i) (sorted_target_ids[source_offset + i])
#define INDEX_FOR_TARGET_PARTICLE(i) (sorted_target_ids[target_offset + i])

#define SOURCE_FOR_CENTER_PARTICLE(i) (i / 2)
#define SIDE_FOR_CENTER_PARTICLE(i) (2 * (i % 2) - 1)

## Convert to dict first, as this may be passed as a tuple-of-tuples.
<% vec_types_dict = dict(vec_types) %>
typedef ${dtype_to_ctype(vec_types_dict[coord_dtype, dimensions])} coord_vec_t;
"""

QBX_TREE_MAKO_DEFS = r"""//CL:mako//
<%def name="load_particle(particle, coords)">
    %for ax in AXIS_NAMES[:dimensions]:
        ${coords}.${ax} = particles_${ax}[${particle}];
    %endfor
</%def>
"""

# }}}


# {{{ make interleaved centers

def get_interleaved_centers(queue, places, dofdesc=None):
    """
    Return an array of shape (dim, ncenters) in which interior centers are placed
    next to corresponding exterior centers.
    """
    from pytential.symbolic.execution import GeometryCollection
    if not isinstance(places, GeometryCollection):
        places = GeometryCollection(places, auto_where=dofdesc)
    dofdesc = dofdesc or places.auto_source
    discr = places.get_discretization(dofdesc)

    from pytential import bind, sym
    int_centers = bind(places, sym.expansion_centers(
        discr.ambient_dim, -1,
        dofdesc=dofdesc))(queue)
    ext_centers = bind(places, sym.expansion_centers(
        discr.ambient_dim, +1,
        dofdesc=dofdesc))(queue)

    from pytential.symbolic.dof_connection import CenterGranularityConnection
    interleaver = CenterGranularityConnection(discr)
    return interleaver(queue, [int_centers, ext_centers])

# }}}


# {{{ make interleaved radii

def get_interleaved_radii(queue, places, dofdesc=None):
    """
    Return an array of shape (dim, ncenters) in which interior centers are placed
    next to corresponding exterior centers.
    """
    from pytential.symbolic.execution import GeometryCollection
    if not isinstance(places, GeometryCollection):
        places = GeometryCollection(places, auto_where=dofdesc)
    dofdesc = dofdesc or places.auto_source
    discr = places.get_discretization(dofdesc)

    from pytential import bind, sym
    radii = bind(places, sym.expansion_radii(
        discr.ambient_dim,
        dofdesc=dofdesc))(queue)

    from pytential.symbolic.dof_connection import CenterGranularityConnection
    interleaver = CenterGranularityConnection(discr)
    return interleaver(queue, radii)

# }}}


# {{{ tree code container

class TreeCodeContainer(object):

    def __init__(self, cl_context):
        self.cl_context = cl_context

    @memoize_method
    def build_tree(self):
        from boxtree.tree_build import TreeBuilder
        return TreeBuilder(self.cl_context)

    @memoize_method
    def peer_list_finder(self):
        from boxtree.area_query import PeerListFinder
        return PeerListFinder(self.cl_context)

    @memoize_method
    def particle_list_filter(self):
        from boxtree.tree import ParticleListFilter
        return ParticleListFilter(self.cl_context)

# }}}


# {{{ tree code container mixin

class TreeCodeContainerMixin(object):
    """Forwards requests for tree-related code to an inner code container named
    self.tree_code_container.
    """

    def build_tree(self):
        return self.tree_code_container.build_tree()

    def peer_list_finder(self):
        return self.tree_code_container.peer_list_finder()

    def particle_list_filter(self):
        return self.tree_code_container.particle_list_filter()

# }}}


# {{{ tree wrangler base class

class TreeWranglerBase(object):

    def __init__(self, code_container, queue):
        self.code_container = code_container
        self.queue = queue

    def build_tree(self, lpot_source, targets_list=(),
                   use_stage2_discr=False):
        tb = self.code_container.build_tree()
        plfilt = self.code_container.particle_list_filter()
        from pytential.qbx.utils import build_tree_with_qbx_metadata
        return build_tree_with_qbx_metadata(
                self.queue, tb, plfilt, lpot_source, targets_list=targets_list,
                use_stage2_discr=use_stage2_discr)

    def find_peer_lists(self, tree):
        plf = self.code_container.peer_list_finder()
        peer_lists, evt = plf(self.queue, tree)
        cl.wait_for_events([evt])
        return peer_lists

# }}}


# {{{ tree-with-metadata: data structure

class TreeWithQBXMetadata(Tree):
    """A subclass of :class:`boxtree.tree.Tree`. Has all of that class's
    attributes, along with the following:

    .. attribute:: nqbxpanels
    .. attribuet:: nqbxsources
    .. attribute:: nqbxcenters
    .. attribute:: nqbxtargets

    .. ------------------------------------------------------------------------
    .. rubric:: Box properties
    .. ------------------------------------------------------------------------

    .. rubric:: Box to QBX panels

    .. attribute:: box_to_qbx_panel_starts

        ``box_id_t [nboxes + 1]``

    .. attribute:: box_to_qbx_panel_lists

        ``particle_id_t [*]``

    .. rubric:: Box to QBX sources

    .. attribute:: box_to_qbx_source_starts

        ``box_id_t [nboxes + 1]``

    .. attribute:: box_to_qbx_source_lists

        ``particle_id_t [*]``

    .. rubric:: Box to QBX centers

    .. attribute:: box_to_qbx_center_starts

        ``box_id_t [nboxes + 1]``

    .. attribute:: box_to_qbx_center_lists

        ``particle_id_t [*]``

    .. rubric:: Box to QBX targets

    .. attribute:: box_to_qbx_target_starts

        ``box_id_t [nboxes + 1]``

    .. attribute:: box_to_qbx_target_lists

        ``particle_id_t [*]``

    .. ------------------------------------------------------------------------
    .. rubric:: Panel properties
    .. ------------------------------------------------------------------------

    .. attribute:: qbx_panel_to_source_starts

        ``particle_id_t [nqbxpanels + 1]``

    .. attribute:: qbx_panel_to_center_starts

        ``particle_id_t [nqbxpanels + 1]``

    .. ------------------------------------------------------------------------
    .. rubric:: Particle order indices
    .. ------------------------------------------------------------------------

    .. attribute:: qbx_user_source_slice
    .. attribute:: qbx_user_center_slice
    .. attribute:: qbx_user_target_slice
    """
    pass

# }}}


# {{{ tree-with-metadata: creation

MAX_REFINE_WEIGHT = 64


@log_process(logger)
def build_tree_with_qbx_metadata(
        queue, tree_builder, particle_list_filter, lpot_source, targets_list=(),
        use_stage2_discr=False):
    """Return a :class:`TreeWithQBXMetadata` built from the given layer
    potential source. This contains particles of four different types:

       * source particles either from
         ``lpot_source.density_discr`` or
         ``lpot_source.stage2_density_discr``
       * centers from ``lpot_source.centers()``
       * targets from ``targets_list``.

    :arg queue: An instance of :class:`pyopencl.CommandQueue`

    :arg lpot_source: An instance of
        :class:`pytential.qbx.NewQBXLayerPotentialSource`.

    :arg targets_list: A list of :class:`pytential.target.TargetBase`

    :arg use_stage2_discr: If *True*, builds a tree with sources
        from ``lpot_source.stage2_density_discr``. If *False* (default),
        they are from ``lpot_source.density_discr``.
    """
    # The ordering of particles is as follows:
    # - sources go first
    # - then centers
    # - then targets

    if use_stage2_discr:
        density_discr = lpot_source.quad_stage2_density_discr
    else:
        density_discr = lpot_source.density_discr

    sources = density_discr.nodes()
    centers = get_interleaved_centers(queue, lpot_source)
    targets = (tgt.nodes() for tgt in targets_list)

    particles = tuple(
            cl.array.concatenate(dim_coords, queue=queue)
            for dim_coords in zip(sources, centers, *targets))

    # Counts
    nparticles = len(particles[0])
    npanels = density_discr.mesh.nelements
    nsources = len(sources[0])
    ncenters = len(centers[0])
    # Each source gets an interior / exterior center.
    assert 2 * nsources == ncenters or use_stage2_discr
    ntargets = sum(tgt.nnodes for tgt in targets_list)

    # Slices
    qbx_user_source_slice = slice(0, nsources)

    center_slice_start = nsources
    qbx_user_center_slice = slice(center_slice_start, center_slice_start + ncenters)

    panel_slice_start = center_slice_start + ncenters
    target_slice_start = panel_slice_start
    qbx_user_target_slice = slice(target_slice_start, target_slice_start + ntargets)

    # Build tree with sources and centers. Split boxes
    # only because of sources.
    refine_weights = cl.array.zeros(queue, nparticles, np.int32)
    refine_weights[:nsources].fill(1)

    refine_weights.finish()

    tree, evt = tree_builder(queue, particles,
            max_leaf_refine_weight=MAX_REFINE_WEIGHT,
            refine_weights=refine_weights)

    # Compute box => particle class relations
    flags = refine_weights
    del refine_weights
    particle_classes = {}

    for class_name, particle_slice, fixup in (
            ("box_to_qbx_source", qbx_user_source_slice, 0),
            ("box_to_qbx_target", qbx_user_target_slice, -target_slice_start),
            ("box_to_qbx_center", qbx_user_center_slice, -center_slice_start)):
        flags.fill(0)
        flags[particle_slice].fill(1)
        flags.finish()

        box_to_class = (
            particle_list_filter
            .filter_target_lists_in_user_order(queue, tree, flags)
            .with_queue(queue))

        if fixup:
            box_to_class.target_lists += fixup
        particle_classes[class_name + "_starts"] = box_to_class.target_starts
        particle_classes[class_name + "_lists"] = box_to_class.target_lists

    del flags
    del box_to_class

    # Compute panel => source relation
    qbx_panel_to_source_starts = cl.array.empty(
            queue, npanels + 1, dtype=tree.particle_id_dtype)
    el_offset = 0
    for group in density_discr.groups:
        qbx_panel_to_source_starts[el_offset:el_offset + group.nelements] = \
                cl.array.arange(queue, group.node_nr_base,
                                group.node_nr_base + group.nnodes,
                                group.nunit_nodes,
                                dtype=tree.particle_id_dtype)
        el_offset += group.nelements
    qbx_panel_to_source_starts[-1] = nsources

    # Compute panel => center relation
    qbx_panel_to_center_starts = (
            2 * qbx_panel_to_source_starts
            if not use_stage2_discr
            else None)

    # Transfer all tree attributes.
    tree_attrs = {}
    for attr_name in tree.__class__.fields:
        try:
            tree_attrs[attr_name] = getattr(tree, attr_name)
        except AttributeError:
            pass

    tree_attrs.update(particle_classes)

    return TreeWithQBXMetadata(
        qbx_panel_to_source_starts=qbx_panel_to_source_starts,
        qbx_panel_to_center_starts=qbx_panel_to_center_starts,
        qbx_user_source_slice=qbx_user_source_slice,
        qbx_user_center_slice=qbx_user_center_slice,
        qbx_user_target_slice=qbx_user_target_slice,
        nqbxpanels=npanels,
        nqbxsources=nsources,
        nqbxcenters=ncenters,
        nqbxtargets=ntargets,
        **tree_attrs).with_queue(None)

# }}}

# vim: foldmethod=marker:filetype=pyopencl
