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
from pytools import memoize, memoize_method
from loopy.version import MOST_RECENT_LANGUAGE_VERSION

import logging
logger = logging.getLogger(__name__)

from pytools import log_process


# {{{ c and mako snippets

QBX_TREE_C_PREAMBLE = r"""//CL:mako//
// A note on node numberings: sources, centers, and panels each
// have their own numbering starting at 0. These macros convert
// the per-class numbering into the internal tree particle number.
#define INDEX_FOR_CENTER_PARTICLE(i) (sorted_target_ids[center_offset + i])
#define INDEX_FOR_PANEL_PARTICLE(i) (sorted_target_ids[panel_offset + i])
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


# {{{ interleaver kernel

@memoize
def get_interleaver_kernel(dtype):
    # NOTE: Returned kernel needs dstlen or dst parameter
    from pymbolic import var
    knl = lp.make_kernel(
        "[srclen,dstlen] -> {[i]: 0<=i<srclen}",
        """
        dst[2*i] = src1[i]
        dst[2*i+1] = src2[i]
        """, [
            lp.GlobalArg("src1", shape=(var("srclen"),), dtype=dtype),
            lp.GlobalArg("src2", shape=(var("srclen"),), dtype=dtype),
            lp.GlobalArg("dst", shape=(var("dstlen"),), dtype=dtype),
            "..."
        ],
        assumptions="2*srclen = dstlen",
        lang_version=MOST_RECENT_LANGUAGE_VERSION)
    knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")
    return knl

# }}}


# {{{ make interleaved centers

def get_interleaved_centers(queue, lpot_source):
    """
    Return an array of shape (dim, ncenters) in which interior centers are placed
    next to corresponding exterior centers.
    """
    knl = get_interleaver_kernel(lpot_source.density_discr.real_dtype)
    int_centers = get_centers_on_side(lpot_source, -1)
    ext_centers = get_centers_on_side(lpot_source, +1)

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


# {{{ make interleaved radii

def get_interleaved_radii(queue, lpot_source):
    """
    Return an array of shape (dim, ncenters) in which interior centers are placed
    next to corresponding exterior centers.
    """
    knl = get_interleaver_kernel(lpot_source.density_discr.real_dtype)
    radii = lpot_source._expansion_radii("nsources")

    result = cl.array.empty(queue, len(radii) * 2, radii.dtype)
    evt, _ = knl(queue, src1=radii, src2=radii, dst=result)
    evt.wait()

    return result

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


# {{{ to_last_dim_length

def to_last_dim_length(discr, vec, last_dim_length, queue=None):
    """Takes a :class:`pyopencl.array.Array` with a last axis that has the same
    length as the number of discretization nodes in the discretization *discr*
    and converts it so that the last axis has a length as specified by
    *last_dim_length*.
    """

    queue = queue or vec.queue

    if last_dim_length == "nsources":
        return vec

    elif last_dim_length == "ncenters":
        knl = get_interleaver_kernel(vec.dtype)
        _, (result,) = knl(queue, dstlen=2*discr.nnodes, src1=vec, src2=vec)
        return result

    elif last_dim_length == "npanels":
        knl = lp.make_kernel(
            "{[i,k]: 0<=i<nelements}",
            "result[i] = a[i,0]",
            [
                lp.GlobalArg("a", shape="nelements, nunit_nodes", dtype=lp.auto),
                lp.ValueArg("nunit_nodes", dtype=np.int32),
                "..."
                ],
            name="subsample_to_panels",
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")

        result = cl.array.empty(queue, discr.mesh.nelements, vec.dtype)
        for group_nr, group in enumerate(discr.groups):
            knl(queue,
                nelements=group.nelements,
                a=group.view(vec),
                result=el_view(discr, group_nr, result))

        return result
    else:
        raise ValueError("unknown dim length specified")

# }}}


# {{{ element centers of mass

def element_centers_of_mass(discr):
    knl = lp.make_kernel(
        """{[dim,k,i]:
            0<=dim<ndims and
            0<=k<nelements and
            0<=i<nunit_nodes}""",
        """
            panels[dim, k] = sum(i, nodes[dim, k, i])/nunit_nodes
            """,
        default_offset=lp.auto, name="find_panel_centers_of_mass",
        lang_version=MOST_RECENT_LANGUAGE_VERSION)

    knl = lp.fix_parameters(knl, ndims=discr.ambient_dim)

    knl = lp.split_iname(knl, "k", 128, inner_tag="l.0", outer_tag="g.0")
    knl = lp.tag_inames(knl, dict(dim="ilp"))

    with cl.CommandQueue(discr.cl_context) as queue:
        mesh = discr.mesh
        panels = cl.array.empty(queue, (mesh.ambient_dim, mesh.nelements),
                                dtype=discr.real_dtype)
        for group_nr, group in enumerate(discr.groups):
            _, (result,) = knl(queue,
                nelements=group.nelements,
                nunit_nodes=group.nunit_nodes,
                nodes=group.view(discr.nodes()),
                panels=el_view(discr, group_nr, panels))
        panels.finish()
        panels = panels.with_queue(None)
        return tuple(panels[d, :] for d in range(mesh.ambient_dim))

# }}}


# {{{ compute center array

def get_centers_on_side(lpot_src, sign):
    adim = lpot_src.density_discr.ambient_dim
    dim = lpot_src.density_discr.dim

    from pytential import sym, bind
    with cl.CommandQueue(lpot_src.cl_context) as queue:
        nodes = bind(lpot_src.density_discr, sym.nodes(adim))(queue)
        normals = bind(lpot_src.density_discr, sym.normal(adim, dim=dim))(queue)
        expansion_radii = lpot_src._expansion_radii("nsources").with_queue(queue)
        return (nodes + normals * sign * expansion_radii).as_vector(np.object)

# }}}


# {{{ el_view

def el_view(discr, group_nr, global_array):
    """Return a view of *global_array* of shape
    ``(..., discr.groups[group_nr].nelements)``
    where *global_array* is of shape ``(..., nelements)``,
    where *nelements* is the global (per-discretization) node count.
    """

    group = discr.mesh.groups[group_nr]

    return global_array[
        ..., group.element_nr_base:group.element_nr_base + group.nelements] \
        .reshape(
            global_array.shape[:-1]
            + (group.nelements,))

# }}}


# {{{ discr plotter

def plot_discr(lpot_source, outfilename="discr.pdf"):
    with cl.CommandQueue(lpot_source.cl_context) as queue:
        from boxtree.tree_builder import TreeBuilder
        tree_builder = TreeBuilder(lpot_source.cl_context)
        tree = tree_builder(queue, lpot_source).get(queue=queue)
        from boxtree.visualization import TreePlotter

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tp = TreePlotter(tree)
        tp.draw_tree()
        sources = (tree.sources[0], tree.sources[1])
        sti = tree.sorted_target_ids
        plt.plot(sources[0][sti[tree.qbx_user_source_slice]],
                 sources[1][sti[tree.qbx_user_source_slice]],
                 lw=0, marker=".", markersize=1, label="sources")
        plt.plot(sources[0][sti[tree.qbx_user_center_slice]],
                 sources[1][sti[tree.qbx_user_center_slice]],
                 lw=0, marker=".", markersize=1, label="centers")
        plt.plot(sources[0][sti[tree.qbx_user_target_slice]],
                 sources[1][sti[tree.qbx_user_target_slice]],
                 lw=0, marker=".", markersize=1, label="targets")
        plt.axis("equal")
        plt.legend()
        plt.savefig(outfilename)

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
    .. attribute:: qbx_user_panel_slice
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

       * source particles and panel centers of mass either from
         ``lpot_source.density_discr`` or
         ``lpot_source.stage2_density_discr``
       * centers from ``lpot_source.centers()``
       * targets from ``targets_list``.

    :arg queue: An instance of :class:`pyopencl.CommandQueue`

    :arg lpot_source: An instance of
        :class:`pytential.qbx.NewQBXLayerPotentialSource`.

    :arg targets_list: A list of :class:`pytential.target.TargetBase`

    :arg use_stage2_discr: If *True*, builds a tree with sources/centers of
        mass from ``lpot_source.stage2_density_discr``. If *False* (default),
        they are from ``lpot_source.density_discr``.
    """
    # The ordering of particles is as follows:
    # - sources go first
    # - then centers
    # - then panels (=centers of mass)
    # - then targets

    sources = (
            lpot_source.density_discr.nodes()
            if not use_stage2_discr
            else lpot_source.quad_stage2_density_discr.nodes())

    centers = get_interleaved_centers(queue, lpot_source)

    centers_of_mass = (
            lpot_source._panel_centers_of_mass()
            if not use_stage2_discr
            else lpot_source._stage2_panel_centers_of_mass())

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
    assert 2 * nsources == ncenters or use_stage2_discr
    ntargets = sum(tgt.nnodes for tgt in targets_list)

    # Slices
    qbx_user_source_slice = slice(0, nsources)

    center_slice_start = nsources
    qbx_user_center_slice = slice(center_slice_start, center_slice_start + ncenters)

    panel_slice_start = center_slice_start + ncenters
    qbx_user_panel_slice = slice(panel_slice_start, panel_slice_start + npanels)

    target_slice_start = panel_slice_start + npanels
    qbx_user_target_slice = slice(target_slice_start, target_slice_start + ntargets)

    # Build tree with sources, centers, and centers of mass. Split boxes
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
            ("box_to_qbx_center", qbx_user_center_slice, -center_slice_start),
            ("box_to_qbx_panel", qbx_user_panel_slice, -panel_slice_start)):
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
    if use_stage2_discr:
        density_discr = lpot_source.quad_stage2_density_discr
    else:
        density_discr = lpot_source.density_discr

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
        qbx_user_panel_slice=qbx_user_panel_slice,
        qbx_user_center_slice=qbx_user_center_slice,
        qbx_user_target_slice=qbx_user_target_slice,
        nqbxpanels=npanels,
        nqbxsources=nsources,
        nqbxcenters=ncenters,
        nqbxtargets=ntargets,
        **tree_attrs).with_queue(None)

# }}}


# {{{ host geo data wrapper

class ToHostTransferredGeoDataWrapper(object):
    """Wraps an instance of :class:`pytential.qbx.geometry.QBXFMMGeometryData`,
    automatically converting returned OpenCL arrays to host data.
    """

    def __init__(self, queue, geo_data):
        self.queue = queue
        self.geo_data = geo_data

    @memoize_method
    def tree(self):
        return self.geo_data.tree().get(queue=self.queue)

    @memoize_method
    def traversal(self):
        return self.geo_data.traversal().get(queue=self.queue)

    @property
    def ncenters(self):
        return self.geo_data.ncenters

    @memoize_method
    def centers(self):
        return np.array([
            ci.get(queue=self.queue)
            for ci in self.geo_data.centers()])

    @memoize_method
    def expansion_radii(self):
        return self.geo_data.expansion_radii().get(queue=self.queue)

    @memoize_method
    def global_qbx_centers(self):
        return self.geo_data.global_qbx_centers().get(queue=self.queue)

    @memoize_method
    def qbx_center_to_target_box(self):
        return self.geo_data.qbx_center_to_target_box().get(queue=self.queue)

    @memoize_method
    def qbx_center_to_target_box_source_level(self, source_level):
        return self.geo_data.qbx_center_to_target_box_source_level(
            source_level).get(queue=self.queue)

    @memoize_method
    def non_qbx_box_target_lists(self):
        return self.geo_data.non_qbx_box_target_lists().get(queue=self.queue)

    @memoize_method
    def center_to_tree_targets(self):
        return self.geo_data.center_to_tree_targets().get(queue=self.queue)

    @memoize_method
    def all_targets(self):
        """All (not just non-QBX) targets packaged into a single array."""
        return np.array(list(self.tree().targets))

# }}}

# vim: foldmethod=marker:filetype=pyopencl
