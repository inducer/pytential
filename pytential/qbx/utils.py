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

from pytools import memoize_method, memoize_in, log_process
from arraycontext import PyOpenCLArrayContext
from meshmode.dof_array import DOFArray

from boxtree.tree import Tree
from boxtree.pyfmmlib_integration import FMMLibRotationDataInterface

import logging
logger = logging.getLogger(__name__)


# {{{ c and mako snippets

QBX_TREE_C_PREAMBLE = r"""//CL:mako//
// A note on node numberings: sources, centers, and elements each
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


# {{{ tree code container

class TreeCodeContainer:

    def __init__(self, actx: PyOpenCLArrayContext):
        self.array_context = actx

    @memoize_method
    def build_tree(self):
        from boxtree.tree_build import TreeBuilder
        return TreeBuilder(self.array_context.context)

    @memoize_method
    def peer_list_finder(self):
        from boxtree.area_query import PeerListFinder
        return PeerListFinder(self.array_context.context)

    @memoize_method
    def particle_list_filter(self):
        from boxtree.tree import ParticleListFilter
        return ParticleListFilter(self.array_context.context)

    @memoize_method
    def build_area_query(self):
        from boxtree.area_query import AreaQueryBuilder
        return AreaQueryBuilder(self.array_context.context)


def tree_code_container(actx: PyOpenCLArrayContext) -> TreeCodeContainer:
    @memoize_in(actx, (TreeCodeContainer, tree_code_container))
    def make_container():
        return TreeCodeContainer(actx)

    return make_container()

# }}}


# {{{ tree code container mixin

class TreeCodeContainerMixin:
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

class TreeWranglerBase:

    def __init__(self, array_context: PyOpenCLArrayContext, code_container):
        self.code_container = code_container
        self.array_context = array_context

    @property
    def queue(self):
        return self.array_context.queue

    def build_tree(self, places, targets_list=(), sources_list=(),
                   use_stage2_discr=False):
        tb = self.code_container.build_tree()
        plfilt = self.code_container.particle_list_filter()

        return build_tree_with_qbx_metadata(
                self.array_context, places, tb, plfilt,
                sources_list=sources_list,
                targets_list=targets_list,
                use_stage2_discr=use_stage2_discr)

    def find_peer_lists(self, tree):
        plf = self.code_container.peer_list_finder()
        peer_lists, evt = plf(self.queue, tree)

        import pyopencl as cl
        cl.wait_for_events([evt])

        return peer_lists

# }}}


# {{{ tree-with-metadata: data structure

class TreeWithQBXMetadata(Tree):
    """A subclass of :class:`boxtree.tree.Tree`. Has all of that class's
    attributes, along with the following:

    .. attribute:: nqbxelements
    .. attribuet:: nqbxsources
    .. attribute:: nqbxcenters
    .. attribute:: nqbxtargets

    .. ------------------------------------------------------------------------
    .. rubric:: Box properties
    .. ------------------------------------------------------------------------

    .. rubric:: Box to QBX elements

    .. attribute:: box_to_qbx_element_starts

        ``box_id_t [nboxes + 1]``

    .. attribute:: box_to_qbx_element_lists

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
    .. rubric:: Element properties
    .. ------------------------------------------------------------------------

    .. attribute:: qbx_element_to_source_starts

        ``particle_id_t [nqbxelements + 1]``

    .. attribute:: qbx_element_to_center_starts

        ``particle_id_t [nqbxelements + 1]``

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
def build_tree_with_qbx_metadata(actx: PyOpenCLArrayContext,
        places, tree_builder, particle_list_filter,
        sources_list=(), targets_list=(),
        use_stage2_discr=False):
    """Return a :class:`TreeWithQBXMetadata` built from the given layer
    potential source. This contains particles of four different types:

       * source particles either from
         :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_STAGE1` or
         :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_QUAD_STAGE2`.
       * centers from
         :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_STAGE1`.
       * targets from ``targets_list``.

    :arg places: An instance of :class:`~pytential.collection.GeometryCollection`.
    :arg targets_list: A list of :class:`pytential.target.TargetBase`

    :arg use_stage2_discr: If *True*, builds a tree with stage 2 sources.
        If *False*, the tree is built with stage 1 sources.
    """

    # The ordering of particles is as follows:
    # - sources go first
    # - then centers
    # - then targets

    from pytential import bind, sym
    stage1_density_discrs = []
    density_discrs = []
    for source_name in sources_list:
        dd = sym.as_dofdesc(source_name)

        discr = places.get_discretization(dd.geometry)
        stage1_density_discrs.append(discr)

        if use_stage2_discr:
            discr = places.get_discretization(
                    dd.geometry, sym.QBX_SOURCE_QUAD_STAGE2)
        density_discrs.append(discr)

    # TODO: update code to work for multiple source discretizations
    if len(sources_list) != 1:
        raise RuntimeError("can only build a tree for a single source")

    def _make_centers(discr):
        return bind(discr, sym.interleaved_expansion_centers(
            discr.ambient_dim))(actx)

    stage1_density_discr = stage1_density_discrs[0]
    density_discr = density_discrs[0]

    from arraycontext import flatten
    sources = flatten(density_discr.nodes(), actx, leaf_class=DOFArray)
    centers = flatten(_make_centers(stage1_density_discr), actx, leaf_class=DOFArray)
    targets = [
            flatten(tgt.nodes(), actx, leaf_class=DOFArray)
            for tgt in targets_list]

    queue = actx.queue
    particles = tuple(
            actx.np.concatenate(dim_coords)
            for dim_coords in zip(sources, centers, *targets))

    # Counts
    nparticles = len(particles[0])
    nelements = density_discr.mesh.nelements
    nsources = len(sources[0])
    ncenters = len(centers[0])
    # Each source gets an interior / exterior center.
    assert 2 * nsources == ncenters or use_stage2_discr
    ntargets = sum(tgt.ndofs for tgt in targets_list)

    # Slices
    qbx_user_source_slice = slice(0, nsources)

    center_slice_start = nsources
    qbx_user_center_slice = slice(center_slice_start, center_slice_start + ncenters)

    element_slice_start = center_slice_start + ncenters
    target_slice_start = element_slice_start
    qbx_user_target_slice = slice(target_slice_start, target_slice_start + ntargets)

    # Build tree with sources and centers. Split boxes
    # only because of sources.
    refine_weights = actx.zeros(nparticles, np.int32)
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
            ).with_queue(actx.queue)

        if fixup:
            box_to_class.target_lists += fixup
        particle_classes[class_name + "_starts"] = box_to_class.target_starts
        particle_classes[class_name + "_lists"] = box_to_class.target_lists

    del flags
    del box_to_class

    # Compute element => source relation
    qbx_element_to_source_starts = actx.zeros(nelements + 1, tree.particle_id_dtype)
    el_offset = 0
    node_nr_base = 0
    for group in density_discr.groups:
        group_element_starts = np.arange(
                node_nr_base, node_nr_base + group.ndofs, group.nunit_dofs,
                dtype=tree.particle_id_dtype)
        qbx_element_to_source_starts[el_offset:el_offset + group.nelements] = \
                actx.from_numpy(group_element_starts)

        node_nr_base += group.ndofs
        el_offset += group.nelements
    qbx_element_to_source_starts[-1] = nsources

    # Compute element => center relation
    qbx_element_to_center_starts = (
            2 * qbx_element_to_source_starts
            if not use_stage2_discr
            else None)
    if qbx_element_to_center_starts is not None:
        assert (qbx_element_to_center_starts.dtype
                == qbx_element_to_source_starts.dtype)

    # Transfer all tree attributes.
    tree_attrs = {}
    for attr_name in tree.__class__.fields:
        try:
            tree_attrs[attr_name] = getattr(tree, attr_name)
        except AttributeError:
            pass

    tree_attrs.update(particle_classes)

    return TreeWithQBXMetadata(
        qbx_element_to_source_starts=qbx_element_to_source_starts,
        qbx_element_to_center_starts=qbx_element_to_center_starts,
        qbx_user_source_slice=qbx_user_source_slice,
        qbx_user_center_slice=qbx_user_center_slice,
        qbx_user_target_slice=qbx_user_target_slice,
        nqbxelements=nelements,
        nqbxsources=nsources,
        nqbxcenters=ncenters,
        nqbxtargets=ntargets,
        **tree_attrs).with_queue(None)

# }}}


# {{{ host geo data wrapper

class ToHostTransferredGeoDataWrapper(FMMLibRotationDataInterface):
    """Wraps an instance of :class:`pytential.qbx.geometry.QBXFMMGeometryData`,
    automatically converting returned OpenCL arrays to host data.
    """

    def __init__(self, geo_data):
        self.geo_data = geo_data

    @property
    def queue(self):
        return self.geo_data._setup_actx.queue

    def to_numpy(self, ary):
        return self.geo_data._setup_actx.to_numpy(ary)

    @memoize_method
    def tree(self):
        return self.geo_data.tree().get(queue=self.queue)

    @memoize_method
    def traversal(self):
        return self.geo_data.traversal().get(queue=self.queue)

    @property
    def lpot_source(self):
        return self.geo_data.lpot_source

    @property
    def ncenters(self):
        return self.geo_data.ncenters

    @memoize_method
    def centers(self):
        return np.stack(self.to_numpy(self.geo_data.flat_centers()))

    @memoize_method
    def expansion_radii(self):
        return self.to_numpy(self.geo_data.flat_expansion_radii())

    @memoize_method
    def global_qbx_centers(self):
        return self.to_numpy(self.geo_data.global_qbx_centers())

    @memoize_method
    def qbx_center_to_target_box(self):
        return self.to_numpy(self.geo_data.qbx_center_to_target_box())

    @memoize_method
    def qbx_center_to_target_box_source_level(self, source_level):
        return self.to_numpy(
            self.geo_data.qbx_center_to_target_box_source_level(source_level)
            )

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

    def eval_qbx_targets(self):
        return self.all_targets()

    def m2l_rotation_lists(self):
        # Already on host
        return self.geo_data.m2l_rotation_lists()

    def m2l_rotation_angles(self):
        # Already on host
        return self.geo_data.m2l_rotation_angles()

# }}}

# vim: foldmethod=marker:filetype=pyopencl
