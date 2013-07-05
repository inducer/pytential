from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
import pyopencl as cl
import pyopencl.array  # noqa
from pytools import memoize_method, Record
import loopy as lp
from cgen import Enum


import logging
logger = logging.getLogger(__name__)


# {{{ code getter

class target_state(Enum):
    # c_name = "particle_id_t" (tree-dependent, intentionally unspecified)
    # dtype intentionally unspecified
    c_value_prefix = "TGT_"

    NO_QBX_NEEDED = -1

    # QBX needed, but no usable center found
    FAILED = -2


class QBXFMMCodeGetter(object):
    def __init__(self, cl_context, ambient_dim):
        self.cl_context = cl_context
        self.ambient_dim = ambient_dim

    @property
    @memoize_method
    def pick_expansion_centers(self):
        knl = lp.make_kernel(self.cl_context.devices[0],
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nout_nodes}""",
            """
                centers[dim, k, i] = all_centers[dim, k, kept_center_indices[i]]
                radii[k, i] = all_radii[k, kept_center_indices[i]]
                """,
            [
                lp.GlobalArg("all_centers", None,
                    shape="ndims,nelements,nunit_nodes"),
                lp.GlobalArg("all_radii", None, shape="nelements,nunit_nodes"),
                lp.ValueArg("nunit_nodes", np.int32),
                "..."
                ],
            default_offset=lp.auto, name="center_pick",
            defines=dict(ndims=self.ambient_dim))

        knl = lp.tag_data_axes(knl, "centers,all_centers", "sep, C, C")

        knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
        return lp.tag_inames(knl, dict(k="g.0", dim="ilp"))

    @property
    @memoize_method
    def find_element_centers(self):
        knl = lp.make_kernel(self.cl_context.devices[0],
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nunit_nodes}""",
            """
                el_centers[dim, k] = sum(i, nodes[dim, k, i])/nunit_nodes
                """,
            default_offset=lp.auto, name="find_element_centers",
            defines=dict(ndims=self.ambient_dim))

        knl = lp.split_iname(knl, "k", 128, inner_tag="l.0", outer_tag="g.0")
        return lp.tag_inames(knl, dict(dim="ilp"))

    @property
    @memoize_method
    def find_element_radii(self):
        knl = lp.make_kernel(self.cl_context.devices[0],
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nunit_nodes}""",
            """
                el_radii[k] = max(dim, max(i, \
                    fabs(nodes[dim, k, i] - el_centers[dim, k])))
                """,
            default_offset=lp.auto, name="find_element_radii",
            defines=dict(ndims=self.ambient_dim))

        knl = lp.split_iname(knl, "k", 128, inner_tag="l.0", outer_tag="g.0")
        return lp.tag_inames(knl, dict(dim="unr"))

    @memoize_method
    def copy_targets_kernel(self, sep_points_axes):
        knl = lp.make_kernel(self.cl_context.devices[0],
            """{[dim,i]:
                0<=dim<ndims and
                0<=i<npoints}""",
            """
                targets[dim, i] = points[dim, i]
                """,
            default_offset=lp.auto, name="copy_targets",
            defines=dict(ndims=self.ambient_dim))

        knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")
        if sep_points_axes:
            knl = lp.tag_data_axes(knl, "points", "sep, C")
        else:
            knl = lp.tag_data_axes(knl, "points", "stride:auto, stride:1")

        knl = lp.tag_data_axes(knl, "targets", "stride:auto, stride:1")
        return lp.tag_inames(knl, dict(dim="ilp"))

    @property
    @memoize_method
    def build_tree(self):
        from boxtree import TreeBuilder
        return TreeBuilder(self.cl_context)

    @property
    @memoize_method
    def build_traversal(self):
        from boxtree.traversal import FMMTraversalBuilder
        return FMMTraversalBuilder(self.cl_context)

    @memoize_method
    def qbx_center_to_target_box_lookup(self, particle_id_dtype, box_id_dtype):
        knl = lp.make_kernel(self.cl_context.devices[0],
            [
                "{[ibox]: 0<=ibox<nboxes}",
                "{[itarget]: b_t_start <= itarget < b_t_start + ntargets}",
                ],
            [
                """
                <> b_t_start = box_target_starts[ibox]
                <> ntargets = box_target_counts_nonchild[ibox]
                """,
                lp.CInstruction(
                    ["itarget", "ibox"],
                    """//CL//
                    if (itarget < ncenters)
                        qbx_center_to_target_box[itarget] = box_to_target_box[ibox];
                    """, assignees="qbx_center_to_target_box[itarget]",
                    id="tgt_write")
                ],
            [
                lp.GlobalArg("qbx_center_to_target_box", box_id_dtype,
                    shape="ncenters"),
                lp.GlobalArg("box_to_target_box", box_id_dtype),
                lp.ValueArg("ncenters", particle_id_dtype),
                "..."
                ],
            name="qbx_center_to_target_box_lookup",
            silenced_warnings="write_race(tgt_write)")

        knl = lp.split_iname(knl, "ibox", 128,
                inner_tag="l.0", outer_tag="g.0")

        return knl

    @property
    @memoize_method
    def build_leaf_to_ball_lookup(self):
        from boxtree.geo_lookup import LeavesToBallsLookupBuilder
        return LeavesToBallsLookupBuilder(self.cl_context)

    # {{{ check if a center may be used with global QBX

    @memoize_method
    def qbx_center_for_global_tester(self,
            coord_dtype, box_id_dtype, particle_id_dtype):
        from pyopencl.elementwise import ElementwiseTemplate
        return ElementwiseTemplate(
            arguments="""//CL:mako//
                /* input */
                %for iaxis in range(ambient_dim):
                    coord_t *center_${iaxis},  /* [ncenters] */
                %endfor
                coord_t *radii,  /* [ncenters] */
                box_id_t *qbx_center_to_target_box, /* [ncenters] */

                box_id_t *neighbor_source_boxes_starts,
                box_id_t *neighbor_source_boxes_lists,
                particle_id_t *box_point_source_starts,
                particle_id_t *box_point_source_counts_cumul,
                %for iaxis in range(ambient_dim):
                    coord_t *point_sources_${iaxis},
                %endfor

                /* output */
                char *center_may_use_global_qbx,
                """,
            operation=r"""//CL:mako//
                particle_id_t icenter = i;

                %for iaxis in range(ambient_dim):
                    coord_t my_center_${iaxis} = center_${iaxis}[icenter];
                %endfor
                coord_t radius = radii[icenter];

                // {{{ see if there are sources close enough to require local QBX

                bool found_too_close = false;

                coord_t radius_squared = radius * radius;

                box_id_t itgt_box = qbx_center_to_target_box[icenter];

                box_id_t nb_src_start = neighbor_source_boxes_starts[itgt_box];
                box_id_t nb_src_stop = neighbor_source_boxes_starts[itgt_box+1];

                for (
                        box_id_t ibox_list = nb_src_start;
                        ibox_list < nb_src_stop && !found_too_close;
                        ++ibox_list)
                {
                    box_id_t neighbor_box_id =
                        neighbor_source_boxes_lists[ibox_list];

                    box_id_t bps_start = box_point_source_starts[neighbor_box_id];
                    box_id_t bps_stop =
                        bps_start + box_point_source_counts_cumul[neighbor_box_id];
                    for (
                            box_id_t ipsrc = bps_start;
                            ipsrc < bps_stop;
                            ++ipsrc)
                    {
                        %for iaxis in range(ambient_dim):
                            coord_t psource_${iaxis} = point_sources_${iaxis}[ipsrc];
                        %endfor

                        coord_t dist_squared = 0
                        %for iaxis in range(ambient_dim):
                            + (my_center_${iaxis} - psource_${iaxis})
                              * (my_center_${iaxis} - psource_${iaxis})
                        %endfor
                            ;

                        const coord_t too_close_slack = (1+1e-2)*(1+1e-2);
                        found_too_close = found_too_close
                            || (dist_squared*too_close_slack  < radius_squared);

                        if (found_too_close)
                            break;
                    }
                }

                // }}}

                center_may_use_global_qbx[icenter] = !found_too_close;
                """,
            name="qbx_test_center_for_global").build(
                    self.cl_context,
                    type_aliases=(
                        ("box_id_t", box_id_dtype),
                        ("particle_id_t", particle_id_dtype),
                        ("coord_t", coord_dtype),
                        ),
                    var_values=(
                        ("ambient_dim", self.ambient_dim),
                        ))

    # }}}

    # {{{ find a QBX center for each target

    @memoize_method
    def centers_for_target_finder(self,
            coord_dtype, box_id_dtype, particle_id_dtype):
        # must be able to represent target_state
        assert int(np.iinfo(particle_id_dtype).min) < 0

        from pyopencl.elementwise import ElementwiseTemplate
        return ElementwiseTemplate(
            arguments="""//CL:mako//
                /* input */
                box_id_t aligned_nboxes,
                box_id_t *box_child_ids, /* [2**dimensions, aligned_nboxes] */
                particle_id_t ntargets,
                coord_t *targets,  /* [dimensions, ntargets] */
                signed char *qbx_forced_limits,
                coord_t root_extent,
                %for iaxis in range(ambient_dim):
                    coord_t lower_left_${iaxis},
                %endfor
                particle_id_t *balls_near_box_starts,
                particle_id_t *balls_near_box_lists,
                %for iaxis in range(ambient_dim):
                    coord_t *center_${iaxis},  /* [ncenters] */
                %endfor
                coord_t *radii,  /* [ncenters] */
                signed char *center_sides,  /* [ncenters] */
                char *center_may_use_global_qbx,

                /* output */
                particle_id_t *center_ids,
                """,
            operation=r"""//CL:mako//
                particle_id_t itgt = i;

                // {{{ compute normalized coordinates of target

                %for iaxis in range(ambient_dim):
                    coord_t my_target_${iaxis} = targets[itgt + ntargets*${iaxis}];
                %endfor
                %for iaxis in range(ambient_dim):
                    coord_t norm_my_target_${iaxis} =
                        (my_target_${iaxis} - lower_left_${iaxis}) / root_extent;
                %endfor

                // }}}

                // {{{ find leaf box containing target

                box_id_t ibox;
                int level = 0;
                box_id_t next_box = 0;

                do
                {
                    ibox = next_box;
                    ++level;

                    int which_child = 0;
                    %for iaxis in range(ambient_dim):
                        which_child +=
                            ((int) (norm_my_target_${iaxis} * (1 << level)) & 1)
                            * ${2**(ambient_dim-1-iaxis)};
                    %endfor

                    next_box =
                        box_child_ids[ibox + aligned_nboxes * which_child];
                }
                while (next_box);

                // }}}

                // {{{ walk list of centers near box

                particle_id_t start = balls_near_box_starts[ibox];
                particle_id_t stop = balls_near_box_starts[ibox+1];

                coord_t min_dist_squared = INFINITY;
                particle_id_t best_center_id = TGT_NO_QBX_NEEDED;
                signed char my_forced_limit = qbx_forced_limits[itgt];

                for (particle_id_t ilist = start; ilist < stop; ++ilist)
                {
                    particle_id_t icenter = balls_near_box_lists[ilist];

                    coord_t dist_squared = 0;
                    %for iaxis in range(ambient_dim):
                        {
                            coord_t axis_dist =
                                center_${iaxis}[icenter] - my_target_${iaxis};
                            dist_squared += axis_dist*axis_dist;
                        }
                    %endfor

                    // check if we're within the (l^2) radius
                    coord_t ball_radius = radii[icenter];
                    bool center_usable = (
                        dist_squared <= ball_radius * ball_radius * (1+1e-5));

                    if (center_usable && best_center_id == TGT_NO_QBX_NEEDED)
                    {
                        // The target is in a region touched by a QBX disk.
                        // Unless we find a usable QBX disk, we'll have to fail
                        // this target.
                        best_center_id = TGT_FAILED;
                    }

                    center_usable = center_usable
                        && (bool) center_may_use_global_qbx[icenter];

                    // check if we're on the desired side
                    center_usable = center_usable && (
                        my_forced_limit == 0
                        || my_forced_limit == center_sides[icenter]);

                    if (dist_squared < min_dist_squared && center_usable)
                    {
                        best_center_id = icenter;
                        min_dist_squared = dist_squared;
                    }
                }

                // }}}

                center_ids[itgt] = best_center_id;
                """,
            name="centers_for_target_finder",
            preamble=target_state.get_c_defines()
            ).build(
                    self.cl_context,
                    type_aliases=(
                        ("box_id_t", box_id_dtype),
                        ("particle_id_t", particle_id_dtype),
                        ("coord_t", coord_dtype),
                        ),
                    var_values=(
                        ("ambient_dim", self.ambient_dim),
                        ))

    # }}}


# }}}


# {{{ geometry data

class TargetInfo(Record):
    """ Targets consist of QBX centers, then target points for each target
    discretization. The starts of the target points for each target
    discretization are given by target_discr_starts.

    .. attribute:: targets
        Shape: ``[dim,ntargets]``

    .. attribute:: target_discr_starts
        Shape: ``[ndiscrs+1]``

    .. attribute:: ntargets
    """


class CenterInfo(Record):
    """
    .. attribute:: centers
        Shape: ``[dim][ncenters]``

    .. attribute:: sides

        -1 for inside, +1 for outside

    .. attribute:: radii

        Shape: ``[ncenters]``
    """

    @property
    def ncenters(self):
        return len(self.radii)


class QBXFMMGeometryData(object):
    def __init__(self, code_getter, source_discr,
            target_discrs_and_qbx_sides):
        """
        :arg targets: a tuple of
            :class:`pytential.discretization.Discretization`
            or
            :class:`pytential.discretization.target.TargetBase`
            instances
        """

        self.code_getter = code_getter
        self.source_discr = source_discr
        self.target_discrs_and_qbx_sides = \
                target_discrs_and_qbx_sides

    @property
    def cl_context(self):
        return self.source_discr.cl_context

    @property
    def coord_dtype(self):
        return self.source_discr.nodes().dtype

    # {{{ center info

    @memoize_method
    def kept_center_indices(self, el_group):
        # FIXME: Be more careful about which nodes to keep
        return np.arange(0, el_group.nunit_nodes, 8)

    @memoize_method
    def center_info(self):
        ncenters = 0
        for el_group in self.source_discr.groups:
            kept_indices = self.kept_center_indices(el_group)
            # two: one for positive side, one for negative side
            ncenters += 2 * len(kept_indices) * el_group.nelements

        from pytential import sym, bind
        from pytools.obj_array import make_obj_array
        with cl.CommandQueue(self.cl_context) as queue:
            radii_sym = sym.cse(2*sym.area_element(), "radii")
            all_radii, all_pos_centers, all_neg_centers = bind(self.source_discr,
                    make_obj_array([
                        radii_sym,
                        sym.Nodes() + radii_sym*sym.normal(),
                        sym.Nodes() - radii_sym*sym.normal()
                        ]))(queue)

            # The centers are returned from the above as multivectors.
            all_pos_centers = all_pos_centers.as_vector(np.object)
            all_neg_centers = all_neg_centers.as_vector(np.object)

            # -1 for inside, +1 for outside
            sides = cl.array.empty(
                    self.cl_context, ncenters, np.int8)
            radii = cl.array.empty(
                    self.cl_context, ncenters, self.coord_dtype)
            centers = make_obj_array([
                cl.array.empty(self.cl_context, ncenters,
                    self.coord_dtype)
                for i in xrange(self.source_discr.ambient_dim)])

            ibase = 0
            for el_group in self.source_discr.groups:
                kept_center_indices = self.kept_center_indices(el_group)
                group_len = len(kept_indices) * el_group.nelements

                for side, all_centers in [
                        (+1, all_pos_centers),
                        (-1, all_neg_centers),
                        ]:

                    sides[ibase:ibase + group_len].fill(side, queue=queue)

                    radii_view = radii[ibase:ibase + group_len] \
                            .reshape(el_group.nelements, len(kept_indices))
                    centers_view = make_obj_array([
                            centers_i[ibase:ibase + group_len]
                            .reshape((el_group.nelements, len(kept_indices)))
                            for centers_i in centers
                        ])
                    all_centers_view = make_obj_array([
                        el_group.view(pos_centers_i)
                        for pos_centers_i in all_centers
                        ])
                    self.code_getter.pick_expansion_centers(queue,
                            centers=centers_view,
                            all_centers=all_centers_view,
                            radii=radii_view,
                            all_radii=el_group.view(all_radii),
                            kept_center_indices=kept_center_indices)

                    ibase += group_len

            assert ibase == ncenters

        return CenterInfo(
                sides=sides,
                radii=radii,
                centers=centers)

    # }}}

    # {{{ target info

    @memoize_method
    def target_info(self):
        code_getter = self.code_getter

        src_discr = self.source_discr

        with cl.CommandQueue(self.cl_context) as queue:
            center_info = self.center_info()

            ntargets = center_info.ncenters
            target_discr_starts = []

            for target_discr, qbx_side in self.target_discrs_and_qbx_sides:
                target_discr_starts.append(ntargets)
                ntargets += target_discr.nnodes

            target_discr_starts.append(ntargets)

            targets = cl.array.empty(
                    self.cl_context, (src_discr.ambient_dim, ntargets),
                    self.coord_dtype)
            code_getter.copy_targets_kernel(sep_points_axes=True)(queue,
                    targets=targets[:, :center_info.ncenters],
                    points=center_info.centers)

            for start, (target_discr, _) in zip(
                    target_discr_starts, self.target_discrs_and_qbx_sides):
                code_getter.copy_targets_kernel(sep_points_axes=False)(
                        queue,
                        targets=targets[:, start:start+target_discr.nnodes],
                        points=target_discr.nodes())

            return TargetInfo(
                    targets=targets,
                    target_discr_starts=target_discr_starts,
                    ntargets=ntargets)

    def target_radii(self):
        """Shape: ``[ntargets]``"""

        tgt_info = self.target_info()
        center_info = self.center_info()

        with cl.CommandQueue(self.cl_context) as queue:
            target_radii = cl.array.zeros(queue, tgt_info.ntargets, self.coord_dtype)
            # all but the centers are zero
            target_radii[:center_info.ncenters] = center_info.radii

            return target_radii.with_queue(None)

    def qbx_forced_limits(self):
        """Shape: ``[ntargets]``, dtype: int8"""

        tgt_info = self.target_info()
        center_info = self.center_info()

        with cl.CommandQueue(self.cl_context) as queue:
            qbx_forced_limits = cl.array.empty(queue, tgt_info.ntargets, np.int8)
            qbx_forced_limits[:center_info.ncenters] = 0

            for tdstart, (target_discr, qbx_side) in \
                    zip(tgt_info.target_discr_starts,
                            self.target_discrs_and_qbx_sides):
                qbx_forced_limits[tdstart:tdstart+target_discr.nnodes] = qbx_side

            return qbx_forced_limits.with_queue(None)

    # }}}

    # {{{ tree

    @memoize_method
    def tree(self):
        code_getter = self.code_getter

        src_discr = self.source_discr

        with cl.CommandQueue(self.cl_context) as queue:
            nelements = sum(grp.nelements for grp in src_discr.groups)

            el_centers = cl.array.empty(
                    self.cl_context, (src_discr.ambient_dim, nelements),
                    self.coord_dtype)
            el_radii = cl.array.empty(self.cl_context, nelements, self.coord_dtype)

            # {{{ find sources and radii (=element 'centroids')

            # FIXME: Should probably use quad weights to find 'centroids' to deal
            # with non-symmetric elements.

            i_el_base = 0
            for grp in src_discr.groups:
                el_centers_view = el_centers[:, i_el_base:i_el_base+grp.nelements]
                el_radii_view = el_radii[i_el_base:i_el_base+grp.nelements]
                nodes_view = grp.view(src_discr.nodes())

                code_getter.find_element_centers(
                        queue, el_centers=el_centers_view, nodes=nodes_view)
                code_getter.find_element_radii(
                        queue, el_centers=el_centers_view, nodes=nodes_view,
                        el_radii=el_radii_view)

                i_el_base += grp.nelements

            # }}}

            target_info = self.target_info()

            tree, _ = code_getter.build_tree(queue,
                    particles=el_centers, source_radii=el_radii,
                    max_particles_in_box=30,
                    targets=target_info.targets,
                    target_radii=self.target_radii())

            # {{{ link point sources

            point_source_starts = cl.array.empty(self.cl_context,
                    nelements+1, tree.particle_id_dtype)

            i_el_base = 0
            for grp in src_discr.groups:
                point_source_starts.setitem(
                        slice(i_el_base, i_el_base+grp.nelements),
                        cl.array.arange(queue,
                            grp.node_nr_base, grp.node_nr_base + grp.nnodes,
                            grp.nunit_nodes,
                            dtype=point_source_starts.dtype),
                        queue=queue)

                i_el_base += grp.nelements

            point_source_starts.setitem(-1, self.source_discr.nnodes, queue=queue)

            tree = tree.link_point_sources(queue,
                    point_source_starts,
                    self.source_discr.nodes())

            # }}}

            return tree

    # }}}

    @memoize_method
    def traversal(self):
        with cl.CommandQueue(self.cl_context) as queue:
            trav, _ = self.code_getter.build_traversal(queue, self.tree())

            trav = trav.merge_close_lists(queue)
            return trav

    def leaf_to_center_lookup(self):
        center_info = self.center_info()

        with cl.CommandQueue(self.cl_context) as queue:
            lbl, _ = self.code_getter.build_leaf_to_ball_lookup(queue,
                    self.tree(), center_info.centers, center_info.radii)
            return lbl

    @memoize_method
    def global_qbx_flags(self):
        tree = self.tree()
        trav = self.traversal()
        center_info = self.center_info()

        qbx_center_to_target_box_lookup = \
                self.code_getter.qbx_center_to_target_box_lookup(
                        box_id_dtype=tree.box_id_dtype,
                        particle_id_dtype=tree.particle_id_dtype)

        qbx_center_for_global_tester = \
                self.code_getter.qbx_center_for_global_tester(
                        coord_dtype=tree.coord_dtype,
                        box_id_dtype=tree.box_id_dtype,
                        particle_id_dtype=tree.particle_id_dtype)

        with cl.CommandQueue(self.cl_context) as queue:
            result = cl.array.empty(queue, center_info.ncenters, np.int8)

            logging.info("find global qbx flags: start")

            box_to_target_box = cl.array.empty(
                    queue, tree.nboxes, tree.box_id_dtype)
            box_to_target_box.fill(-1)
            box_to_target_box[trav.target_boxes] = cl.array.arange(
                    queue, len(trav.target_boxes), dtype=tree.box_id_dtype)

            evt, (qbx_center_to_target_box,) = qbx_center_to_target_box_lookup(
                    queue,
                    box_to_target_box=box_to_target_box,
                    box_target_starts=tree.box_target_starts,
                    box_target_counts_nonchild=tree.box_target_counts_nonchild,
                    ncenters=center_info.ncenters)

            qbx_center_for_global_tester(*(
                        tuple(center_info.centers)
                    + (
                        center_info.radii,
                        qbx_center_to_target_box,

                        trav.neighbor_source_boxes_starts,
                        trav.neighbor_source_boxes_lists,
                        tree.box_point_source_starts,
                        tree.box_point_source_counts_cumul,
                    ) + tuple(tree.point_sources) + (
                        result,
                    )),
                    **dict(
                        queue=queue,
                        range=slice(center_info.ncenters)
                    ))

            logging.info("find global qbx flags: done")

            return result

    @memoize_method
    def target_to_center(self):
        """Find which QBX center, if any, is to be used for each target.
        -1 if none. -2 if a center needs to be used, but none was found.

        Shape: ``[ntargets]`` of :attr:`boxtree.Tree.particle_id_dtype`.
        """
        ltc = self.leaf_to_center_lookup()

        tree = self.tree()
        tgt_info = self.target_info()
        center_info = self.center_info()

        assert ltc.balls_near_box_starts.dtype == ltc.balls_near_box_lists.dtype
        assert ltc.balls_near_box_starts.dtype == tree.particle_id_dtype

        centers_for_target_finder = self.code_getter.centers_for_target_finder(
                coord_dtype=tree.coord_dtype,
                box_id_dtype=tree.box_id_dtype,
                particle_id_dtype=tree.particle_id_dtype)

        logging.info("find center for each target: start")
        with cl.CommandQueue(self.cl_context) as queue:
            result = cl.array.zeros(queue, tgt_info.ntargets, tree.particle_id_dtype)

            centers_for_target_finder(
                    *((
                        tree.aligned_nboxes,
                        tree.box_child_ids,
                        tgt_info.ntargets,
                        tgt_info.targets,
                        self.qbx_forced_limits(),
                        tree.root_extent,
                    ) + tuple(tree.bounding_box[0]) + (
                        ltc.balls_near_box_starts,
                        ltc.balls_near_box_lists,
                    ) + tuple(center_info.centers) + (
                        center_info.radii,
                        center_info.sides,
                        self.global_qbx_flags(),

                        result
                    )),
                    **dict(
                        queue=queue,
                        range=slice(center_info.ncenters, tgt_info.ntargets)))

            logging.info("find center for each target: done")

            return result

    # {{{ plotting (for debugging)

    def plot(self):
        with cl.CommandQueue(self.cl_context) as queue:
            import matplotlib.pyplot as pt
            nodes_host = self.source_discr.nodes().get(queue)
            pt.plot(nodes_host[0], nodes_host[1], "x-")

            global_flags = self.global_qbx_flags().get()

            tree = self.tree().get()
            from boxtree.visualization import TreePlotter
            tp = TreePlotter(tree)
            tp.draw_tree()

            # {{{ draw centers and circles

            center_info = self.center_info()
            centers = [
                    center_info.centers[0].get(queue),
                    center_info.centers[1].get(queue)]
            pt.plot(centers[0][global_flags == 0],
                    centers[1][global_flags == 0], "or")
            ax = pt.gca()
            for cx, cy, r in zip(
                    centers[0], centers[1], center_info.radii.get(queue)):
                ax.add_artist(
                        pt.Circle((cx, cy), r, fill=False, ls="dotted", lw=1))

            # }}}

            # {{{ draw target-to-center arrows

            ttc = self.target_to_center().get()
            targets = self.target_info().targets.get(queue)

            pt.plot(targets[0], targets[1], "x")

            tccount = 0
            checked = 0
            for tx, ty, tcenter in zip(
                    targets[0][center_info.ncenters:],
                    targets[1][center_info.ncenters:],
                    ttc[center_info.ncenters:]):
                checked += 1
                if tcenter >= 0:
                    tccount += 1
                    ax.add_artist(
                            pt.Line2D(
                                (tx, centers[0][tcenter]),
                                (ty, centers[1][tcenter]),
                                ))

            print "found a center for %d/%d targets" % (tccount, checked)

            # }}}

            pt.gca().set_aspect("equal")
            pt.show()

    # }}}

# }}}

# vim: foldmethod=marker:filetype=pyopencl
