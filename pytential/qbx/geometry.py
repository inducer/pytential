from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from six.moves import zip

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
from pytools import memoize_method
from boxtree.tools import DeviceDataRecord
import loopy as lp
from cgen import Enum


import logging
logger = logging.getLogger(__name__)

# Targets match centers if they are within the center's circle, which, for matching
# purposes, is enlarged by this fraction.
QBX_CENTER_MATCH_THRESHOLD = 1 + 0.05


# {{{

__doc__ = """

.. note::

    This module documents :mod:`pytential` internals and is not typically
    needed in end-user applications.

This module documents data structures created for the execution of the QBX
FMM.  For each pair of (target discretizations, kernels),
:class:`pytential.discretization.qbx.QBXDiscretization` creates an instance of
:class:`QBXFMMGeometryData`.

The module is described in top-down fashion, with the (conceptually)
highest-level objects first.

Geometry data
^^^^^^^^^^^^^

.. autoclass:: QBXFMMGeometryData

Subordinate data structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CenterInfo()

.. autoclass:: TargetInfo()

.. autoclass:: CenterToTargetList()

Enums of special values
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: target_state

.. |cached| replace::
    Output is cached. Use ``obj.<method_name>.clear_cache(obj)`` to clear.

Geometry description code container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: QBXFMMGeometryCodeGetter
    :members:
    :undoc-members:

"""

# }}}


# {{{ code getter

class target_state(Enum):  # noqa
    """This enumeration contains special values that are used in
    the array returned by :meth:`QBXFMMGeometryData.target_to_center`.

    .. attribute:: NO_QBX_NEEDED

    .. attribute:: FAILED

        The code is unable to compute an accurate potential for this target.
        This happens if it is determined that QBX is required to compute
        an accurate potential, but no suitable center is found.
    """

    # c_name = "particle_id_t" (tree-dependent, intentionally unspecified)
    # dtype intentionally unspecified
    c_value_prefix = "TGT_"

    NO_QBX_NEEDED = -1

    FAILED = -2


class QBXFMMGeometryCodeGetter(object):
    def __init__(self, cl_context, ambient_dim, debug):
        self.cl_context = cl_context
        self.ambient_dim = ambient_dim
        self.debug = debug

    @property
    @memoize_method
    def pick_expansion_centers(self):
        knl = lp.make_kernel(
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
            default_offset=lp.auto,
            name="center_pick")

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.tag_array_axes(knl, "centers,all_centers", "sep, C, C")

        knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
        return lp.tag_inames(knl, dict(k="g.0", dim="ilp"))

    @property
    @memoize_method
    def find_element_centers(self):
        knl = lp.make_kernel(
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nunit_nodes}""",
            """
                el_centers[dim, k] = sum(i, nodes[dim, k, i])/nunit_nodes
                """,
            default_offset=lp.auto, name="find_element_centers")

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.split_iname(knl, "k", 128, inner_tag="l.0", outer_tag="g.0")
        return lp.tag_inames(knl, dict(dim="ilp"))

    @property
    @memoize_method
    def find_element_radii(self):
        knl = lp.make_kernel(
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nunit_nodes}""",
            """
                el_radii[k] = max(dim, max(i, \
                    fabs(nodes[dim, k, i] - el_centers[dim, k])))
                """,
            default_offset=lp.auto, name="find_element_radii")

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.split_iname(knl, "k", 128, inner_tag="l.0", outer_tag="g.0")
        return lp.tag_inames(knl, dict(dim="unr"))

    @memoize_method
    def copy_targets_kernel(self):
        knl = lp.make_kernel(
            """{[dim,i]:
                0<=dim<ndims and
                0<=i<npoints}""",
            """
                targets[dim, i] = points[dim, i]
                """,
            default_offset=lp.auto, name="copy_targets")

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")
        knl = lp.tag_array_axes(knl, "points", "sep, C")

        knl = lp.tag_array_axes(knl, "targets", "stride:auto, stride:1")
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
        # FIXME Iterating over all boxes to find which ones have QBX centers
        # is inefficient.

        knl = lp.make_kernel(
            [
                "{[ibox]: 0<=ibox<nboxes}",
                "{[itarget_tree]: b_t_start <= itarget_tree < b_t_start + ntargets}",
                ],
            """
            for ibox
                <> b_t_start = box_target_starts[ibox]
                <> ntargets = box_target_counts_nonchild[ibox]

                for itarget_tree
                    <> itarget_user = user_target_from_tree_target[itarget_tree]
                    <> in_bounds = itarget_user < ncenters

                    # This write is race-free because each center only belongs
                    # to one box.
                    qbx_center_to_target_box[itarget_user] = \
                            box_to_target_box[ibox] {id=tgt_write,if=in_bounds}
                end
            end
            """,
            [
                lp.GlobalArg("qbx_center_to_target_box", box_id_dtype,
                    shape="ncenters"),
                lp.GlobalArg("box_to_target_box", box_id_dtype),
                lp.GlobalArg("user_target_from_tree_target", None, shape=None),
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
        from boxtree.area_query import LeavesToBallsLookupBuilder
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
        # must be able to represent target_state values, which are negative
        assert int(np.iinfo(particle_id_dtype).min) < 0

        from pyopencl.elementwise import ElementwiseTemplate
        return ElementwiseTemplate(
            arguments="""//CL:mako//
                /* input */
                box_id_t aligned_nboxes,
                box_id_t *box_child_ids, /* [2**dimensions, aligned_nboxes] */
                particle_id_t ntargets,
                coord_t *targets,  /* [dimensions, ntargets] */
                signed char *target_side_preferences,
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
                /* passing this as a parameter (instead of baking it */
                /* into the source) works around an AMD bug */
                const coord_t match_threshold_squared,

                /* output */
                particle_id_t *center_ids,
                """,
            operation=r"""//CL:mako//
                #if 0
                    #define dbg_printf(ARGS) printf ARGS
                #else
                    #define dbg_printf(ARGS) /* */
                #endif


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

                dbg_printf(("[itgt=%d] leaf box=%d\n", itgt, ibox));

                // }}}

                // {{{ walk list of centers near box

                particle_id_t start = balls_near_box_starts[ibox];
                particle_id_t stop = balls_near_box_starts[ibox+1];

                coord_t min_dist_squared = INFINITY;
                particle_id_t best_center_id = TGT_NO_QBX_NEEDED;
                signed char my_side_preference = target_side_preferences[itgt];

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
                        dist_squared
                        <= ball_radius * ball_radius * match_threshold_squared);

                    dbg_printf(("[itgt=%d] icenter=%d "
                        "dist_squared=%g ball_rad_sq=%g match_threshold_squared=%g "
                        "usable=%d\n",
                        itgt, icenter, dist_squared, ball_radius*ball_radius,
                        match_threshold_squared,
                        center_usable));

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
                    int center_side = center_sides[icenter];
                    center_usable = center_usable && (

                           // accept volume or QBX, either side
                           my_side_preference == 0

                           // accept only QBX
                        || my_side_preference == center_side

                           // accept volume or QBX from a specific side
                        || (
                            (abs((int) my_side_preference) == 2)
                            && (my_side_preference == 2*center_side
                                || my_side_preference == 0))
                        );

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

    @property
    @memoize_method
    def key_value_sort(self):
        from pyopencl.algorithm import KeyValueSorter
        return KeyValueSorter(self.cl_context)

    @memoize_method
    def filter_center_and_target_ids(self, particle_id_dtype):
        from pyopencl.scan import GenericScanKernel
        from pyopencl.tools import VectorArg
        return GenericScanKernel(
                self.cl_context, particle_id_dtype,
                arguments=[
                    VectorArg(particle_id_dtype, "target_to_center"),
                    VectorArg(particle_id_dtype, "filtered_target_to_center"),
                    VectorArg(particle_id_dtype, "filtered_target_id"),
                    VectorArg(particle_id_dtype, "count"),
                    ],
                input_expr="(target_to_center[i] >= 0) ? 1 : 0",
                scan_expr="a+b", neutral="0",
                output_statement="""
                    if (prev_item != item)
                    {
                        filtered_target_to_center[item-1] = target_to_center[i];
                        filtered_target_id[item-1] = i;
                    }
                    if (i+1 == N) *count = item;
                    """)

# }}}


# {{{ geometry data

class TargetInfo(DeviceDataRecord):
    """Describes the internal structure of the QBX FMM's list of :attr:`targets`.
    The list consists of QBX centers, then target
    points for each target discretization. The starts of the target points for
    each target discretization are given by :attr:`target_discr_starts`.

    .. attribute:: targets

        Shape: ``[dim,ntargets]``

    .. attribute:: target_discr_starts

        Shape: ``[ndiscrs+1]``

        Start indices of targets for each target discretization.

        The first entry here is the start of the targets for the
        first target discretization. (The QBX centers start at index 0,
        a fact which is not explicitly represented.)

    .. attribute:: ntargets
    """


class CenterInfo(DeviceDataRecord):
    """Information on location of QBX centers.
    Returned from :meth:`QBXFMMGeometryData.center_info`.

    .. attribute:: centers

        Shape: ``[dim][ncenters]``

    .. attribute:: sides

        Shape: ``[ncenters]``

        -1 for inside, +1 for outside (relative to normal)

    .. attribute:: radii

        Shape: ``[ncenters]``

    .. attribute:: ncenters
    """

    @property
    def ncenters(self):
        return len(self.radii)


class CenterToTargetList(DeviceDataRecord):
    """A lookup table of targets covered by each QBX disk. Indexed by global
    number of QBX center, ``lists[start[i]:start[i+1]]`` indicates numbers
    of the overlapped targets in tree target order.

    See :meth:`QBXFMMGeometryData.center_to_tree_targets`.

    .. attribute:: starts

        Shape: ``[ncenters+1]``

    .. attribute:: lists

        Lists of targets in tree order. Use with :attr:`starts`.
    """


class QBXFMMGeometryData(object):
    """

    .. rubric :: Attributes

    .. attribute:: code_getter

        The :class:`QBXFMMGeometryCodeGetter` for this object.

    .. attribute:: lpot_source

        The :class:`pytential.discretization.qbx.QBXDiscretization`
        acting as the source geometry.

    .. attribute:: target_discrs_and_qbx_sides

        a list of tuples ``(discr, sides)``, where
        *discr* is a
        :class:`pytential.discretization.Discretization`
        or a
        :class:`pytential.discretization.target.TargetBase` instance, and
        *sides* is an array of (:class:`numpy.int8`) side requests for each
        target.

        The side request can take the following values for each target:

        ===== ==============================================
        Value Meaning
        ===== ==============================================
        0     Volume target. If near a QBX center,
              the value from the QBX expansion is returned,
              otherwise the volume potential is returned.

        -1    Surface target. Return interior limit from
              interior-side QBX expansion.

        +1    Surface target. Return exterior limit from
              exterior-side QBX expansion.

        -2    Volume target. If within an *interior* QBX disk,
              the value from the QBX expansion is returned,
              otherwise the volume potential is returned.

        +2    Volume target. If within an *exterior* QBX disk,
              the value from the QBX expansion is returned,
              otherwise the volume potential is returned.
        ===== ==============================================

    .. attribute:: cl_context

        A :class:`pyopencl.Context`.

    .. attribute:: coord_dtype

    .. rubric :: Methods

    .. automethod:: center_info()
    .. automethod:: target_info()
    .. automethod:: tree()
    .. automethod:: traversal()
    .. automethod:: leaf_to_center_lookup
    .. automethod:: qbx_center_to_target_box()
    .. automethod:: global_qbx_flags()
    .. automethod:: global_qbx_centers()
    .. automethod:: user_target_to_center()
    .. automethod:: center_to_tree_targets()
    .. automethod:: global_qbx_centers_box_target_lists()
    .. automethod:: non_qbx_box_target_lists()
    .. automethod:: plot()
    """

    def __init__(self, code_getter, lpot_source,
            target_discrs_and_qbx_sides, debug):
        """
        .. rubric:: Constructor arguments

        See the attributes of the same name for the meaning of most
        of the constructor arguments.

        :arg debug: a :class:`bool` flag for whether to enable
            potentially costly self-checks
        """

        self.code_getter = code_getter
        self.lpot_source = lpot_source
        self.target_discrs_and_qbx_sides = \
                target_discrs_and_qbx_sides
        self.debug = debug

    @property
    def cl_context(self):
        return self.lpot_source.cl_context

    @property
    def coord_dtype(self):
        return self.lpot_source.fine_density_discr.nodes().dtype

    # {{{ center info

    @memoize_method
    def kept_center_indices(self, el_group):
        """Return indices of unit nodes (of the target discretization)
        that will give rise to QBX centers.
        """

        # FIXME: Be more careful about which nodes to keep
        return np.arange(0, el_group.nunit_nodes)

    @memoize_method
    def center_info(self):
        """Return a :class:`CenterInfo`. |cached|

        """
        self_discr = self.lpot_source.density_discr

        ncenters = 0
        for el_group in self_discr.groups:
            kept_indices = self.kept_center_indices(el_group)
            # two: one for positive side, one for negative side
            ncenters += 2 * len(kept_indices) * el_group.nelements

        adim = self_discr.ambient_dim
        dim = self_discr.dim

        from pytential import sym, bind
        from pytools.obj_array import make_obj_array
        with cl.CommandQueue(self.cl_context) as queue:
            radii_sym = sym.cse(2*sym.area_element(adim, dim), "radii")
            all_radii, all_pos_centers, all_neg_centers = bind(self_discr,
                    make_obj_array([
                        radii_sym,
                        sym.nodes(adim) + radii_sym*sym.normal(adim),
                        sym.nodes(adim) - radii_sym*sym.normal(adim)
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
                for i in range(self_discr.ambient_dim)])

            ibase = 0
            for el_group in self_discr.groups:
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
                centers=centers).with_queue(None)

    # }}}

    # {{{ target info

    @memoize_method
    def target_info(self):
        """Return a :class:`TargetInfo`. |cached|"""

        code_getter = self.code_getter

        lpot_src = self.lpot_source

        with cl.CommandQueue(self.cl_context) as queue:
            center_info = self.center_info()

            ntargets = center_info.ncenters
            target_discr_starts = []

            for target_discr, qbx_side in self.target_discrs_and_qbx_sides:
                target_discr_starts.append(ntargets)
                ntargets += target_discr.nnodes

            target_discr_starts.append(ntargets)

            targets = cl.array.empty(
                    self.cl_context, (lpot_src.ambient_dim, ntargets),
                    self.coord_dtype)
            code_getter.copy_targets_kernel()(
                    queue,
                    targets=targets[:, :center_info.ncenters],
                    points=center_info.centers)

            for start, (target_discr, _) in zip(
                    target_discr_starts, self.target_discrs_and_qbx_sides):
                code_getter.copy_targets_kernel()(
                        queue,
                        targets=targets[:, start:start+target_discr.nnodes],
                        points=target_discr.nodes())

            return TargetInfo(
                    targets=targets,
                    target_discr_starts=target_discr_starts,
                    ntargets=ntargets).with_queue(None)

    def target_radii(self):
        """Shape: ``[ntargets]``

        A list of target radii for FMM tree construction. In this list, the QBX
        centers have the radii determined by :meth:`center_info`, and all other
        targets have radius zero.
        """

        tgt_info = self.target_info()
        center_info = self.center_info()

        with cl.CommandQueue(self.cl_context) as queue:
            target_radii = cl.array.zeros(queue, tgt_info.ntargets, self.coord_dtype)
            target_radii[:center_info.ncenters] = center_info.radii

            return target_radii.with_queue(None)

    def target_side_preferences(self):
        """Return one big array combining all the data from
        the *side* part of :attr:`TargetInfo.target_discrs_and_qbx_sides`.

        Shape: ``[ntargets]``, dtype: int8"""

        tgt_info = self.target_info()
        center_info = self.center_info()

        with cl.CommandQueue(self.cl_context) as queue:
            target_side_preferences = cl.array.empty(
                    queue, tgt_info.ntargets, np.int8)
            target_side_preferences[:center_info.ncenters] = 0

            for tdstart, (target_discr, qbx_side) in \
                    zip(tgt_info.target_discr_starts,
                            self.target_discrs_and_qbx_sides):
                target_side_preferences[tdstart:tdstart+target_discr.nnodes] \
                    = qbx_side

            return target_side_preferences.with_queue(None)

    # }}}

    # {{{ tree

    @memoize_method
    def tree(self):
        """Build and return a :class:`boxtree.tree.TreeWithLinkedPointSources`
        for this source with these targets.

        |cached|
        """

        code_getter = self.code_getter

        lpot_src = self.lpot_source

        with cl.CommandQueue(self.cl_context) as queue:
            nelements = sum(grp.nelements
                    for grp in lpot_src.fine_density_discr.groups)

            el_centers = cl.array.empty(
                    self.cl_context, (lpot_src.ambient_dim, nelements),
                    self.coord_dtype)
            el_radii = cl.array.empty(self.cl_context, nelements, self.coord_dtype)

            # {{{ find sources and radii (=element 'centroids')

            # FIXME: Should probably use quad weights to find 'centroids' to deal
            # with non-symmetric elements.

            i_el_base = 0
            for grp in lpot_src.fine_density_discr.groups:
                el_centers_view = el_centers[:, i_el_base:i_el_base+grp.nelements]
                el_radii_view = el_radii[i_el_base:i_el_base+grp.nelements]
                nodes_view = grp.view(lpot_src.fine_density_discr.nodes())

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
                    target_radii=self.target_radii(),
                    debug=self.debug)

            # {{{ link point sources

            point_source_starts = cl.array.empty(self.cl_context,
                    nelements+1, tree.particle_id_dtype)

            i_el_base = 0
            for grp in lpot_src.fine_density_discr.groups:
                point_source_starts.setitem(
                        slice(i_el_base, i_el_base+grp.nelements),
                        cl.array.arange(queue,
                            grp.node_nr_base, grp.node_nr_base + grp.nnodes,
                            grp.nunit_nodes,
                            dtype=point_source_starts.dtype),
                        queue=queue)

                i_el_base += grp.nelements

            point_source_starts.setitem(
                    -1, self.lpot_source.fine_density_discr.nnodes, queue=queue)

            from boxtree.tree import link_point_sources
            tree = link_point_sources(queue, tree,
                    point_source_starts,
                    self.lpot_source.fine_density_discr.nodes())

            # }}}

            return tree

    # }}}

    @memoize_method
    def traversal(self):
        """Return a :class:`boxtree.traversal.FMMTraversalInfo` with merged
        close lists.
        (See :meth:`boxtree.traversal.FMMTraversalInfo.merge_close_lists`)
        |cached|
        """

        with cl.CommandQueue(self.cl_context) as queue:
            trav, _ = self.code_getter.build_traversal(queue, self.tree(),
                    debug=self.debug)

            trav = trav.merge_close_lists(queue)
            return trav

    def leaf_to_center_lookup(self):
        """Return a :class:`boxtree.area_query.LeavesToBallsLookup` to look up
        which which QBX disks overlap each leaf box.
        """

        center_info = self.center_info()

        with cl.CommandQueue(self.cl_context) as queue:
            lbl, _ = self.code_getter.build_leaf_to_ball_lookup(queue,
                    self.tree(), center_info.centers, center_info.radii)
            return lbl

    @memoize_method
    def qbx_center_to_target_box(self):
        """Return a lookup table of length :attr:`CenterInfo.ncenters`
        (see :meth:`center_info`) indicating the target box in which each
        QBX disk is located.

        |cached|
        """
        tree = self.tree()
        trav = self.traversal()
        center_info = self.center_info()

        qbx_center_to_target_box_lookup = \
                self.code_getter.qbx_center_to_target_box_lookup(
                        # particle_id_dtype:
                        tree.particle_id_dtype,
                        # box_id_dtype:
                        tree.box_id_dtype,
                        )

        with cl.CommandQueue(self.cl_context) as queue:
            box_to_target_box = cl.array.empty(
                    queue, tree.nboxes, tree.box_id_dtype)
            if self.debug:
                box_to_target_box.fill(-1)
            box_to_target_box[trav.target_boxes] = cl.array.arange(
                    queue, len(trav.target_boxes), dtype=tree.box_id_dtype)

            sorted_target_ids = self.tree().sorted_target_ids
            user_target_from_tree_target = \
                    cl.array.empty_like(sorted_target_ids).with_queue(queue)

            user_target_from_tree_target[sorted_target_ids] = \
                    cl.array.arange(
                            queue, len(sorted_target_ids),
                            user_target_from_tree_target.dtype)

            evt, (qbx_center_to_target_box,) = qbx_center_to_target_box_lookup(
                    queue,
                    box_to_target_box=box_to_target_box,
                    box_target_starts=tree.box_target_starts,
                    box_target_counts_nonchild=tree.box_target_counts_nonchild,
                    user_target_from_tree_target=user_target_from_tree_target,
                    ncenters=center_info.ncenters)

            return qbx_center_to_target_box.with_queue(None)

    @memoize_method
    def global_qbx_flags(self):
        """Return an array of :class:`numpy.int8` of length
        :attr:`CenterInfo.ncenters` (see :meth:`center_info`) indicating
        whether each center can use gloal QBX, i.e. whether a single expansion
        can mediate interactions from *all* sources to all targets for which it
        is valid. If global QBX can be used, the center's entry will be 1,
        otherwise it will be 0.

        (If not, local QBX is needed, and the center may only be
        able to mediate some of the interactions to a given target.)

        |cached|
        """

        tree = self.tree()
        trav = self.traversal()
        center_info = self.center_info()

        qbx_center_for_global_tester = \
                self.code_getter.qbx_center_for_global_tester(
                        # coord_dtype:
                        tree.coord_dtype,
                        # box_id_dtype:
                        tree.box_id_dtype,
                        # particle_id_dtype:
                        tree.particle_id_dtype)

        with cl.CommandQueue(self.cl_context) as queue:
            result = cl.array.empty(queue, center_info.ncenters, np.int8)

            logger.info("find global qbx flags: start")

            qbx_center_for_global_tester(*(
                    tuple(center_info.centers)
                    + (
                        center_info.radii,
                        self.qbx_center_to_target_box(),

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

            logger.info("find global qbx flags: done")

            return result.with_queue(None)

    @memoize_method
    def global_qbx_centers(self):
        """Build a list of indices of QBX centers that use global QBX.  This
        indexes into the global list of targets, (see :meth:`target_info`) of
        which the QBX centers occupy the first *ncenters*.
        """

        tree = self.tree()

        with cl.CommandQueue(self.cl_context) as queue:
            from pyopencl.algorithm import copy_if

            logger.info("find global qbx centers: start")
            result, count, _ = copy_if(
                    cl.array.arange(queue, self.center_info().ncenters,
                        tree.particle_id_dtype),
                    "global_qbx_flags[i] != 0",
                    extra_args=[
                        ("global_qbx_flags", self.global_qbx_flags())
                        ],
                    queue=queue)

            logger.info("find global qbx centers: done")

            return result[:count.get()].with_queue(None)

    @memoize_method
    def user_target_to_center(self):
        """Find which QBX center, if any, is to be used for each target.
        :attr:`target_state.NO_QBX_NEEDED` if none. :attr:`target_state.FAILED`
        if a center needs to be used, but none was found.
        See :meth:`center_to_tree_targets` for the reverse look-up table.

        Shape: ``[ntargets]`` of :attr:`boxtree.Tree.particle_id_dtype`, with extra
        values from :class:`target_state` allowed. Targets occur in user order.
        """

        ltc = self.leaf_to_center_lookup()

        tree = self.tree()
        tgt_info = self.target_info()
        center_info = self.center_info()

        assert ltc.balls_near_box_starts.dtype == ltc.balls_near_box_lists.dtype
        assert ltc.balls_near_box_starts.dtype == tree.particle_id_dtype

        centers_for_target_finder = self.code_getter.centers_for_target_finder(
                tree.coord_dtype,
                tree.box_id_dtype,
                tree.particle_id_dtype)

        logger.info("find center for each target: start")
        with cl.CommandQueue(self.cl_context) as queue:
            result = cl.array.empty(queue, tgt_info.ntargets, tree.particle_id_dtype)
            result[:center_info.ncenters].fill(target_state.NO_QBX_NEEDED)

            centers_for_target_finder(
                    *((
                        tree.aligned_nboxes,
                        tree.box_child_ids,
                        tgt_info.ntargets,
                        tgt_info.targets,
                        self.target_side_preferences(),
                        tree.root_extent,
                    ) + tuple(tree.bounding_box[0]) + (
                        ltc.balls_near_box_starts,
                        ltc.balls_near_box_lists,
                    ) + tuple(center_info.centers) + (
                        center_info.radii,
                        center_info.sides,
                        self.global_qbx_flags(),
                        QBX_CENTER_MATCH_THRESHOLD**2,

                        result
                    )),
                    **dict(
                        queue=queue,
                        range=slice(center_info.ncenters, tgt_info.ntargets)))

            logger.info("find center for each target: done")

            return result.with_queue(None)

    @memoize_method
    def center_to_tree_targets(self):
        """Return a :class:`CenterToTargetList`. See :meth:`target_to_center`
        for the reverse look-up table with targets in user order.

        |cached|
        """

        center_info = self.center_info()
        user_ttc = self.user_target_to_center()

        with cl.CommandQueue(self.cl_context) as queue:
            logger.info("build center -> targets lookup table: start")

            tree_ttc = cl.array.empty_like(user_ttc).with_queue(queue)
            tree_ttc[self.tree().sorted_target_ids] = user_ttc

            filtered_tree_ttc = cl.array.empty(queue, tree_ttc.shape, tree_ttc.dtype)
            filtered_target_ids = cl.array.empty(
                    queue, tree_ttc.shape, tree_ttc.dtype)
            count = cl.array.empty(queue, 1, tree_ttc.dtype)

            self.code_getter.filter_center_and_target_ids(tree_ttc.dtype)(
                    tree_ttc, filtered_tree_ttc, filtered_target_ids, count,
                    queue=queue, size=len(tree_ttc))

            count = count.get()

            filtered_tree_ttc = filtered_tree_ttc[:count]
            filtered_target_ids = filtered_target_ids[:count].copy()

            center_target_starts, targets_sorted_by_center, _ = \
                    self.code_getter.key_value_sort(queue,
                            filtered_tree_ttc, filtered_target_ids,
                            center_info.ncenters, tree_ttc.dtype)

            logger.info("build center -> targets lookup table: done")

            result = CenterToTargetList(
                    starts=center_target_starts,
                    lists=targets_sorted_by_center).with_queue(None)

            return result

    @memoize_method
    def global_qbx_centers_box_target_lists(self):
        """Build a list of targets per box consisting only of global QBX centers.
        Returns a :class:`boxtree.tree.FilteredTargetListsInUserOrder`.
        (I.e. no new target order is created for these targets, as we expect
        there to be (relatively) not many of them.)

        |cached|
        """

        center_info = self.center_info()
        with cl.CommandQueue(self.cl_context) as queue:
            logger.info("find global qbx centers box target list: start")

            flags = cl.array.zeros(queue, self.tree().ntargets, np.int8)

            flags[:center_info.ncenters] = self.global_qbx_flags()

            from boxtree.tree import filter_target_lists_in_user_order
            result = filter_target_lists_in_user_order(queue, self.tree(), flags)

            logger.info("find global qbx centers box target list: done")

            return result.with_queue(None)

    @memoize_method
    def non_qbx_box_target_lists(self):
        """Build a list of targets per box that don't need to bother with QBX.
        Returns a :class:`boxtree.tree.FilteredTargetListsInTreeOrder`.
        (I.e. a new target order is created for these targets, as we expect
        there to be many of them.)

        |cached|
        """

        with cl.CommandQueue(self.cl_context) as queue:
            logger.info("find non-qbx box target lists: start")

            flags = (self.user_target_to_center().with_queue(queue)
                    == target_state.NO_QBX_NEEDED)

            # The QBX centers come up with NO_QBX_NEEDED, but they don't
            # belong in this target list.

            # 'flags' is in user order, and should be.

            nqbx_centers = self.center_info().ncenters
            flags[:nqbx_centers] = 0

            from boxtree.tree import filter_target_lists_in_tree_order
            result = filter_target_lists_in_tree_order(queue, self.tree(), flags)

            logger.info("find non-qbx box target lists: done")

            return result.with_queue(None)

    # {{{ plotting (for debugging)

    def plot(self):
        """Plot most of the information contained in a :class:`QBXFMMGeometryData`
        object, for debugging.

        .. note::

            This only works for two-dimensional geometries.
        """

        dims = self.tree().targets.shape[0]
        if dims != 2:
            raise ValueError("only 2-dimensional geometry info can be plotted")

        with cl.CommandQueue(self.cl_context) as queue:
            import matplotlib.pyplot as pt
            nodes_host = self.lpot_source.fine_density_discr.nodes().get(queue)
            pt.plot(nodes_host[0], nodes_host[1], "x-")

            global_flags = self.global_qbx_flags().get(queue=queue)

            tree = self.tree().get(queue=queue)
            from boxtree.visualization import TreePlotter
            tp = TreePlotter(tree)
            tp.draw_tree()

            # {{{ draw centers and circles

            center_info = self.center_info()
            centers = [
                    center_info.centers[0].get(queue),
                    center_info.centers[1].get(queue)]
            pt.plot(centers[0][global_flags == 0],
                    centers[1][global_flags == 0], "oc",
                    label="centers needing local qbx")
            ax = pt.gca()
            for icenter, (cx, cy, r) in enumerate(zip(
                    centers[0], centers[1], center_info.radii.get(queue))):
                ax.add_artist(
                        pt.Circle((cx, cy), r, fill=False, ls="dotted", lw=1))
                pt.text(cx, cy,
                        str(icenter), fontsize=8,
                        ha="left", va="center",
                        bbox=dict(facecolor='white', alpha=0.5, lw=0))

            # }}}

            # {{{ draw target-to-center arrows

            ttc = self.user_target_to_center().get(queue)
            tinfo = self.target_info()
            targets = tinfo.targets.get(queue)

            pt.plot(targets[0], targets[1], "+")
            pt.plot(
                    targets[0][ttc == target_state.FAILED],
                    targets[1][ttc == target_state.FAILED],
                    "dr", markersize=15, label="failed targets")

            for itarget in np.where(ttc == target_state.FAILED)[0]:
                pt.text(
                        targets[0][itarget],
                        targets[1][itarget],
                        str(itarget), fontsize=8,
                        ha="left", va="center",
                        bbox=dict(facecolor='white', alpha=0.5, lw=0))

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

            print("found a center for %d/%d targets" % (tccount, checked))

            # }}}

            pt.gca().set_aspect("equal")
            pt.legend()
            pt.show()

    # }}}

# }}}

# vim: foldmethod=marker:filetype=pyopencl
