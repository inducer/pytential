from __future__ import division, absolute_import, print_function

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

from six.moves import zip

import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa
from pytools import memoize_method
from boxtree.tools import DeviceDataRecord
import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from cgen import Enum
from time import time


from pytential.qbx.utils import TreeCodeContainerMixin


import logging
logger = logging.getLogger(__name__)


# {{{ docs

__doc__ = """
For each invocation of the QBX FMM with a distinct set of (target, side request)
pairs, :class:`pytential.qbx.QBXLayerPotentialSource` creates an instance of
:class:`QBXFMMGeometryData`.

The module is described in top-down fashion, with the (conceptually)
highest-level objects first.

Geometry data
^^^^^^^^^^^^^

.. autoclass:: QBXFMMGeometryData

Subordinate data structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


class QBXFMMGeometryCodeGetter(TreeCodeContainerMixin):
    def __init__(self, cl_context, ambient_dim, tree_code_container, debug,
            _well_sep_is_n_away, _from_sep_smaller_crit):
        self.cl_context = cl_context
        self.ambient_dim = ambient_dim
        self.tree_code_container = tree_code_container
        self.debug = debug
        self._well_sep_is_n_away = _well_sep_is_n_away
        self._from_sep_smaller_crit = _from_sep_smaller_crit

    @memoize_method
    def copy_targets_kernel(self):
        knl = lp.make_kernel(
            """{[dim,i]:
                0<=dim<ndims and
                0<=i<npoints}""",
            """
                targets[dim, i] = points[dim, i]
                """,
            default_offset=lp.auto, name="copy_targets",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")
        knl = lp.tag_array_axes(knl, "points", "sep, C")

        knl = lp.tag_array_axes(knl, "targets", "stride:auto, stride:1")
        return lp.tag_inames(knl, dict(dim="ilp"))

    @property
    @memoize_method
    def build_traversal(self):
        from boxtree.traversal import FMMTraversalBuilder
        return FMMTraversalBuilder(
                self.cl_context,
                well_sep_is_n_away=self._well_sep_is_n_away,
                from_sep_smaller_crit=self._from_sep_smaller_crit,
                )

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
                    if in_bounds
                        qbx_center_to_target_box[itarget_user] = \
                                box_to_target_box[ibox] {id=tgt_write}
                    end
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
            silenced_warnings="write_race(tgt_write)",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.split_iname(knl, "ibox", 128,
                inner_tag="l.0", outer_tag="g.0")

        return knl

    @property
    @memoize_method
    def build_leaf_to_ball_lookup(self):
        from boxtree.area_query import LeavesToBallsLookupBuilder
        return LeavesToBallsLookupBuilder(self.cl_context)

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

                # "Does this target have a QBX center?"
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

    @property
    @memoize_method
    def pick_used_centers(self):
        knl = lp.make_kernel(
            """{[i]: 0<=i<ntargets}""",
            """
                <>target_has_center = (target_to_center[i] >= 0)
                center_is_used[target_to_center[i]] = 1 \
                    {id=center_is_used_write,if=target_has_center}
            """,
            [
                lp.GlobalArg("target_to_center", shape="ntargets", offset=lp.auto),
                lp.GlobalArg("center_is_used", shape="ncenters"),
                lp.ValueArg("ncenters", np.int32),
                lp.ValueArg("ntargets", np.int32),
            ],
            name="pick_used_centers",
            silenced_warnings="write_race(center_is_used_write)",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

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

        The :class:`pytential.qbx.QBXLayerPotentialSource`
        acting as the source geometry.

    .. attribute:: target_discrs_and_qbx_sides

        a list of tuples ``(discr, sides)``, where
        *discr* is a
        :class:`pytential.discretization.Discretization`
        or a
        :class:`pytential.discretization.target.TargetBase` instance, and
        *sides* is an array of (:class:`numpy.int8`) side requests for each
        target.

        The side request can take on the values
        found in :ref:`qbx-side-request-table`.

    .. attribute:: cl_context

        A :class:`pyopencl.Context`.

    .. attribute:: coord_dtype

    .. rubric :: Expansion centers

    .. attribute:: ncenters
    .. automethod:: centers()

    .. rubric :: Methods

    .. automethod:: target_info()
    .. automethod:: tree()
    .. automethod:: traversal()
    .. automethod:: qbx_center_to_target_box()
    .. automethod:: global_qbx_flags()
    .. automethod:: global_qbx_centers()
    .. automethod:: user_target_to_center()
    .. automethod:: center_to_tree_targets()
    .. automethod:: non_qbx_box_target_lists()
    .. automethod:: plot()
    """

    def __init__(self, code_getter, lpot_source,
            target_discrs_and_qbx_sides,
            target_association_tolerance,
            tree_kind, debug):
        """
        .. rubric:: Constructor arguments

        See the attributes of the same name for the meaning of most
        of the constructor arguments.

        :arg tree_kind: The tree kind to pass to the tree builder

        :arg debug: a :class:`bool` flag for whether to enable
            potentially costly self-checks
        """

        self.code_getter = code_getter
        self.lpot_source = lpot_source
        self.target_discrs_and_qbx_sides = \
                target_discrs_and_qbx_sides
        self.target_association_tolerance = target_association_tolerance
        self.tree_kind = tree_kind
        self.debug = debug

    @property
    def cl_context(self):
        return self.lpot_source.cl_context

    @property
    def coord_dtype(self):
        return self.lpot_source.quad_stage2_density_discr.nodes().dtype

    # {{{ centers/radii

    @property
    def ncenters(self):
        return len(self.centers()[0])

    @memoize_method
    def centers(self):
        """ Return an object array of (interleaved) center coordinates.

        ``coord_t [ambient_dim][ncenters]``
        """

        with cl.CommandQueue(self.cl_context) as queue:
            from pytential.qbx.utils import get_interleaved_centers
            from pytools.obj_array import make_obj_array
            return make_obj_array([
                ccomp.with_queue(None)
                for ccomp in get_interleaved_centers(queue, self.lpot_source)])

    @memoize_method
    def expansion_radii(self):
        """Return an  array of radii associated with the (interleaved)
        expansion centers.

        ``coord_t [ncenters]``
        """
        with cl.CommandQueue(self.cl_context) as queue:
            from pytential.qbx.utils import get_interleaved_radii
            return get_interleaved_radii(queue, self.lpot_source)

    # }}}

    # {{{ target info

    @memoize_method
    def target_info(self):
        """Return a :class:`TargetInfo`. |cached|"""

        code_getter = self.code_getter
        lpot_src = self.lpot_source

        with cl.CommandQueue(self.cl_context) as queue:
            ntargets = self.ncenters
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
                    targets=targets[:, :self.ncenters],
                    points=self.centers())

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

    def target_side_preferences(self):
        """Return one big array combining all the data from
        the *side* part of :attr:`TargetInfo.target_discrs_and_qbx_sides`.

        Shape: ``[ntargets]``, dtype: int8"""

        tgt_info = self.target_info()

        with cl.CommandQueue(self.cl_context) as queue:
            target_side_preferences = cl.array.empty(
                    queue, tgt_info.ntargets, np.int8)
            target_side_preferences[:self.ncenters] = 0

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
        """Build and return a :class:`boxtree.tree.Tree`
        for this source with these targets.

        |cached|
        """

        code_getter = self.code_getter
        lpot_src = self.lpot_source
        target_info = self.target_info()

        with cl.CommandQueue(self.cl_context) as queue:
            nsources = lpot_src.quad_stage2_density_discr.nnodes
            nparticles = nsources + target_info.ntargets

            target_radii = None
            if self.lpot_source._expansions_in_tree_have_extent:
                target_radii = cl.array.zeros(queue, target_info.ntargets,
                        self.coord_dtype)
                target_radii[:self.ncenters] = self.expansion_radii()

            refine_weights = cl.array.empty(queue, nparticles, dtype=np.int32)

            # Assign a weight of 1 to all sources, QBX centers, and conventional
            # (non-QBX) targets. Assign a weight of 0 to all targets that need
            # QBX centers. The potential at the latter targets is mediated
            # entirely by the QBX center, so as a matter of evaluation cost,
            # their location in the tree is irrelevant.
            refine_weights[:-target_info.ntargets] = 1
            user_ttc = self.user_target_to_center().with_queue(queue)
            refine_weights[-target_info.ntargets:] = (
                    user_ttc == target_state.NO_QBX_NEEDED).astype(np.int32)

            refine_weights.finish()

            tree, _ = code_getter.build_tree()(queue,
                    particles=lpot_src.quad_stage2_density_discr.nodes(),
                    targets=target_info.targets,
                    target_radii=target_radii,
                    max_leaf_refine_weight=lpot_src._max_leaf_refine_weight,
                    refine_weights=refine_weights,
                    debug=self.debug,
                    stick_out_factor=lpot_src._expansion_stick_out_factor,
                    extent_norm=lpot_src._box_extent_norm,
                    kind=self.tree_kind)

            if self.debug:
                tgt_count_2 = cl.array.sum(
                        tree.box_target_counts_nonchild, queue=queue).get()

                assert (tree.ntargets == tgt_count_2), (tree.ntargets, tgt_count_2)

            return tree

    # }}}

    @memoize_method
    def traversal(self, merge_close_lists=True):
        """Return a :class:`boxtree.traversal.FMMTraversalInfo`.

        :arg merge_close_lists: Use merged close lists. (See
            :meth:`boxtree.traversal.FMMTraversalInfo.merge_close_lists`)

        |cached|
        """

        with cl.CommandQueue(self.cl_context) as queue:
            trav, _ = self.code_getter.build_traversal(queue, self.tree(),
                    debug=self.debug,
                    _from_sep_smaller_min_nsources_cumul=(
                        self.lpot_source._from_sep_smaller_min_nsources_cumul))

            if (merge_close_lists
                    and self.lpot_source._expansions_in_tree_have_extent):
                trav = trav.merge_close_lists(queue)

            return trav

    @memoize_method
    def qbx_center_to_target_box(self):
        """Return a lookup table of length :attr:`ncenters`
        indicating the target box in which each
        QBX disk is located.

        |cached|
        """
        tree = self.tree()
        trav = self.traversal()

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

            qbx_center_to_target_box = cl.array.empty(
                    queue, self.ncenters, tree.box_id_dtype)

            if self.lpot_source.debug:
                qbx_center_to_target_box.fill(-1)

            evt, _ = qbx_center_to_target_box_lookup(
                    queue,
                    qbx_center_to_target_box=qbx_center_to_target_box,
                    box_to_target_box=box_to_target_box,
                    box_target_starts=tree.box_target_starts,
                    box_target_counts_nonchild=tree.box_target_counts_nonchild,
                    user_target_from_tree_target=user_target_from_tree_target,
                    ncenters=self.ncenters)

            if self.lpot_source.debug:
                assert 0 <= cl.array.min(qbx_center_to_target_box).get()
                assert (
                        cl.array.max(qbx_center_to_target_box).get()
                        < len(trav.target_boxes))

            return qbx_center_to_target_box.with_queue(None)

    @memoize_method
    def qbx_center_to_target_box_source_level(self, source_level):
        """Return an array for mapping qbx centers to indices into
        interaction lists as found in
        ``traversal.from_sep_smaller_by_level[source_level].``
        -1 if no such interaction list exist on *source_level*.
        """
        traversal = self.traversal()
        sep_smaller = traversal.from_sep_smaller_by_level[source_level]
        qbx_center_to_target_box = self.qbx_center_to_target_box()

        with cl.CommandQueue(self.cl_context) as queue:
            target_box_to_target_box_source_level = cl.array.empty(
                queue, len(traversal.target_boxes),
                dtype=traversal.tree.box_id_dtype
            )
            target_box_to_target_box_source_level.fill(-1)
            target_box_to_target_box_source_level[sep_smaller.nonempty_indices] = (
                cl.array.arange(queue, sep_smaller.num_nonempty_lists,
                                dtype=traversal.tree.box_id_dtype)
            )

            qbx_center_to_target_box_source_level = (
                target_box_to_target_box_source_level[
                    qbx_center_to_target_box
                ]
            )

            return qbx_center_to_target_box_source_level.with_queue(None)

    @memoize_method
    def global_qbx_flags(self):
        """Return an array of :class:`numpy.int8` of length
        :attr:`ncenters` indicating whether each center can use gloal QBX, i.e.
        whether a single expansion can mediate interactions from *all* sources
        to all targets for which it is valid. If global QBX can be used, the
        center's entry will be 1, otherwise it will be 0.

        (If not, local QBX is needed, and the center may only be
        able to mediate some of the interactions to a given target.)

        |cached|
        """

        with cl.CommandQueue(self.cl_context) as queue:
            result = cl.array.empty(queue, self.ncenters, np.int8)
            result.fill(1)

        return result.with_queue(None)

    @memoize_method
    def global_qbx_centers(self):
        """Build a list of indices of QBX centers that use global QBX.  This
        indexes into the global list of targets, (see :meth:`target_info`) of
        which the QBX centers occupy the first *ncenters*.

        Centers without any associated targets are excluded.
        """

        tree = self.tree()
        user_target_to_center = self.user_target_to_center()

        with cl.CommandQueue(self.cl_context) as queue:
            logger.debug("find global qbx centers: start")
            center_find_start_time = time()

            tgt_assoc_result = (
                    user_target_to_center.with_queue(queue)[self.ncenters:])

            center_is_used = cl.array.zeros(queue, self.ncenters, np.int8)

            self.code_getter.pick_used_centers(
                    queue,
                    center_is_used=center_is_used,
                    target_to_center=tgt_assoc_result,
                    ncenters=self.ncenters,
                    ntargets=len(tgt_assoc_result))

            from pyopencl.algorithm import copy_if

            result, count, _ = copy_if(
                    cl.array.arange(queue, self.ncenters,
                        tree.particle_id_dtype),
                    "global_qbx_flags[i] != 0 && center_is_used[i] != 0",
                    extra_args=[
                        ("global_qbx_flags", self.global_qbx_flags()),
                        ("center_is_used", center_is_used)
                        ],
                    queue=queue)

            center_find_elapsed = time() - center_find_start_time
            if center_find_elapsed > 0.1:
                done_logger = logger.info
            else:
                done_logger = logger.debug

            done_logger("find global qbx centers: done after %g seconds",
                    center_find_elapsed)

            if self.debug:
                logger.debug(
                        "find global qbx centers: using %d/%d centers"
                        % (int(count.get()), self.ncenters))

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
        from pytential.qbx.target_assoc import associate_targets_to_qbx_centers
        tgt_info = self.target_info()

        from pytential.target import PointsTarget

        with cl.CommandQueue(self.cl_context) as queue:
            target_side_prefs = (self
                    .target_side_preferences()[self.ncenters:].get(queue=queue))

            target_discrs_and_qbx_sides = [(
                    PointsTarget(tgt_info.targets[:, self.ncenters:]),
                    target_side_prefs.astype(np.int32))]

            target_association_wrangler = (
                    self.lpot_source.target_association_code_container
                    .get_wrangler(queue))

            tgt_assoc_result = associate_targets_to_qbx_centers(
                    self.lpot_source,
                    target_association_wrangler,
                    target_discrs_and_qbx_sides,
                    target_association_tolerance=(
                        self.target_association_tolerance))

            result = cl.array.empty(queue, tgt_info.ntargets,
                    tgt_assoc_result.target_to_center.dtype)
            result[:self.ncenters].fill(target_state.NO_QBX_NEEDED)
            result[self.ncenters:] = tgt_assoc_result.target_to_center

        return result.with_queue(None)

    @memoize_method
    def center_to_tree_targets(self):
        """Return a :class:`CenterToTargetList`. See :meth:`target_to_center`
        for the reverse look-up table with targets in user order.

        |cached|
        """

        user_ttc = self.user_target_to_center()

        with cl.CommandQueue(self.cl_context) as queue:
            logger.debug("build center -> targets lookup table: start")
            c2t_start_time = time()

            tree_ttc = cl.array.empty_like(user_ttc).with_queue(queue)
            tree_ttc[self.tree().sorted_target_ids] = user_ttc

            filtered_tree_ttc = cl.array.empty(queue, tree_ttc.shape, tree_ttc.dtype)
            filtered_target_ids = cl.array.empty(
                    queue, tree_ttc.shape, tree_ttc.dtype)
            count = cl.array.empty(queue, 1, tree_ttc.dtype)

            self.code_getter.filter_center_and_target_ids(tree_ttc.dtype)(
                    tree_ttc, filtered_tree_ttc, filtered_target_ids, count,
                    queue=queue, size=len(tree_ttc))

            count = np.asscalar(count.get())

            filtered_tree_ttc = filtered_tree_ttc[:count]
            filtered_target_ids = filtered_target_ids[:count].copy()

            center_target_starts, targets_sorted_by_center, _ = \
                    self.code_getter.key_value_sort(queue,
                            filtered_tree_ttc, filtered_target_ids,
                            self.ncenters, tree_ttc.dtype)

            c2t_elapsed = time() - c2t_start_time
            if c2t_elapsed > 0.1:
                done_logger = logger.info
            else:
                done_logger = logger.debug
            done_logger("build center -> targets lookup table: "
                    "done after %g seconds", c2t_elapsed)

            result = CenterToTargetList(
                    starts=center_target_starts,
                    lists=targets_sorted_by_center).with_queue(None)

            return result

    @memoize_method
    def non_qbx_box_target_lists(self):
        """Build a list of targets per box that don't need to bother with QBX.
        Returns a :class:`boxtree.tree.FilteredTargetListsInTreeOrder`.
        (I.e. a new target order is created for these targets, as we expect
        there to be many of them.)

        |cached|
        """

        with cl.CommandQueue(self.cl_context) as queue:
            logger.debug("find non-qbx box target lists: start")
            nonqbx_start_time = time()

            flags = (self.user_target_to_center().with_queue(queue)
                    == target_state.NO_QBX_NEEDED)

            # The QBX centers come up with NO_QBX_NEEDED, but they don't
            # belong in this target list.

            # 'flags' is in user order, and should be.

            nqbx_centers = self.ncenters
            flags[:nqbx_centers] = 0

            tree = self.tree()
            plfilt = self.code_getter.particle_list_filter()
            result = plfilt.filter_target_lists_in_tree_order(queue, tree, flags)

            nonqbx_elapsed = time() - nonqbx_start_time
            if nonqbx_elapsed > 0.1:
                done_logger = logger.info
            else:
                done_logger = logger.debug
            done_logger("find non-qbx box target lists: done after %g seconds",
                    nonqbx_elapsed)

            return result.with_queue(None)

    # {{{ plotting (for debugging)

    def plot(self, draw_circles=False, draw_center_numbers=False,
            highlight_centers=None):
        """Plot most of the information contained in a :class:`QBXFMMGeometryData`
        object, for debugging.

        :arg highlight_centers: If not *None*, an object with which the array of
            centers can be indexed to find the highlighted centers.

        .. note::

            This only works for two-dimensional geometries.
        """

        dims = self.tree().targets.shape[0]
        if dims != 2:
            raise ValueError("only 2-dimensional geometry info can be plotted")

        with cl.CommandQueue(self.cl_context) as queue:
            import matplotlib.pyplot as pt
            from meshmode.discretization.visualization import draw_curve
            draw_curve(self.lpot_source.quad_stage2_density_discr)

            global_flags = self.global_qbx_flags().get(queue=queue)

            tree = self.tree().get(queue=queue)
            from boxtree.visualization import TreePlotter
            tp = TreePlotter(tree)
            tp.draw_tree()

            # {{{ draw centers and circles

            centers = self.centers()
            centers = [
                    centers[0].get(queue),
                    centers[1].get(queue)]
            pt.plot(centers[0][global_flags == 0],
                    centers[1][global_flags == 0], "oc",
                    label="centers needing local qbx")

            if highlight_centers is not None:
                pt.plot(centers[0][highlight_centers],
                        centers[1][highlight_centers], "oc",
                        label="highlighted centers",
                        markersize=15)

            ax = pt.gca()

            if draw_circles:
                for icenter, (cx, cy, r) in enumerate(zip(
                        centers[0], centers[1],
                        self.expansion_radii().get(queue))):
                    ax.add_artist(
                            pt.Circle((cx, cy), r, fill=False, ls="dotted", lw=1))

            if draw_center_numbers:
                for icenter, (cx, cy, r) in enumerate(zip(centers[0], centers[1])):
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
                    targets[0][self.ncenters:],
                    targets[1][self.ncenters:],
                    ttc[self.ncenters:]):
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
