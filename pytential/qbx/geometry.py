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
from cgen import Enum


import logging
logger = logging.getLogger(__name__)


# {{{ docs

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

    .. attribute:: ncenters

    .. automethod:: centers()
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
            target_stick_out_factor, debug):
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
        self.target_stick_out_factor = target_stick_out_factor
        self.debug = debug

    @property
    def cl_context(self):
        return self.lpot_source.cl_context

    @property
    def coord_dtype(self):
        return self.lpot_source.fine_density_discr.nodes().dtype

    # {{{ centers

    @memoize_method
    def centers(self):
        """ Return a :class:`CenterInfo`. |cached|
        """

        lpot_source = self.lpot_source

        with cl.CommandQueue(self.cl_context) as queue:
            from pytential.qbx.utils import get_interleaved_centers
            from pytools.obj_array import make_obj_array
            return make_obj_array([
                ccomp.with_queue(None)
                for ccomp in get_interleaved_centers(queue, lpot_source)])

    @property
    def ncenters(self):
        return len(self.centers()[0])

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

    # FIXME unused
    def xxx_target_radii(self):
        """Shape: ``[ntargets]``

        A list of target radii for FMM tree construction. In this list, the QBX
        centers have nonzero radii, and all other targets have radius zero.
        """

        tgt_info = self.target_info()

        with cl.CommandQueue(self.cl_context) as queue:
            target_radii = cl.array.zeros(queue, tgt_info.ntargets, self.coord_dtype)
            # target_radii[:self.ncenters] = center_info.radii

            return target_radii.with_queue(None)

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
            nsources = lpot_src.fine_density_discr.nnodes
            nparticles = nsources + target_info.ntargets

            refine_weights = cl.array.zeros(queue, nparticles, dtype=np.int32)
            refine_weights[:nsources] = 1
            refine_weights.finish()

            # NOTE: max_leaf_refine_weight has an impact on accuracy.
            # For instance, if a panel contains 64*4 = 256 nodes, then
            # a box with 128 sources will contain at most half a
            # panel, meaning that its width will be on the order h/2,
            # which means many QBX disks (diameter h) will be forced
            # to cross boxes.
            # So we set max_leaf_refine weight comfortably large
            # to avoid having too many disks overlap more than one box.
            #
            # FIXME: Should investigate this further.
            tree, _ = code_getter.build_tree(queue,
                    particles=lpot_src.fine_density_discr.nodes(),
                    targets=target_info.targets,
                    max_leaf_refine_weight=256,
                    refine_weights=refine_weights,
                    debug=self.debug,
                    kind="adaptive-level-restricted")

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

            evt, (qbx_center_to_target_box,) = qbx_center_to_target_box_lookup(
                    queue,
                    box_to_target_box=box_to_target_box,
                    box_target_starts=tree.box_target_starts,
                    box_target_counts_nonchild=tree.box_target_counts_nonchild,
                    user_target_from_tree_target=user_target_from_tree_target,
                    ncenters=self.ncenters)

            return qbx_center_to_target_box.with_queue(None)

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
        """

        tree = self.tree()

        with cl.CommandQueue(self.cl_context) as queue:
            from pyopencl.algorithm import copy_if

            logger.info("find global qbx centers: start")
            result, count, _ = copy_if(
                    cl.array.arange(queue, self.ncenters,
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
        from pytential.qbx.target_assoc import QBXTargetAssociator

        # FIXME: kernel ownership...
        tgt_assoc = QBXTargetAssociator(self.cl_context)

        tgt_info = self.target_info()

        from pytential.target import PointsTarget

        with cl.CommandQueue(self.cl_context) as queue:
            target_side_prefs = (self
                .target_side_preferences()[self.ncenters:].get(queue=queue))

        target_discrs_and_qbx_sides = [(
                PointsTarget(tgt_info.targets[:, self.ncenters:]),
                target_side_prefs.astype(np.int32))]

        # FIXME: try block...
        tgt_assoc_result = tgt_assoc(self.lpot_source,
                                     target_discrs_and_qbx_sides,
                                     stick_out_factor=self.target_stick_out_factor)

        tree = self.tree()

        with cl.CommandQueue(self.cl_context) as queue:
            result = cl.array.empty(queue, tgt_info.ntargets, tree.particle_id_dtype)
            result[:self.ncenters].fill(target_state.NO_QBX_NEEDED)
            result[self.ncenters:] = tgt_assoc_result.target_to_center

        return result

    @memoize_method
    def center_to_tree_targets(self):
        """Return a :class:`CenterToTargetList`. See :meth:`target_to_center`
        for the reverse look-up table with targets in user order.

        |cached|
        """

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

            count = np.asscalar(count.get())

            filtered_tree_ttc = filtered_tree_ttc[:count]
            filtered_target_ids = filtered_target_ids[:count].copy()

            center_target_starts, targets_sorted_by_center, _ = \
                    self.code_getter.key_value_sort(queue,
                            filtered_tree_ttc, filtered_target_ids,
                            self.ncenters, tree_ttc.dtype)

            logger.info("build center -> targets lookup table: done")

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
            logger.info("find non-qbx box target lists: start")

            flags = (self.user_target_to_center().with_queue(queue)
                    == target_state.NO_QBX_NEEDED)

            # The QBX centers come up with NO_QBX_NEEDED, but they don't
            # belong in this target list.

            # 'flags' is in user order, and should be.

            nqbx_centers = self.ncenters
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

            centers = self.centers()
            centers = [
                    centers[0].get(queue),
                    centers[1].get(queue)]
            pt.plot(centers[0][global_flags == 0],
                    centers[1][global_flags == 0], "oc",
                    label="centers needing local qbx")
            ax = pt.gca()
            # for icenter, (cx, cy, r) in enumerate(zip(
            #         centers[0], centers[1], center_info.radii.get(queue))):
            #     ax.add_artist(
            #             pt.Circle((cx, cy), r, fill=False, ls="dotted", lw=1))
            #     pt.text(cx, cy,
            #             str(icenter), fontsize=8,
            #             ha="left", va="center",
            #             bbox=dict(facecolor='white', alpha=0.5, lw=0))

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
