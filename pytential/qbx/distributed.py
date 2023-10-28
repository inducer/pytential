__copyright__ = "Copyright (C) 2022 Hao Gao"

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

from pytential.qbx import QBXLayerPotentialSource
from arraycontext import PyOpenCLArrayContext, unflatten
from typing import Optional, Any
from dataclasses import dataclass
import numpy as np
import pyopencl as cl
from boxtree.tools import DeviceDataRecord
from boxtree.timing import TimingRecorder
from pytools import memoize_method


@dataclass
class GlobalQBXFMMGeometryData:
    """A trimmed-down version of :class:`QBXFMMGeometryData` to be broadcasted for
    the distributed implementation. Each rank should have the same global geometry
    data.
    """
    global_traversal: Any
    centers: Any
    expansion_radii: Any
    global_qbx_centers: Any
    qbx_center_to_target_box: Any
    non_qbx_box_target_lists: Any
    center_to_tree_targets: Any


class LocalQBXFMMGeometryData(DeviceDataRecord):
    """A subset of the global geometry data used by each rank to calculate potentials
    using FMM. Each rank should have its own version of the local geometry data.
    """
    def non_qbx_box_target_lists(self):
        return self._non_qbx_box_target_lists

    def traversal(self):
        return self.local_trav

    def tree(self):
        return self.traversal().tree

    def centers(self):
        return self._local_centers

    @property
    def ncenters(self):
        return self._local_centers.shape[1]

    def global_qbx_centers(self):
        return self._global_qbx_centers

    def expansion_radii(self):
        return self._expansion_radii

    def qbx_center_to_target_box(self):
        return self._local_qbx_center_to_target_box

    def center_to_tree_targets(self):
        return self._local_center_to_tree_targets

    def qbx_targets(self):
        return self._qbx_targets

    def qbx_center_to_target_box_source_level(self, source_level):
        return self._qbx_center_to_target_box_source_level[source_level]

    @memoize_method
    def build_rotation_classes_lists(self):
        with cl.CommandQueue(self.cl_context) as queue:
            trav = self.traversal().to_device(queue)
            tree = self.tree().to_device(queue)

            from boxtree.rotation_classes import RotationClassesBuilder
            return RotationClassesBuilder(self.cl_context)(
                queue, trav, tree)[0].get(queue)

    def eval_qbx_targets(self):
        return self.qbx_targets()

    @memoize_method
    def m2l_rotation_lists(self):
        return self.build_rotation_classes_lists().from_sep_siblings_rotation_classes

    @memoize_method
    def m2l_rotation_angles(self):
        return (self
                .build_rotation_classes_lists()
                .from_sep_siblings_rotation_class_to_angle)


# {{{ Traversal builder

class QBXFMMGeometryDataTraversalBuilder:
    # Could we use QBXFMMGeometryDataCodeContainer instead?
    def __init__(self, context, well_sep_is_n_away=1, from_sep_smaller_crit=None,
                 _from_sep_smaller_min_nsources_cumul=0):
        from boxtree.traversal import FMMTraversalBuilder
        self.traversal_builder = FMMTraversalBuilder(
            context,
            well_sep_is_n_away=well_sep_is_n_away,
            from_sep_smaller_crit=from_sep_smaller_crit)
        self._from_sep_smaller_min_nsources_cumul = (
            _from_sep_smaller_min_nsources_cumul)

    def __call__(self, queue, tree, **kwargs):
        trav, evt = self.traversal_builder(
            queue, tree,
            _from_sep_smaller_min_nsources_cumul=(
                self._from_sep_smaller_min_nsources_cumul),
            **kwargs)

        return trav, evt

# }}}


def broadcast_global_geometry_data(
        comm, actx: PyOpenCLArrayContext, traversal_builder, global_geometry_data):
    """Broadcasts useful fields of global geometry data from the root rank to the
    worker ranks, so that each rank can form local geometry data independently.

    This function should be called collectively by all ranks in `comm`.

    :arg comm: MPI communicator.
    :arg traversal_builder: a :class:`QBXFMMGeometryDataTraversalBuilder` object,
        used for constructing the global traversal object from the broadcasted global
        tree. This argument is significant on all ranks.
    :arg global_geometry_data: an object of :class:`ToHostTransferredGeoDataWrapper`,
        the global geometry data on host memory. This argument is only significant on
        the root rank.
    :returns: a :class:`GlobalQBXFMMGeometryData` object on each worker
        rank, representing the broadcasted subset of the global geometry data, used
        for constructing the local geometry data independently. See
        :func:`compute_local_geometry_data`.
    """
    mpi_rank = comm.Get_rank()
    queue = actx.queue

    global_traversal = None
    global_tree = None
    centers = None
    expansion_radii = None
    global_qbx_centers = None
    qbx_center_to_target_box = None
    non_qbx_box_target_lists = None
    center_to_tree_targets = None

    # {{{ Broadcast necessary fields from the root rank to worker ranks

    if mpi_rank == 0:
        global_traversal = global_geometry_data.traversal()
        global_tree = global_traversal.tree

        centers = global_geometry_data.centers()
        expansion_radii = global_geometry_data.expansion_radii()
        global_qbx_centers = global_geometry_data.global_qbx_centers()
        qbx_center_to_target_box = global_geometry_data.qbx_center_to_target_box()
        non_qbx_box_target_lists = global_geometry_data.non_qbx_box_target_lists()
        center_to_tree_targets = global_geometry_data.center_to_tree_targets()

    global_tree = comm.bcast(global_tree, root=0)
    centers = comm.bcast(centers, root=0)
    expansion_radii = comm.bcast(expansion_radii, root=0)
    global_qbx_centers = comm.bcast(global_qbx_centers, root=0)
    qbx_center_to_target_box = comm.bcast(qbx_center_to_target_box, root=0)
    non_qbx_box_target_lists = comm.bcast(non_qbx_box_target_lists, root=0)
    center_to_tree_targets = comm.bcast(center_to_tree_targets, root=0)

    # }}}

    # {{{ Each rank constructs the global traversal object independently

    global_tree_dev = global_tree.to_device(queue).with_queue(queue)
    if mpi_rank != 0:
        global_traversal, _ = traversal_builder(queue, global_tree_dev)

        if global_tree_dev.targets_have_extent:
            global_traversal = global_traversal.merge_close_lists(queue)

        global_traversal = global_traversal.get(queue)

    # }}}

    return GlobalQBXFMMGeometryData(
        global_traversal,
        centers,
        expansion_radii,
        global_qbx_centers,
        qbx_center_to_target_box,
        non_qbx_box_target_lists,
        center_to_tree_targets)


def compute_local_geometry_data(
        actx: PyOpenCLArrayContext, comm, global_geometry_data, boxes_time,
        traversal_builder):
    """Compute the local geometry data of the current rank from the global geometry
    data.

    :arg comm: MPI communicator.
    :arg global_geometry_data: Global geometry data from which the local geometry
        data is generated.
    :arg boxes_time: Predicated cost of each box. Used by partitioning to improve
        load balancing.
    :arg traversal_builder: Used to construct local tranversal.
    """
    queue = actx.queue

    global_traversal = global_geometry_data.global_traversal
    global_tree = global_traversal.tree
    centers = global_geometry_data.centers
    ncenters = len(centers[0])
    expansion_radii = global_geometry_data.expansion_radii
    global_qbx_centers = global_geometry_data.global_qbx_centers
    qbx_center_to_target_box = global_geometry_data.qbx_center_to_target_box
    non_qbx_box_target_lists = global_geometry_data.non_qbx_box_target_lists
    center_to_tree_targets = global_geometry_data.center_to_tree_targets

    # {{{ Generate local tree and local traversal

    from boxtree.distributed.partition import partition_work
    responsible_boxes_list = partition_work(boxes_time, global_traversal, comm)

    from boxtree.distributed.local_tree import generate_local_tree
    local_tree, src_idx, tgt_idx = generate_local_tree(
        queue, global_traversal, responsible_boxes_list, comm)

    src_idx_all_ranks = comm.gather(src_idx, root=0)
    tgt_idx_all_ranks = comm.gather(tgt_idx, root=0)

    from boxtree.distributed.local_traversal import generate_local_travs
    local_trav = generate_local_travs(
        queue, local_tree, traversal_builder,
        # TODO: get whether to merge close lists from root instead of
        # hard-coding?
        merge_close_lists=True).get(queue=queue)

    # }}}

    # {{{ Form non_qbx_box_target_lists

    from boxtree.distributed.local_tree import LocalTreeGeneratorCodeContainer
    code = LocalTreeGeneratorCodeContainer(
        queue.context,
        global_tree.dimensions,
        global_tree.particle_id_dtype,
        global_tree.coord_dtype)

    box_target_starts = cl.array.to_device(
        queue, non_qbx_box_target_lists.box_target_starts)
    box_target_counts_nonchild = cl.array.to_device(
        queue, non_qbx_box_target_lists.box_target_counts_nonchild)
    nfiltered_targets = non_qbx_box_target_lists.nfiltered_targets
    targets = non_qbx_box_target_lists.targets

    particle_mask = cl.array.zeros(
        queue, (nfiltered_targets,), dtype=global_tree.particle_id_dtype)

    responsible_boxes_mask = np.zeros(global_tree.nboxes, dtype=np.int8)
    responsible_boxes_mask[responsible_boxes_list] = 1
    responsible_boxes_mask = cl.array.to_device(queue, responsible_boxes_mask)

    code.particle_mask_kernel()(
        responsible_boxes_mask,
        box_target_starts,
        box_target_counts_nonchild,
        particle_mask)

    particle_scan = cl.array.empty(
        queue, (nfiltered_targets + 1,),
        dtype=global_tree.particle_id_dtype)
    particle_scan[0] = 0
    code.mask_scan_kernel()(particle_mask, particle_scan)

    local_box_target_starts = particle_scan[box_target_starts]

    lobal_box_target_counts_all_zeros = cl.array.zeros(
        queue, (global_tree.nboxes,), dtype=global_tree.particle_id_dtype)

    local_box_target_counts_nonchild = cl.array.if_positive(
        responsible_boxes_mask,
        box_target_counts_nonchild,
        lobal_box_target_counts_all_zeros)

    local_nfiltered_targets = int(particle_scan[-1].get(queue))

    particle_mask = particle_mask.get().astype(bool)
    particle_mask_all_ranks = comm.gather(particle_mask, root=0)
    local_targets = np.empty((global_tree.dimensions,), dtype=object)
    for idimension in range(global_tree.dimensions):
        local_targets[idimension] = targets[idimension][particle_mask]

    from boxtree.tree import FilteredTargetListsInTreeOrder
    non_qbx_box_target_lists = FilteredTargetListsInTreeOrder(
        nfiltered_targets=local_nfiltered_targets,
        box_target_starts=local_box_target_starts.get(),
        box_target_counts_nonchild=local_box_target_counts_nonchild.get(),
        targets=local_targets,
        unfiltered_from_filtered_target_indices=None)

    # }}}

    tgt_mask = np.zeros((global_tree.ntargets,), dtype=bool)
    tgt_mask[tgt_idx] = True

    tgt_mask_user_order = tgt_mask[global_tree.sorted_target_ids]
    centers_mask = tgt_mask_user_order[:ncenters]
    centers_scan = np.empty(
        (ncenters + 1,), dtype=global_tree.particle_id_dtype)
    centers_scan[1:] = np.cumsum(
        centers_mask.astype(global_tree.particle_id_dtype))
    centers_scan[0] = 0

    # {{{ local centers

    nlocal_centers = np.sum(centers_mask.astype(np.int32))
    centers_dims = centers.shape[0]
    local_centers = np.empty(
        (centers_dims, nlocal_centers), dtype=centers[0].dtype)
    for idims in range(centers_dims):
        local_centers[idims, :] = centers[idims][centers_mask]

    # }}}

    # {{{ local global_qbx_centers

    local_global_qbx_centers = centers_scan[
        global_qbx_centers[centers_mask[global_qbx_centers]]]

    # }}}

    # {{{ local expansion_radii

    local_expansion_radii = expansion_radii[centers_mask]

    # }}}

    # {{{ local qbx_center_to_target_box

    # Transform local qbx_center_to_target_box to global indexing
    local_qbx_center_to_target_box = global_traversal.target_boxes[
        qbx_center_to_target_box[centers_mask]]

    # Transform local_qbx_center_to_target_box to local target_boxes indexing
    global_boxes_to_target_boxes = np.ones(
        (global_tree.nboxes,), dtype=local_tree.particle_id_dtype)
    # make sure accessing invalid position raises an error
    global_boxes_to_target_boxes *= -1
    global_boxes_to_target_boxes[local_trav.target_boxes] = \
        np.arange(local_trav.target_boxes.shape[0])
    local_qbx_center_to_target_box = \
        global_boxes_to_target_boxes[local_qbx_center_to_target_box]

    # }}}

    # {{{ local_qbx_targets and local center_to_tree_targets

    starts = center_to_tree_targets.starts
    lists = center_to_tree_targets.lists
    local_starts = np.empty((nlocal_centers + 1,), dtype=starts.dtype)
    local_lists = np.empty(lists.shape, dtype=lists.dtype)

    qbx_target_mask = np.zeros((global_tree.ntargets,), dtype=bool)
    current_start = 0  # index into local_lists
    ilocal_center = 0
    local_starts[0] = 0

    for icenter in range(ncenters):
        # skip the current center if the current rank is not responsible for
        # processing it
        if not centers_mask[icenter]:
            continue

        current_center_targets = lists[starts[icenter]:starts[icenter + 1]]
        qbx_target_mask[current_center_targets] = True
        current_stop = current_start + starts[icenter + 1] - starts[icenter]
        local_starts[ilocal_center + 1] = current_stop
        local_lists[current_start:current_stop] = \
            lists[starts[icenter]:starts[icenter + 1]]

        current_start = current_stop
        ilocal_center += 1

    qbx_target_mask_all_ranks = comm.gather(qbx_target_mask, root=0)

    local_lists = local_lists[:current_start]

    qbx_target_scan = np.empty(
        (global_tree.ntargets + 1,), dtype=lists.dtype
    )
    qbx_target_scan[0] = 0
    qbx_target_scan[1:] = np.cumsum(qbx_target_mask.astype(lists.dtype))
    nlocal_qbx_target = qbx_target_scan[-1]

    local_qbx_targets = np.empty(
        (global_tree.dimensions, nlocal_qbx_target),
        dtype=global_tree.targets[0].dtype
    )
    for idim in range(global_tree.dimensions):
        local_qbx_targets[idim, :] = global_tree.targets[idim][qbx_target_mask]

    local_lists = qbx_target_scan[local_lists]

    from pytential.qbx.geometry import CenterToTargetList
    local_center_to_tree_targets = CenterToTargetList(
        starts=local_starts,
        lists=local_lists)

    # }}}

    # }}}

    # {{{ Construct qbx_center_to_target_box_source_level

    # This is modified from pytential.geometry.QBXFMMGeometryData.
    # qbx_center_to_target_box_source_level but on host using Numpy instead of
    # PyOpenCL.

    tree = local_trav.tree

    qbx_center_to_target_box_source_level = np.empty(
        (tree.nlevels,), dtype=object)

    for source_level in range(tree.nlevels):
        sep_smaller = local_trav.from_sep_smaller_by_level[source_level]

        target_box_to_target_box_source_level = np.empty(
            len(local_trav.target_boxes),
            dtype=tree.box_id_dtype)
        target_box_to_target_box_source_level.fill(-1)
        target_box_to_target_box_source_level[sep_smaller.nonempty_indices] = (
            np.arange(sep_smaller.num_nonempty_lists,
                      dtype=tree.box_id_dtype))

        qbx_center_to_target_box_source_level[source_level] = (
            target_box_to_target_box_source_level[
                local_qbx_center_to_target_box])

    # }}}

    return LocalQBXFMMGeometryData(
            cl_context=queue.context,
            local_tree=local_tree,
            local_trav=local_trav,
            _local_centers=local_centers,
            _global_qbx_centers=local_global_qbx_centers,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            src_idx_all_ranks=src_idx_all_ranks,
            tgt_idx_all_ranks=tgt_idx_all_ranks,
            particle_mask=particle_mask_all_ranks,
            qbx_target_mask=qbx_target_mask_all_ranks,
            _non_qbx_box_target_lists=non_qbx_box_target_lists,
            _local_qbx_center_to_target_box=local_qbx_center_to_target_box,
            _expansion_radii=local_expansion_radii,
            _qbx_targets=local_qbx_targets,
            _local_center_to_tree_targets=local_center_to_tree_targets,
            _qbx_center_to_target_box_source_level=(
                qbx_center_to_target_box_source_level))


class DistributedQBXLayerPotentialSource(QBXLayerPotentialSource):
    def __init__(self, comm, cl_context, *args,
                 _use_target_specific_qbx: Optional[bool] = None,
                 fmm_backend: str = "fmmlib",
                 **kwargs):
        """
        :arg comm: MPI communicator.
        :arg cl_context: This argument is necessary because although the root rank
        can deduce the CL context from density, worker ranks do not have a valid
        density, so we specify there explicitly.

        `*args` and `**kwargs` will be forwarded to
        `QBXLayerPotentialSource.__init__` on the root rank.
        """
        self.comm = comm
        self._cl_context = cl_context

        # "_from_sep_smaller_min_nsources_cumul" can only be 0 for the distributed
        # implementation. If not, the potential contribution of a list 3 box may be
        # computed particle-to-particle instead of using its multipole expansion.
        # However, the source particles may not be distributed to the target rank.
        if "_from_sep_smaller_min_nsources_cumul" not in kwargs:
            kwargs["_from_sep_smaller_min_nsources_cumul"] = 0
        elif kwargs["_from_sep_smaller_min_nsources_cumul"] != 0:
            raise ValueError(
                "_from_sep_smaller_min_nsources_cumul has to be 0 for the "
                "distributed implementation")

        # Only fmmlib is supported
        assert fmm_backend == "fmmlib"

        if self.comm.Get_rank() == 0:
            super().__init__(
                *args,
                _use_target_specific_qbx=_use_target_specific_qbx,
                fmm_backend=fmm_backend,
                **kwargs)
        else:
            self._use_target_specific_qbx = _use_target_specific_qbx
            self.fmm_backend = fmm_backend

            self.qbx_order = kwargs.get("qbx_order", None)

            fmm_order = kwargs.get("fmm_order", None)
            fmm_level_to_order = kwargs.get("fmm_level_to_order", None)

            if fmm_level_to_order is None:
                if fmm_order is False:
                    fmm_level_to_order = False
                else:
                    assert (isinstance(fmm_order, int)
                            and not isinstance(fmm_order, bool))

                    def fmm_level_to_order(kernel, kernel_args, tree, level):  # noqa pylint:disable=function-redefined
                        return fmm_order
            assert (isinstance(fmm_level_to_order, bool)
                    or callable(fmm_level_to_order))

            self.fmm_order = fmm_order
            self.fmm_level_to_order = fmm_level_to_order

        # Broadcast expansion_factory
        expansion_factory = None
        if self.comm.Get_rank() == 0:
            expansion_factory = self.expansion_factory
        expansion_factory = comm.bcast(expansion_factory, root=0)
        self.expansion_factory = expansion_factory

    @property
    def cl_context(self):
        return self._cl_context

    def get_local_fmm_expansion_wrangler_extra_kwargs(
            self, actx, src_idx, target_kernels, tree_user_source_ids, arguments,
            evaluator):
        mpi_rank = self.comm.Get_rank()

        kernel_extra_kwargs = {}
        source_extra_kwargs = {}

        if mpi_rank == 0:
            kernel_extra_kwargs, source_extra_kwargs = \
                self.get_fmm_expansion_wrangler_extra_kwargs(
                    actx, target_kernels, tree_user_source_ids,
                    arguments, evaluator)

        # kernel_extra_kwargs contains information like helmholtz k, which should be
        # picklable and cheap to broadcast
        kernel_extra_kwargs = self.comm.bcast(kernel_extra_kwargs, root=0)

        # Broadcast the keys in `source_extra_kwargs` to worker ranks
        source_arg_names = None
        if mpi_rank == 0:
            source_arg_names = list(source_extra_kwargs.keys())
        source_arg_names = self.comm.bcast(source_arg_names, root=0)

        for arg_name in source_arg_names:
            if arg_name != "dsource_vec":
                raise NotImplementedError

            # Broadcast the global source array to worker ranks
            global_array_host = None
            if mpi_rank == 0:
                global_array_host = actx.to_numpy(source_extra_kwargs[arg_name])
            global_array_host = self.comm.bcast(global_array_host, root=0)

            # Compute the local source array independently on each worker rank
            local_array_host = np.empty_like(global_array_host)
            for idim, global_array_idim in enumerate(global_array_host):
                local_array_host[idim] = global_array_idim[src_idx]

            source_extra_kwargs[arg_name] = actx.from_numpy(local_array_host)

        return kernel_extra_kwargs, source_extra_kwargs

    def exec_compute_potential_insn(self, actx, insn, bound_expr, evaluate,
            return_timing_data):
        extra_args = {}

        # Broadcast whether to use direct evaluation or FMM
        use_direct = True
        if self.comm.Get_rank() == 0:
            use_direct = self.fmm_level_to_order is False
        use_direct = self.comm.bcast(use_direct, root=0)

        if use_direct:
            func = self.exec_compute_potential_insn_direct
            extra_args["return_timing_data"] = return_timing_data
        else:
            func = self.exec_compute_potential_insn_fmm
            extra_args["fmm_driver"] = None

        if self.comm.Get_rank() == 0:
            return self._dispatch_compute_potential_insn(
                    actx, insn, bound_expr, evaluate, func, extra_args)
        else:
            return func(actx, insn, bound_expr, evaluate, **extra_args)

    def exec_compute_potential_insn_direct(self, *args, **kwargs):
        if self.comm.Get_rank() == 0:
            return super().exec_compute_potential_insn_direct(*args, **kwargs)
        else:
            results = []
            timing_data = {}
            return results, timing_data

    def exec_compute_potential_insn_fmm(self, actx: PyOpenCLArrayContext,
            insn, bound_expr, evaluate, fmm_driver):
        """
        :returns: a tuple ``(assignments, extra_outputs)``, where *assignments*
            is a list of tuples containing pairs ``(name, value)`` representing
            assignments to be performed in the evaluation context.
            *extra_outputs* is data that *fmm_driver* may return
            (such as timing data), passed through unmodified.
        """
        from pytential.qbx import get_flat_strengths_from_densities
        from meshmode.discretization import Discretization

        target_name_and_side_to_number = None
        target_discrs_and_qbx_sides = None
        global_geo_data_device = None
        global_geo_data = None
        local_geo_data = None
        boxes_time = None
        output_and_expansion_dtype = None
        flat_strengths = None

        if self.comm.Get_rank() == 0:
            target_name_and_side_to_number, target_discrs_and_qbx_sides = (
                    self.get_target_discrs_and_qbx_sides(insn, bound_expr))

            global_geo_data_device = self.qbx_fmm_geometry_data(
                    bound_expr.places,
                    insn.source.geometry,
                    target_discrs_and_qbx_sides)

            # Use the cost model to estimate execution time for partitioning
            from pytential.qbx.cost import AbstractQBXCostModel, QBXCostModel

            # FIXME: If the expansion wrangler is not FMMLib, the argument
            # 'uses_pde_expansions' might be different
            cost_model = QBXCostModel()

            import warnings
            warnings.warn(
                "Kernel-specific calibration parameters are not supplied when"
                "using distributed FMM.")
            # TODO: supply better default calibration parameters
            calibration_params = AbstractQBXCostModel.get_unit_calibration_params()

            kernel_args = {}
            for arg_name, arg_expr in insn.kernel_arguments.items():
                kernel_args[arg_name] = evaluate(arg_expr)

            boxes_time, _ = cost_model.qbx_cost_per_box(
                actx.queue, global_geo_data_device, insn.target_kernels[0],
                kernel_args, calibration_params)
            boxes_time = boxes_time.get()

            from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
            global_geo_data = ToHostTransferredGeoDataWrapper(global_geo_data_device)

        # FIXME Exert more positive control over geo_data attribute lifetimes using
        # geo_data.<method>.clear_cache(geo_data).

        # FIXME Synthesize "bad centers" around corners and edges that have
        # inadequate QBX coverage.

        # FIXME don't compute *all* output kernels on all targets--respect that
        # some target discretizations may only be asking for derivatives (e.g.)

        # {{{ Construct a traversal builder

        # NOTE: The distributed implementation relies on building the same traversal
        # objects as the one on the root rank. This means here the traversal builder
        # should use the same parameters as `QBXFMMGeometryData.traversal`. To make
        # it consistent across ranks, we broadcast the parameters here.

        trav_param = None
        if self.comm.Get_rank() == 0:
            trav_param = {
                "well_sep_is_n_away":
                    global_geo_data.geo_data.code_getter.build_traversal
                    .well_sep_is_n_away,
                "from_sep_smaller_crit":
                    global_geo_data.geo_data.code_getter.build_traversal.
                    from_sep_smaller_crit,
                "_from_sep_smaller_min_nsources_cumul":
                    global_geo_data.geo_data.lpot_source.
                    _from_sep_smaller_min_nsources_cumul}
        trav_param = self.comm.bcast(trav_param, root=0)

        traversal_builder = QBXFMMGeometryDataTraversalBuilder(
            actx.context,
            well_sep_is_n_away=trav_param["well_sep_is_n_away"],
            from_sep_smaller_crit=trav_param["from_sep_smaller_crit"],
            _from_sep_smaller_min_nsources_cumul=trav_param[
                "_from_sep_smaller_min_nsources_cumul"])

        # }}}

        # {{{ Broadcast the subset of the global geometry data to worker ranks

        global_geo_data = broadcast_global_geometry_data(
            self.comm, actx, traversal_builder, global_geo_data)

        # }}}

        # {{{ Compute the local geometry data from the global geometry data

        if self.comm.Get_rank() != 0:
            boxes_time = np.empty(
                global_geo_data.global_traversal.tree.nboxes, dtype=np.float64)

        self.comm.Bcast(boxes_time, root=0)

        local_geo_data = compute_local_geometry_data(
            actx, self.comm, global_geo_data, boxes_time, traversal_builder)

        # }}}

        tree_indep = self._tree_indep_data_for_wrangler(
                target_kernels=insn.target_kernels,
                source_kernels=insn.source_kernels)

        user_source_ids = None
        if self.comm.Get_rank() == 0:
            user_source_ids = global_geo_data_device.tree().user_source_ids

        kernel_extra_kwargs, source_extra_kwargs = (
                self.get_local_fmm_expansion_wrangler_extra_kwargs(
                    actx, local_geo_data.src_idx,
                    insn.target_kernels + insn.source_kernels,
                    user_source_ids, insn.kernel_arguments, evaluate))

        if self.comm.Get_rank() == 0:
            flat_strengths = get_flat_strengths_from_densities(
                    actx, bound_expr.places, evaluate, insn.densities,
                    dofdesc=insn.source)

            output_and_expansion_dtype = (
                    self.get_fmm_output_and_expansion_dtype(
                        insn.source_kernels, flat_strengths[0]))

        output_and_expansion_dtype = self.comm.bcast(
            output_and_expansion_dtype, root=0)

        wrangler = tree_indep.wrangler_cls(
                        self._cl_context, self.comm, tree_indep,
                        local_geo_data, global_geo_data,
                        output_and_expansion_dtype,
                        self.qbx_order,
                        self.fmm_level_to_order,
                        source_extra_kwargs,
                        kernel_extra_kwargs,
                        _use_target_specific_qbx=self._use_target_specific_qbx)

        if self.comm.Get_rank() == 0:
            from pytential.qbx.geometry import target_state
            if actx.to_numpy(actx.np.any(
                    actx.thaw(global_geo_data_device.user_target_to_center())
                    == target_state.FAILED)):
                raise RuntimeError("geometry has failed targets")

            # {{{ geometry data inspection hook

            if self.geometry_data_inspector is not None:
                perform_fmm = self.geometry_data_inspector(
                    insn, bound_expr, global_geo_data_device)
                if not perform_fmm:
                    return [(o.name, 0) for o in insn.outputs]

            # }}}

        # Execute global QBX.
        timing_data = {}
        all_potentials_on_every_target = drive_dfmm(
            self.comm, flat_strengths, wrangler, timing_data)

        if self.comm.Get_rank() == 0:
            results = []

            for o in insn.outputs:
                target_side_number = target_name_and_side_to_number[
                        o.target_name, o.qbx_forced_limit]
                target_discr, _ = target_discrs_and_qbx_sides[target_side_number]
                target_slice = slice(
                    *global_geo_data_device.target_info().target_discr_starts[
                        target_side_number:target_side_number+2])

                result = all_potentials_on_every_target[
                    o.target_kernel_index][target_slice]

                if isinstance(target_discr, Discretization):
                    template_ary = actx.thaw(target_discr.nodes()[0])
                    result = unflatten(template_ary, result, actx, strict=False)

                results.append((o.name, result))

            return results, timing_data
        else:
            results = []
            return results, timing_data


MPITags = {
    "non_qbx_potentials": 0,
    "qbx_potentials": 1
}


def drive_dfmm(comm, src_weight_vecs, wrangler, timing_data=None):
    # TODO: Integrate the distributed functionality with `qbx.fmm.drive_fmm`,
    # similar to that in `boxtree`.

    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()
    local_traversal = wrangler.traversal

    # {{{ Distribute source weights

    if current_rank == 0:
        template_ary = src_weight_vecs[0]

        src_weight_vecs = [wrangler.reorder_sources(weight)
            for weight in src_weight_vecs]

    src_weight_vecs = wrangler.distribute_source_weights(
        src_weight_vecs, wrangler.geo_data.src_idx_all_ranks)

    # }}}

    recorder = TimingRecorder()

    # {{{ construct local multipoles

    mpole_exps, timing_future = wrangler.form_multipoles(
        local_traversal.level_start_source_box_nrs,
        local_traversal.source_boxes,
        src_weight_vecs)

    recorder.add("form_multipoles", timing_future)

    # }}}

    # {{{ propagate multipoles upward

    mpole_exps, timing_future = wrangler.coarsen_multipoles(
        local_traversal.level_start_source_parent_box_nrs,
        local_traversal.source_parent_boxes,
        mpole_exps)

    recorder.add("coarsen_multipoles", timing_future)

    # }}}

    # {{{ Communicate mpoles

    wrangler.communicate_mpoles(mpole_exps)

    # }}}

    # {{{ direct evaluation from neighbor source boxes ("list 1")

    non_qbx_potentials, timing_future = wrangler.eval_direct(
        local_traversal.target_boxes,
        local_traversal.neighbor_source_boxes_starts,
        local_traversal.neighbor_source_boxes_lists,
        src_weight_vecs)

    recorder.add("eval_direct", timing_future)

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    local_exps, timing_future = wrangler.multipole_to_local(
        local_traversal.level_start_target_or_target_parent_box_nrs,
        local_traversal.target_or_target_parent_boxes,
        local_traversal.from_sep_siblings_starts,
        local_traversal.from_sep_siblings_lists,
        mpole_exps)

    recorder.add("multipole_to_local", timing_future)

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    mpole_result, timing_future = wrangler.eval_multipoles(
        local_traversal.target_boxes_sep_smaller_by_source_level,
        local_traversal.from_sep_smaller_by_level,
        mpole_exps)

    recorder.add("eval_multipoles", timing_future)

    non_qbx_potentials = non_qbx_potentials + mpole_result

    # assert that list 3 close has been merged into list 1
    assert local_traversal.from_sep_close_smaller_starts is None

    # }}}

    # {{{ form locals for separated bigger source boxes ("list 4")

    local_result, timing_future = wrangler.form_locals(
        local_traversal.level_start_target_or_target_parent_box_nrs,
        local_traversal.target_or_target_parent_boxes,
        local_traversal.from_sep_bigger_starts,
        local_traversal.from_sep_bigger_lists,
        src_weight_vecs)

    recorder.add("form_locals", timing_future)

    local_exps = local_exps + local_result

    # assert that list 4 close has been merged into list 1
    assert local_traversal.from_sep_close_bigger_starts is None

    # }}}

    # {{{ propagate local_exps downward

    local_exps, timing_future = wrangler.refine_locals(
        local_traversal.level_start_target_or_target_parent_box_nrs,
        local_traversal.target_or_target_parent_boxes,
        local_exps)

    recorder.add("refine_locals", timing_future)

    # }}}

    # {{{ evaluate locals

    local_result, timing_future = wrangler.eval_locals(
        local_traversal.level_start_target_box_nrs,
        local_traversal.target_boxes,
        local_exps)

    recorder.add("eval_locals", timing_future)

    non_qbx_potentials = non_qbx_potentials + local_result

    # }}}

    # {{{ wrangle qbx expansions

    # form_global_qbx_locals and eval_target_specific_qbx_locals are responsible
    # for the same interactions (directly evaluated portion of the potentials
    # via unified List 1).  Which one is used depends on the wrangler. If one of
    # them is unused the corresponding output entries will be zero.

    qbx_expansions, timing_future = \
        wrangler.form_global_qbx_locals(src_weight_vecs)

    recorder.add("form_global_qbx_locals", timing_future)

    local_result, timing_future = \
        wrangler.translate_box_multipoles_to_qbx_local(mpole_exps)

    recorder.add("translate_box_multipoles_to_qbx_local", timing_future)

    qbx_expansions = qbx_expansions + local_result

    local_result, timing_future = \
        wrangler.translate_box_local_to_qbx_local(local_exps)

    recorder.add("translate_box_local_to_qbx_local", timing_future)

    qbx_expansions = qbx_expansions + local_result

    qbx_potentials, timing_future = wrangler.eval_qbx_expansions(qbx_expansions)

    recorder.add("eval_qbx_expansions", timing_future)

    ts_result, timing_future = \
        wrangler.eval_target_specific_qbx_locals(src_weight_vecs)

    recorder.add("eval_target_specific_qbx_locals", timing_future)

    qbx_potentials = qbx_potentials + ts_result

    # }}}

    if current_rank != 0:  # worker process
        comm.send(non_qbx_potentials, dest=0, tag=MPITags["non_qbx_potentials"])
        comm.send(qbx_potentials, dest=0, tag=MPITags["qbx_potentials"])
        result = None

    else:  # master process

        all_potentials_in_tree_order = wrangler.full_output_zeros(template_ary)

        nqbtl = wrangler.global_geo_data.non_qbx_box_target_lists

        from pytools.obj_array import make_obj_array
        non_qbx_potentials_all_rank = make_obj_array([
            np.zeros(nqbtl.nfiltered_targets, wrangler.tree_indep.dtype)
            for k in wrangler.tree_indep.outputs]
        )

        for irank in range(total_rank):

            if irank == 0:
                non_qbx_potentials_cur_rank = non_qbx_potentials
            else:
                non_qbx_potentials_cur_rank = comm.recv(
                    source=irank, tag=MPITags["non_qbx_potentials"])

            for idim in range(len(wrangler.tree_indep.outputs)):
                non_qbx_potentials_all_rank[idim][
                    wrangler.geo_data.particle_mask[irank]
                ] = non_qbx_potentials_cur_rank[idim]

        for ap_i, nqp_i in zip(
                all_potentials_in_tree_order, non_qbx_potentials_all_rank):
            ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

        for irank in range(total_rank):

            if irank == 0:
                qbx_potentials_cur_rank = qbx_potentials
            else:
                qbx_potentials_cur_rank = comm.recv(
                    source=irank, tag=MPITags["qbx_potentials"]
                )

            for idim in range(len(wrangler.tree_indep.outputs)):
                all_potentials_in_tree_order[idim][
                    wrangler.geo_data.qbx_target_mask[irank]
                ] += qbx_potentials_cur_rank[idim]

        def reorder_and_finalize_potentials(x):
            # "finalize" gives host FMMs (like FMMlib) a chance to turn the
            # potential back into a CL array.
            return wrangler.finalize_potentials(
                x[wrangler.global_traversal.tree.sorted_target_ids], template_ary)

        from pytools.obj_array import with_object_array_or_scalar
        result = with_object_array_or_scalar(
            reorder_and_finalize_potentials, all_potentials_in_tree_order)

    if timing_data is not None:
        timing_data.update(recorder.summarize())

    return result
