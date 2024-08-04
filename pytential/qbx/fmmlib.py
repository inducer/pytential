__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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

from pytools import memoize_method
import pyopencl as cl
import pyopencl.array

from boxtree.pyfmmlib_integration import (
        Kernel,
        FMMLibTreeIndependentDataForWrangler,
        FMMLibExpansionWrangler)
from boxtree.distributed.calculation import DistributedFMMLibExpansionWrangler
from sumpy.kernel import (
        LaplaceKernel, HelmholtzKernel, AxisTargetDerivative,
        DirectionalSourceDerivative)
import pytential.qbx.target_specific as ts


from boxtree.timing import return_timing_data
from pytools import log_process

import logging
logger = logging.getLogger(__name__)


class QBXFMMLibTreeIndependentDataForWrangler(FMMLibTreeIndependentDataForWrangler):
    def __init__(self, cl_context, *,
            multipole_expansion_factory, local_expansion_factory,
            qbx_local_expansion_factory, target_kernels,
            _use_target_specific_qbx):
        self.cl_context = cl_context
        self.multipole_expansion_factory = multipole_expansion_factory
        self.local_expansion_factory = local_expansion_factory
        self.qbx_local_expansion_factory = qbx_local_expansion_factory

        kernel = target_kernels[0].get_base_kernel()
        self.target_kernels = target_kernels

        # {{{ digest target_kernels

        ifgrad = False
        outputs = []
        source_deriv_names = []
        k_names = []

        using_tsqbx = (
                _use_target_specific_qbx
                # None means use by default if possible
                or _use_target_specific_qbx is None)

        for out_knl in target_kernels:
            if not self.is_supported_helmknl_for_tsqbx(out_knl):
                if _use_target_specific_qbx:
                    raise ValueError("not all kernels passed support TSQBX")
                using_tsqbx = False

            if self.is_supported_helmknl(out_knl):
                outputs.append(())
                no_target_deriv_knl = out_knl

            elif (isinstance(out_knl, AxisTargetDerivative)
                    and self.is_supported_helmknl(out_knl.inner_kernel)):
                outputs.append((out_knl.axis,))
                ifgrad = True
                no_target_deriv_knl = out_knl.inner_kernel

            else:
                raise ValueError(
                        "only the 2/3D Laplace and Helmholtz kernel "
                        "and their derivatives are supported")

            source_deriv_names.append(no_target_deriv_knl.dir_vec_name
                    if isinstance(no_target_deriv_knl, DirectionalSourceDerivative)
                    else None)

            base_knl = out_knl.get_base_kernel()
            k_names.append(base_knl.helmholtz_k_name
                    if isinstance(base_knl, HelmholtzKernel)
                    else None)

        self.using_tsqbx = using_tsqbx
        self.outputs = outputs

        from pytools import is_single_valued

        if not is_single_valued(source_deriv_names):
            raise ValueError("not all kernels passed are the same in "
                    "whether they represent a source derivative")

        self.source_deriv_name = source_deriv_names[0]

        if not is_single_valued(k_names):
            raise ValueError("not all kernels passed have the same "
                    "Helmholtz parameter")

        self.k_name = k_names[0]

        # }}}

        super().__init__(kernel.dim, {
            LaplaceKernel: Kernel.LAPLACE,
            HelmholtzKernel: Kernel.HELMHOLTZ,
            }[type(kernel)],
            ifgrad=ifgrad)

    @staticmethod
    def is_supported_helmknl(knl):
        if isinstance(knl, DirectionalSourceDerivative):
            knl = knl.inner_kernel

        return (isinstance(knl, (LaplaceKernel, HelmholtzKernel))
                and knl.dim in (2, 3))

    @staticmethod
    def is_supported_helmknl_for_tsqbx(knl):
        # Supports at most one derivative.
        if isinstance(knl, (DirectionalSourceDerivative, AxisTargetDerivative)):
            knl = knl.inner_kernel

        return (isinstance(knl, (LaplaceKernel, HelmholtzKernel))
                and knl.dim == 3)

    @property
    def wrangler_cls(self):
        return QBXFMMLibExpansionWrangler

# }}}


# {{{ fmmlib expansion wrangler

def boxtree_fmm_level_to_order(fmm_level_to_order, helmholtz_k):
    def inner_fmm_level_to_order(tree, level):
        if helmholtz_k == 0:
            return fmm_level_to_order(
                    LaplaceKernel(tree.dimensions),
                    frozenset(), tree, level)
        else:
            return fmm_level_to_order(
                    HelmholtzKernel(tree.dimensions),
                    frozenset([("k", helmholtz_k)]), tree, level)

    return inner_fmm_level_to_order


class QBXFMMLibExpansionWrangler(FMMLibExpansionWrangler):
    def __init__(self, tree_indep, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs,
            kernel_extra_kwargs,
            _use_target_specific_qbx=None):
        # FMMLib is CPU-only. This wrapper gets the geometry out of
        # OpenCL-land.
        if hasattr(geo_data, "_setup_actx"):
            from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
            geo_data = ToHostTransferredGeoDataWrapper(geo_data)

        self.geo_data = geo_data
        self.qbx_order = qbx_order

        if tree_indep.k_name is None:
            helmholtz_k = 0
        else:
            helmholtz_k = kernel_extra_kwargs[tree_indep.k_name]

        dipole_vec = None
        if tree_indep.source_deriv_name is not None:
            with cl.CommandQueue(tree_indep.cl_context) as queue:
                dipole_vec = np.array([
                        d_i.get(queue=queue)
                        for d_i in source_extra_kwargs[
                            tree_indep.source_deriv_name]],
                        order="F")

        FMMLibExpansionWrangler.__init__(
                self,
                tree_indep,
                geo_data.traversal(),

                helmholtz_k=helmholtz_k,
                dipole_vec=dipole_vec,
                dipoles_already_reordered=True,

                fmm_level_to_order=boxtree_fmm_level_to_order(
                    fmm_level_to_order, helmholtz_k),
                rotation_data=geo_data)

    # {{{ data vector helpers

    def output_zeros(self):
        """This ought to be called ``non_qbx_output_zeros``, but since
        it has to override the superclass's behavior to integrate seamlessly,
        it needs to be called just :meth:`output_zeros`.
        """

        nqbtl = self.geo_data.non_qbx_box_target_lists()

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                np.zeros(nqbtl.nfiltered_targets, self.tree_indep.dtype)
                for k in self.tree_indep.outputs])

    def full_output_zeros(self, template_ary):
        """This includes QBX and non-QBX targets."""

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                np.zeros(self.tree.ntargets, self.tree_indep.dtype)
                for k in self.tree_indep.outputs])

    def eval_qbx_output_zeros(self, template_ary):
        return self.full_output_zeros(template_ary)

    def reorder_sources(self, source_array):
        if isinstance(source_array, cl.array.Array):
            source_array = source_array.get()

        return super().reorder_sources(source_array)

    def reorder_potentials(self, potentials):
        raise NotImplementedError("reorder_potentials should not "
            "be called on a QBXFMMLibExpansionWrangler")

        # Because this is a multi-stage, more complicated process that combines
        # potentials from non-QBX targets and QBX targets.

    def add_potgrad_onto_output(self, output, output_slice, pot, grad):
        for i_out, out in enumerate(self.tree_indep.outputs):
            if len(out) == 0:
                output[i_out][output_slice] += pot.squeeze()
            elif len(out) == 1:
                axis, = out
                if isinstance(grad, np.ndarray):
                    output[i_out][output_slice] += grad[axis].squeeze()
                else:
                    assert grad == 0
            else:
                raise ValueError("element '%s' of outputs array not "
                        "understood" % out)

    @memoize_method
    def _get_single_centers_array(self):
        return np.array([
            self.geo_data.centers()[idim]
            for idim in range(self.dim)
            ], order="F")

    # }}}

    # {{{ override target lists to only hit non-QBX targets

    def box_target_starts(self):
        nqbtl = self.geo_data.non_qbx_box_target_lists()
        return nqbtl.box_target_starts

    def box_target_counts_nonchild(self):
        nqbtl = self.geo_data.non_qbx_box_target_lists()
        return nqbtl.box_target_counts_nonchild

    def targets(self):
        nqbtl = self.geo_data.non_qbx_box_target_lists()
        return nqbtl.targets

    # }}}

    def qbx_local_expansion_zeros(self):
        return np.zeros(
                    (self.geo_data.ncenters, *self.expansion_shape(self.qbx_order)),
                    dtype=self.tree_indep.dtype)

    # {{{ p2qbxl

    @log_process(logger)
    @return_timing_data
    def form_global_qbx_locals(self, src_weight_vecs):
        src_weights, = src_weight_vecs
        if self.tree_indep.using_tsqbx:
            return self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        trav = geo_data.traversal()

        if len(geo_data.global_qbx_centers()) == 0:
            return self.qbx_local_expansion_zeros()

        formta_qbx = self.tree_indep.get_routine(
                "%ddformta" + ("_dp" if self.dipole_vec is not None else ""),
                suffix="_qbx")

        kwargs = {}
        kwargs.update(self.kernel_kwargs)

        if self.dipole_vec is None:
            kwargs["charge"] = src_weights

        else:
            if self.dim == 2 and self.tree_indep.eqn_letter == "l":
                kwargs["dipstr"] = -src_weights * (
                        self.dipole_vec[0] + 1j*self.dipole_vec[1])
            else:
                kwargs["dipstr"] = src_weights
                kwargs["dipvec"] = self.dipole_vec

        ier, qbx_exps = formta_qbx(
                sources=self._get_single_sources_array(),
                qbx_centers=geo_data.centers().T,
                global_qbx_centers=geo_data.global_qbx_centers(),
                qbx_expansion_radii=geo_data.expansion_radii(),
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                nterms=self.qbx_order,
                source_box_starts=trav.neighbor_source_boxes_starts,
                source_box_lists=trav.neighbor_source_boxes_lists,
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=self.tree.box_source_counts_nonchild,
                **kwargs)
        qbx_exps = qbx_exps.T

        if np.any(ier != 0):
            raise RuntimeError("formta for p2qbxl returned an error (ier=%d)" % ier)

        qbx_exps_2 = self.qbx_local_expansion_zeros()
        assert qbx_exps.shape == qbx_exps_2.shape

        return qbx_exps

    # }}}

    # {{{ m2qbxl

    @log_process(logger)
    @return_timing_data
    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        qbx_exps = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        qbx_centers = geo_data.centers()
        centers = self.tree.box_centers
        ngqbx_centers = len(geo_data.global_qbx_centers())
        traversal = geo_data.traversal()

        if ngqbx_centers == 0:
            return qbx_exps

        mploc = self.tree_indep.get_translation_routine(
                self, "%ddmploc", vec_suffix="_imany")

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):
            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(multipole_exps, isrc_level)

            tgt_icenter_vec = geo_data.global_qbx_centers()
            qbx_center_to_target_box_source_level = (
                geo_data.qbx_center_to_target_box_source_level(isrc_level)
            )
            icontaining_tgt_box_vec = qbx_center_to_target_box_source_level[
                tgt_icenter_vec
            ]

            rscale2 = geo_data.expansion_radii()[geo_data.global_qbx_centers()]

            kwargs = {}
            if self.dim == 3 and self.tree_indep.eqn_letter == "h":
                kwargs["radius"] = (0.5
                        * geo_data.expansion_radii()[geo_data.global_qbx_centers()])

            nsrc_boxes_per_gqbx_center = np.zeros(icontaining_tgt_box_vec.shape,
                                                  dtype=traversal.tree.box_id_dtype)
            mask = (icontaining_tgt_box_vec != -1)
            nsrc_boxes_per_gqbx_center[mask] = (
                ssn.starts[icontaining_tgt_box_vec[mask] + 1]
                - ssn.starts[icontaining_tgt_box_vec[mask]]
            )
            nsrc_boxes = np.sum(nsrc_boxes_per_gqbx_center)

            src_boxes_starts = np.empty(ngqbx_centers+1, dtype=np.int32)
            src_boxes_starts[0] = 0
            src_boxes_starts[1:] = np.cumsum(nsrc_boxes_per_gqbx_center)

            rscale1 = np.ones(nsrc_boxes) * self.level_to_rscale(isrc_level)
            rscale1_offsets = np.arange(nsrc_boxes)

            src_ibox = np.empty(nsrc_boxes, dtype=np.int32)
            for itgt_center, tgt_icenter in enumerate(
                    geo_data.global_qbx_centers()):
                icontaining_tgt_box = qbx_center_to_target_box_source_level[
                    tgt_icenter
                ]
                src_ibox[
                        src_boxes_starts[itgt_center]:
                        src_boxes_starts[itgt_center+1]] = (
                    ssn.lists[
                        ssn.starts[icontaining_tgt_box]:
                        ssn.starts[icontaining_tgt_box+1]])

            del itgt_center
            del tgt_icenter
            del icontaining_tgt_box

            if self.dim == 3:
                # This gets max'd onto: pass initialized version.
                ier = np.zeros(ngqbx_centers, dtype=np.int32)
                kwargs["ier"] = ier

            # This gets added onto: pass initialized version.
            expn2 = np.zeros(
                    (ngqbx_centers, *self.expansion_shape(self.qbx_order)),
                    dtype=self.tree_indep.dtype)

            kwargs.update(self.kernel_kwargs)

            expn2 = mploc(
                    rscale1=rscale1,
                    rscale1_offsets=rscale1_offsets,
                    rscale1_starts=src_boxes_starts,

                    center1=centers,
                    center1_offsets=src_ibox,
                    center1_starts=src_boxes_starts,

                    expn1=source_mpoles_view.T,
                    expn1_offsets=src_ibox - source_level_start_ibox,
                    expn1_starts=src_boxes_starts,

                    rscale2=rscale2,
                    # FIXME: center2 has wrong layout, will copy
                    center2=qbx_centers[:, tgt_icenter_vec],
                    expn2=expn2.T,

                    nterms2=self.qbx_order,

                    **kwargs).T

            if self.dim == 3:
                if ier.any():
                    raise RuntimeError("m2qbxl failed")

            qbx_exps[geo_data.global_qbx_centers()] += expn2

        return qbx_exps

    # }}}

    @log_process(logger)
    @return_timing_data
    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        global_qbx_centers = geo_data.global_qbx_centers()

        if global_qbx_centers.size == 0:
            return qbx_expansions

        trav = geo_data.traversal()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        qbx_centers = geo_data.centers()
        qbx_radii = geo_data.expansion_radii()

        is_global_qbx_center = np.zeros(geo_data.ncenters, dtype=int)
        is_global_qbx_center[global_qbx_centers] = 1

        locloc = self.tree_indep.get_translation_routine(
                self, "%ddlocloc", vec_suffix="_qbx")

        nlevels = geo_data.tree().nlevels

        box_to_rscale = np.empty(geo_data.tree().nboxes, dtype=np.float64)
        for isrc_level in range(nlevels):
            lev_box_start, lev_box_stop = self.tree.level_start_box_nrs[
                    isrc_level:isrc_level+2]
            box_to_rscale[lev_box_start:lev_box_stop] = (
                    self.level_to_rscale(isrc_level))

        box_centers = self._get_single_box_centers_array()

        # This translates from target box to global box numbers.
        qbx_center_to_box = trav.target_boxes[qbx_center_to_target_box]

        kwargs = {}
        kwargs.update(self.kernel_kwargs)

        for isrc_level in range(nlevels):
            lev_box_start, lev_box_stop = self.tree.level_start_box_nrs[
                    isrc_level:isrc_level+2]

            locals_level_start_ibox, locals_view = \
                    self.local_expansions_view(local_exps, isrc_level)

            assert locals_level_start_ibox == lev_box_start

            # Find used QBX centers that are on this level. (This is not ideal,
            # but we're supplied a mapping of QBX centers to boxes and we have
            # to invert that in some way.)
            curr_level_qbx_centers = np.flatnonzero(
                    is_global_qbx_center
                    & (lev_box_start <= qbx_center_to_box)
                    & (qbx_center_to_box < lev_box_stop))

            if curr_level_qbx_centers.size == 0:
                continue

            icurr_level_qbx_center_to_box = (
                    qbx_center_to_box[curr_level_qbx_centers])

            if self.dim == 3 and self.tree_indep.eqn_letter == "h":
                kwargs["radius"] = 0.5 * (
                        geo_data.expansion_radii()[curr_level_qbx_centers])

            # This returns either the expansion or a tuple (ier, expn).
            rvals = locloc(
                    rscale1=box_to_rscale,
                    rscale1_offsets=icurr_level_qbx_center_to_box,
                    center1=box_centers,
                    center1_offsets=icurr_level_qbx_center_to_box,
                    expn1=locals_view.T,
                    expn1_offsets=icurr_level_qbx_center_to_box - lev_box_start,
                    nterms1=self.level_orders[isrc_level],
                    nterms2=self.qbx_order,
                    rscale2=qbx_radii,
                    rscale2_offsets=curr_level_qbx_centers,
                    center2=qbx_centers,
                    center2_offsets=curr_level_qbx_centers,
                    **kwargs)

            if isinstance(rvals, tuple):
                ier, expn2 = rvals
                if ier.any():
                    raise RuntimeError("locloc failed")
            else:
                expn2 = rvals

            qbx_expansions[curr_level_qbx_centers] += expn2.T

        return qbx_expansions

    @log_process(logger)
    @return_timing_data
    def eval_qbx_expansions(self, qbx_expansions):
        output = self.eval_qbx_output_zeros(template_ary=qbx_expansions)

        geo_data = self.geo_data
        ctt = geo_data.center_to_tree_targets()
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_centers = geo_data.centers()
        qbx_radii = geo_data.expansion_radii()

        all_targets = geo_data.eval_qbx_targets()

        taeval = self.tree_indep.get_expn_eval_routine("ta")

        for src_icenter in global_qbx_centers:
            for icenter_tgt in range(
                    ctt.starts[src_icenter],
                    ctt.starts[src_icenter+1]):

                center_itgt = ctt.lists[icenter_tgt]

                center = qbx_centers[:, src_icenter]

                pot, grad = taeval(
                        rscale=qbx_radii[src_icenter],
                        center=center,
                        expn=qbx_expansions[src_icenter].T,
                        ztarg=all_targets[:, center_itgt],
                        **self.kernel_kwargs)

                self.add_potgrad_onto_output(output, center_itgt, pot, grad)

        return output

    @log_process(logger)
    @return_timing_data
    def eval_target_specific_qbx_locals(self, src_weight_vecs):
        src_weights, = src_weight_vecs
        output = self.eval_qbx_output_zeros(template_ary=src_weights)
        noutput_targets = len(output[0])

        if not self.tree_indep.using_tsqbx:
            return output

        geo_data = self.geo_data
        trav = geo_data.traversal()

        ctt = geo_data.center_to_tree_targets()

        src_weights = src_weights.astype(np.complex128)

        ifcharge = self.dipole_vec is None
        ifdipole = self.dipole_vec is not None

        ifpot = any(not output for output in self.tree_indep.outputs)
        ifgrad = self.tree_indep.ifgrad

        # Create temporary output arrays for potential / gradient.
        pot = np.zeros(noutput_targets, np.complex128) if ifpot else None
        grad = (
                np.zeros((self.dim, noutput_targets), np.complex128)
                if ifgrad else None)

        ts.eval_target_specific_qbx_locals(
                ifpot=ifpot,
                ifgrad=ifgrad,
                ifcharge=ifcharge,
                ifdipole=ifdipole,
                order=self.qbx_order,
                sources=self._get_single_sources_array(),
                targets=geo_data.eval_qbx_targets(),
                centers=self._get_single_centers_array(),
                qbx_centers=geo_data.global_qbx_centers(),
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                center_to_target_starts=ctt.starts,
                center_to_target_lists=ctt.lists,
                source_box_starts=trav.neighbor_source_boxes_starts,
                source_box_lists=trav.neighbor_source_boxes_lists,
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=self.tree.box_source_counts_nonchild,
                helmholtz_k=self.kernel_kwargs.get("zk", 0),
                charge=src_weights,
                dipstr=src_weights,
                dipvec=self.dipole_vec,
                pot=pot,
                grad=grad)

        self.add_potgrad_onto_output(output, slice(None), pot, grad)

        return output

    def finalize_potentials(self, potential, template_ary):
        potential = super().finalize_potentials(potential, template_ary)
        return cl.array.to_device(template_ary.queue, potential)

# }}}


class DistributedQBXFMMLibExpansionWrangler(
        QBXFMMLibExpansionWrangler, DistributedFMMLibExpansionWrangler):
    MPITags = {
        "non_qbx_potentials": 0,
        "qbx_potentials": 1
    }

    def __init__(
            self, context, comm, tree_indep, local_geo_data, global_geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs,
            kernel_extra_kwargs,
            _use_target_specific_qbx=None,
            communicate_mpoles_via_allreduce=False):
        self.global_geo_data = global_geo_data

        QBXFMMLibExpansionWrangler.__init__(
            self, tree_indep, local_geo_data, dtype, qbx_order, fmm_level_to_order,
            source_extra_kwargs, kernel_extra_kwargs,
            _use_target_specific_qbx=_use_target_specific_qbx)

        # This is blatantly copied from QBXFMMLibExpansionWrangler, is it worthwhile
        # to refactor this?
        if tree_indep.k_name is None:
            helmholtz_k = 0
        else:
            helmholtz_k = kernel_extra_kwargs[tree_indep.k_name]

        DistributedFMMLibExpansionWrangler.__init__(
            self, context, comm, tree_indep,
            local_geo_data.local_trav, global_geo_data.global_traversal,
            boxtree_fmm_level_to_order(fmm_level_to_order, helmholtz_k),
            communicate_mpoles_via_allreduce=communicate_mpoles_via_allreduce)

    def reorder_sources(self, source_array):
        if self.comm.Get_rank() == 0:
            return super().reorder_sources(source_array)
        else:
            return None

    def eval_qbx_output_zeros(self, template_ary):
        from pytools.obj_array import make_obj_array
        ctt = self.geo_data.center_to_tree_targets()
        output = make_obj_array([np.zeros(len(ctt.lists), self.tree_indep.dtype)
                                 for k in self.tree_indep.outputs])
        return output

    def full_output_zeros(self, template_ary):
        """This includes QBX and non-QBX targets."""

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                np.zeros(self.global_traversal.tree.ntargets, self.tree_indep.dtype)
                for k in self.tree_indep.outputs])

    def _gather_tgt_potentials(self, ntargets, potentials, mask, mpi_tag):
        mpi_rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()

        if mpi_rank == 0:
            from pytools.obj_array import make_obj_array
            potentials_all_rank = make_obj_array([
                np.zeros(ntargets, self.tree_indep.dtype)
                for k in self.tree_indep.outputs])

            for irank in range(mpi_size):
                if irank == 0:
                    potentials_cur_rank = potentials
                else:
                    potentials_cur_rank = self.comm.recv(source=irank, tag=mpi_tag)

                for idim in range(len(self.tree_indep.outputs)):
                    potentials_all_rank[idim][mask[irank]] = \
                        potentials_cur_rank[idim]

            return potentials_all_rank
        else:
            self.comm.send(potentials, dest=0, tag=mpi_tag)
            return None

    def gather_non_qbx_potentials(self, non_qbx_potentials):
        ntargets = 0
        if self.comm.Get_rank() == 0:
            nqbtl = self.global_geo_data.non_qbx_box_target_lists
            ntargets = nqbtl.nfiltered_targets

        return self._gather_tgt_potentials(
            ntargets, non_qbx_potentials,
            self.geo_data.particle_mask, self.MPITags["non_qbx_potentials"])

    def gather_qbx_potentials(self, qbx_potentials):
        ntargets = 0
        if self.comm.Get_rank() == 0:
            ntargets = self.global_traversal.tree.ntargets

        return self._gather_tgt_potentials(
            ntargets, qbx_potentials,
            self.geo_data.qbx_target_mask, self.MPITags["qbx_potentials"])

# vim: foldmethod=marker
