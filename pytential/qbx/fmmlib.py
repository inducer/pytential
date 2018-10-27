from __future__ import division, absolute_import

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
from pytools import Record
import pyopencl as cl  # noqa
import pyopencl.array  # noqa: F401
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
from sumpy.kernel import LaplaceKernel, HelmholtzKernel


from boxtree.tools import return_timing_data
from pytools import log_process

import logging
logger = logging.getLogger(__name__)


class P2QBXLInfo(Record):
    pass


class QBXFMMLibExpansionWranglerCodeContainer(object):
    def __init__(self, cl_context,
            multipole_expansion_factory, local_expansion_factory,
            qbx_local_expansion_factory, out_kernels):
        self.cl_context = cl_context
        self.multipole_expansion_factory = multipole_expansion_factory
        self.local_expansion_factory = local_expansion_factory
        self.qbx_local_expansion_factory = qbx_local_expansion_factory

        self.out_kernels = out_kernels

    def get_wrangler(self, queue, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs={},
            kernel_extra_kwargs=None):

        return QBXFMMLibExpansionWrangler(self, queue, geo_data, dtype,
                qbx_order, fmm_level_to_order,
                source_extra_kwargs,
                kernel_extra_kwargs)

# }}}


# {{{ fmmlib expansion wrangler

class QBXFMMLibExpansionWrangler(FMMLibExpansionWrangler):
    def __init__(self, code, queue, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs,
            kernel_extra_kwargs):

        self.code = code
        self.queue = queue

        # FMMLib is CPU-only. This wrapper gets the geometry out of
        # OpenCL-land.
        from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
        self.geo_data = ToHostTransferredGeoDataWrapper(queue, geo_data)

        self.qbx_order = qbx_order

        # {{{ digest out_kernels

        from sumpy.kernel import AxisTargetDerivative, DirectionalSourceDerivative

        k_names = []
        source_deriv_names = []

        def is_supported_helmknl(knl):
            if isinstance(knl, DirectionalSourceDerivative):
                source_deriv_name = knl.dir_vec_name
                knl = knl.inner_kernel
            else:
                source_deriv_name = None

            if isinstance(knl, HelmholtzKernel) and knl.dim in [2, 3]:
                k_names.append(knl.helmholtz_k_name)
                source_deriv_names.append(source_deriv_name)
                return True
            elif isinstance(knl, LaplaceKernel) and knl.dim in [2, 3]:
                k_names.append(None)
                source_deriv_names.append(source_deriv_name)
                return True

            return False

        ifgrad = False
        outputs = []
        for out_knl in self.code.out_kernels:
            if is_supported_helmknl(out_knl):
                outputs.append(())
            elif (isinstance(out_knl, AxisTargetDerivative)
                    and is_supported_helmknl(out_knl.inner_kernel)):
                outputs.append((out_knl.axis,))
                ifgrad = True
            else:
                raise NotImplementedError(
                        "only the 2/3D Laplace and Helmholtz kernel "
                        "and their derivatives are supported")

        from pytools import is_single_valued
        if not is_single_valued(source_deriv_names):
            raise ValueError("not all kernels passed are the same in "
                    "whether they represent a source derivative")

        source_deriv_name = source_deriv_names[0]
        self.outputs = outputs

        # }}}

        from pytools import single_valued
        k_name = single_valued(k_names)
        if k_name is None:
            helmholtz_k = 0
        else:
            helmholtz_k = kernel_extra_kwargs[k_name]

        dipole_vec = None
        if source_deriv_name is not None:
            dipole_vec = np.array([
                    d_i.get(queue=queue)
                    for d_i in source_extra_kwargs[source_deriv_name]],
                    order="F")

        def inner_fmm_level_to_nterms(tree, level):
            from sumpy.kernel import LaplaceKernel, HelmholtzKernel
            if helmholtz_k == 0:
                return fmm_level_to_order(
                        LaplaceKernel(tree.dimensions),
                        frozenset(), tree, level)
            else:
                return fmm_level_to_order(
                        HelmholtzKernel(tree.dimensions),
                        frozenset([("k", helmholtz_k)]), tree, level)

        super(QBXFMMLibExpansionWrangler, self).__init__(
                self.geo_data.tree(),

                helmholtz_k=helmholtz_k,
                dipole_vec=dipole_vec,
                dipoles_already_reordered=True,

                fmm_level_to_nterms=inner_fmm_level_to_nterms,

                ifgrad=ifgrad)

    # {{{ data vector helpers

    def output_zeros(self):
        """This ought to be called ``non_qbx_output_zeros``, but since
        it has to override the superclass's behavior to integrate seamlessly,
        it needs to be called just :meth:`output_zeros`.
        """

        nqbtl = self.geo_data.non_qbx_box_target_lists()

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                np.zeros(nqbtl.nfiltered_targets, self.dtype)
                for k in self.outputs])

    def full_output_zeros(self):
        """This includes QBX and non-QBX targets."""

        from pytools.obj_array import make_obj_array
        return make_obj_array([
                np.zeros(self.tree.ntargets, self.dtype)
                for k in self.outputs])

    def reorder_sources(self, source_array):
        if isinstance(source_array, cl.array.Array):
            source_array = source_array.get(queue=self.queue)

        return (super(QBXFMMLibExpansionWrangler, self)
                .reorder_sources(source_array))

    def reorder_potentials(self, potentials):
        raise NotImplementedError("reorder_potentials should not "
            "be called on a QBXFMMLibHelmholtzExpansionWrangler")

        # Because this is a multi-stage, more complicated process that combines
        # potentials from non-QBX targets and QBX targets.

    def add_potgrad_onto_output(self, output, output_slice, pot, grad):
        for i_out, out in enumerate(self.outputs):
            if len(out) == 0:
                output[i_out][output_slice] += pot
            elif len(out) == 1:
                axis, = out
                if isinstance(grad, np.ndarray):
                    output[i_out][output_slice] += grad[axis]
                else:
                    assert grad == 0
            else:
                raise ValueError("element '%s' of outputs array not "
                        "understood" % out)

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
                    (self.geo_data.ncenters,) + self.expansion_shape(self.qbx_order),
                    dtype=self.dtype)

    # {{{ p2qbxl

    @log_process(logger)
    @return_timing_data
    def form_global_qbx_locals(self, src_weights):
        geo_data = self.geo_data
        trav = geo_data.traversal()

        if len(geo_data.global_qbx_centers()) == 0:
            return self.qbx_local_expansion_zeros()

        formta_qbx = self.get_routine("%ddformta" + self.dp_suffix,
                suffix="_qbx")

        kwargs = {}
        kwargs.update(self.kernel_kwargs)

        if self.dipole_vec is None:
            kwargs["charge"] = src_weights

        else:
            if self.dim == 2 and self.eqn_letter == "l":
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

        mploc = self.get_translation_routine("%ddmploc", vec_suffix="_imany")

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
            if self.dim == 3 and self.eqn_letter == "h":
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
                    (ngqbx_centers,) + self.expansion_shape(self.qbx_order),
                    dtype=self.dtype)

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

        locloc = self.get_translation_routine("%ddlocloc", vec_suffix="_qbx")

        nlevels = geo_data.tree().nlevels

        box_to_rscale = np.empty(geo_data.tree().nboxes, dtype=np.float)
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

            if self.dim == 3 and self.eqn_letter == "h":
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
                    nterms1=self.level_nterms[isrc_level],
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
        output = self.full_output_zeros()

        geo_data = self.geo_data
        ctt = geo_data.center_to_tree_targets()
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_centers = geo_data.centers()
        qbx_radii = geo_data.expansion_radii()

        all_targets = geo_data.all_targets()

        taeval = self.get_expn_eval_routine("ta")

        for isrc_center, src_icenter in enumerate(global_qbx_centers):
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

    def finalize_potentials(self, potential):
        potential = super(QBXFMMLibExpansionWrangler, self).finalize_potentials(
                potential)

        return cl.array.to_device(self.queue, potential)

# }}}

# vim: foldmethod=marker
