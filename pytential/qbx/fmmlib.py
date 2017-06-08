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
from pytools import memoize_method
import pyopencl as cl  # noqa
import pyopencl.array  # noqa: F401
from boxtree.pyfmmlib_integration import HelmholtzExpansionWrangler
from sumpy.kernel import HelmholtzKernel


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

        return QBXFMMLibHelmholtzExpansionWrangler(self, queue, geo_data, dtype,
                qbx_order, fmm_level_to_order,
                source_extra_kwargs,
                kernel_extra_kwargs)

# }}}


# {{{ host geo data wrapper

class ToHostTransferredGeoDataWrapper(object):
    def __init__(self, queue, geo_data):
        self.queue = queue
        self.geo_data = geo_data

    @memoize_method
    def tree(self):
        return self.traversal().tree

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
    def global_qbx_centers(self):
        return self.geo_data.global_qbx_centers().get(queue=self.queue)

    @memoize_method
    def qbx_center_to_target_box(self):
        return self.geo_data.qbx_center_to_target_box().get(queue=self.queue)

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


# {{{ fmmlib expansion wrangler

class QBXFMMLibHelmholtzExpansionWrangler(HelmholtzExpansionWrangler):
    def __init__(self, code, queue, geo_data, dtype,
            qbx_order, fmm_level_to_order,
            source_extra_kwargs,
            kernel_extra_kwargs):

        self.code = code
        self.queue = queue

        # FMMLib is CPU-only. This wrapper gets the geometry out of
        # OpenCL-land.
        self.geo_data = ToHostTransferredGeoDataWrapper(queue, geo_data)

        self.qbx_order = qbx_order

        # {{{ digest out_kernels

        from sumpy.kernel import AxisTargetDerivative

        k_names = []

        def is_supported_helmknl(knl):
            result = isinstance(knl, HelmholtzKernel) and knl.dim == 3
            if result:
                k_names.append(knl.helmholtz_k_name)
            return result

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
                        "only the 3D Helmholtz kernel and its target derivatives "
                        "are supported for now")

        self.outputs = outputs

        # }}}

        from pytools import single_valued
        k_name = single_valued(k_names)
        helmholtz_k = kernel_extra_kwargs[k_name]

        self.level_orders = [
                fmm_level_to_order(level)
                for level in range(self.geo_data.tree().nlevels)]

        # FIXME: For now
        from pytools import single_valued
        assert single_valued(self.level_orders)

        super(QBXFMMLibHelmholtzExpansionWrangler, self).__init__(
                self.geo_data.tree(),

                helmholtz_k=helmholtz_k,

                # FIXME
                nterms=fmm_level_to_order(0),

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
        source_array = source_array.get(queue=self.queue)
        return (super(QBXFMMLibHelmholtzExpansionWrangler, self)
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

    # {{{ qbx-related

    def qbx_local_expansion_zeros(self):
        return np.zeros(
                    (self.geo_data.ncenters,) + self.expansion_shape(self.qbx_order),
                    dtype=self.dtype)

    def form_global_qbx_locals(self, starts, lists, src_weights):
        local_exps = self.qbx_local_expansion_zeros()

        rscale = 1  # FIXME
        geo_data = self.geo_data

        if len(geo_data.global_qbx_centers()) == 0:
            return local_exps

        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        qbx_centers = geo_data.centers()

        formta = self.get_routine("%ddformta")

        for itgt_center, tgt_icenter in enumerate(geo_data.global_qbx_centers()):
            itgt_box = qbx_center_to_target_box[tgt_icenter]

            isrc_box_start = starts[itgt_box]
            isrc_box_stop = starts[itgt_box+1]

            tgt_center = qbx_centers[:, tgt_icenter]

            ctr_coeffs = 0

            for isrc_box in range(isrc_box_start, isrc_box_stop):
                src_ibox = lists[isrc_box]

                src_pslice = self._get_source_slice(src_ibox)

                ier, coeffs = formta(
                        self.helmholtz_k, rscale,
                        self._get_sources(src_pslice), src_weights[src_pslice],
                        tgt_center, self.nterms)
                if ier:
                    raise RuntimeError("formta failed")

                ctr_coeffs += coeffs

            local_exps[tgt_icenter] += ctr_coeffs

        return local_exps

    def translate_box_multipoles_to_qbx_local(self, multipole_exps):
        local_exps = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        qbx_centers = geo_data.centers()
        centers = self.tree.box_centers

        mploc = self.get_translation_routine("%ddmploc")

        for isrc_level, ssn in enumerate(geo_data.traversal().sep_smaller_by_level):
            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(multipole_exps, isrc_level)

            # FIXME
            rscale = 1

            kwargs = {}
            if self.dim == 3:
                # FIXME Is this right?
                kwargs["radius"] = self.tree.root_extent * 2**(-isrc_level)

            for itgt_center, tgt_icenter in enumerate(geo_data.global_qbx_centers()):
                ctr_loc = 0

                icontaining_tgt_box = qbx_center_to_target_box[tgt_icenter]

                tgt_center = qbx_centers[:, tgt_icenter]

                for isrc_box in range(
                        ssn.starts[icontaining_tgt_box],
                        ssn.starts[icontaining_tgt_box+1]):

                    src_ibox = ssn.lists[isrc_box]
                    src_center = centers[:, src_ibox]

                    ctr_loc = ctr_loc + mploc(
                        self.helmholtz_k,
                        rscale, src_center, multipole_exps[src_ibox],
                        rscale, tgt_center, self.nterms, **kwargs)[..., 0]

                local_exps[tgt_icenter] += ctr_loc

        return local_exps

    def translate_box_local_to_qbx_local(self, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros()

        geo_data = self.geo_data
        if geo_data.ncenters == 0:
            return qbx_expansions
        trav = geo_data.traversal()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        qbx_centers = geo_data.centers()

        rscale = 1  # FIXME

        locloc = self.get_translation_routine("%ddlocloc")

        for isrc_level in range(geo_data.tree().nlevels):
            local_order = self.level_orders[isrc_level]

            lev_box_start, lev_box_stop = self.tree.level_start_box_nrs[
                    isrc_level:isrc_level+2]
            target_level_start_ibox, target_locals_view = \
                    self.local_expansions_view(local_exps, isrc_level)
            assert target_level_start_ibox == lev_box_start

            kwargs = {}
            if self.dim == 3:
                # FIXME Is this right?
                kwargs["radius"] = self.tree.root_extent * 2**(-isrc_level)

            for tgt_icenter in range(geo_data.ncenters):
                isrc_box = qbx_center_to_target_box[tgt_icenter]

                tgt_center = qbx_centers[:, tgt_icenter]

                # The box's expansions which we're translating here
                # (our source) is, globally speaking, a target box.

                src_ibox = trav.target_boxes[isrc_box]

                # Is the box number on the level currently under
                # consideration?
                in_range = (lev_box_start <= src_ibox and src_ibox < lev_box_stop)

                if in_range:
                    src_center = self.tree.box_centers[:, src_ibox]
                    tmp_loc_exp = locloc(
                                self.helmholtz_k,
                                rscale, src_center, local_exps[src_ibox],
                                rscale, tgt_center, local_order, **kwargs)[..., 0]

                    qbx_expansions[tgt_icenter] += tmp_loc_exp

        return qbx_expansions

    def eval_qbx_expansions(self, qbx_expansions):
        output = self.full_output_zeros()

        geo_data = self.geo_data
        ctt = geo_data.center_to_tree_targets()
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_centers = geo_data.centers()

        all_targets = geo_data.all_targets()

        rscale = 1  # FIXME

        taeval = self.get_expn_eval_routine("ta")

        for isrc_center, src_icenter in enumerate(global_qbx_centers):
            for icenter_tgt in range(
                    ctt.starts[src_icenter],
                    ctt.starts[src_icenter+1]):

                center_itgt = ctt.lists[icenter_tgt]

                center = qbx_centers[:, src_icenter]

                pot, grad = taeval(self.helmholtz_k, rscale,
                        center, qbx_expansions[src_icenter],
                        all_targets[:, center_itgt])

                self.add_potgrad_onto_output(output, center_itgt, pot, grad)

        return output

    def finalize_potential(self, potential):
        if self.dim == 3:
            scale_factor = 1/(4*np.pi)
        else:
            raise NotImplementedError(
                    "scale factor for pyfmmlib for %d dimensions" % self.dim)

        return cl.array.to_device(self.queue, potential) * scale_factor

    # }}}

# }}}

# vim: foldmethod=marker
