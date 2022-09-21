from __future__ import annotations


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

import loopy as lp
from pytools import memoize_method, obj_array
from sumpy.e2e import E2EBase
from sumpy.e2p import E2PBase
from sumpy.p2e import P2EBase

from pytential.array_context import PyOpenCLArrayContext, make_loopy_program
from pytential.version import PYTENTIAL_KERNEL_VERSION


# {{{ form qbx expansions from points

class P2QBXLFromCSR(P2EBase):
    @property
    def default_name(self):
        return "p2qbxl_from_csr"

    def get_cache_key(self):
        return (*super().get_cache_key(), PYTENTIAL_KERNEL_VERSION)

    def get_kernel(self):
        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        arguments = (
                [
                    lp.GlobalArg("sources", None,
                        shape=(self.dim, "nsources")),
                    lp.GlobalArg("strengths", None,
                        shape=(self.strength_count, "nsources")),
                    lp.GlobalArg("qbx_center_to_target_box",
                        None, shape=None),
                    lp.GlobalArg("source_box_starts,source_box_lists",
                        None, shape=None),
                    lp.GlobalArg("box_source_starts,box_source_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("qbx_centers", None,
                        shape=("dim", "ncenters"), dim_tags="sep,C"),
                    lp.GlobalArg("qbx_expansion_radii", None, shape="ncenters"),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeffs)),
                    lp.ValueArg("ncenters", np.int32),
                    lp.ValueArg("nsources", np.int32),
                    *loopy_args,
                    ...
                ])

        loopy_knl = make_loopy_program(
                [
                    "{[itgt_center]: 0<=itgt_center<ntgt_centers}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_stop}",
                    "{[isrc]: isrc_start<=isrc<isrc_end}",
                    "{[idim]: 0<=idim<dim}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[istrength]: 0<=istrength<nstrengths}",
                ],
                ["""
                for itgt_center
                    <> tgt_icenter = global_qbx_centers[itgt_center]

                    <> center[idim] = qbx_centers[idim, tgt_icenter] \
                            {id=fetch_center}
                    <> rscale = qbx_expansion_radii[tgt_icenter] {id=fetch_rscale}

                    <> itgt_box = qbx_center_to_target_box[tgt_icenter]

                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_stop = source_box_starts[itgt_box+1]

                    <> coeffs[icoeff] = 0  {id=init_coeffs,dup=icoeff}
                    for isrc_box
                        <> src_ibox = source_box_lists[isrc_box]
                        <> isrc_start = box_source_starts[src_ibox]
                        <> isrc_end = isrc_start+box_source_counts_nonchild[src_ibox]

                        for isrc
                            <> source[idim] = sources[idim, isrc] \
                                    {dup=idim,id=fetch_src}
                            <> strength[istrength] = strengths[istrength, isrc] \
                                    {dup=istrength,id=fetch_strength}
                            [icoeff]: coeffs[icoeff] = p2e(
                                    [icoeff]: coeffs[icoeff],
                                    [idim]: center[idim],
                                    [idim]: source[idim],
                                    [istrength]: strength[istrength],
                                    rscale,
                                    isrc,
                                    nsources,
                                    sources,
                    """ + ",".join(arg.name for arg in loopy_args) + """
                                )  {id=update_result,dep=fetch_*:init_coeffs}
                        end
                    end
                    qbx_expansions[tgt_icenter, icoeff] = \
                            coeffs[icoeff] {id=write_expn,dup=icoeff, \
                            dep=update_result:init_coeffs}
                end
                """],
                arguments,
                name=self.name,
                assumptions="ntgt_centers>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={
                    "dim": self.dim,
                    "nstrengths": self.strength_count,
                    "ncoeffs": ncoeffs})

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "istrength*:unr")
        loopy_knl = self.add_loopy_form_callable(loopy_knl)

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self, is_sources_obj_array, is_centers_obj_array):
        # FIXME
        knl = self.get_kernel()

        if is_sources_obj_array:
            knl = lp.tag_array_axes(knl, "sources", "sep,C")
        if is_centers_obj_array:
            knl = lp.tag_array_axes(knl, "qbx_centers", "sep,C")

        knl = lp.split_iname(knl, "itgt_center", 16, outer_tag="g.0")
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        sources = kwargs.pop("sources")
        qbx_centers = kwargs.pop("qbx_centers")

        from sumpy.tools import is_obj_array_like
        knl = self.get_cached_kernel(
                is_sources_obj_array=is_obj_array_like(sources),
                is_centers_obj_array=is_obj_array_like(qbx_centers),
                )

        result = actx.call_loopy(
            knl,
            sources=sources,
            qbx_centers=qbx_centers, **kwargs)

        return result["qbx_expansions"]

# }}}


# {{{ translation from (likely, list 3) multipoles to qbx expansions

class M2QBXL(E2EBase):
    """Implements translation from a "compressed sparse row"-like source box
    list.
    """

    @property
    def default_name(self):
        return "m2qbxl_from_csr"

    def get_cache_key(self):
        return (*super().get_cache_key(), PYTENTIAL_KERNEL_VERSION)

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = make_loopy_program([
                "{[icenter]: 0<=icenter<ncenters}",
                "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                "{[idim]: 0<=idim<dim}",
                ],
                ["""
                for icenter
                    <> icontaining_tgt_box = \
                        qbx_center_to_target_box_source_level[icenter]

                    if icontaining_tgt_box != -1
                        <> tgt_center[idim] = qbx_centers[idim, icenter] \
                                {id=fetch_tgt_center}
                        <> tgt_rscale = qbx_expansion_radii[icenter]

                        <> isrc_start = src_box_starts[icontaining_tgt_box]
                        <> isrc_stop = src_box_starts[icontaining_tgt_box+1]

                        for isrc_box
                            <> src_ibox = src_box_lists[isrc_box] \
                                    {id=read_src_ibox}
                            <> src_center[idim] = centers[idim, src_ibox] {dup=idim}
                            <> d[idim] = tgt_center[idim] - src_center[idim] \
                                    {dup=idim}
                            """] + ["""

                            <> src_coeff{i} = \
                                src_expansions[src_ibox - src_base_ibox, {i}] \
                                {{dep=read_src_ibox}}

                            """.format(i=i) for i in range(ncoeff_src)] + [

                            *self.get_translation_loopy_insns(), """

                        end
                        """] + ["""
                        qbx_expansions[icenter, {i}] = \
                                qbx_expansions[icenter, {i}] + \
                                simul_reduce(sum, isrc_box, coeff{i}) \
                                {{id_prefix=write_expn}}
                        """.format(i=i)
                                for i in range(ncoeff_tgt)] + ["""
                    end
                end
                """],
                [
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("src_rscale", None),
                    lp.GlobalArg("src_box_starts, src_box_lists",
                        None, shape=None, strides=(1,)),
                    lp.GlobalArg("qbx_centers", None,
                        shape="dim, ncenters", dim_tags="sep,C"),
                    lp.GlobalArg("qbx_expansion_radii", None, shape="ncenters"),
                    lp.ValueArg("aligned_nboxes,nsrc_level_boxes", np.int32),
                    lp.ValueArg("src_base_ibox", np.int32),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeff_src), offset=lp.auto),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeff_tgt)),
                    *gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                    ...
                ],
                name=self.name,
                assumptions="ncenters>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={"dim": self.dim},
                )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        for knl in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self, is_centers_obj_array):
        # FIXME
        knl = self.get_kernel()

        if is_centers_obj_array:
            knl = lp.tag_array_axes(knl, "qbx_centers", "sep,C")

        knl = lp.split_iname(knl, "icenter", 16, outer_tag="g.0")
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))

        from sumpy.tools import is_obj_array_like
        qbx_centers = kwargs.pop("qbx_centers")
        knl = self.get_cached_kernel(
                is_centers_obj_array=is_obj_array_like(qbx_centers),
                )

        result = actx.call_loopy(
            knl,
            centers=centers,
            qbx_centers=qbx_centers,
            src_rscale=src_rscale, **kwargs)

        return result["qbx_expansions"]

# }}}


# {{{ translation from a center's box

class L2QBXL(E2EBase):
    @property
    def default_name(self):
        return "l2qbxl"

    def get_cache_key(self):
        return (*super().get_cache_key(), PYTENTIAL_KERNEL_VERSION)

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = make_loopy_program([
                "{[icenter]: 0<=icenter<ncenters}",
                "{[idim]: 0<=idim<dim}",
                ],
                ["""
                for icenter
                    <> isrc_box = qbx_center_to_target_box[icenter]

                    # The box's expansions which we're translating here
                    # (our source) is, globally speaking, a target box.

                    <> src_ibox = target_boxes[isrc_box] \
                        {id=read_src_ibox}

                    # Is the box number on the level currently under
                    # consideration?
                    <> in_range = (target_base_ibox <= src_ibox
                            and src_ibox < target_base_ibox + nboxes)

                    if in_range
                        <> tgt_center[idim] = qbx_centers[idim, icenter]
                        <> src_center[idim] = centers[idim, src_ibox] {dup=idim}

                        <> tgt_rscale = qbx_expansion_radii[icenter]

                        <> d[idim] = tgt_center[idim] - src_center[idim] {dup=idim}

                        """] + ["""
                        <> src_coeff{i} = \
                                expansions[src_ibox - target_base_ibox, {i}] \
                                {{dep=read_src_ibox}}
                        """.format(i=i) for i in range(ncoeff_src)] + [
                        *self.get_translation_loopy_insns(),
                        ] + [
                        """
                        qbx_expansions[icenter, {i}] = \
                            qbx_expansions[icenter, {i}] + coeff{i} \
                            {{id_prefix=write_expn}}
                        """.format(i=i)
                            for i in range(ncoeff_tgt)] + ["""
                    end
                end
                """],
                [
                    lp.GlobalArg("target_boxes", None, shape=None,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.ValueArg("src_rscale", None),
                    lp.GlobalArg("qbx_centers", None,
                        shape="dim, ncenters", dim_tags="sep,C"),
                    lp.GlobalArg("qbx_expansion_radii", None, shape="ncenters"),
                    lp.ValueArg("naligned_boxes,target_base_ibox,nboxes", np.int32),
                    lp.GlobalArg("expansions", None,
                        shape=("nboxes", ncoeff_src), offset=lp.auto),
                    *gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                    ...
                ],
                name=self.name,
                assumptions="ncenters>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={"dim": self.dim, "nchildren": 2**self.dim},
                )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        for knl in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self, is_centers_obj_array):
        # FIXME
        knl = self.get_kernel()

        if is_centers_obj_array:
            knl = lp.tag_array_axes(knl, "qbx_centers", "sep,C")

        knl = lp.split_iname(knl, "icenter", 16, outer_tag="g.0")
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))

        from sumpy.tools import is_obj_array_like
        qbx_centers = kwargs.pop("qbx_centers")
        knl = self.get_cached_kernel(
                is_centers_obj_array=is_obj_array_like(qbx_centers),
                )

        result = actx.call_loopy(
            knl,
            centers=centers,
            qbx_centers=qbx_centers,
            src_rscale=src_rscale, **kwargs)

        return result["qbx_expansions"]

# }}}


# {{{ evaluation of qbx expansions

class QBXL2P(E2PBase):
    @property
    def default_name(self):
        return "qbx_potential_from_local"

    def get_cache_key(self):
        return (*super().get_cache_key(), PYTENTIAL_KERNEL_VERSION)

    def get_kernel(self):
        ncoeffs = len(self.expansion)
        loopy_args = [arg.loopy_arg for arg in self.expansion.get_args()]

        loopy_knl = make_loopy_program(
                [
                    "{[iglobal_center]: 0<=iglobal_center<nglobal_qbx_centers}",
                    "{[icenter_tgt]: \
                            icenter_tgt_start<=icenter_tgt<icenter_tgt_end}",
                    "{[idim]: 0<=idim<dim}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[iknl]: 0<=iknl<nresults}",
                    ],
                [
                *self.get_kernel_scaling_assignment(),
                """
                for iglobal_center
                    <> src_icenter = global_qbx_centers[iglobal_center]

                    <> icenter_tgt_start = center_to_targets_starts[src_icenter]
                    <> icenter_tgt_end = center_to_targets_starts[src_icenter+1]

                    <> center[idim] = qbx_centers[idim, src_icenter] \
                            {dup=idim,id=fetch_center}
                    <> rscale = qbx_expansion_radii[src_icenter]

                    for icenter_tgt
                        <> itgt = center_to_targets_lists[icenter_tgt]

                        <> coeffs[icoeff] = qbx_expansions[src_icenter, icoeff] \
                            {id=fetch_coeffs,dup=icoeff}

                        <> tgt[idim] = targets[idim, itgt] {id=fetch_tgt,dup=idim}
                        <> result_temp[iknl] = 0  {id=init_result,dup=iknl}

                        [iknl]: result_temp[iknl] = e2p(
                            [iknl]: result_temp[iknl],
                            [icoeff]: coeffs[icoeff],
                            [idim]: center[idim],
                            [idim]: tgt[idim],
                            rscale,
                            itgt,
                            ntargets,
                            targets,
                """ + ",".join(arg.name for arg in loopy_args)
                + """
                        )  {dep=fetch_coeffs:fetch_center:fetch_tgt:init_result,\
                                id=write_result}
                        result[iknl, itgt] = result_temp[iknl] * kernel_scaling \
                            {dep=write_result}
                    end
                end
                """],
                [
                    lp.GlobalArg("result", None,
                        shape=("nresults", "ntargets"), dim_tags="sep,C"),
                    lp.GlobalArg("qbx_centers", None,
                        shape=("dim", "ncenters"), dim_tags="sep,C"),
                    lp.GlobalArg("center_to_targets_starts,center_to_targets_lists",
                        None, shape=None),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeffs)),
                    lp.GlobalArg("targets", None,
                        shape=(self.dim, "ntargets")),
                    lp.ValueArg("ncenters,ntargets", np.int32),
                    *loopy_args,
                    ...
                ],
                name=self.name,
                assumptions="nglobal_qbx_centers>=1",
                silenced_warnings="write_race(write_result*)",
                fixed_parameters={
                    "dim": self.dim,
                    "ncoeffs": ncoeffs,
                    "nresults": len(self.kernels)})

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr,iknl:unr")
        loopy_knl = self.add_loopy_eval_callable(loopy_knl)

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self, is_targets_obj_array, is_centers_obj_array):
        # FIXME
        knl = self.get_kernel()

        if is_targets_obj_array:
            knl = lp.tag_array_axes(knl, "targets", "sep,C")
        if is_centers_obj_array:
            knl = lp.tag_array_axes(knl, "qbx_centers", "sep,C")

        knl = lp.tag_inames(knl, {"iglobal_center": "g.0"})
        knl = lp.add_inames_to_insn(knl, "iglobal_center", "id:kernel_scaling")
        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        targets = kwargs.pop("targets")
        qbx_centers = kwargs.pop("qbx_centers")

        from sumpy.tools import is_obj_array_like
        knl = self.get_cached_kernel(
                is_targets_obj_array=is_obj_array_like(targets),
                is_centers_obj_array=is_obj_array_like(qbx_centers),
                )

        result = actx.call_loopy(
            knl, targets=targets, qbx_centers=qbx_centers, **kwargs)

        return obj_array.new_1d([result[f"result_s{i}"] for i in range(self.nresults)])

# }}}

# vim: foldmethod=marker
