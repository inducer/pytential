from __future__ import division
from __future__ import absolute_import
from six.moves import range

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
from pytools import memoize_method

from sumpy.p2e import P2EBase
from sumpy.e2e import E2EBase
from sumpy.e2p import E2PBase


# {{{ form qbx expansions from points

class P2QBXLFromCSR(P2EBase):
    default_name = "p2qbxl_from_csr"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        from sumpy.tools import gather_source_arguments
        arguments = (
                [
                    lp.GlobalArg("sources", None, shape=(self.dim, "nsources"),
                        dim_tags="sep,c"),
                    lp.GlobalArg("strengths", None, shape="nsources"),
                    lp.GlobalArg("qbx_center_to_target_box",
                        None, shape=None),
                    lp.GlobalArg("source_box_starts,source_box_lists",
                        None, shape=None),
                    lp.GlobalArg("box_source_starts,box_source_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("qbx_centers", None, shape="dim, ncenters",
                        dim_tags="sep,c"),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeffs)),
                    lp.ValueArg("ncenters", np.int32),
                    lp.ValueArg("nsources", np.int32),
                    "..."
                ] + gather_source_arguments([self.expansion]))

        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_center]: 0<=itgt_center<ntgt_centers}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_stop}",
                    "{[isrc,idim]: isrc_start<=isrc<isrc_end and 0<=idim<dim}",
                    ],
                self.get_loopy_instructions()
                + ["""
                    <> tgt_icenter = global_qbx_centers[itgt_center]

                    <> itgt_box = qbx_center_to_target_box[tgt_icenter]

                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_stop = source_box_starts[itgt_box+1]

                    <> src_ibox = source_box_lists[isrc_box]
                    <> isrc_start = box_source_starts[src_ibox]
                    <> isrc_end = isrc_start+box_source_counts_nonchild[src_ibox]

                    <> center[idim] = qbx_centers[idim, tgt_icenter] \
                            {id=fetch_center}
                    <> a[idim] = center[idim] - sources[idim, isrc] {id=compute_a}
                    <> strength = strengths[isrc]
                    qbx_expansions[tgt_icenter, ${COEFFIDX}] = \
                            sum((isrc_box, isrc), strength*coeff${COEFFIDX}) \
                            {id_prefix=write_expn}
                    """],
                arguments,
                name=self.name, assumptions="ntgt_centers>=1",
                defines=dict(
                    dim=self.dim,
                    COEFFIDX=[str(i) for i in range(ncoeffs)]
                    ),
                silenced_warnings="write_race(write_expn*)")

        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)
        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_center",
                tags={"idim": "unr"})
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itgt_center", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        return self.get_cached_optimized_kernel()(queue, **kwargs)

# }}}


# {{{ translation from (likely, list 3) multipoles to qbx expansions

class M2QBXL(E2EBase):
    """Implements translation from a "compressed sparse row"-like source box
    list.
    """

    default_name = "m2qbxl_from_csr"

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)

        from sumpy.tools import gather_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[icenter]: 0<=icenter<ncenters}",
                    "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                self.get_translation_loopy_insns()
                + ["""
                    <> icontaining_tgt_box = qbx_center_to_target_box[icenter]

                    <> tgt_center[idim] = qbx_centers[idim, icenter] \
                            {id=fetch_tgt_center}

                    <> isrc_start = src_box_starts[icontaining_tgt_box]
                    <> isrc_stop = src_box_starts[icontaining_tgt_box+1]

                    <> src_ibox = src_box_lists[isrc_box] \
                            {id=read_src_ibox}
                    <> src_center[idim] = centers[idim, src_ibox] \
                            {id=fetch_src_center}
                    <> d[idim] = tgt_center[idim] - src_center[idim]
                    <> src_coeff${SRC_COEFFIDX} = \
                        src_expansions[src_ibox, ${SRC_COEFFIDX}] \
                        {dep=read_src_ibox}

                    qbx_expansions[icenter, ${TGT_COEFFIDX}] = \
                            sum(isrc_box, coeff${TGT_COEFFIDX}) \
                            {id_prefix=write_expn}
                    """],
                [
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.GlobalArg("src_box_starts, src_box_lists",
                        None, shape=None, strides=(1,)),
                    lp.GlobalArg("qbx_centers", None, shape="dim, ncenters",
                        dim_tags="sep,c"),
                    lp.ValueArg("aligned_nboxes,nboxes", np.int32),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nboxes", ncoeff_src)),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeff_tgt)),
                    "..."
                ] + gather_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ncenters>=1",
                defines=dict(
                    dim=self.dim,
                    SRC_COEFFIDX=[str(i) for i in range(ncoeff_src)],
                    TGT_COEFFIDX=[str(i) for i in range(ncoeff_tgt)],
                    ),
                silenced_warnings="write_race(write_expn*)")

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_tgt_center",
                tags={"idim": "unr"})
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "icenter", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        return self.get_cached_optimized_kernel()(queue, **kwargs)

# }}}


# {{{ translation from a box's parent

class L2QBXL(E2EBase):
    default_name = "l2qbxl"

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)

        from sumpy.tools import gather_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[icenter]: 0<=icenter<ncenters}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                self.get_translation_loopy_insns()
                + ["""
                    <> isrc_box = qbx_center_to_target_box[icenter]

                    # The box's expansions which we're translating here
                    # (our source) is, globally speaking, a target box.

                    <> src_ibox = target_boxes[isrc_box] \
                        {id=read_src_ibox}

                    <> tgt_center[idim] = qbx_centers[idim, icenter] \
                        {id=fetch_tgt_center}

                    <> src_center[idim] = centers[idim, src_ibox] \
                        {id=fetch_src_center}
                    <> d[idim] = tgt_center[idim] - src_center[idim]

                    <> src_coeff${SRC_COEFFIDX} = \
                        expansions[src_ibox, ${SRC_COEFFIDX}] \
                        {dep=read_src_ibox}
                    qbx_expansions[icenter, ${TGT_COEFFIDX}] = \
                        qbx_expansions[icenter, ${TGT_COEFFIDX}] \
                        + coeff${TGT_COEFFIDX} \
                        {id_prefix=write_expn}
                    """],
                [
                    lp.GlobalArg("target_boxes", None, shape=None,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.GlobalArg("qbx_centers", None, shape="dim, ncenters",
                        dim_tags="sep,c"),
                    lp.ValueArg("naligned_boxes,nboxes", np.int32),
                    lp.GlobalArg("expansions", None,
                        shape=("nboxes", ncoeff_src)),
                    "..."
                ] + gather_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ncenters>=1",
                defines=dict(
                    dim=self.dim,
                    nchildren=2**self.dim,
                    SRC_COEFFIDX=[str(i) for i in range(ncoeff_src)],
                    TGT_COEFFIDX=[str(i) for i in range(ncoeff_tgt)],
                    ),
                silenced_warnings="write_race(write_expn*)")

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_tgt_center",
                tags={"idim": "unr"})
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "icenter", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        return self.get_cached_optimized_kernel()(queue, **kwargs)

# }}}


# {{{ evaluation of qbx expansions

class QBXL2P(E2PBase):
    default_name = "qbx_potential_from_local"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        loopy_knl = lp.make_kernel(
                [
                    "{[iglobal_center]: 0<=iglobal_center<nglobal_qbx_centers}",
                    "{[icenter_tgt]: \
                            icenter_tgt_start<=icenter_tgt<icenter_tgt_end}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                loopy_insns
                + ["""
                    <> src_icenter = global_qbx_centers[iglobal_center]

                    <> icenter_tgt_start = center_to_targets_starts[src_icenter]
                    <> icenter_tgt_end = center_to_targets_starts[src_icenter+1]

                    <> center_itgt = center_to_targets_lists[icenter_tgt]

                    <> center[idim] = qbx_centers[idim, src_icenter] \
                            {id=fetch_center}
                    <> b[idim] = targets[idim, center_itgt] - center[idim] \
                            {id=compute_b}
                    <> coeff${COEFFIDX} = qbx_expansions[src_icenter, ${COEFFIDX}]
                    result[${RESULTIDX},center_itgt] = \
                            kernel_scaling * result_${RESULTIDX}_p \
                            {id_prefix=write_result}
                """],
                [
                    lp.GlobalArg("result", None, shape="nresults, ntargets",
                        dim_tags="sep,C"),
                    lp.GlobalArg("qbx_centers", None, shape="dim, ncenters",
                        dim_tags="sep,c"),
                    lp.GlobalArg("center_to_targets_starts,center_to_targets_lists",
                        None, shape=None),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeffs)),
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets"),
                        dim_tags="sep,C"),
                    lp.ValueArg("ncenters,ntargets", np.int32),
                    "..."
                ] + self.expansion.get_args(),
                name=self.name, assumptions="nglobal_qbx_centers>=1",
                defines=dict(
                    dim=self.dim,
                    COEFFIDX=[str(i) for i in range(ncoeffs)],
                    RESULTIDX=[str(i) for i in range(len(result_names))],
                    nresults=len(result_names),
                    ),
                silenced_warnings="write_race(write_result*)")

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "compute_b",
                tags={"idim": "unr"})
        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_center",
                tags={"idim": "unr"})
        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, dict(iglobal_center="g.0"))
        return knl

    def __call__(self, queue, **kwargs):
        return self.get_cached_optimized_kernel()(queue, **kwargs)

# }}}

# vim: foldmethod=marker
