from __future__ import division, absolute_import

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
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from pytools import memoize_method
from six.moves import range

from sumpy.p2e import P2EBase
from sumpy.e2e import E2EBase
from sumpy.e2p import E2PBase


PYTENTIAL_KERNEL_VERSION = 6


# {{{ form qbx expansions from points

class P2QBXLFromCSR(P2EBase):
    default_name = "p2qbxl_from_csr"

    def get_cache_key(self):
        return super(P2QBXLFromCSR, self).get_cache_key() + (
                PYTENTIAL_KERNEL_VERSION,)

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        from sumpy.tools import gather_loopy_source_arguments
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
                    lp.GlobalArg("qbx_expansion_radii", None, shape="ncenters"),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeffs)),
                    lp.ValueArg("ncenters", np.int32),
                    lp.ValueArg("nsources", np.int32),
                    "..."
                ] + gather_loopy_source_arguments([self.expansion]))

        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_center]: 0<=itgt_center<ntgt_centers}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_stop}",
                    "{[isrc]: isrc_start<=isrc<isrc_end}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                ["""
                for itgt_center
                    <> tgt_icenter = global_qbx_centers[itgt_center]

                    <> center[idim] = qbx_centers[idim, tgt_icenter] {dup=idim}
                    <> rscale = qbx_expansion_radii[tgt_icenter]

                    <> itgt_box = qbx_center_to_target_box[tgt_icenter]

                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_stop = source_box_starts[itgt_box+1]

                    for isrc_box
                        <> src_ibox = source_box_lists[isrc_box]
                        <> isrc_start = box_source_starts[src_ibox]
                        <> isrc_end = isrc_start+box_source_counts_nonchild[src_ibox]

                        for isrc
                            <> a[idim] = center[idim] - sources[idim, isrc] \
                                    {dup=idim}
                            <> strength = strengths[isrc]

                            """] + self.get_loopy_instructions() + ["""
                        end
                    end

                    """] + ["""
                    qbx_expansions[tgt_icenter, {i}] = \
                            simul_reduce(sum, (isrc_box, isrc), strength*coeff{i}) \
                            {{id_prefix=write_expn{nosync}}}
                    """.format(i=i,
                               nosync=",nosync=write_expn*"
                               if ncoeffs > 1 else "")
                        for i in range(ncoeffs)] + ["""

                end
                """],
                arguments,
                name=self.name, assumptions="ntgt_centers>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters=dict(dim=self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)
        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

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

    def get_cache_key(self):
        return super(M2QBXL, self).get_cache_key() + (PYTENTIAL_KERNEL_VERSION,)

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[icenter]: 0<=icenter<ncenters}",
                    "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                ["""
                for icenter
                    <> icontaining_tgt_box = qbx_center_to_target_box[icenter]

                    <> tgt_center[idim] = qbx_centers[idim, icenter] \
                            {id=fetch_tgt_center}
                    <> tgt_rscale = qbx_expansion_radii[icenter]

                    <> isrc_start = src_box_starts[icontaining_tgt_box]
                    <> isrc_stop = src_box_starts[icontaining_tgt_box+1]

                    for isrc_box
                        <> src_ibox = src_box_lists[isrc_box] \
                                {id=read_src_ibox}
                        <> src_center[idim] = centers[idim, src_ibox] {dup=idim}
                        <> d[idim] = tgt_center[idim] - src_center[idim] {dup=idim}
                        """] + ["""

                        <> src_coeff{i} = \
                            src_expansions[src_ibox - src_base_ibox, {i}] \
                            {{dep=read_src_ibox}}

                        """.format(i=i) for i in range(ncoeff_src)] + [

                        ] + self.get_translation_loopy_insns() + ["""

                    end
                    """] + ["""
                    qbx_expansions[icenter, {i}] = qbx_expansions[icenter, {i}] + \
                            simul_reduce(sum, isrc_box, coeff{i}) \
                            {{id_prefix=write_expn{nosync}}}
                    """.format(i=i,
                               nosync=",nosync=write_expn*"
                               if ncoeff_tgt > 1 else "")
                            for i in range(ncoeff_tgt)] + ["""

                end
                """],
                [
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("src_rscale", None),
                    lp.GlobalArg("src_box_starts, src_box_lists",
                        None, shape=None, strides=(1,)),
                    lp.GlobalArg("qbx_centers", None, shape="dim, ncenters",
                        dim_tags="sep,c"),
                    lp.GlobalArg("qbx_expansion_radii", None, shape="ncenters"),
                    lp.ValueArg("aligned_nboxes,nsrc_level_boxes", np.int32),
                    lp.ValueArg("src_base_ibox", np.int32),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeff_src), offset=lp.auto),
                    lp.GlobalArg("qbx_expansions", None,
                        shape=("ncenters", ncoeff_tgt)),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ncenters>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters=dict(dim=self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "icenter", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))

        return self.get_cached_optimized_kernel()(queue,
                centers=centers,
                src_rscale=src_rscale,
                **kwargs)

# }}}


# {{{ translation from a center's box

class L2QBXL(E2EBase):
    default_name = "l2qbxl"

    def get_cache_key(self):
        return super(L2QBXL, self).get_cache_key() + (PYTENTIAL_KERNEL_VERSION,)

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = lp.make_kernel(
                [
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
                        ] + self.get_translation_loopy_insns() + ["""
                        qbx_expansions[icenter, {i}] = \
                            qbx_expansions[icenter, {i}] + coeff{i} \
                            {{id_prefix=write_expn{nosync}}}
                        """.format(i=i,
                                   nosync=",nosync=write_expn*"
                                   if ncoeff_tgt > 1 else "")
                            for i in range(ncoeff_tgt)] + ["""
                    end
                end
                """],
                [
                    lp.GlobalArg("target_boxes", None, shape=None,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.ValueArg("src_rscale", None),
                    lp.GlobalArg("qbx_centers", None, shape="dim, ncenters",
                        dim_tags="sep,c"),
                    lp.GlobalArg("qbx_expansion_radii", None, shape="ncenters"),
                    lp.ValueArg("naligned_boxes,target_base_ibox,nboxes", np.int32),
                    lp.GlobalArg("expansions", None,
                        shape=("nboxes", ncoeff_src), offset=lp.auto),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ncenters>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters=dict(dim=self.dim, nchildren=2**self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "icenter", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))

        return self.get_cached_optimized_kernel()(queue,
                centers=centers,
                src_rscale=src_rscale,
                **kwargs)

# }}}


# {{{ evaluation of qbx expansions

class QBXL2P(E2PBase):
    default_name = "qbx_potential_from_local"

    def get_cache_key(self):
        return super(QBXL2P, self).get_cache_key() + (PYTENTIAL_KERNEL_VERSION,)

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
                self.get_kernel_scaling_assignment()
                + ["""
                for iglobal_center
                    <> src_icenter = global_qbx_centers[iglobal_center]

                    <> icenter_tgt_start = center_to_targets_starts[src_icenter]
                    <> icenter_tgt_end = center_to_targets_starts[src_icenter+1]

                    <> center[idim] = qbx_centers[idim, src_icenter] {dup=idim}
                    <> rscale = qbx_expansion_radii[src_icenter]

                    for icenter_tgt

                        <> center_itgt = center_to_targets_lists[icenter_tgt]

                        <> b[idim] = targets[idim, center_itgt] - center[idim]

                        """] + ["""
                        <> coeff{i} = qbx_expansions[src_icenter, {i}]
                        """.format(i=i) for i in range(ncoeffs)] + [

                        ] + loopy_insns + ["""

                        result[{i},center_itgt] = kernel_scaling * result_{i}_p \
                                {{id_prefix=write_result{nosync}}}
                        """.format(i=i,
                                   nosync=",nosync=write_result*"
                                   if len(result_names) > 1 else "")
                            for i in range(len(result_names))] + ["""
                    end
                end
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
                ] + [arg.loopy_arg for arg in self.expansion.get_args()],
                name=self.name,
                assumptions="nglobal_qbx_centers>=1",
                silenced_warnings="write_race(write_result*)",
                fixed_parameters=dict(dim=self.dim, nresults=len(result_names)),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
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
