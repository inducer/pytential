from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

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

import loopy as lp
import numpy as np

from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from sumpy.qbx import LayerPotentialBase
from pytential.version import PYTENTIAL_KERNEL_VERSION


# {{{ qbx applier on a target/center subset

class LayerPotentialOnTargetAndCenterSubset(LayerPotentialBase):
    default_name = "qbx_tgt_ctr_subset"

    def get_cache_key(self):
        return super(LayerPotentialOnTargetAndCenterSubset, self).get_cache_key() + (
                PYTENTIAL_KERNEL_VERSION,)

    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        kernel_exprs = self.get_kernel_exprs(result_names)

        from sumpy.tools import gather_loopy_source_arguments
        arguments = (
            gather_loopy_source_arguments(self.kernels)
            + [
                lp.GlobalArg("src", None,
                    shape=(self.dim, "nsources"), order="C"),
                lp.GlobalArg("tgt", None,
                    shape=(self.dim, "ntargets_total"), order="C"),
                lp.GlobalArg("center", None,
                    shape=(self.dim, "ncenters_total"), dim_tags="sep,C"),
                lp.GlobalArg("expansion_radii", None,
                    shape="ncenters_total"),
                lp.GlobalArg("qbx_tgt_numbers", None,
                    shape="ntargets"),
                lp.GlobalArg("qbx_center_numbers", None,
                    shape="ntargets"),
                lp.ValueArg("nsources", np.int32),
                lp.ValueArg("ntargets", np.int32),
                lp.ValueArg("ntargets_total", np.int32),
                lp.ValueArg("ncenters_total", np.int32)]
            + [lp.GlobalArg("strength_%d" % i, None,
                shape="nsources", order="C")
            for i in range(self.strength_count)]
            + [lp.GlobalArg("result_%d" % i, self.value_dtypes[i],
                shape="ntargets_total", order="C")
            for i in range(len(self.kernels))])

        loopy_knl = lp.make_kernel([
            "{[itgt]: 0 <= itgt < ntargets}",
            "{[isrc]: 0 <= isrc < nsources}",
            "{[idim]: 0 <= idim < dim}"
            ],
            self.get_kernel_scaling_assignments()
            + ["for itgt, isrc"]
            + ["""
                <> icenter = qbx_center_numbers[itgt]
                <> itgt_overall = qbx_tgt_numbers[itgt]

                <> a[idim] = center[idim, icenter] - src[idim, isrc] \
                        {dup=idim}
                <> b[idim] = tgt[idim, itgt_overall] - center[idim, icenter] \
                        {dup=idim}
                <> rscale = expansion_radii[icenter]
            """]
            + loopy_insns + kernel_exprs
            + ["""
                result_{i}[itgt_overall] = knl_{i}_scaling * \
                    simul_reduce(sum, isrc, pair_result_{i})  \
                        {{inames=itgt}}
                """.format(i=iknl)
                for iknl in range(len(self.expansions))]
            + ["end"],
            arguments,
            name=self.name,
            assumptions="ntargets>=1 and nsources>=1",
            fixed_parameters=dict(dim=self.dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        for expn in self.expansions:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def __call__(self, queue, targets, sources, centers, strengths, expansion_radii,
            **kwargs):
        knl = self.get_cached_optimized_kernel()

        for i, dens in enumerate(strengths):
            kwargs["strength_%d" % i] = dens

        return knl(queue, src=sources, tgt=targets, center=centers,
                expansion_radii=expansion_radii, **kwargs)

# }}}

# vim: foldmethod=marker
