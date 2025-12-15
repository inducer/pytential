from __future__ import annotations


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

from typing import TYPE_CHECKING

import numpy as np

import loopy as lp
from pytools import obj_array
from sumpy.qbx import LayerPotentialBase


if TYPE_CHECKING:
    from pytential.array_context import PyOpenCLArrayContext


# {{{ qbx applier on a target/center subset

class LayerPotentialOnTargetAndCenterSubset(LayerPotentialBase):
    @property
    def default_name(self):
        return "qbx_tgt_ctr_subset"

    def get_cache_key(self):
        from pytential.version import PYTENTIAL_KERNEL_VERSION
        return (*super().get_cache_key(), PYTENTIAL_KERNEL_VERSION)

    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        kernel_exprs = self.get_kernel_exprs(result_names)

        from sumpy.tools import gather_loopy_source_arguments
        arguments = (
            gather_loopy_source_arguments(
                (self.expansion, *self.source_kernels, *self.target_kernels))
            + [
                lp.GlobalArg("sources", None,
                    shape=(self.dim, "nsources"), order="C"),
                lp.GlobalArg("targets", None,
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
            + [lp.GlobalArg(f"strength_{i}", None,
                shape="nsources", order="C")
            for i in range(self.strength_count)]
            + [lp.GlobalArg(f"result_{i}", self.value_dtypes[i],
                shape="ntargets_total", order="C")
            for i in range(self.nresults)])

        from pytential.array_context import make_loopy_program

        loopy_knl = make_loopy_program([
            "{[itgt_local]: 0 <= itgt_local < ntargets}",
            "{[isrc]: 0 <= isrc < nsources}",
            "{[idim]: 0 <= idim < dim}"
            ],
            self.get_kernel_scaling_assignments()
            + ["for itgt_local, isrc"]
            + ["""
                <> icenter = qbx_center_numbers[itgt_local]
                <> itgt = qbx_tgt_numbers[itgt_local]

                <> a[idim] = center[idim, icenter] - sources[idim, isrc]
                <> b[idim] = targets[idim, itgt] - center[idim, icenter] \
                        {dup=idim}
                <> rscale = expansion_radii[icenter]
            """]
            + [f"<> strength_{i}_isrc = strength_{i}[isrc]"
                for i in range(self.strength_count)]
            + [*loopy_insns, *kernel_exprs]
            + ["""
                result_{i}[itgt] = knl_{i}_scaling * \
                    simul_reduce(sum, isrc, pair_result_{i})  \
                        {{inames=itgt_local}}
                """.format(i=iknl)
                for iknl in range(self.nresults)]
            + ["end"],
            kernel_data=arguments,
            name=self.name,
            assumptions="ntargets>=1 and nsources>=1",
            fixed_parameters={"dim": self.dim},
            )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        for expn in self.source_kernels + self.target_kernels:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self, *,
            is_cpu: bool,
            targets_is_obj_array: bool,
            sources_is_obj_array: bool,
            centers_is_obj_array: bool):
        return super().get_optimized_kernel(
                is_cpu=is_cpu,
                targets_is_obj_array=targets_is_obj_array,
                sources_is_obj_array=sources_is_obj_array,
                centers_is_obj_array=centers_is_obj_array,
                itgt_name="itgt_local")

    def __call__(self, actx: PyOpenCLArrayContext,
            targets, sources, centers, strengths, expansion_radii,
            **kwargs):
        from sumpy.array_context import is_cl_cpu
        from sumpy.tools import is_obj_array_like

        knl = self.get_cached_kernel(
                is_cpu=is_cl_cpu(actx),
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources),
                centers_is_obj_array=is_obj_array_like(centers))

        for i, dens in enumerate(strengths):
            kwargs[f"strength_{i}"] = dens

        result = actx.call_loopy(
                knl,
                sources=sources, targets=targets, center=centers,
                expansion_radii=expansion_radii, **kwargs)

        return obj_array.new_1d([result[f"result_{i}"] for i in range(self.nresults)])

# }}}

# vim: foldmethod=marker
