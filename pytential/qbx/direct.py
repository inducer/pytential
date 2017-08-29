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

from sumpy.qbx import LayerPotential as LayerPotentialBase


# {{{ qbx applier on a target/center subset

class LayerPotentialOnTargetAndCenterSubset(LayerPotentialBase):
    def get_compute_a_and_b_vecs(self):
        return """
            <> icenter = qbx_center_numbers[itgt]
            <> itgt_overall = qbx_tgt_numbers[itgt]
            for idim
            <> a[idim] = center[idim,icenter] - src[idim,isrc] {id=compute_a}
            <> b[idim] = tgt[idim,itgt_overall] - center[idim,icenter] \
                    {id=compute_b}
            <> rscale = expansion_radii[icenter]
            end
            """

    def get_src_tgt_arguments(self):
        return [
                lp.GlobalArg("src", None,
                    shape=(self.dim, "nsources"), order="C"),
                lp.GlobalArg("tgt", None,
                    shape=(self.dim, "ntargets_total"), order="C"),
                lp.GlobalArg("center", None,
                    shape=(self.dim, "ncenters_total"), order="C"),
                lp.GlobalArg("expansion_radii", None, shape="ncenters_total"),
                lp.GlobalArg("qbx_tgt_numbers", None, shape="ntargets"),
                lp.GlobalArg("qbx_center_numbers", None, shape="ntargets"),
                lp.ValueArg("nsources", np.int32),
                lp.ValueArg("ntargets", np.int32),
                lp.ValueArg("ntargets_total", np.int32),
                lp.ValueArg("ncenters_total", np.int32),
                ]

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("strength_%d" % i, None, shape="nsources", order="C")
                for i in range(self.strength_count)
                ]+[
                lp.GlobalArg("result_%d" % i, None, shape="ntargets_total",
                    order="C")
                for i in range(len(self.kernels))
                ]

    def get_result_store_instructions(self):
        return [
                """
                result_KNLIDX[itgt_overall] = \
                        knl_KNLIDX_scaling*simul_reduce(\
                            sum, isrc, pair_result_KNLIDX)  {inames=itgt}
                """.replace("KNLIDX", str(iknl))
                for iknl in range(len(self.expansions))
                ]

# }}}


# vim: foldmethod=marker
