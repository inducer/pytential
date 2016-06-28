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


class CenterFinder:
    def __init__(self, dim):
        self.dim = dim

    def get_kernel(self):
        knl = lp.make_kernel(
                "{[ictr,itgt,idim]: "
                "0<=itgt<ntargets "
                "and 0<=ictr<ncenters "
                "and 0<=idim<dim}",

                """
                for itgt
                    for ictr
                        <> dist_sq = sum(idim, \
                                (tgt[idim,itgt] - center[idim,ictr])**2)
                        <> in_disk = dist < radius[ictr]**2
                        <> post_dist_sq = if(in_disk, dist_sq, HUGE)
                    end
                    min_dist_sq, min_ictr = argmax(ictr, post_dist_sq)

                    qbx_center[itgt] = if(min_dist_sq < HUGE, min_ictr, -1)
                    tgt_center
                end
                """)

        knl = lp.fix_parameters(knl,
                dim=self.dim,
                HUGE=2**30)

        knl = lp.tag_array_axes(knl, "center", "sep,C")

        return knl

    def get_optimized_kernel(self):
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itgt", 128, outer_tag="g.0")
        return knl

    def __call__(self, queue, tgt, center, radius):
        return self.get_optimized_kernel()(queue, tgt=tgt, center=center, radius=radius)
