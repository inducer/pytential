from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner, Michael Tom"

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




#import numpy as np
from pytools import memoize_method

import modepy as mp




class RefElementBase(object):
    def __init__(self, order):
        self.order = order

class SimplexRefElementBase(RefElementBase):
    @memoize_method
    def node_tuples(self):
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam
        return list(gnitstam(self.order, self.dimensions))

    @property
    def nvertices(self):
        return self.dimensions + 1

    @memoize_method
    def unit_nodes(self):
        return mp.get_warp_and_blend_nodes(
                self.dimensions, self.order, self.node_tuples())




class RefInterval(SimplexRefElementBase):
    dimensions = 1

class RefTriangle(SimplexRefElementBase):
    dimensions = 2
