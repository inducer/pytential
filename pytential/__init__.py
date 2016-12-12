from __future__ import division
from __future__ import absolute_import

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

import pytential.symbolic.primitives as sym
from pytential.symbolic.execution import bind

from pytools import memoize_on_first_arg


@memoize_on_first_arg
def _integral_op(discr):
    from pytential import sym, bind
    return bind(discr,
            sym.integral(
                discr.ambient_dim, discr.dim, sym.var("integrand")))


def integral(discr, queue, x):
    return _integral_op(discr)(queue, integrand=x)


@memoize_on_first_arg
def _norm_op(discr, num_components):
    from pytential import sym, bind
    if num_components is not None:
        from pymbolic.primitives import make_sym_vector
        v = make_sym_vector("integrand", num_components)
        integrand = sym.real(np.dot(sym.conj(v), v))
    else:
        integrand = sym.abs(sym.var("integrand"))**2

    return bind(discr,
            sym.integral(discr.ambient_dim, discr.dim, integrand))


def norm(discr, queue, x):
    from pymbolic.geometric_algebra import MultiVector
    if isinstance(x, MultiVector):
        x = x.as_vector(np.object)

    num_components = None
    if isinstance(x, np.ndarray):
        num_components, = x.shape

    norm_op = _norm_op(discr, num_components)
    from math import sqrt
    return sqrt(norm_op(queue, integrand=x))


__all__ = ["sym", "bind"]
