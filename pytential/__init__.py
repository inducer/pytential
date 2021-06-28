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
from pytential.symbolic.execution import GeometryCollection

from pytools import memoize_on_first_arg


def _set_up_logging_from_environment():
    import logging
    import os
    from pytential.log import set_up_logging

    for level_name, level in (
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL)):

        pytential_log_var = os.environ.get(f"PYTENTIAL_LOG_{level_name}")
        if pytential_log_var is not None:
            set_up_logging(pytential_log_var.split(":"), level=level)


def _set_up_errors():
    import warnings
    from pytential.qbx.refinement import RefinerNotConvergedWarning
    warnings.filterwarnings("error", category=RefinerNotConvergedWarning)


_set_up_logging_from_environment()
_set_up_errors()


@memoize_on_first_arg
def _integral_op(discr):
    from pytential import sym, bind
    return bind(discr,
            sym.integral(
                discr.ambient_dim, discr.dim, sym.var("integrand")))


def integral(discr, x):
    return _integral_op(discr)(integrand=x)


@memoize_on_first_arg
def _norm_2_op(discr, num_components):
    from pytential import sym, bind
    if num_components is not None:
        from pymbolic.primitives import make_sym_vector
        v = make_sym_vector("integrand", num_components)
        integrand = sym.real(np.dot(sym.conj(v), v))
    else:
        integrand = sym.abs(sym.var("integrand"))**2

    return bind(discr,
            sym.integral(discr.ambient_dim, discr.dim, integrand))


@memoize_on_first_arg
def _norm_inf_op(discr, num_components):
    from pytential import sym, bind
    if num_components is not None:
        from pymbolic.primitives import make_sym_vector
        v = make_sym_vector("arg", num_components)
        max_arg = sym.abs(v)
    else:
        max_arg = sym.abs(sym.var("arg"))

    return bind(discr, sym.NodeMax(max_arg))


def norm(discr, x, p=2):
    from pymbolic.geometric_algebra import MultiVector
    if isinstance(x, MultiVector):
        x = x.as_vector(object)

    from meshmode.dof_array import DOFArray
    num_components = None
    if (isinstance(x, np.ndarray)
            and x.dtype.char == "O"
            and not isinstance(x, DOFArray)):
        num_components, = x.shape

    if p == 2:
        norm_op = _norm_2_op(discr, num_components)
        return norm_op(integrand=x)**(1/2)

    elif p == np.inf or p == "inf":
        norm_op = _norm_inf_op(discr, num_components)
        norm_res = norm_op(arg=x)
        if isinstance(norm_res, np.ndarray):
            return max(norm_res)
        else:
            return norm_res

    else:
        raise ValueError(f"unsupported norm order: {p}")


__all__ = ["sym", "bind", "GeometryCollection"]
