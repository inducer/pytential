__copyright__ = "Copyright (C) 2022 University of Illinois Board of Trustees"

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


from typing import Tuple

import numpy as np
from pytools.obj_array import make_obj_array           # noqa: F401
from pytential import bind, sym, norm
from sumpy.kernel import HelmholtzKernel, AxisSourceDerivative

from meshmode import _acf
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])

def dyadic_layer_pot(helm_k, vec, *,
                           transpose: bool, source_curl: bool = False) -> np.ndarray:
    k_sym = HelmholtzKernel(3)

    if transpose:
        outer = lambda i, j: j
        inner = lambda i, j: i
    else:
        outer = lambda i, j: i
        inner = lambda i, j: j

    def helmholtz_vec(source_derivative_axes: Tuple[int, ...]):
        knl = k_sym
        for axis in source_derivative_axes:
            knl = AxisSourceDerivative(axis, knl)

        return sym.int_g_vec(knl, vec, qbx_forced_limit=None,
                   kernel_arguments={"k": helm_k})

    from pytools import levi_civita
    # FIXME Could/should use Schwarz's theorem to optimize, but probably
    # only off-surface where potential is smooth.
    if source_curl:
        return make_obj_array([
            sum(
                levi_civita((ell, m, n))
                * (helmholtz_vec((m,))[n]
                + 1/helm_k**2 * sum(
                    helmholtz_vec((m, outer(n, j), inner(n, j)))[j]
                    for j in range(3)))
                for m in range(3)
                for n in range(3))
            for ell in range(3)
            ])
    else:
        return make_obj_array([
            helmholtz_vec(())[i]
            + 1/helm_k**2 * sum(
                helmholtz_vec((outer(i, j), inner(i, j)))[j]
                for j in range(3))
            for i in range(3)])


def test_dyadic_green(actx_factory):
    actx = actx_factory()

    helm_k = sym.var("k")
    trace_e = sym.make_sym_vector("trace_e", 3)
    trace_curl_e = sym.make_sym_vector("trace_curl_e", 3)

    normal = sym.normal(3).as_vector()

    nxe = sym.cross(normal, trace_e)
    nxcurl_e = sym.cross(normal, trace_curl_e)

    # Monk, Finite element methods for Maxwell's equations (2003)
    # https://doi.org/10.1093/acprof:oso/9780198508885.001.0001
    # Theorem 12.2.
    zero_op = (dyadic_layer_pot(helm_k, nxe, transpose=True)
               + dyadic_layer_pot(helm_k, nxcurl_e, transpose=True, source_curl=True)
               - trace_e)

    print(sym.pretty(zero_op))



# You can test individual routines by typing
# $ python test_maxwell_green.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
