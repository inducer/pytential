__copyright__ = "Copyright (C) 2022 Isuru Fernando"

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

from pytential.symbolic.pde.system_utils import (
    convert_target_transformation_to_source, rewrite_int_g_using_base_kernel)
from pytential.symbolic.primitives import IntG
from pytential import sym
import pytential
import numpy as np

from sumpy.kernel import (
    LaplaceKernel, HelmholtzKernel, ExpressionKernel, BiharmonicKernel,
    StokesletKernel,
    AxisTargetDerivative, TargetPointMultiplier, AxisSourceDerivative)
from sumpy.symbolic import USE_SYMENGINE

from pymbolic.primitives import make_sym_vector
import pymbolic.primitives as prim


def test_convert_target_deriv():
    knl = LaplaceKernel(2)
    int_g = IntG(AxisTargetDerivative(0, knl), [AxisSourceDerivative(1, knl), knl],
        [1, 2], qbx_forced_limit=1)
    expected_int_g = IntG(knl,
        [AxisSourceDerivative(0, AxisSourceDerivative(1, knl)),
        AxisSourceDerivative(0, knl)], [-1, -2], qbx_forced_limit=1)

    assert sum(convert_target_transformation_to_source(int_g)) == expected_int_g


def test_convert_target_point_multiplier():
    xs = sym.nodes(3).as_vector()

    knl = LaplaceKernel(3)
    int_g = IntG(TargetPointMultiplier(0, knl), [AxisSourceDerivative(1, knl), knl],
        [1, 2], qbx_forced_limit=1)

    d = make_sym_vector("d", 3)

    if USE_SYMENGINE:
        r2 = d[2]**2 + d[1]**2 + d[0]**2
        eknl1 = ExpressionKernel(3, d[1]*d[0]*r2**prim.Quotient(-3, 2),
            knl.global_scaling_const, False)
    else:
        r2 = d[0]**2 + d[1]**2 + d[2]**2
        eknl1 = ExpressionKernel(3, d[0]*d[1]*r2**prim.Quotient(-3, 2),
            knl.global_scaling_const, False)
    eknl2 = ExpressionKernel(3, d[0]*r2**prim.Quotient(-1, 2),
        knl.global_scaling_const, False)
    expected_int_g = IntG(eknl1, [eknl1], [1], qbx_forced_limit=1) + \
        IntG(eknl2, [eknl2], [2], qbx_forced_limit=1) + \
        IntG(knl, [AxisSourceDerivative(1, knl), knl],
        [xs[0], 2*xs[0]], qbx_forced_limit=1)

    assert expected_int_g == sum(convert_target_transformation_to_source(int_g))


def test_product_rule():
    xs = sym.nodes(3).as_vector()

    knl = LaplaceKernel(3)
    int_g = IntG(AxisTargetDerivative(0, TargetPointMultiplier(0, knl)), [knl], [1],
        qbx_forced_limit=1)

    d = make_sym_vector("d", 3)
    if USE_SYMENGINE:
        r2 = d[2]**2 + d[1]**2 + d[0]**2
    else:
        r2 = d[0]**2 + d[1]**2 + d[2]**2
    eknl = ExpressionKernel(3, d[0]**2*r2**prim.Quotient(-3, 2),
        knl.global_scaling_const, False)
    expected_int_g = IntG(eknl, [eknl], [-1], qbx_forced_limit=1) + \
        IntG(knl, [AxisSourceDerivative(0, knl)], [xs[0]*(-1)], qbx_forced_limit=1)

    assert expected_int_g == sum(convert_target_transformation_to_source(int_g))


def test_convert_helmholtz():
    xs = sym.nodes(3).as_vector()

    knl = HelmholtzKernel(3)
    int_g = IntG(TargetPointMultiplier(0, knl), [knl], [1],
        qbx_forced_limit=1, k=1)

    d = make_sym_vector("d", 3)
    exp = prim.Variable("exp")

    if USE_SYMENGINE:
        r2 = d[2]**2 + d[1]**2 + d[0]**2
        eknl = ExpressionKernel(3, exp(1j*r2**prim.Quotient(1, 2))*d[0]
            * r2**prim.Quotient(-1, 2),
            knl.global_scaling_const, knl.is_complex_valued)
    else:
        r2 = d[0]**2 + d[1]**2 + d[2]**2
        eknl = ExpressionKernel(3, d[0]*r2**prim.Quotient(-1, 2)
            * exp(1j*r2**prim.Quotient(1, 2)),
            knl.global_scaling_const, knl.is_complex_valued)

    expected_int_g = IntG(eknl, [eknl], [1], qbx_forced_limit=1,
            kernel_arguments={"k": 1}) + \
        IntG(knl, [knl], [xs[0]], qbx_forced_limit=1, k=1)

    assert expected_int_g == sum(convert_target_transformation_to_source(int_g))


def test_convert_int_g_base():
    knl = LaplaceKernel(3)
    int_g = IntG(knl, [knl], [1], qbx_forced_limit=1)

    base_knl = BiharmonicKernel(3)
    expected_int_g = sum(
        IntG(base_knl, [AxisSourceDerivative(d, AxisSourceDerivative(d, base_knl))],
            [-1], qbx_forced_limit=1) for d in range(3))

    assert expected_int_g == rewrite_int_g_using_base_kernel(int_g,
                                                             base_kernel=base_knl)


def test_convert_int_g_base_with_const():
    knl = StokesletKernel(2, 0, 0)
    int_g = IntG(knl, [knl], [1], qbx_forced_limit=1, mu=2)

    base_knl = BiharmonicKernel(2)
    dim = 2
    dd = pytential.sym.DOFDescriptor(geometry=pytential.sym.DEFAULT_SOURCE)

    expected_int_g = (-0.1875)*prim.Power(np.pi, -1) * \
        pytential.sym.integral(dim, dim-1, 1, dofdesc=dd) + \
        IntG(base_knl,
            [AxisSourceDerivative(1, AxisSourceDerivative(1, base_knl))], [0.5],
            qbx_forced_limit=1)
    assert rewrite_int_g_using_base_kernel(int_g,
                                           base_kernel=base_knl) == expected_int_g


def test_convert_int_g_base_with_const_and_deriv():
    knl = StokesletKernel(2, 0, 0)
    int_g = IntG(knl, [AxisSourceDerivative(0, knl)], [1], qbx_forced_limit=1, mu=2)

    base_knl = BiharmonicKernel(2)

    expected_int_g = IntG(base_knl,
            [AxisSourceDerivative(1, AxisSourceDerivative(1,
                AxisSourceDerivative(0, base_knl)))], [0.5],
            qbx_forced_limit=1)
    assert rewrite_int_g_using_base_kernel(int_g,
                                           base_kernel=base_knl) == expected_int_g
