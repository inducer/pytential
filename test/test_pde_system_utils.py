from __future__ import annotations


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

from functools import partial

import numpy as np

import pymbolic.primitives as prim
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    BiharmonicKernel,
    ExpressionKernel,
    HelmholtzKernel,
    LaplaceKernel,
    StokesletKernel,
    TargetPointMultiplier,
)
from sumpy.symbolic import USE_SYMENGINE

from pytential import sym
from pytential.symbolic.mappers import flatten
from pytential.symbolic.pde.system_utils import (
    convert_target_transformation_to_source,
    rewrite_int_g_using_base_kernel,
)


class RealExpressionKernel(ExpressionKernel):
    @property
    def is_complex_valued(self) -> bool:
        return False


class ComplexExpressionKernel(ExpressionKernel):
    @property
    def is_complex_valued(self) -> bool:
        return True


def test_convert_target_deriv():
    knl = LaplaceKernel(2)
    dsource = partial(AxisSourceDerivative, inner_kernel=knl)

    int_g = sym.IntG(
        AxisTargetDerivative(0, knl),
        (dsource(1), knl),
        (1, 2), qbx_forced_limit=1)
    expected_int_g = sym.IntG(
        knl,
        (AxisSourceDerivative(0, dsource(1)), dsource(0)),
        (-1, -2), qbx_forced_limit=1)

    expr = sum(convert_target_transformation_to_source(int_g))
    assert flatten(expr) == expected_int_g


def test_convert_target_point_multiplier():
    ambient_dim = 3
    knl = LaplaceKernel(ambient_dim)
    dsource = partial(AxisSourceDerivative, inner_kernel=knl)

    int_g = sym.IntG(
        TargetPointMultiplier(0, knl),
        (dsource(1), knl),
        (1, 2), qbx_forced_limit=1)

    d = sym.make_sym_vector("d", ambient_dim)
    r2 = d[2]**2 + d[1]**2 + d[0]**2

    eknl0 = RealExpressionKernel(ambient_dim,
                                 d[1]*d[0]*r2**prim.Quotient(-3, 2),
                                 knl.global_scaling_const)
    eknl2 = RealExpressionKernel(ambient_dim,
                                 d[0]*r2**prim.Quotient(-1, 2),
                                 knl.global_scaling_const)

    r2 = d[0]**2 + d[1]**2 + d[2]**2

    eknl1 = RealExpressionKernel(ambient_dim,
                                 d[0]*d[1]*r2**prim.Quotient(-3, 2),
                                 knl.global_scaling_const)
    eknl3 = RealExpressionKernel(ambient_dim,
                                 d[0]*r2**prim.Quotient(-1, 2),
                                 knl.global_scaling_const)

    xs = sym.nodes(3).as_vector()

    possible_int_g1 = flatten(
        sym.IntG(eknl0, (eknl0,), (1,), qbx_forced_limit=1)
        + sym.IntG(eknl2, (eknl2,), (2,), qbx_forced_limit=1)
        + sym.IntG(knl, (dsource(1), knl), (xs[0], 2*xs[0]), qbx_forced_limit=1))
    possible_int_g2 = flatten(
        sym.IntG(eknl1, (eknl1,), (1,), qbx_forced_limit=1)
        + sym.IntG(eknl3, (eknl3,), (2,), qbx_forced_limit=1)
        + sym.IntG(knl, (dsource(1), knl), (xs[0], 2*xs[0]), qbx_forced_limit=1))

    expr = flatten(sum(convert_target_transformation_to_source(int_g)))
    assert expr in (possible_int_g1, possible_int_g2)


def test_product_rule():
    ambient_dim = 3
    knl = LaplaceKernel(ambient_dim)
    dsource = partial(AxisSourceDerivative, inner_kernel=knl)

    int_g = sym.IntG(
        AxisTargetDerivative(0, TargetPointMultiplier(0, knl)),
        (knl,), (1,), qbx_forced_limit=1)

    d = sym.make_sym_vector("d", 3)
    r2 = d[2]**2 + d[1]**2 + d[0]**2

    eknl0 = RealExpressionKernel(ambient_dim,
                                 d[0]**2*r2**prim.Quotient(-3, 2),
                                 knl.global_scaling_const)

    r2 = d[0]**2 + d[1]**2 + d[2]**2

    eknl1 = RealExpressionKernel(ambient_dim,
                                 d[0]**2*r2**prim.Quotient(-3, 2),
                                 knl.global_scaling_const)

    xs = sym.nodes(3).as_vector()

    possible_int_g1 = flatten(
        sym.IntG(eknl0, (eknl0,), (-1,), qbx_forced_limit=1)
        + sym.IntG(knl, (dsource(0),), (xs[0]*(-1),), qbx_forced_limit=1)
        )
    possible_int_g2 = flatten(
        sym.IntG(eknl1, (eknl1,), (-1,), qbx_forced_limit=1)
        + sym.IntG(knl, (dsource(0),), (xs[0]*(-1),), qbx_forced_limit=1))

    expr = flatten(sum(convert_target_transformation_to_source(int_g)))
    assert expr in [possible_int_g1, possible_int_g2]


def test_convert_helmholtz():
    ambient_dim = 3
    knl = HelmholtzKernel(ambient_dim)
    int_g = sym.IntG(
        TargetPointMultiplier(0, knl),
        (knl,), (1,), qbx_forced_limit=1, kernel_arguments={"k": 1})

    d = sym.make_sym_vector("d", 3)
    exp = prim.Variable("exp")

    if USE_SYMENGINE:
        r2 = d[2]**2 + d[1]**2 + d[0]**2
        eknl = ComplexExpressionKernel(
            ambient_dim,
            exp(1j*r2**prim.Quotient(1, 2))*d[0] * r2**prim.Quotient(-1, 2),
            knl.global_scaling_const)
    else:
        r2 = d[0]**2 + d[1]**2 + d[2]**2
        eknl = ComplexExpressionKernel(
            ambient_dim,
            d[0]*r2**prim.Quotient(-1, 2) * exp(1j*r2**prim.Quotient(1, 2)),
            knl.global_scaling_const)

    xs = sym.nodes(3).as_vector()
    expected_int_g = flatten(
        sym.IntG(eknl, (eknl,), (1,), qbx_forced_limit=1, kernel_arguments={"k": 1})
        + sym.IntG(knl, (knl,), (xs[0],), qbx_forced_limit=1, kernel_arguments={"k": 1})
    )

    expr = flatten(sum(convert_target_transformation_to_source(int_g)))
    assert expr == expected_int_g


def test_convert_int_g_base():
    ambient_dim = 3
    knl = LaplaceKernel(ambient_dim)
    int_g = sym.IntG(knl, (knl,), (1,), qbx_forced_limit=1)

    base_knl = BiharmonicKernel(ambient_dim)
    expected_int_g = sum(sym.IntG(
        base_knl,
        (AxisSourceDerivative(d, AxisSourceDerivative(d, base_knl)),),
        (-1,), qbx_forced_limit=1)
        for d in range(ambient_dim))

    expr = flatten(rewrite_int_g_using_base_kernel(int_g, base_kernel=base_knl))
    assert expr == flatten(expected_int_g)


def test_convert_int_g_base_with_const():
    ambient_dim = 2
    knl = StokesletKernel(ambient_dim, 0, 0)
    base_knl = BiharmonicKernel(ambient_dim)

    dd = sym.DOFDescriptor(sym.DEFAULT_SOURCE)
    int_g = sym.IntG(knl, (knl,), (1,), qbx_forced_limit=1, kernel_arguments={"mu": 2})

    expected_int_g = flatten(
        (-0.1875) / np.pi * sym.integral(ambient_dim, ambient_dim - 1, 1, dofdesc=dd)
        + sym.IntG(base_knl,
                   (AxisSourceDerivative(1, AxisSourceDerivative(1, base_knl)),),
                   (0.5,), qbx_forced_limit=1))

    expr = flatten(rewrite_int_g_using_base_kernel(int_g, base_kernel=base_knl))
    assert expr == expected_int_g


def test_convert_int_g_base_with_const_and_deriv():
    ambient_dim = 2
    knl = StokesletKernel(ambient_dim, 0, 0)
    base_knl = BiharmonicKernel(ambient_dim)

    int_g = sym.IntG(
        knl,
        (AxisSourceDerivative(0, knl),), (1,), qbx_forced_limit=1,
        kernel_arguments={"mu": 2})

    expected_int_g = sym.IntG(
        base_knl,
        (AxisSourceDerivative(
            1, AxisSourceDerivative(
                1, AxisSourceDerivative(0, base_knl))),),
        (0.5,), qbx_forced_limit=1)

    expr = flatten(rewrite_int_g_using_base_kernel(int_g, base_kernel=base_knl))
    assert expr == expected_int_g
