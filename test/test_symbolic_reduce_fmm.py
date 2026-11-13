from __future__ import annotations


__copyright__ = """
Copyright (C) 2021 Isuru Fernando
Copyright (C) 2026 Alexandru Fikl
"""

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

import pytest
from constantdict import constantdict

from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    BiharmonicKernel,
    HelmholtzKernel,
    LaplaceKernel,
    ScalarKernel,
    StokesletComponentKernel,
    YukawaKernel,
)
from sumpy.kernel_rewrite import rewrite_using_base_kernel
from sumpy.symbolic import SpatialConstant

from pytential import sym
from pytential.symbolic.reduce_fmm import ReduceFailedError, reduce_number_of_fmms


# {{{ test_paper_reduce_example

def test_paper_reduce_example() -> None:
    import sympy as sp

    from pytential.symbolic.reduce_fmm import _make_syzygy_factorization

    y = y1, y2, y3 = sp.symbols("y1, y2, y3")
    m = sp.Matrix([[y1 * y2, -2*y1**2-2*y3**2], [y1*y3, 2*y2*y3]])
    poly_ring = sp.EX.old_poly_ring(*y)
    ring = poly_ring / [y1**2 + y2**2 + y3**2]

    lhs, _rhs = _make_syzygy_factorization(m, ring)
    assert lhs == sp.Matrix([[y2], [y3]])

    # When the PDE is not taken into account, rank is 2
    lhs, _rhs = _make_syzygy_factorization(m, poly_ring)
    assert lhs == m

# }}}


# {{{ test_reduce_number_of_fmms_scalar_pde

@pytest.mark.parametrize("kernel_name", ["laplace", "helmholtz", "yukawa"])
def test_reduce_number_of_fmms_scalar_pde(kernel_name: str) -> None:
    # The matrix for this system is
    #
    #   [[y0*y1, -2*y0**2 - 2*y2**2 + c],
    #    [y0*y2,  2*y1*y2]]
    #
    # where c = 0 (Laplace), -2*k**2 (Helmholtz), or 2*lam**2 (Yukawa).
    # It has rank 2 as a polynomial matrix, but rank 1 modulo the scalar PDE
    # (y0**2 + y1**2 + y2**2 + ... = 0).
    #
    # The reduction should therefore produce one source FMM and two target derivatives.

    dim = 3
    sigma0 = sym.var("sigma0")
    sigma1 = sym.var("sigma1")

    if kernel_name == "laplace":
        knl = LaplaceKernel(dim)
        kernel_arguments = constantdict({})
        extra_source_kernels = []
        extra_densities = []
    elif kernel_name == "helmholtz":
        knl = HelmholtzKernel(dim)
        k = sym.SpatialConstant("k")
        kernel_arguments = constantdict({"k": k})
        extra_source_kernels = [knl]
        extra_densities = [-2 * k**2 * sigma1]
    elif kernel_name == "yukawa":
        knl = YukawaKernel(dim)
        lam = sym.SpatialConstant("lam")
        kernel_arguments = constantdict({"lam": lam})
        extra_source_kernels = [knl]
        extra_densities = [2 * lam**2 * sigma1]
    else:
        raise ValueError(f"unknown kernel type: {kernel_name!r}")

    def dsource(*axes: int) -> AxisSourceDerivative:
        kernel = knl
        for axis in axes:
            kernel = AxisSourceDerivative(axis, kernel)

        return kernel

    int_gs = [
        sym.IntG(
            target_kernel=knl,
            source_kernels=(
                dsource(0, 1), dsource(0, 0), dsource(2, 2),
                *extra_source_kernels,
            ),
            densities=(sigma0, -2 * sigma1, -2 * sigma1, *extra_densities),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 2), dsource(1, 2)),
            densities=(sigma0, 2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
    ]

    result = reduce_number_of_fmms(
        int_gs, source_dependent_variables=[sigma0, sigma1]
    )

    expected = (
        sym.IntG(
            target_kernel=AxisTargetDerivative(1, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
        sym.IntG(
            target_kernel=AxisTargetDerivative(2, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
    )

    assert result == expected

# }}}


# {{{ test_reduce_number_of_fmms_polynomial

@pytest.mark.parametrize("kernel_name", ["laplace", "helmholtz", "yukawa"])
def test_reduce_number_of_fmms_polynomial(kernel_name: str) -> None:
    # The matrix for this system is
    #
    #   [[y0*y1, 2*y1**2],
    #    [y0*y2, 2*y1*y2]]
    #
    # It already has rank 1 as a polynomial matrix (without using the PDE
    # quotient), factoring as [[y1], [y2]] * [[y0, 2*y1]].

    dim = 3
    sigma0 = sym.var("sigma0")
    sigma1 = sym.var("sigma1")

    if kernel_name == "laplace":
        knl = LaplaceKernel(dim)
        kernel_arguments = constantdict()
    elif kernel_name == "helmholtz":
        knl = HelmholtzKernel(dim)
        kernel_arguments = constantdict({"k": sym.SpatialConstant("k")})
    elif kernel_name == "yukawa":
        knl = YukawaKernel(dim)
        kernel_arguments = constantdict({"lam": sym.SpatialConstant("lam")})
    else:
        raise ValueError(f"unknown kernel type: {kernel_name!r}")

    def dsource(*axes: int) -> AxisSourceDerivative:
        kernel = knl
        for axis in axes:
            kernel = AxisSourceDerivative(axis, kernel)
        return kernel

    int_gs = [
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 1), dsource(1, 1)),
            densities=(sigma0, 2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 2), dsource(1, 2)),
            densities=(sigma0, 2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
    ]

    result = reduce_number_of_fmms(
        int_gs, source_dependent_variables=[sigma0, sigma1]
    )

    expected = (
        sym.IntG(
            target_kernel=AxisTargetDerivative(1, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
        sym.IntG(
            target_kernel=AxisTargetDerivative(2, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=kernel_arguments),
    )

    assert result == expected

# }}}


# {{{ test_reduce_number_of_fmms_stokeslet

def test_reduce_number_of_fmms_stokeslet() -> None:
    dim = 2
    bh = BiharmonicKernel(dim)
    mu = SpatialConstant("mu")
    sigma = [sym.var(f"sigma{j}") for j in range(dim)]

    def dsource(*axes: int) -> AxisSourceDerivative:
        kernel = bh
        for axis in axes:
            kernel = AxisSourceDerivative(axis, kernel)
        return kernel

    def dsource_mi(mi: tuple[int, ...]) -> AxisSourceDerivative:
        axes = [axis for axis, count in enumerate(mi) for _ in range(count)]
        return dsource(*axes)

    int_gs: list[sym.IntG] = []
    for i in range(dim):
        source_kernels: list[ScalarKernel] = []
        densities: list[sym.ArithmeticExpression] = []

        for j in range(dim):
            rep = rewrite_using_base_kernel(StokesletComponentKernel(dim, i, j), bh)
            for mi, coeff in zip(rep.mis, rep.coeffs, strict=True):
                source_kernels.append(dsource_mi(mi))
                densities.append(coeff * sigma[j])

        int_gs.append(
            sym.IntG(
                target_kernel=bh,
                source_kernels=tuple(source_kernels),
                densities=tuple(densities),
                qbx_forced_limit=1,
            )
        )

    result = reduce_number_of_fmms(int_gs, source_dependent_variables=sigma)

    expected = (
        (-1 / mu) * sym.IntG(
            target_kernel=AxisTargetDerivative(1, bh),
            source_kernels=(dsource(1), dsource(0)),
            densities=(sigma[0], -sigma[1]),
            qbx_forced_limit=1,
        ),
        (1 / mu) * sym.IntG(
            target_kernel=AxisTargetDerivative(0, bh),
            source_kernels=(dsource(1), dsource(0)),
            densities=(sigma[0], -sigma[1]),
            qbx_forced_limit=1,
        ),
    )

    assert result == expected

# }}}


# {{{ test_reduce_number_of_fmms_rectangular

def test_reduce_number_of_fmms_rectangular() -> None:
    dim = 3
    knl = LaplaceKernel(dim)
    sigma0 = sym.var("sigma0")
    sigma1 = sym.var("sigma1")
    sigma2 = sym.var("sigma2")

    def dsource(*axes: int) -> AxisSourceDerivative:
        kernel = knl
        for axis in axes:
            kernel = AxisSourceDerivative(axis, kernel)
        return kernel

    # Wide matrix (m=2 < n=3):
    #
    #   [[y0**2, 2*y0*y1, 3*y0*y2],
    #    [y1*y0, 2*y1**2, 3*y1*y2]]
    #
    # factors as [[y0], [y1]] * [[y0, 2*y1, 3*y2]].

    wide_int_gs = [
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 0), dsource(0, 1), dsource(0, 2)),
            densities=(sigma0, 2 * sigma1, 3 * sigma2),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(1, 0), dsource(1, 1), dsource(1, 2)),
            densities=(sigma0, 2 * sigma1, 3 * sigma2),
            qbx_forced_limit=1,
        ),
    ]

    wide_result = reduce_number_of_fmms(
        wide_int_gs, source_dependent_variables=[sigma0, sigma1, sigma2]
    )
    wide_expected = (
        sym.IntG(
            target_kernel=AxisTargetDerivative(0, knl),
            source_kernels=(dsource(0), dsource(1), dsource(2)),
            densities=(-sigma0, -2 * sigma1, -3 * sigma2),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=AxisTargetDerivative(1, knl),
            source_kernels=(dsource(0), dsource(1), dsource(2)),
            densities=(-sigma0, -2 * sigma1, -3 * sigma2),
            qbx_forced_limit=1,
        ),
    )
    assert wide_result == wide_expected

    # Tall matrix (m=3 > n=2) modulo the Laplace PDE:
    #
    #   [[y0*y1, -2*y0**2 - 2*y2**2],
    #    [y0*y2,  2*y1*y2],
    #    [y0**2,  2*y0*y1]]
    #
    # factors as [[y1], [y2], [y0]] * [[y0, 2*y1]].

    tall_int_gs = [
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 1), dsource(0, 0), dsource(2, 2)),
            densities=(sigma0, -2 * sigma1, -2 * sigma1),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 2), dsource(1, 2)),
            densities=(sigma0, 2 * sigma1),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 0), dsource(0, 1)),
            densities=(sigma0, 2 * sigma1),
            qbx_forced_limit=1,
        ),
    ]

    tall_result = reduce_number_of_fmms(
        tall_int_gs, source_dependent_variables=[sigma0, sigma1]
    )
    tall_expected = (
        sym.IntG(
            target_kernel=AxisTargetDerivative(1, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=AxisTargetDerivative(2, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=AxisTargetDerivative(0, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
        ),
    )
    assert tall_result == tall_expected

# }}}


# {{{ test_reduce_number_of_fmms_auto_source_vars

def test_reduce_number_of_fmms_auto_source_vars() -> None:
    dim = 3
    knl = HelmholtzKernel(dim)
    k = SpatialConstant("k")
    sigma0 = sym.var("sigma0")
    sigma1 = sym.var("sigma1")

    def dsource(*axes: int) -> AxisSourceDerivative:
        kernel = knl
        for axis in axes:
            kernel = AxisSourceDerivative(axis, kernel)
        return kernel

    int_gs = [
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 1), dsource(0, 0), dsource(2, 2), knl),
            densities=(sigma0, -2 * sigma1, -2 * sigma1, -2 * k**2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=constantdict({"k": k}),
        ),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0, 2), dsource(1, 2)),
            densities=(sigma0, 2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=constantdict({"k": k}),
        ),
    ]

    result = reduce_number_of_fmms(int_gs)
    expected = (
        sym.IntG(
            target_kernel=AxisTargetDerivative(1, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=constantdict({"k": k}),
        ),
        sym.IntG(
            target_kernel=AxisTargetDerivative(2, knl),
            source_kernels=(dsource(0), dsource(1)),
            densities=(-sigma0, -2 * sigma1),
            qbx_forced_limit=1,
            kernel_arguments=constantdict({"k": k}),
        ),
    )

    assert result == expected

# }}}


# {{{ test_reduce_number_of_fmms_failed

def test_reduce_number_of_fmms_failed() -> None:
    # A 4x3 system modulo the Laplace PDE whose syzygy module has k = 4 > n = 3
    # minimal generators:
    #
    #   [[y1**2,   -y0,     0],
    #    [y2**2,     0,   -y0],
    #    [   y0,     1,     1],
    #    [    0, -y2**2, y1**2]]

    dim = 3
    knl = LaplaceKernel(dim)
    sigma0 = sym.var("sigma0")
    sigma1 = sym.var("sigma1")
    sigma2 = sym.var("sigma2")

    def dsource(*axes: int) -> AxisSourceDerivative:
        kernel = knl
        for axis in axes:
            kernel = AxisSourceDerivative(axis, kernel)
        return kernel

    int_gs = [
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(1, 1), dsource(0)),
            densities=(sigma0, sigma1),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(2, 2), dsource(0)),
            densities=(sigma0, sigma2),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(0), knl, knl),
            densities=(-sigma0, sigma1, sigma2),
            qbx_forced_limit=1,
        ),
        sym.IntG(
            target_kernel=knl,
            source_kernels=(dsource(2, 2), dsource(1, 1)),
            densities=(-sigma1, sigma2),
            qbx_forced_limit=1,
        ),
    ]

    with pytest.raises(ReduceFailedError):
        reduce_number_of_fmms(
            int_gs, source_dependent_variables=[sigma0, sigma1, sigma2]
        )

# }}}


if __name__ == "__main__":
    import sys

    from pytential.array_context import _acf  # ruff:ignore[unused-import]

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
