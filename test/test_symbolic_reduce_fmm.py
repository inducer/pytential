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

import pymbolic.primitives as prim
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

from pytential import sym
from pytential.symbolic.reduce_fmm import (
    GatherAllSourceDependentVariables,
    ReduceFailedError,
    collect,
    get_all_source_dependent_variables,
    reduce_number_of_fmms,
)


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

    result = reduce_number_of_fmms(int_gs)

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

    result = reduce_number_of_fmms(int_gs)

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
    mu = sym.SpatialConstant("mu")
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

    result = reduce_number_of_fmms(int_gs)

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

    wide_result = reduce_number_of_fmms(wide_int_gs)
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

    tall_result = reduce_number_of_fmms(tall_int_gs)
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
    k = sym.SpatialConstant("k")
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
        reduce_number_of_fmms(int_gs)

# }}}


# {{{ test_coefficient_collector

def test_coefficient_collector() -> None:
    s = sym.var("s")
    a, b, c, d, e, f = (sym.var(name) for name in "abcdef")
    n = sym.var("n")
    k = sym.SpatialConstant("k")
    mu = sym.SpatialConstant("mu")
    nu = sym.SpatialConstant("nu")

    # simple distributed product
    terms = collect(s * (s + 2) + 3, [s])
    assert terms == {s * s: 1, s: 2, 1: 3}

    # products of more than two sums
    terms = collect((a + b) * (c + d) * (e + f), [a, b, c, d, e, f])
    assert terms == {
        # NOTE: a * c * e isn't flattened, this looks a bit nasty TBH
        prim.Product((a, c, e)): 1, prim.Product((a, c, f)): 1,
        prim.Product((a, d, e)): 1, prim.Product((a, d, f)): 1,
        prim.Product((b, c, e)): 1, prim.Product((b, c, f)): 1,
        prim.Product((b, d, e)): 1, prim.Product((b, d, f)): 1,
    }

    # a sum of monomials is collected term by term
    terms = collect(a * b + a * c + b * c, [a, b, c])
    assert terms == {a * b: 1, a * c: 1, b * c: 1}

    # product factor ordering is canonicalized
    assert collect(b * a, [a, b]) == {a * b: 1}
    assert collect(c * (a * b), [a, b, c]) == {prim.Product((a, b, c)): 1}
    assert collect(a * b + b * a, [a, b]) == {a * b: 2}

    # powers
    assert collect(a**3, [a]) == {a**3: 1}
    assert collect(a**n, [a]) == {a**n: 1}

    with pytest.raises(NotImplementedError):
        collect((a + b)**2, [a, b])

    with pytest.raises(RuntimeError):
        collect(a**n, [a, n])

    # quotients by source-dependent expressions are not supported
    with pytest.raises(RuntimeError):
        collect((a + b) / c, [a, b, c])

    with pytest.raises(RuntimeError):
        collect((a + b) / (c * c), [a, b, c])

    with pytest.raises(RuntimeError):
        collect((a + b) / (c + d), [a, b, c, d])

    # quotients with source-independent denominators
    terms = collect((a + b) / mu, [a, b])
    assert terms == {a: 1 / mu, b: 1 / mu}
    terms = collect((a + b) / (mu + nu), [a, b])
    assert terms == {a: 1 / (mu + nu), b: 1 / (mu + nu)}

    # constants
    assert collect(0, []) == {}
    assert collect(3, []) == {1: 3}
    assert collect(k, []) == {1: k}

    # coefficients can be spatial constants
    sigma = sym.var("sigma")
    assert collect(-2 * k**2 * sigma, [sigma]) == {sigma: -2 * k**2}
    assert collect(-1 / mu * sigma, [sigma]) == {sigma: -1 / mu}
    assert collect(sigma / mu, [sigma]) == {sigma: 1 / mu}

# }}}


# {{{ test_coefficient_collector_basis

def test_coefficient_collector_basis() -> None:
    a, b, c = sym.var("a"), sym.var("b"), sym.var("c")
    sigma = sym.var("sigma")

    # a source expression can itself be compound
    assert collect((a + b) * c, [a + b]) == {a + b: c}

    # subscripts can be in the basis directly or through their aggregate
    sub = prim.Subscript(sigma, 0)
    assert collect(sub, [sub]) == {sub: 1}
    assert collect(sub, [sigma]) == {sub: 1}
    assert collect(sub, []) == {1: sub}

    # common subexpressions are atomic
    cse = prim.CommonSubexpression(a + b)
    assert collect(cse * a, [a, b]) == {cse * a: 1}

# }}}


# {{{ test_coefficient_collector_elasticity_densities

def test_coefficient_collector_elasticity_densities() -> None:
    dim = 2

    q = sym.make_sym_vector("q", dim)
    normal = sym.make_sym_vector("normal", dim)
    source = [
        sym.NodeCoordinateComponent(i, sym.as_dofdesc("src"))
        for i in range(dim)
    ]

    mu = sym.SpatialConstant("mu")
    nu = sym.SpatialConstant("nu")
    w = sym.SpatialConstant("w")

    basis = [sym.var("q"), sym.var("normal"), *source]

    # stresslet-style: w / 6 * (sigma[k] * normal[j] + sigma[j] * normal[k])
    expr = w / 6 * (q[0] * normal[1] + q[1] * normal[0])
    assert collect(expr, basis) == {
        normal[1] * q[0]: w / 6,
        normal[0] * q[1]: w / 6,
    }

    # Tornberg-style: products of normal, density and source coordinates
    expr = (
        mu * normal[0] * q[1] * source[0]
        + (mu + nu) * normal[1] * q[0] * source[1]
    )
    assert collect(expr, basis) == {
        prim.Product((normal[0], q[1], source[0])): mu,
        prim.Product((normal[1], q[0], source[1])): mu + nu,
    }

    # common-density products: (sum_k source[k] * sigma[k]) * normal[j]
    expr = (source[0] * q[0] + source[1] * q[1]) * normal[0]
    assert collect(expr, basis) == {
        prim.Product((normal[0], q[0], source[0])): 1,
        prim.Product((normal[0], q[1], source[1])): 1,
    }

    # stokeslet-style: w * source[k] * sigma[k]
    assert collect(w * source[0] * q[0], basis) == {q[0] * source[0]: w}

    # CSE densities (e.g. `sym.normal` components or Hsiao-Kress' meanless
    # sigma) are treated as a single source-dependent expression
    cse = prim.CommonSubexpression(normal[0] + normal[1])
    assert collect(cse * q[0], basis) == {cse * q[0]: 1}

# }}}


# {{{ test_gather_all_source_dependent_variables

def test_gather_all_source_dependent_variables() -> None:
    a, b = sym.var("a"), sym.var("b")
    sigma = sym.var("sigma")
    tau = sym.var("tau")
    k = sym.SpatialConstant("k")

    # plain variables are collected, SpatialConstants are not
    mapper = GatherAllSourceDependentVariables()
    mapper(sigma + k * tau + 1)
    assert set(mapper.vars) == {sigma, tau}

    # discretization-like nodes are collected as a single expression
    src = sym.as_dofdesc("src")
    tgt = sym.as_dofdesc("tgt")
    nodes = [
        sym.NodeCoordinateComponent(0, src),
        sym.num_reference_derivative(sigma + tau, 0, src),
        sym.Interpolation(src, tgt, sigma + tau),
        prim.CommonSubexpression(sigma + tau),
    ]
    for node in nodes:
        mapper = GatherAllSourceDependentVariables()
        mapper(node)
        assert set(mapper.vars) == {node}

    # subscripts and powers are traversed
    mapper = GatherAllSourceDependentVariables()
    mapper(prim.Subscript(sigma, 0) + a**b)
    assert set(mapper.vars) == {sigma, a, b}

    # IntGs are collected and their densities are traversed
    knl = LaplaceKernel(2)
    inner = sym.IntG(
        target_kernel=knl,
        source_kernels=(knl,),
        densities=(tau,),
        qbx_forced_limit=1,
    )
    outer = sym.IntG(
        target_kernel=knl,
        source_kernels=(knl,),
        densities=(sigma + inner,),
        qbx_forced_limit=1,
    )

    mapper = GatherAllSourceDependentVariables()
    mapper(outer)
    assert set(mapper.vars) == {outer, inner, sigma, tau}

    assert set(get_all_source_dependent_variables([outer])) == {inner, sigma, tau}
    assert get_all_source_dependent_variables([]) == ()

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
