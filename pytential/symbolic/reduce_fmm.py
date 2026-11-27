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

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, TypeAlias, cast

import sympy as sp
from typing_extensions import override

import pymbolic.primitives as prim
from pymbolic.geometric_algebra.mapper import WalkMapper
from pymbolic.mapper import Mapper
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    KernelWrapper,
    ScalarKernel,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set as AbstractSet

    from sympy.polys.agca.modules import SubModule
    from sympy.polys.domains.quotientring import QuotientRingElement
    from sympy.polys.domains.ring import Ring
    from sympy.polys.polyclasses import DMP

    from pymbolic.typing import ArithmeticExpression, Expression

    from pytential.symbolic.primitives import (
        IntG,
        NodeCoordinateComponent,
        NumReferenceDerivative,
    )

logger = logging.getLogger(__name__)


__all__ = (
    "IncompatibleIntGError",
    "ReduceFailedError",
    "reduce_number_of_fmms",
)

__doc__ = """
.. autofunction:: reduce_number_of_fmms
.. autoexception:: ReduceFailedError
.. autoexception:: IncompatibleIntGError
"""


# {{{ Reduce number of FMMs - main routine


class ReduceFailedError(RuntimeError):
    """Exception raised when the number of FMMs could not be reduced."""


class IncompatibleIntGError(ReduceFailedError):
    """Exception raised when FMMs could not be reduced due to incompatible inputs."""


def reduce_number_of_fmms(int_gs: Sequence[IntG]) -> Sequence[ArithmeticExpression]:
    r"""
    Reduce the number of FMMs needed for a system of
    :class:`~pytential.symbolic.primitives.IntG` objects.

    This is done by converting the ``IntG`` objects to a matrix of polynomials
    :math:`M` in :math:`d` variables :math:`x_0, \dots, x_{d - 1}`, where each
    variable represents a target derivative operator along one of the axes.
    Source derivatives on the kernel are mapped to target derivatives using
    translation invariance, i.e. :math:`\partial_{y_j} G(x - y) =
    -\partial_{x_j} G(x - y)`.

    All the properties of the derivative operator that we want are reflected in
    the properties of polynomials: addition, multiplication, and
    exact polynomial division.

    The matrix is factored as :math:`M = L R` over the polynomial ring, where
    :math:`L` represents target derivatives applied to :math:`k` intermediate
    ``IntG`` potentials, and :math:`R` represents source derivatives and
    source-dependent density combinations. When :math:`k < n` (with :math:`n`
    the number of source-dependent density expressions), fewer FMM evaluations
    are required.

    All variables in the densities, other than
    :class:`~sumpy.symbolic.SpatialConstant`, are assumed to be source-dependent.

    :raises IncompatibleIntGError: if the input ``IntG`` objects do not share
        the same kernel or kernel arguments.
    :raises ReduceFailedError: if no reducing factorization (:math:`k < n`)
        could be found.
    """
    # {{{ check inputs

    # NOTE: might want to refactor this into its own function?

    if not int_gs:
        return int_gs

    ref_int_g = int_gs[0]
    target_kernel = ref_int_g.target_kernel
    dim = target_kernel.dim
    base_kernel = target_kernel.get_base_kernel()

    if not all(int_g.target_kernel.dim == dim for int_g in int_gs):
        raise IncompatibleIntGError(f"all IntGs must have the same dimension: {dim}")

    if not base_kernel.is_translation_invariant:
        raise IncompatibleIntGError(
            f"kernel must be translation invariant: {base_kernel!r}"
        )

    kernel_arguments = dict(ref_int_g.kernel_arguments)
    for i, int_g in enumerate(int_gs[1:], start=1):
        # NOTE: in each IntG, source_kernels are guaranteed to have the same
        # base_kernel as the target_kernel, so we don't need to check more
        if int_g.target_kernel != target_kernel:
            raise IncompatibleIntGError(
                f"IntG[{i}] has a mismatched 'target_kernel': "
                f"{int_g.target_kernel!r} (expected {target_kernel!r})"
            )

        if ref_int_g.qbx_forced_limit != int_g.qbx_forced_limit:
            raise IncompatibleIntGError(
                f"IntG[{i}] has a mismatched 'qbx_forced_limit': "
                f"{int_g.qbx_forced_limit!r} (expected {ref_int_g.qbx_forced_limit!r})"
            )

        if ref_int_g.source != int_g.source:
            raise IncompatibleIntGError(
                f"IntG[{i}] has a mismatched 'source' geometry: "
                f"{int_g.source!r} (expected {ref_int_g.source!r})"
            )

        # NOTE: check that all common kernel arguments agree
        for k, v in int_g.kernel_arguments.items():
            if (ref := kernel_arguments.get(k)) is not None and v != ref:
                raise IncompatibleIntGError(
                    f"IntG[{i}] has a mismatched kernel argument '{k}': "
                    f"{v!r} (expected {ref!r})"
                )
            else:
                kernel_arguments[k] = v

    # }}}

    # gather variables that make up the columns of the matrix
    source_dependent_variables = get_all_source_dependent_variables(int_gs)

    # convert to a matrix
    x = cast("tuple[sp.Symbol, ...]", sp.symbols(f"_x0:{dim}"))
    mat, source_exprs = make_int_g_poly_matrix(
        int_gs, source_dependent_variables, x
    )
    mat = cast("sp.Matrix", sp.nsimplify(mat))

    # create polynomial ring
    poly_ring = sp.EX.old_poly_ring(*x, order=sp.grevlex)

    # quotient out the kernel PDE, so the factorization knows those terms are zero
    try:
        pde_poly = get_kernel_pde_as_poly(base_kernel, x)
    except NotImplementedError:
        pde_poly = None

    if pde_poly:
        ring = poly_ring / [pde_poly]
    else:
        ring = poly_ring

    # factor the matrix into two
    try:
        left_factor, right_factor = _make_syzygy_factorization(mat, ring)
    except ValueError:
        raise ReduceFailedError(
            "could not find a factorization for the given kernels"
        ) from None

    # If there are n inputs and m outputs,
    #
    #  - matrix: R^{m x n},
    #  - LHS: R^{m x k},
    #  - RHS: R^{k x n}.
    #
    # If k is greater than or equal to n, we are gaining nothing
    m, k = left_factor.shape
    _, n = right_factor.shape

    # NOTE: this can happen if there is nothing to factorize in the int_gs
    if k == n:
        return tuple(int_gs)

    # NOTE: this can happen because the syzygy factorization algorithm is not
    # optimal, so it can sometimes find factorizations with larger k
    if k > n:
        raise ReduceFailedError(
            "could not find a factorization that reduces the number of FMMs"
        )

    # create a base IntG on which we will apply source/target derivatives
    base_int_g = replace(ref_int_g,
        target_kernel=base_kernel,
        source_kernels=(base_kernel,),
        densities=(1,)
    )

    # convert polynomials back to IntGs with source derivatives
    source_int_gs = [
        [
            _convert_source_poly_to_int_g(as_poly(expr, x), base_int_g)
            for expr in row
        ] for row in right_factor.tolist()
    ]

    # for each row in the right factor, merge the IntGs to get a total of k IntGs.
    summed_source_int_gs: list[IntG] = []
    for i in range(k):
        source_kernels: list[ScalarKernel] = []
        densities: list[ArithmeticExpression] = []

        for j in range(n):
            source_int_g = source_int_gs[i][j]
            if source_int_g is None:
                continue

            source_kernels.extend(source_int_g.source_kernels)
            densities.extend(d * source_exprs[j] for d in source_int_g.densities)

        summed_source_int_gs.append(
            replace(
                base_int_g,
                source_kernels=tuple(source_kernels),
                densities=tuple(densities)
            )
        )

    # Now that we have the IntG expressions depending on the source, we have
    # to attach the target-dependent derivatives.
    from pytential.symbolic.mappers import flatten

    result: list[ArithmeticExpression] = []
    for i in range(m):
        result_i = 0
        for j in range(k):
            result_i += _convert_target_poly_to_int_g(
                left_factor[i, j].as_poly(*x, domain=sp.EX),
                int_gs[i],
                summed_source_int_gs[j]
            )

        result.append(flatten(result_i))

    return tuple(result)


def as_poly(expr: sp.Expr, x: tuple[sp.Symbol, ...]) -> sp.Poly | None:
    res = expr.as_poly(*x, domain=sp.EX)
    if res is None:
        return expr.simplify().as_poly(*x, domain=sp.EX)
    else:
        return res


def get_kernel_pde_as_poly(kernel: ScalarKernel, x: Sequence[sp.Symbol]) -> sp.Expr:
    pde = kernel.get_pde_as_diff_op()
    if len(pde.eqs) > 1:
        raise NotImplementedError

    from sumpy.tools import mi_power
    return sum((coeff * mi_power(x, d.mi) for d, coeff in pde.eqs[0].items()),
               sp.sympify(0))

# }}}


# {{{ get_all_source_dependent_variables

class GatherAllSourceDependentVariables(WalkMapper[[]]):
    vars: dict[ArithmeticExpression, bool]

    def __init__(self) -> None:
        self.vars = {}

    @override
    def map_variable(self, expr: prim.Variable, /) -> None:
        from sumpy.symbolic import SpatialConstant
        if not isinstance(expr, SpatialConstant):
            self.vars[expr] = True

    @override
    def map_list(self, exprs: list[Expression], /) -> None:
        for expr in exprs:
            self.rec(expr)

    def map_int_g(self, expr: IntG, /) -> None:
        self.vars[expr] = True
        for density in expr.densities:
            self.rec(density)

    def map_node_coordinate_component(self, expr: NodeCoordinateComponent) -> None:
        self.vars[expr] = True

    map_num_reference_derivative = map_node_coordinate_component
    map_interpolation = map_node_coordinate_component
    map_common_subexpression = map_node_coordinate_component


def get_all_source_dependent_variables(
        int_gs: Sequence[IntG]
    ) -> Sequence[ArithmeticExpression]:
    # NOTE: kernel_arguments are ignored here. We're obviously interested
    # in things like the normal in the double layer. This assumes that these
    # have been turned into a more explicit expression in terms of
    # axial derivatives previously.
    #
    # Code in the caller makes sure that kernel arguments match across
    # all IntGs involved, so ignoring them is safe no matter what.

    mapper = GatherAllSourceDependentVariables()
    for int_g in int_gs:
        for density in int_g.densities:
            mapper(density)

    return tuple(mapper.vars)


# }}}


# {{{ convert IntG expressions to a matrix

# NOTE: this is not a full "polynomial coefficient collector", but it should
# be enough for what this algorithm needs: sympy will reduce the remaining
# variables itself and then we clean up the resulting M = L R factorization.

PolyCoeffMapping: TypeAlias = "Mapping[ArithmeticExpression, ArithmeticExpression]"


class CoefficientCollector(Mapper[PolyCoeffMapping, []]):
    """Extract expressions that need to be evaluated for each source and their
    coefficients from a density expression.

    For example, when this mapper is given ``s*(s + 2) + 3`` as input,
    it returns ``{s*s: 1, s: 2, 1: 3}``.

    This is more general than
    :class:`pymbolic.mapper.coefficient.CoefficientCollector` as that deals
    only with linear expressions, but this collector works for polynomial
    expressions too.
    """

    basis: AbstractSet[ArithmeticExpression]

    def __init__(self, basis: Sequence[ArithmeticExpression]) -> None:
        from pytential.symbolic.mappers import flatten

        # NOTE: try to avoid simple cases like `1 * x` or `x * 1` not matching
        self.basis = {flatten(b) for b in basis}

    @override
    def __call__(self, expr: Expression, /) -> PolyCoeffMapping:
        if prim.is_arithmetic_expression(expr) and expr in self.basis:
            return {expr: 1}

        return super().__call__(expr)

    @override
    def rec(self, expr: Expression, /) -> PolyCoeffMapping:
        return self(expr)

    @override
    def map_algebraic_leaf(self, expr: prim.AlgebraicLeaf, /) -> PolyCoeffMapping:
        if expr in self.basis:
            return {expr: 1}
        else:
            return {1: expr}

    def map_common_subexpression(
            self, expr: prim.CommonSubexpression, /
        ) -> PolyCoeffMapping:
        return {expr: 1}

    @override
    def map_constant(self, expr: object, /) -> PolyCoeffMapping:
        assert prim.is_arithmetic_expression(expr)
        return {} if prim.is_zero(expr) else {1: expr}

    @override
    def map_subscript(self, expr: prim.Subscript, /) -> PolyCoeffMapping:
        if expr in self.basis or expr.aggregate in self.basis:
            return {expr: 1}
        else:
            return {1: expr}

    @override
    def map_quotient(self, expr: prim.Quotient, /) -> PolyCoeffMapping:
        d_num = self.rec(expr.numerator)
        d_den = self.rec(expr.denominator)

        if not d_den:
            raise ZeroDivisionError(f"division by zero: {expr!r}")

        # d_den should look like {1: k}
        if len(d_den) != 1 or 1 not in d_den:
            raise RuntimeError("nonlinear expression")

        (den_coeff,) = d_den.values()
        return {
            num_var: num_coeff / den_coeff
            for num_var, num_coeff in d_num.items()
        }

    @override
    def map_power(self, expr: prim.Power, /) -> PolyCoeffMapping:
        d_base = self.rec(expr.base)
        d_exponent = self.rec(expr.exponent)

        # d_exponent should look like {1: k}
        if len(d_exponent) > 1 or 1 not in d_exponent:
            raise RuntimeError("nonlinear expression")

        exp, = d_exponent.values()
        if exp == 1:
            return d_base

        if not isinstance(exp, int):
            return {expr: 1}

        if len(d_base) > 1:
            raise NotImplementedError("powers are not implemented")

        (var, coeff), = d_base.items()
        return {var**exp: coeff**exp}

    @override
    def map_sum(self, expr: prim.Sum, /) -> PolyCoeffMapping:
        result: PolyCoeffMapping = {}
        for child in expr.children:
            ch = self.rec(child)
            for var, stride in ch.items():
                result[var] = result.get(var, 0) + stride

        return result

    @override
    def map_product(self, expr: prim.Product, /) -> PolyCoeffMapping:
        children: list[PolyCoeffMapping] = []
        for child in expr.children:
            child_rec = self.rec(child)
            if not child_rec:
                return {}

            children.append(child_rec)

        from itertools import product

        result: PolyCoeffMapping = {}
        for coeffs in product(*(c.items() for c in children)):
            # gather terms
            factors: list[ArithmeticExpression] = []
            coeff = 1
            for var_i, coeff_i in coeffs:
                if var_i != 1:
                    factors.append(var_i)

                coeff = coeff * coeff_i

            # normalize the key so that `a * b <=> b * a`
            key = prim.flattened_product(factors)
            if isinstance(key, prim.Product):
                key = prim.Product(tuple(sorted(key.children, key=str)))

            result[key] = result.get(key, 0) + coeff

        return result

    def map_node_coordinate_component(
            self, expr: NodeCoordinateComponent
        ) -> PolyCoeffMapping:
        return {expr: 1}

    def map_num_reference_derivative(
        self, expr: NumReferenceDerivative
        ) -> PolyCoeffMapping:
        return {expr: 1}


def collect(
        expr: Expression,
        source_dependent_variables: Sequence[ArithmeticExpression] | None = None,
    ) -> PolyCoeffMapping:
    from pytential.symbolic.mappers import flatten

    collector = CoefficientCollector(source_dependent_variables or ())
    return {flatten(k): flatten(v) for k, v in collector(expr).items()}


def _kernel_source_derivatives_as_poly(
        kernel: ScalarKernel, x: Sequence[sp.Symbol]
    ) -> sp.Expr:
    r"""Convert a :class:`sumpy.kernel.ScalarKernel` object to a polynomial.

    A :class:`~sumpy.kernel.ScalarKernel` represents a source derivative
    operator, which is converted to a polynomial in the target-derivative
    variables *x*. Each :class:`~sumpy.kernel.AxisSourceDerivative` along axis
    *j* contributes a factor of ``-x[j]`` via translation invariance
    (:math:`\partial_{y_j} G(x - y) = -\partial_{x_j} G(x - y)`). For example:

    .. math::

        \frac{\mathrm{d}}{\mathrm{d} y_1}
        \frac{\mathrm{d}}{\mathrm{d} y_2}
        +
        \frac{\mathrm{d}}{\mathrm{d} y_1}
        \quad \implies \quad
        x_1 x_2 - x_1.
    """

    if isinstance(kernel, AxisSourceDerivative):
        poly = _kernel_source_derivatives_as_poly(kernel.inner_kernel, x)
        return -x[kernel.axis] * poly

    if isinstance(kernel, KernelWrapper):
        raise TypeError(f"{type(kernel).__name__!r} not supported")

    return sp.sympify(1)


def make_int_g_poly_matrix(
        int_gs: Sequence[IntG],
        source_dependent_variables: Sequence[ArithmeticExpression],
        x: Sequence[sp.Symbol],
    ) -> tuple[sp.Matrix, Sequence[ArithmeticExpression]]:
    """Create a matrix from a list of :class:`~pytential.symbolic.primitives.IntG`
    objects.

    Each density expression may contain ``source_dependent_variables``. Each
    element in the matrix is a multivariate polynomial in the *x* input
    variables, representing a derivative operator.

    The number of rows of the returned matrix is equal to the length of
    ``int_gs`` and the number of columns is equal to the number of extracted
    source-dependent expressions.
    """
    from sumpy.symbolic import PymbolicToSympyMapper

    to_sympy = PymbolicToSympyMapper()
    coefficient_collector = CoefficientCollector(source_dependent_variables)

    source_exprs: list[ArithmeticExpression] = []
    matrix: list[list[sp.Expr]] = []

    for int_g in int_gs:
        row: list[sp.Expr] = [sp.sympify(0)] * len(source_exprs)
        for density, source_kernel in zip(int_g.densities, int_g.source_kernels,
                                          strict=True):
            d = coefficient_collector(density)
            for source_expr, coeff in d.items():
                if source_expr not in source_exprs:
                    source_exprs.append(source_expr)
                    row += [sp.sympify(0)]

                poly = _kernel_source_derivatives_as_poly(source_kernel, x)
                row[source_exprs.index(source_expr)] += poly * to_sympy(coeff)

        matrix.append(row)

    # NOTE: pad the matrix so that all the rows have equal length
    for row in matrix:
        row += [sp.sympify(0)]*(len(source_exprs) - len(row))

    return sp.Matrix(matrix), source_exprs


# }}}


# {{{ factor the matrix

def _get_minimal_generating_set(mod: SubModule) -> SubModule:
    """Computes a module with a minimal generating set as its generators
    from an input module with possibly redundant generators.

    The output does not necessarily have the smallest minimal generating set.
    """
    nonzero = [x for x in mod.gens if any(y != mod.ring.zero for y in x)]
    basis = nonzero.copy()

    for x in nonzero:
        others = nonzero.copy()
        others.remove(x)
        if x in mod.container.submodule(*others):
            basis = others

    return mod.container.submodule(*basis)


def _convert_to_matrix(
        generators: Sequence[Iterable[DMP | QuotientRingElement]],
        ring: Ring,
    ) -> sp.Matrix:
    return sp.Matrix([
        [ring.to_sympy(dmp) for dmp in generator]
        for generator in generators
    ])


def _get_syzygy_module(mat: sp.Matrix, ring: Ring) -> SubModule:
    """Compute the syzygy module for the module represented by *mat* in *ring*.

    The syzygy module *S* satisfies ``S * mat = 0`` and is a left nullspace of
    *mat*. The domain of *ring* is expected to be :class:`sympy.EX`, since that
    represents a domain with arbitrary symbolic elements.
    """
    m, n = mat.shape
    module = ring.free_module(n).submodule(*[mat[i, :] for i in range(m)])

    return _get_minimal_generating_set(module.syzygy_module())


def _make_syzygy_factorization(
        mat: sp.Matrix,
        ring: Ring,
    ) -> tuple[sp.Matrix, sp.Matrix]:
    """Return a "rank-revealing" factorization of the matrix.

    For a matrix *M*, we want to find a factorization such that ``M = L * R``
    with the minimum number of columns of *L*. The polynomials represent
    derivative operators and therefore division is not well defined.
    To avoid divisions, we work in a polynomial ring, which does not
    have division either. The revealed rank might not be the actual
    rank.

    To get a good factorization, what we do is first find a matrix *S*
    such that ``S * M.T = 0``, where *S* is the syzygy module of ``M.T``
    converted to a matrix (i.e., ``S.T`` is the right nullspace of *M*).
    Then, ``M * S.T = 0``, which implies that the rows of *M* are in the
    syzygy module of ``S.T``, giving *R* directly.
    """
    # NOTE: for sympy performance, we prefer a tall and skinny matrix
    if mat.shape[0] < mat.shape[1]:
        L, R = _make_syzygy_factorization(mat.T, ring)
        return R.T, L.T

    S_module = _get_syzygy_module(mat.T, ring)
    S = _convert_to_matrix(S_module.gens, ring)

    if len(S) == 0:
        return mat, sp.eye(mat.shape[1])

    R_module = _get_syzygy_module(S.T, ring)
    R = _convert_to_matrix(R_module.gens, ring)

    L_module = [
        R_module.in_terms_of_generators(mat[i, :])
        for i in range(mat.shape[0])
    ]
    L = _convert_to_matrix(L_module, ring)

    # NOTE: if L has zero columns then those columns do not matter when we
    # do the factorization M = L R. We prune them here so that we can actually
    # get `k < n` for some cases (in particular the Stokeslet falls in here)
    nonzero_cols = [
        j for j in range(L.shape[1]) if any(L[i, j] != 0 for i in range(L.shape[0]))
    ]
    if len(nonzero_cols) < L.shape[1]:
        L = L[:, nonzero_cols]
        R = R[nonzero_cols, :]

    return L, R

# }}}


# {{{ convert factors back into IntGs

def _convert_source_poly_to_int_g(
        poly: sp.Poly | None,
        orig_int_g: IntG,
    ) -> IntG | None:
    """Convert a polynomial back into source derivatives of *orig_int_g*.

    This does the opposite of :func:`_kernel_source_derivatives_as_poly`. First
    it is converted to a :class:`sumpy.kernel.ScalarKernel` and then to a
    :class:`~pytential.symbolic.primitives.IntG`.
    """

    if poly is None:
        return None

    from sumpy.symbolic import SympyToPymbolicMapper
    to_pymbolic = SympyToPymbolicMapper()

    assert len(orig_int_g.source_kernels) == 1
    orig_kernel = orig_int_g.source_kernels[0]

    from pytential.symbolic.mappers import flatten

    source_kernels: list[ScalarKernel] = []
    densities: list[ArithmeticExpression] = []

    for monom, coeff in poly.terms():
        kernel = orig_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisSourceDerivative(idim, kernel)
        source_kernels.append(kernel)

        # (-1) below is because d/dx f(c - x) = - f'(c - x)
        densities.append(flatten(to_pymbolic(coeff) * (-1)**sum(monom)))

    return replace(
        orig_int_g,
        source_kernels=tuple(source_kernels),
        densities=tuple(densities),
    )


def _convert_target_poly_to_int_g(
        poly: sp.Poly,
        orig_int_g: IntG,
        rhs_int_g: IntG,
    ) -> ArithmeticExpression:
    """Convert a polynomial back into target derivatives of *orig_int_g*.

    Each term is converted to a :class:`sumpy.kernel.ScalarKernel` with
    :class:`~sumpy.kernel.AxisTargetDerivative`, and the resulting
    :class:`~pytential.symbolic.primitives.IntG` terms are summed into an
    :class:`~pymbolic.typing.ArithmeticExpression`.
    """
    from sumpy.symbolic import SympyToPymbolicMapper
    to_pymbolic = SympyToPymbolicMapper()

    result: ArithmeticExpression = 0
    for monom, coeff in poly.terms():
        kernel = orig_int_g.target_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisTargetDerivative(idim, kernel)

        result += to_pymbolic(coeff) * replace(
                orig_int_g,
                target_kernel=kernel,
                source_kernels=rhs_int_g.source_kernels,
                densities=rhs_int_g.densities
        )

    return result

# }}}

# vim: fdm=marker
