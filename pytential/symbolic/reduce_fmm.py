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
from typing import TYPE_CHECKING, cast

import sympy as sp
from typing_extensions import override

import pymbolic.primitives as prim
from pymbolic.geometric_algebra.mapper import WalkMapper
from pymbolic.mapper import Mapper
from pymbolic.typing import ArithmeticExpression, Expression
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    KernelWrapper,
    ScalarKernel,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sympy.polys.agca.modules import SubModulePolyRing
    from sympy.polys.domains.ring import Ring

    from pytential.symbolic.primitives import IntG, NodeCoordinateComponent

logger = logging.getLogger(__name__)


__all__ = (
    "reduce_number_of_fmms",
)

__doc__ = """
.. autofunction:: reduce_number_of_fmms
"""


# {{{ Reduce number of FMMs - main routine


class ReduceFailedError(RuntimeError):
    """Exception raised when the number of FMMs could not be reduced."""


class IncompatibleIntGError(ReduceFailedError):
    """Exception raised when FMMs could not be reduced due to incompatible inputs."""


def reduce_number_of_fmms(
        int_gs: Sequence[IntG], *,
        source_dependent_variables: Sequence[ArithmeticExpression] | None = None,
    ) -> Sequence[ArithmeticExpression]:
    """
    Reduce the number of FMMs needed for a system of
    :class:`~pytential.symbolic.primitives.IntG` objects.

    This is done by converting the ``IntG`` objects to a matrix of polynomials
    with :math:`d` variables corresponding to :math:`d` dimensions, where each
    variable represents a (target) derivative operator along one of the axes.
    All the properties of the derivative operator that we want are reflected in
    the properties of polynomials, including addition, multiplication, and exact
    polynomial division.

    This matrix is factored into two matrices, where the first matrix
    represents a transformation at the target, and the second matrix represents
    a transformation at the source.

    .. note::

        There is no argument for target-dependent variables as the algorithm
        assumes that there are no target-dependent variables passed to this
        function (where a "source/target-dependent variable" is a symbolic
        variable that evaluates to a vector discretized on the sources/targets).

    :arg source_dependent_variables: sequence of
        :class:`~pymbolic.typing.ArithmeticExpression` objects (or *None* to
        collect all source-dependent variables automatically). When reducing
        FMMs, consider only these variables as dependent on the source, e.g.,
        densities or source derivative vectors.
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

    if source_dependent_variables is None:
        source_dependent_variables = get_all_source_dependent_variables(int_gs)

    # }}}

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
        ring = poly_ring / [pde_poly]
    except NotImplementedError:
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
    merged_source_int_gs: list[IntG] = []
    for i in range(k):
        source_kernels: list[ScalarKernel] = []
        densities: list[ArithmeticExpression] = []

        for j in range(n):
            source_int_g = source_int_gs[i][j]
            if source_int_g is None:
                continue

            source_kernels.extend(source_int_g.source_kernels)
            densities.extend(d * source_exprs[j] for d in source_int_g.densities)

        merged_source_int_gs.append(
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
                merged_source_int_gs[j]
            )

        result.append(flatten(result_i))

    return tuple(result)


def as_poly(expr: sp.Expr, x: tuple[sp.Symbol, ...]) -> sp.Expr | None:
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
    # FIXME: what about kernel arguments? normal, parameters, etc.
    mapper = GatherAllSourceDependentVariables()
    for int_g in int_gs:
        for density in int_g.densities:
            mapper(density)

    return tuple(mapper.vars)


# }}}


# {{{ convert IntG expressions to a matrix


# FIXME: make pymbolic.CoefficientCollector more general?

class CoefficientCollector(Mapper[ArithmeticExpression, []]):
    """Extract expressions that need to be evaluated for each source and their
    coefficients from a density expression.

    For example, when this mapper is given ``s*(s + 2) + 3`` as input,
    it returns ``{s**2: 1, s: 2, 1: 3}``.

    This is more general than
    :class:`pymbolic.mapper.coefficient.CoefficientCollector` as that deals
    only with linear expressions, but this collector works for polynomial
    expressions too.
    """

    source_dependent_variables: Sequence[ArithmeticExpression]

    def __init__(
            self,
            source_dependent_variables: Sequence[ArithmeticExpression]
        ) -> None:
        self.source_dependent_variables = source_dependent_variables

    def __call__(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        return super().__call__(expr)

    def map_sum(self, expr):
        stride_dicts = [self.rec(ch) for ch in expr.children]

        result = {}
        for stride_dict in stride_dicts:
            for var, stride in stride_dict.items():
                result[var] = result.get(var, 0) + stride

        return result

    def map_algebraic_leaf(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        else:
            return {1: expr}

    def map_node_coordinate_component(self, expr):
        return {expr: 1}

    map_num_reference_derivative = map_node_coordinate_component

    def map_common_subexpression(self, expr):
        return {expr: 1}

    def map_subscript(self, expr):
        if expr in self.source_dependent_variables or \
                expr.aggregate in self.source_dependent_variables:
            return {expr: 1}
        else:
            return {1: expr}

    def map_constant(self, expr):
        return {1: expr}

    def map_product(self, expr):
        if len(expr.children) > 2:
            # rewrite products of more than two children as a nested
            # product and recurse to make it easier to handle.
            left = expr.children[0]
            right = prim.Product(tuple(expr.children[1:]))
            new_prod = prim.Product((left, right))
            return self.rec(new_prod)
        elif len(expr.children) == 1:
            return self.rec(expr.children[0])
        elif len(expr.children) == 0:
            return {1: 1}

        left, right = expr.children
        d_left = self.rec(left)
        d_right = self.rec(right)

        from pytential.symbolic.mappers import flatten

        d = {}
        for var_left, coeff_left in d_left.items():
            for var_right, coeff_right in d_right.items():
                key = flatten(var_left * var_right)
                d[key] = d.get(key, 0) + coeff_left * coeff_right

        return d

    def map_quotient(self, expr):
        d_num = self.rec(expr.numerator)
        d_den = self.rec(expr.denominator)
        if len(d_den) > 1:
            raise ValueError
        den_var, den_coeff = next(iter(d_den.items()))

        return {num_var/den_var: num_coeff/den_coeff for
            num_var, num_coeff in d_num.items()}

    def map_power(self, expr):
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

    rec = __call__


def _kernel_source_derivatives_as_poly(
        kernel: ScalarKernel, x: Sequence[sp.Symbol]
    ) -> sp.Expr:
    r"""Convert a :class:`sumpy.kernel.ScalarKernel` object to a polynomial.

    A :class:`~sumpy.kernel.ScalarKernel` represents a derivative operator, which
    is converted to a polynomial in the variables *x*. For example, for *x*,
    the derivative operator is converted as follows:

    .. math::

        \frac{\mathrm{d}}{\mathrm{d} x_1}
        \frac{\mathrm{d}}{\mathrm{d} x_2}
        +
        \frac{\mathrm{d}}{\mathrm{d} x_1}
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
                    row += [0]

                poly = _kernel_source_derivatives_as_poly(source_kernel, x)
                row[source_exprs.index(source_expr)] += poly * to_sympy(coeff)

        matrix.append(row)

    # NOTE: pad the matrix so that all the rows have equal length
    for row in matrix:
        row += [sp.sympify(0)]*(len(source_exprs) - len(row))

    return sp.Matrix(matrix), source_exprs


# }}}


# {{{ factor the matrix

def _get_minimal_generating_set(mod: SubModulePolyRing) -> SubModulePolyRing:
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


def _convert_to_matrix(module: SubModulePolyRing, ring: Ring) -> sp.Matrix:
    return sp.Matrix([
        [ring.to_sympy(dmp) for dmp in syzygy]
        for syzygy in module
    ])


def _get_syzygy_module(mat: sp.Matrix, ring: Ring) -> SubModulePolyRing:
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
        poly: sp.Expr | None,
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
        poly: sp.Expr,
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
