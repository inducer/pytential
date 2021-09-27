__copyright__ = "Copyright (C) 2021 Isuru Fernando"

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

from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative,
        KernelWrapper)

from pymbolic.interop.sympy import PymbolicToSympyMapper, SympyToPymbolicMapper
from pymbolic.mapper import Mapper
from pymbolic.primitives import Product
import sympy
import functools
from collections import defaultdict


__all__ = (
    "reduce_number_of_fmms",
    )

__doc__ = """
.. autofunction:: reduce_number_of_fmms
"""


# {{{ Reduce number of FMMs - main routine

def reduce_number_of_fmms(int_gs, source_dependent_variables):
    """
    Reduce the number of FMMs needed for a system of
    :class:`~pytential.symbolic.primitives.IntG` objects.

    This is done by converting the ``IntG`` object to a matrix of polynomials
    with d variables corresponding to d dimensions, where each variable represents
    a (target) derivative operator along one of the axes. All the properties of
    derivative operator that we want are reflected in the properties of the
    polynomial including addition, multiplication and exact polynomial division.

    This matrix is factored into two matrices, where the left hand side matrix
    represents a transformation at the target, and the right hand side matrix
    represents a transformation at the source.

    If the expressions given are not linear, then the input expressions are
    returned as is.

    :arg int_gs: list of ``IntG`` objects.

    :arg source_dependent_variables: list of :class:`pymbolic.primitives.Expression`
        objects. When reducing FMMs, consider only these variables as dependent
        on source. For eg: densities, source derivative vectors.

    Note: there is no argument for target-dependent variables as the algorithm
    assumes that there are no target-dependent variables passed to this function.
    (where a "source/target-dependent variable" is a symbolic variable that evaluates
    to a vector discretized on the sources/targets)
    """

    dim = int_gs[0].target_kernel.dim
    axis_vars = sympy.symbols(f"_x0:{dim}")

    # A high level driver for this function should send int_gs that share all the
    # properties except for the densities, source transformations and target
    # transformations.
    assert _check_int_gs_common(int_gs)

    try:
        mat, source_exprs = _create_matrix(int_gs, source_dependent_variables,
            axis_vars)
    except ValueError:
        return int_gs

    mat = sympy.Matrix(mat)
    # Factor the matrix into two
    try:
        left_factor = _factor_left(mat, axis_vars)
        right_factor = _factor_right(mat, left_factor)
    except ValueError:
        return int_gs

    # If there are n inputs and m outputs,
    #
    #  - matrix: R^{m x n},
    #  - LHS: R^{m x k},
    #  - RHS: R^{k x n}.
    #
    # If k is greater than or equal to n we are gaining nothing.
    # Return as is.
    if right_factor.shape[0] >= mat.shape[0]:
        return int_gs

    # cast to SymPy Polynomial objects
    right_factor = right_factor.applyfunc(lambda x: x.as_poly(*axis_vars,
        domain=sympy.EX))
    left_factor = left_factor.applyfunc(lambda x: x.as_poly(*axis_vars,
        domain=sympy.EX))

    base_kernel = int_gs[0].source_kernels[0].get_base_kernel()
    base_int_g = int_gs[0].copy(target_kernel=base_kernel,
            source_kernels=(base_kernel,), densities=(1,))

    # Convert polynomials back to IntGs with source derivatives
    source_int_gs = [[_convert_source_poly_to_int_g_derivs(poly, base_int_g,
        axis_vars) for poly in row] for row in right_factor.tolist()]

    # For each row in the right factor, merge the IntGs to one IntG
    # to get a total of k IntGs.
    source_int_gs_merged = []
    for i in range(right_factor.shape[0]):
        source_kernels = []
        densities = []
        for j in range(right_factor.shape[1]):
            new_densities = [density * source_exprs[j] for density in
                    source_int_gs[i][j].densities]
            source_kernels.extend(source_int_gs[i][j].source_kernels)
            densities.extend(new_densities)
        source_int_gs_merged.append(source_int_gs[i][0].copy(
            source_kernels=tuple(source_kernels), densities=tuple(densities)))

    # Now that we have the IntG expressions depending on the source
    # we now have to attach the target dependent derivatives.
    res = [0]*left_factor.shape[0]
    for i in range(left_factor.shape[0]):
        for j in range(left_factor.shape[1]):
            res[i] += _convert_target_poly_to_int_g_derivs(left_factor[i, j],
                    int_gs[i], source_int_gs_merged[j])

    return res


# }}}

# {{{ convert IntG expressions to a matrix

def _check_int_gs_common(int_gs):
    """Checks that the :class:`~pytential.symbolic.primtive.IntG` objects
    have the same base kernel and other properties that would allow
    merging them.
    """
    base_kernel = int_gs[0].source_kernels[0].get_base_kernel()
    common_int_g = int_gs[0].copy(target_kernel=base_kernel,
            source_kernels=(base_kernel,), densities=(1,))
    for int_g in int_gs:
        for source_kernel in int_g.source_kernels:
            if source_kernel.get_base_kernel() != base_kernel:
                return False
        if common_int_g != int_g.copy(target_kernel=base_kernel,
                source_kernels=(base_kernel,), densities=(1,)):
            return False
    return True


def _create_matrix(int_gs, source_dependent_variables, axis_vars):
    """Create a matrix from a list of :class:`~pytential.symbolic.primitives.IntG`
    objects and returns the matrix and the expressions corresponding to each column.
    Each expression is an expression containing ``source_dependent_variables``.
    Each element in the matrix is a multi-variate polynomial and the variables
    in the polynomial are from ``axis_vars`` input. Each polynomial represents
    a derivative operator.

    Number of rows of the returned matrix is equal to the number of ``int_gs`` and
    the number of columns is equal to the number of input source dependent
    expressions.
    """
    source_exprs = []
    coefficient_collector = CoefficientCollector(source_dependent_variables)
    to_sympy = PymbolicToSympyMapper()
    matrix = []

    for int_g in int_gs:
        row = [0]*len(source_exprs)
        for density, source_kernel in zip(int_g.densities, int_g.source_kernels):
            d = coefficient_collector(density)
            for source_expr, coeff in d.items():
                if source_expr not in source_exprs:
                    source_exprs.append(source_expr)
                    row += [0]
                poly = _kernel_source_derivs_as_poly(source_kernel, axis_vars)
                row[source_exprs.index(source_expr)] += poly * to_sympy(coeff)
        matrix.append(row)

    # At the beginning, we didn't know the number of columns of the matrix.
    # Therefore we used a list for rows and they kept expanding.
    # Here we are adding zero padding to make the result look like a matrix.
    for row in matrix:
        row += [0]*(len(source_exprs) - len(row))

    return matrix, source_exprs


class CoefficientCollector(Mapper):
    """From a density expression, extracts expressions that need to be
    evaluated for each source and coefficients for each expression.

    For eg: when this mapper is given as ``s*(s + 2) + 3`` input,
    it returns {s**2: 1, s: 2, 1: 3}.

    This is more general than
    :class:`pymbolic.mapper.coefficient.CoefficientCollector` as that deals
    only with linear expressions, but this collector works for polynomial
    expressions too.
    """
    def __init__(self, source_dependent_variables):
        self.source_dependent_variables = source_dependent_variables

    def __call__(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        return super().__call__(expr)

    def map_sum(self, expr):
        stride_dicts = [self.rec(ch) for ch in expr.children]

        result = defaultdict(lambda: 0)
        for stride_dict in stride_dicts:
            for var, stride in stride_dict.items():
                result[var] += stride
        return dict(result)

    def map_algebraic_leaf(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        else:
            return {1: expr}

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
            right = Product(tuple(expr.children[1:]))
            new_prod = Product((left, right))
            return self.rec(new_prod)
        elif len(expr.children) == 1:
            return self.rec(expr.children[0])
        elif len(expr.children) == 0:
            return {1: 1}
        left, right = expr.children
        d_left = self.rec(left)
        d_right = self.rec(right)
        d = defaultdict(lambda: 0)
        for var_left, coeff_left in d_left.items():
            for var_right, coeff_right in d_right.items():
                d[var_left*var_right] += coeff_left*coeff_right
        return dict(d)

    def map_quotient(self, expr):
        d_num = self.rec(expr.numerator)
        d_den = self.rec(expr.denominator)
        if len(d_den) > 1:
            raise ValueError
        den_var, den_coeff = list(d_den.items())[0]

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
        if len(d_base) > 1:
            raise NotImplementedError("powers are not implemented")
        (var, coeff), = d_base.items()
        return {var**exp: coeff**exp}

    rec = __call__


def _kernel_source_derivs_as_poly(kernel, axis_vars):
    """Converts a :class:`sumpy.kernel.Kernel` object to a polynomial.
    A :class:`sumpy.kernel.Kernel` represents a derivative operator
    and the derivative operator is converted to a polynomial with
    variables given by `axis_vars`.

    For eg: for source x the derivative operator,
    d/dx_1 dx_2 + d/dx_1 is converted to x_2 * x_1 + x_1.
    """
    if isinstance(kernel, AxisSourceDerivative):
        poly = _kernel_source_derivs_as_poly(kernel.inner_kernel, axis_vars)
        return -axis_vars[kernel.axis]*poly
    if isinstance(kernel, KernelWrapper):
        raise ValueError
    return 1

# }}}


# {{{ factor the matrix

def _syzygy_module_groebner_basis_mat(m, generators):
    """Takes as input a module of polynomials with domain :class:`sympy.EX`
    represented as a matrix and returns the syzygy module as a matrix of polynomials
    in the same domain. The syzygy module *S* that is returned as a matrix
    satisfies S m = 0 and is a left nullspace of the matrix.
    Using :class:`sympy.EX` because that represents the domain with any symbolic
    element. Usually we need an Integer or Rational domain, but since there can be
    unrelated symbols like *mu* in the expression, we need to use a symbolic domain.
    """
    from sympy.polys.orderings import grevlex

    def _convert_to_matrix(module, *generators):
        result = []
        for syzygy in module:
            row = []
            for dmp in syzygy.data:
                row.append(sympy.Poly(dmp.to_dict(), *generators,
                    domain=sympy.EX).as_expr())
            result.append(row)
        return sympy.Matrix(result)

    ring = sympy.EX.old_poly_ring(*generators, order=grevlex)
    column_ideals = [ring.free_module(1).submodule(*m[:, i].tolist(), order=grevlex)
                for i in range(m.shape[1])]
    column_syzygy_modules = [ideal.syzygy_module() for ideal in column_ideals]

    intersection = functools.reduce(lambda x, y: x.intersect(y),
            column_syzygy_modules)

    # _groebner_vec returns a groebner basis of the syzygy module as a list
    groebner_vec = intersection._groebner_vec()
    return _convert_to_matrix(groebner_vec, *generators)


def _factor_left(mat, axis_vars):
    """Return the left hand side of the factorisation of the matrix
    For a matrix M, we want to find a factorisation such that M = L R
    with minimum number of columns of L. The polynomials represent
    derivative operators and therefore division is not well defined.
    To avoid divisions, we work in a polynomial ring which doesn't
    have division either.

    To get a good factorisation, what we do is first find a matrix
    such that S M = 0 where S is the syzygy module converted to a matrix.
    It can also be referred to as the left nullspace of the matrix.
    Then, M.T S.T = 0 which implies that M.T is in the space spanned by
    the syzygy module of S.T and to get M we get the transpose of that.
    """
    syzygy_of_mat = _syzygy_module_groebner_basis_mat(mat, axis_vars)
    if len(syzygy_of_mat) == 0:
        raise ValueError("could not find a factorization")
    return _syzygy_module_groebner_basis_mat(syzygy_of_mat.T, axis_vars).T


def _factor_right(mat, factor_left):
    """Return the right hand side of the factorisation of the matrix
    Note that from the construction of *factor_left*, we know that
    the factor on the right has elements in the same polynomial ring
    as the input matrix *mat*. Therefore, doing divisions are fine
    as they should result in exact polynomial divisions and will have
    no remainders.
    """
    return factor_left.LUsolve(sympy.Matrix(mat),
            iszerofunc=lambda x: x.simplify() == 0)

# }}}


# {{{ convert factors back into IntGs

def _convert_source_poly_to_int_g_derivs(poly, orig_int_g, axis_vars):
    """This does the opposite of :func:`_kernel_source_derivs_as_poly`
    and converts a polynomial back to a source derivative
    operator. First it is converted to a :class:`sumpy.kernel.Kernel`
    and then to a :class:`~pytential.symbolic.primitives.IntG`.
    """
    from pytential.symbolic.pde.system_utils import simplify_densities
    to_pymbolic = SympyToPymbolicMapper()

    orig_kernel = orig_int_g.source_kernels[0]
    source_kernels = []
    densities = []
    for monom, coeff in poly.terms():
        kernel = orig_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisSourceDerivative(idim, kernel)
        source_kernels.append(kernel)
        # (-1) below is because d/dx f(c - x) = - f'(c - x)
        densities.append(to_pymbolic(coeff) * (-1)**sum(monom))
    return orig_int_g.copy(source_kernels=tuple(source_kernels),
            densities=tuple(simplify_densities(densities)))


def _convert_target_poly_to_int_g_derivs(poly, orig_int_g, rhs_int_g):
    """This does the opposite of :func:`_kernel_source_derivs_as_poly`
    and converts a polynomial back to a target derivative
    operator. It is applied to a :class:`~pytential.symbolic.primitives.IntG`
    object and returns a new instance.
    """
    to_pymbolic = SympyToPymbolicMapper()

    result = 0
    for monom, coeff in poly.terms():
        kernel = orig_int_g.target_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisTargetDerivative(idim, kernel)
        result += orig_int_g.copy(target_kernel=kernel,
                source_kernels=rhs_int_g.source_kernels,
                densities=rhs_int_g.densities) * to_pymbolic(coeff)

    return result

# }}}

# vim: fdm=marker
