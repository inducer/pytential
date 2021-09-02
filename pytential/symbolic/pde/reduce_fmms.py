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

from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative)

from pymbolic.mapper import Mapper
from pymbolic.primitives import Product
from pymbolic.interop.sympy import PymbolicToSympyMapper, SympyToPymbolicMapper
from pymbolic.mapper.coefficient import (
        CoefficientCollector as CoefficientCollectorBase)
import sympy

from collections import defaultdict


__all__ = (
    "reduce_number_of_fmms",
    )


def reduce_number_of_fmms(int_gs, source_dependent_variables):
    """
    Reduce the number of FMMs needed for a system of expressions with
    :class:`~pytential.symbolic.primitives.IntG` objects.

    This is done by converting the `IntG` expression to a matrix of polynomials
    with d variables corresponding to d dimensions and each polynomial represents
    a derivative operator. All the properties of derivative operator that we want
    are reflected in the properties of the polynomial including addition,
    multiplication and exact polynomial division.

    This matrix is factored into two matrices where the left hand side matrix
    represents a transformation at target and the right hand side matrix represents
    a transformation at source.

    If the expressions given are not linear, then the input expressions are
    returned as is.

    :arg source_dependent_variables: When reducing FMMs, consider only these
        variables as dependent on source.
    """

    source_exprs = []
    mapper = CoefficientCollector(source_dependent_variables)
    matrix = []
    dim = int_gs[0].target_kernel.dim
    axis_vars = sympy.symbols(f"_x0:{dim}")
    to_sympy = PymbolicToSympyMapper()

    if not _check_int_gs_common(int_gs):
        return int_gs

    for int_g in int_gs:
        row = [0]*len(source_exprs)
        for density, source_kernel in zip(int_g.densities, int_g.source_kernels):
            try:
                d = mapper(density)
            except ImportError:
                return int_gs
            for source_expr, coeff in d.items():
                if source_expr not in source_exprs:
                    source_exprs.append(source_expr)
                    row += [0]
                poly = _convert_kernel_to_poly(source_kernel, axis_vars)
                row[source_exprs.index(source_expr)] += poly * to_sympy(coeff)
        matrix.append(row)

    for row in matrix:
        row += [0]*(len(source_exprs) - len(row))

    mat = sympy.Matrix(matrix)
    lhs_mat = _factor_left(mat, axis_vars)
    rhs_mat = _factor_right(mat, lhs_mat)

    if rhs_mat.shape[0] >= mat.shape[0]:
        return int_gs

    rhs_mat = rhs_mat.applyfunc(lambda x: x.as_poly(*axis_vars, domain=sympy.EX))
    lhs_mat = lhs_mat.applyfunc(lambda x: x.as_poly(*axis_vars, domain=sympy.EX))

    base_kernel = int_gs[0].source_kernels[0].get_base_kernel()
    base_int_g = int_gs[0].copy(target_kernel=base_kernel,
            source_kernels=(base_kernel,), densities=(1,))
    rhs_mat_int_gs = [[_convert_source_poly_to_int_g(poly, base_int_g, axis_vars)
            for poly in row] for row in rhs_mat.tolist()]

    rhs_int_gs = []
    for i in range(rhs_mat.shape[0]):
        source_kernels = []
        densities = []
        for j in range(rhs_mat.shape[1]):
            new_densities = [density * source_exprs[j] for density in
                    rhs_mat_int_gs[i][j].densities]
            source_kernels.extend(rhs_mat_int_gs[i][j].source_kernels)
            densities.extend(new_densities)
        rhs_int_gs.append(rhs_mat_int_gs[i][0].copy(
            source_kernels=tuple(source_kernels), densities=tuple(densities)))

    res = [0]*lhs_mat.shape[0]
    for i in range(lhs_mat.shape[0]):
        for j in range(lhs_mat.shape[1]):
            res[i] += _convert_target_poly_to_int_g(lhs_mat[i, j],
                    int_gs[i], rhs_int_gs[j])

    return res


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


def _convert_kernel_to_poly(kernel, axis_vars):
    """Converts a :class:`sumpy.kernel.Kernel` object to a polynomial.
    A :class:`sumpy.kernel.Kernel` represents a derivative operator
    and the derivative operator is converted to a polynomial with
    variables given by `axis_vars`.

    For eg: for target y and source x the derivative operator,
    d/dx_1 dy_2 + d/dy_1 is converted to -y_2 * y_1 + y_1.
    """
    if isinstance(kernel, AxisTargetDerivative):
        poly = _convert_kernel_to_poly(kernel.inner_kernel, axis_vars)
        return axis_vars[kernel.axis]*poly
    elif isinstance(kernel, AxisSourceDerivative):
        poly = _convert_kernel_to_poly(kernel.inner_kernel, axis_vars)
        return -axis_vars[kernel.axis]*poly
    return 1


def _convert_source_poly_to_int_g(poly, orig_int_g, axis_vars):
    """This does the opposite of :func:`_convert_kernel_to_poly`
    and converts a polynomial back to a source derivative
    operator. First it is converted to a :class:`sumpy.kernel.Kernel`
    and then to a :class:`~pytential.symbolic.primitives.IntG`.
    """
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
        densities.append(to_pymbolic(coeff) * (-1)**sum(monom))
    return orig_int_g.copy(source_kernels=tuple(source_kernels),
            densities=tuple(densities))


def _convert_target_poly_to_int_g(poly, orig_int_g, rhs_int_g):
    """This does the opposite of :func:`_convert_kernel_to_poly`
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


class CoefficientCollector(CoefficientCollectorBase):
    """This extends :class:`pymbolic.mapper.coefficient.CoefficientCollector`
    by extending the targets to be any expression instead of just algebraic
    leafs
    """
    def __init__(self, source_dependent_variables):
        self.source_dependent_variables = source_dependent_variables

    def __call__(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        return super().__call__(expr)

    rec = __call__


def _syzygy_module(m, gens):
    """Takes as input a module of polynomials with domain `sympy.EX` represented
    as a matrix and returns the syzygy module as a matrix of polynomials in the
    same domain.
    """
    from sympy.polys.orderings import grevlex

    def _convert_to_matrix(module, *gens):
        result = []
        for syzygy in module:
            row = []
            for dmp in syzygy.data:
                row.append(sympy.Poly(dmp.to_dict(), *gens,
                    domain=sympy.EX).as_expr())
            result.append(row)
        return sympy.Matrix(result)

    ring = sympy.EX.old_poly_ring(*gens, order=grevlex)
    column_ideals = [ring.free_module(1).submodule(*m[:, i].tolist(), order=grevlex)
                for i in range(m.shape[1])]
    column_syzygy_modules = [ideal.syzygy_module() for ideal in column_ideals]

    intersection = column_syzygy_modules[0]
    for i in range(1, len(column_syzygy_modules)):
        intersection = intersection.intersect(column_syzygy_modules[i])

    m2 = intersection._groebner_vec()
    m3 = _convert_to_matrix(m2, *gens)
    return m3


def _factor_left(mat, axis_vars):
    """Return the left hand side of the factorisation of the matrix"""
    return _syzygy_module(_syzygy_module(mat, axis_vars).T, axis_vars).T


def _factor_right(mat, factor_left):
    """Return the right hand side of the factorisation of the matrix"""
    ys = sympy.symbols(f"_y{{0:{factor_left.shape[1]}}}")
    factor_right = []
    for i in range(mat.shape[1]):
        aug_mat = sympy.zeros(factor_left.shape[0], factor_left.shape[1] + 1)
        aug_mat[:, :factor_left.shape[1]] = factor_left
        aug_mat[:, factor_left.shape[1]] = mat[:, i]
        res_map = sympy.solve_linear_system(aug_mat, *ys)
        row = []
        for y in ys:
            row.append(res_map[y])
        factor_right.append(row)
    factor_right = sympy.Matrix(factor_right).T
    return factor_right
