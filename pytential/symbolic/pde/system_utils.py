__copyright__ = "Copyright (C) 2020 Isuru Fernando"

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

from sumpy.symbolic import make_sym_vector, sym, SympyToPymbolicMapper
from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative,
    DirectionalSourceDerivative)
from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

from pymbolic.mapper import CombineMapper
from pymbolic.primitives import Sum, Product
from pytential.symbolic.primitives import IntG


def _chop(expr, tol):
    nums = expr.atoms(sym.Number)
    replace_dict = {}
    for num in nums:
        if abs(num) < tol:
            replace_dict[num] = 0
        else:
            new_num = float(num)
            if abs((int(new_num) - new_num)/new_num) < tol:
                new_num = int(new_num)
            replace_dict[num] = new_num
    return expr.xreplace(replace_dict)


def _n(expr):
    from sumpy.symbolic import USE_SYMENGINE
    if USE_SYMENGINE:
        # 100 bits
        return expr.n(prec=100)
    else:
        # 30 decimal places
        return expr.n(n=30)


def get_deriv_relation(kernels, base_kernel, tol=1e-10, order=4, verbose=False):
    dim = base_kernel.dim

    mis = sorted(gnitstam(order, dim), key=sum)

    # (-1, -1, -1) represent a constant
    mis.append((-1, -1, -1))

    rand = np.random.randint(1, 100, (dim, len(mis)))
    sym_vec = make_sym_vector("d", dim)

    base_expr = base_kernel.get_expression(sym_vec)
    sympy_conv = SympyToPymbolicMapper()

    mat = []
    for rand_vec_idx in range(rand.shape[1]):
        row = []
        for mi in mis[:-1]:
            expr = base_expr
            for var_idx, nderivs in enumerate(mi):
                if nderivs == 0:
                    continue
                expr = expr.diff(sym_vec[var_idx], nderivs)
            replace_dict = dict(
                (k, v) for k, v in zip(sym_vec, rand[:, rand_vec_idx])
            )
            eval_expr = _n(expr.xreplace(replace_dict))
            row.append(eval_expr)
        row.append(1)
        mat.append(row)

    mat = sym.Matrix(mat)
    res = []

    for kernel in kernels:
        expr = kernel.get_expression(sym_vec)
        vec = []
        for i in range(len(mis)):
            vec.append(_n(expr.xreplace(dict((k, v) for
                k, v in zip(sym_vec, rand[:, i])))))
        vec = sym.Matrix(vec)
        result = []
        const = 0
        if verbose:
            print(kernel, end=" = ", flush=True)
        for i, coeff in enumerate(mat.solve(vec)):
            coeff = _chop(coeff.expand(), tol)
            if coeff == 0:
                continue
            if mis[i] != (-1, -1, -1):
                coeff *= kernel.get_global_scaling_const()
                coeff /= base_kernel.get_global_scaling_const()
                result.append((mis[i], sympy_conv(coeff)))
                if verbose:
                    print(f"{base_kernel}.diff({mis[i]})*{coeff}", end=" + ")
            else:
                const = sympy_conv(coeff * kernel.get_global_scaling_const())
        if verbose:
            print(const)
        res.append((const, result))

    return res


class HaveIntGs(CombineMapper):
    def combine(self, values):
        return sum(values)

    def map_int_g(self, expr):
        return 1

    def map_constant(self, expr):
        return 0

    def map_variable(self, expr):
        return 0

    def handle_unsupported_expression(self, expr, *args, **kwargs):
        return 0


def merge_int_g_exprs(exprs):
    return np.array([merge_int_g_expr(expr) for expr in exprs])


def _convert_target_deriv_to_source(int_g):
    knl = int_g.target_kernel
    source_kernels = list(int_g.source_kernels)
    coeff = 1
    while isinstance(knl, AxisTargetDerivative):
        coeff *= 1
        source_kernels = [AxisSourceDerivative(knl.axis, source_knl) for
                source_knl in source_kernels]
        knl = knl.inner_kernel
    return coeff * int_g.copy(target_kernel=knl,
                        source_kernels=tuple(source_kernels))


def _convert_axis_source_to_directional_source(int_g):
    if not isinstance(int_g, IntG):
        return int_g
    knls = list(int_g.source_kernels)
    dim = knls[0].dim
    if len(knls) != dim:
        return int_g
    if not any(isinstance(knl, AxisSourceDerivative) for knl in knls):
        return int_g
    # TODO: sort
    axes = [knl.axis for knl in knls]
    if axes != list(range(dim)):
        return int_g
    base_knls = set(knl.inner_kernel for knl in knls)
    if len(base_knls) > 1:
        return int_g
    base_knl = base_knls.pop()
    kernel_arguments = int_g.kernel_arguments.copy()
    name = "generated_dir_vec"
    kernel_arguments[name] = \
            np.array(int_g.densities, dtype=np.object)
    res = int_g.copy(
            source_kernels=(
                DirectionalSourceDerivative(base_knl, dir_vec_name=name),),
            densities=(1,),
            kernel_arguments=kernel_arguments)
    return res


def merge_int_g_expr(expr):
    have_int_g = HaveIntGs()
    if have_int_g(expr) <= 1:
        return expr
    try:
        result_coeff, result_int_g = _merge_int_g_expr(expr)
        result_int_g = _convert_axis_source_to_directional_source(result_int_g)
        return result_coeff + result_int_g
    except AssertionError:
        return expr


def _merge_source_kernel_duplicates(source_kernels, densities):
    new_source_kernels = []
    new_densities = []
    for knl, density in zip(source_kernels, densities):
        if knl not in new_source_kernels:
            new_source_kernels.append(knl)
            new_densities.append(density)
        else:
            idx = new_source_kernels.index(knl)
            new_densities[idx] += density
    return new_source_kernels, new_densities


def _merge_kernel_arguments(x, y):
    res = x.copy()
    for k, v in y.items():
        if k in res:
            assert res[k] == v
        else:
            res[k] = v
    return res


def _merge_int_g_expr(expr):
    if isinstance(expr, Sum):
        result_coeff = 0
        result_int_g = 0
        for c in expr.children:
            coeff, int_g = _merge_int_g_expr(c)
            result_coeff += coeff
            if int_g == 0:
                continue
            if result_int_g == 0:
                result_int_g = int_g
                continue
            assert result_int_g.source == int_g.source
            assert result_int_g.target == int_g.target
            assert result_int_g.qbx_forced_limit == int_g.qbx_forced_limit
            assert result_int_g.target_kernel == int_g.target_kernel
            kernel_arguments = _merge_kernel_arguments(result_int_g.kernel_arguments,
                    int_g.kernel_arguments)
            source_kernels = result_int_g.source_kernels + int_g.source_kernels
            densities = result_int_g.densities + int_g.densities
            new_source_kernels, new_densities = \
                    _merge_source_kernel_duplicates(source_kernels, densities)
            result_int_g = result_int_g.copy(
                source_kernels=tuple(new_source_kernels),
                densities=tuple(new_densities),
                kernel_arguments=kernel_arguments,
            )
        return result_coeff, result_int_g
    elif isinstance(expr, Product):
        mult = 1
        found_int_g = None
        have_int_g = HaveIntGs()
        for c in expr.children:
            if not have_int_g(c):
                mult *= c
            elif found_int_g:
                raise RuntimeError("Not a linear expression.")
            else:
                found_int_g = c
        if not found_int_g:
            return expr, 0
        else:
            coeff, new_int_g = _merge_int_g_expr(found_int_g)
            new_densities = (density * mult for density in new_int_g.densities)
            return coeff*mult, new_int_g.copy(densities=new_densities)
    elif isinstance(expr, IntG):
        return 0, _convert_target_deriv_to_source(expr)
    else:
        return expr, 0


if __name__ == "__main__":
    from sumpy.kernel import StokesletKernel, BiharmonicKernel, StressletKernel
    base_kernel = BiharmonicKernel(3)
    kernels = [StokesletKernel(3, 0, 1), StokesletKernel(3, 0, 0)]
    kernels = [StressletKernel(3, 0, 1, 0), StressletKernel(3, 0, 0, 0),
            StressletKernel(3, 0, 1, 2)]
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=2, verbose=True)
