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

from sumpy.symbolic import Matrix, make_sym_vector, sym, SympyToPymbolicMapper
from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)


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

    mat = Matrix(mat)
    res = []

    for kernel in kernels:
        expr = kernel.get_expression(sym_vec)
        vec = []
        for a in rand.T:
            vec.append(_n(expr.xreplace(dict((k, v) for k, v in zip(sym_vec, a)))))
        vec = Matrix(vec)
        result = []
        const = 0
        if verbose:
            print(kernel, end=" = ")
        for i, coeff in enumerate(mat.solve(vec)):
            coeff = _chop(coeff, tol)
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


if __name__ == "__main__":
    from sumpy.kernel import StokesletKernel, BiharmonicKernel, StressletKernel
    base_kernel = BiharmonicKernel(2)
    kernels = [StokesletKernel(2, 0, 1), StokesletKernel(2, 0, 0)]
    kernels = [StressletKernel(2, 0, 1, 0)]
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=3, verbose=True)
