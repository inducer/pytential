__copyright__ = """
Copyright (C) 2020 Isuru Fernando
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

import sumpy.symbolic as sym
from typing import Iterable, Callable


def sort_arrays_together(*arys, key=None):
    """Sort a sequence of arrays by considering them
    as an array of sequences using the given sorting key

    :param key: a function that takes in a tuple of values
                and returns a value to compare.
    """
    return zip(*sorted(zip(*arys), key=key))


def chop(expr: sym.Basic, tol) -> sym.Basic:
    """Given a symbolic expression, remove all occurences of numbers
    with absolute value less than a given tolerance and replace floating
    point numbers that are close to an integer up to a given relative
    tolerance by the integer.
    """
    nums = expr.atoms(sym.Number)
    replace_dict = {}
    for num in nums:
        if float(abs(num)) < tol:
            replace_dict[num] = 0
        else:
            new_num = float(num)
            if abs((int(new_num) - new_num)/new_num) < tol:
                new_num = int(new_num)
            replace_dict[num] = new_num
    return expr.xreplace(replace_dict)


def forward_substitution(
        L: sym.Matrix,
        b: sym.Matrix,
        postprocess_division: Callable[[sym.Basic], sym.Basic],
        ) -> sym.Matrix:
    """Given a lower triangular matrix *L* and a column vector *b*,
    solve the system ``Lx = b`` and apply the callable *postprocess_division*
    on each expression at the end of division calls.
    """
    n = len(b)
    res = sym.Matrix(b)
    for i in range(n):
        for j in range(i):
            res[i] -= L[i, j]*res[j]
        res[i] = postprocess_division(res[i] / L[i, i])
    return res


def backward_substitution(
        U: sym.Matrix,
        b: sym.Matrix,
        postprocess_division: Callable[[sym.Basic], sym.Basic],
        ) -> sym.Matrix:
    """Given an upper triangular matrix *U* and a column vector *b*,
    solve the system ``Ux = b`` and apply the callable *postprocess_division*
    on each expression at the end of division calls.
    """
    n = len(b)
    res = sym.Matrix(b)
    for i in range(n-1, -1, -1):
        for j in range(n - 1, i, -1):
            res[i] -= U[i, j]*res[j]
        res[i] = callback(res[i] / U[i, i])
    return res


def solve_from_lu(
            L: sym.Matrix,
            U: sym.Matrix,
            perm: Iterable[int],
            b: sym.Matrix,
            postprocess_division: Callable[[sym.Basic], sym.Basic]
        ) -> sym.Matrix:
    """Given an LU factorization and a vector, solve a linear
    system with intermediate results expanded to avoid
    an explosion of the expression trees

    :param L: lower triangular matrix
    :param U: upper triangular matrix
    :param perm: permutation matrix
    :param b: a column vector to solve for
    :param postprocess_division: callable that is called after each division
    """
    # Permute first
    res = sym.Matrix(b)
    for p, q in perm:
        res[p], res[q] = res[q], res[p]

    return backward_substitution(U, forward_substitution(L, res))

# vim: foldmethod=marker
