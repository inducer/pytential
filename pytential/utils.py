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


def sort_arrays_together(*arys, key=None):
    """Sort a sequence of arrays by considering them
    as an array of sequences using the given sorting key

    :param key: a function that takes in a tuple of values
                and returns a value to compare.
    """
    return zip(*sorted([x for x in zip(*arys)], key=key))


def chop(expr, tol):
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


def lu_solve_with_expand(L, U, perm, b):
    """Given an LU factorization and a vector, solve a linear
    system with intermediate results expanded to avoid
    an explosion of the expression trees

    :param L: lower triangular matrix
    :param U: upper triangular matrix
    :param perm: permutation matrix
    :param b: column vector to solve for
    """
    def forward_substitution(L, b):
        n = len(b)
        res = sym.Matrix(b)
        for i in range(n):
            for j in range(i):
                res[i] -= L[i, j]*res[j]
            res[i] = (res[i] / L[i, i]).expand()
        return res

    def backward_substitution(U, b):
        n = len(b)
        res = sym.Matrix(b)
        for i in range(n-1, -1, -1):
            for j in range(n - 1, i, -1):
                res[i] -= U[i, j]*res[j]
            res[i] = (res[i] / U[i, i]).expand()
        return res

    def permuteFwd(b, perm):
        res = sym.Matrix(b)
        for p, q in perm:
            res[p], res[q] = res[q], res[p]
        return res

    return backward_substitution(U,
            forward_substitution(L, permuteFwd(b, perm)))

# vim: foldmethod=marker
