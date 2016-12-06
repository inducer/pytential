from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
import numpy.linalg as la


def test_gmres():
    n = 200
    A = (  # noqa
            n * (np.eye(n) + 2j * np.eye(n))
            + np.random.randn(n, n) + 1j * np.random.randn(n, n))

    true_sol = np.random.randn(n) + 1j * np.random.randn(n)
    b = np.dot(A, true_sol)

    A_func = lambda x: np.dot(A, x)  # noqa
    A_func.shape = A.shape
    A_func.dtype = A.dtype

    from pytential.solve import gmres, ResidualPrinter
    tol = 1e-6
    sol = gmres(A_func, b, callback=ResidualPrinter(),
            maxiter=5*n, tol=tol).solution

    assert la.norm(true_sol - sol) / la.norm(sol) < tol


# You can test individual routines by typing
# $ python test_tools.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
