__copyright__ = "Copyright (C) 2014 Shidong Jiang, Andreas Kloeckner"

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

from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar("T", float, complex)


def muller_deflate(
        f: Callable[[T], T], n: int, *,
        maxiter: int = 100,
        eps: float = 1.0e-14,
        z_start: Optional[np.ndarray] = None,
        ) -> Tuple[List[T], List[int]]:
    """
    :arg n: number of zeros sought.
    :returns: a tuple of ``(roots, niter)``, where *roots* is a list of roots
        of the given function and *niter* is the number of iterations required to
        find each root.
    """
    # initialize variables
    roots: List[T] = []
    niter: List[int] = []

    def f_deflated(z: T) -> T:
        y = f(z)
        for r in roots:
            y = y/(z-r)

        return y

    # finds n roots
    # checks for NaN which signifies the end of the root finding process.
    # Truncates the zero arrays created above if necessary.
    for i in range(n):
        miter = 0
        roots0, niter0 = muller(f_deflated, maxiter=maxiter, tol=eps, z_start=z_start)
        roots.append(roots0)
        niter.append(niter0)

        while (np.isnan(roots[i]) or niter[i] == maxiter) and miter < 50:
            roots0, niter0 = muller(f_deflated,
                                    maxiter=maxiter, tol=eps, z_start=z_start)
            roots[i] = roots0
            niter[i] = niter0
            miter = miter+1

    return roots, niter


def muller(f: Callable[[T], T], *,
           maxiter: int = 100,
           tol: float = 1.0e-11,
           z_start: Optional[np.ndarray] = None) -> Tuple[T, int]:
    """Find a root of the complex-valued function *f* defined in the complex
    plane using Muller's method.

    :arg z_start: *None* or a 3-vector of complex numbers used as a
        starting guess.
    :returns: ``(roots, niter)`` root of the given function; number of
        iterations used.

    [1] https://en.wikipedia.org/wiki/Muller%27s_method
    """
    # initialize variables
    niter = 0

    if z_start is None:
        rng = np.random.default_rng()
        z_start = rng.random(3) + 1j*rng.random(3)

    z1, z2, z3 = z_start

    w1 = f(z1)
    w2 = f(z2)
    w3 = f(z3)

    while True:
        niter = niter + 1
        if niter >= maxiter:
            raise RuntimeError(f"convergence not achieved in {maxiter} iterations")

        h1 = z2 - z1
        h2 = z3 - z2
        lambda_ = h2/h1
        g = w1*lambda_*lambda_ - w2*(1+lambda_)*(1+lambda_)+w3*(1+2*lambda_)
        det = g*g - 4*w3*(1+lambda_)*lambda_*(w1*lambda_-w2*(1+lambda_)+w3)

        h1 = g + np.sqrt(det)
        h2 = g - np.sqrt(det)

        if np.abs(h1) > np.abs(h2):
            lambda_ = -2*w3*(1.0+lambda_)/h1
        else:
            lambda_ = -2*w3*(1.0+lambda_)/h2

        z1 = z2
        w1 = w2
        z2 = z3
        w2 = w3
        z3 = z2+lambda_*(z2-z1)
        w3 = f(z3)

        if np.abs(z3) < 1e-14:
            move_mag = np.abs(z3-z2)
        else:
            move_mag = np.abs((z3-z2)/z3)

        if move_mag < tol:
            return z3, niter
