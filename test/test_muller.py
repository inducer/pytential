from __future__ import division

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

import numpy as np
import pytest


@pytest.mark.parametrize("true_roots", [
    np.linspace(1, 20, 19),
    np.exp(1j*np.linspace(0, 2*np.pi, 5)), # double root at 1
    np.exp(1j*np.linspace(0, 2*np.pi, 15, endpoint=False)),
    ])
def test_muller(true_roots):
    """
    :arg n: number of zeros sought
    :return: (roots, niter, err)
    """
    maxiter = 100
    eps = 1e-12
    from pytential.muller import muller_deflate
    roots, niter, err = muller_deflate(
        lambda z: poly_with_roots(z, true_roots), len(true_roots))

    for r_i in roots:
        min_dist, true_root = min(
            (abs(r_i - root), root) for root in true_roots)
        print min_dist, true_root
        assert min_dist < eps * abs(true_root)


def poly_with_roots(z, roots):
    """
    :a polynomial with 0, ..., n-1 as its roots
    """
    y = 1.0
    for root in roots:
        y = y*(z-root)

    return y

def fun1(z,n):
    """
    :a polynomial with exp(1j i 2 pi/n), i=0,...,n-1 as its roots
    """
    theta = np.linspace(0,2*np.pi,n,endpoint=False)
    
    y = 1.0
    for i in range(n):
        y = y*(z-np.exp(1j*theta[i]))

    return y

if __name__ == "__main__":
    test_muller()
