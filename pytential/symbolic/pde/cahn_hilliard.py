from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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


__doc__ = """
.. autoclass:: CahnHilliardOperator
"""

import numpy as np
from pytential.symbolic.pde.scalar import L2WeightedPDEOperator
from pytential import sym
from functools import partial


class CahnHilliardOperator(L2WeightedPDEOperator):
    def __init__(self, b, c):
        self.b = b
        self.c = c

        # Issue:
        # - let crat = np.abs(4.*c) / ( b**2 + 1e-12 )
        # - when crat is close to zero, sqrt(b**2-4*c) is close to abs(b),
        #   then for b>=0, sqrt(b**2-4*c) - b is inaccurate.
        # - similarly, when crat is close to one, sqrt(b**2-4*c) is close to zero,
        #   then for b>0, sqrt(b**2-4*c) + b is inaccurate.
        # Solution:
        # - find a criteria for crat to choose from formulae, or
        # - use the computed root with smaller residual
        def quadratic_formula_1(a, b, c):
            return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

        def quadratic_formula_2(a, b, c):
            return (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

        def citardauq_formula_1(a, b, c):
            return 2*c / (-b - np.sqrt(b**2-4*a*c))

        def citardauq_formula_2(a, b, c):
            return 2*c / (-b + np.sqrt(b**2-4*a*c))

        def f(x):
            return x**2 - b*x + c

        root11 = quadratic_formula_1(1, -b, c)
        root12 = citardauq_formula_1(1, -b, c)
        if np.abs(f(root11)) < np.abs(f(root12)):
            lam1 = np.sqrt(root11)
        else:
            lam1 = np.sqrt(root12)

        root21 = quadratic_formula_2(1, -b, c)
        root22 = citardauq_formula_2(1, -b, c)
        if np.abs(f(root21)) < np.abs(f(root22)):
            lam1 = np.sqrt(root21)
        else:
            lam2 = np.sqrt(root22)

        assert np.abs(f(lam1**2)) < 1e-12
        assert np.abs(f(lam2**2)) < 1e-12

        self.lambdas = sorted([lam1, lam2], key=abs, reverse=True)  # biggest first

    def make_unknown(self, name):
        return sym.make_sym_vector(name, 2)

    def S_G(self, i, density, qbx_forced_limit, op_map=None):  # noqa: N802
        if op_map is None:
            op_map = lambda x: x  # noqa: E731

        from sumpy.kernel import YukawaKernel
        knl = YukawaKernel(2)

        if i == 0:
            lam1, lam2 = self.lambdas
            return (
                    1/(lam1**2-lam2**2)
                    * (
                        op_map(sym.S(knl, density, lam=lam1,
                            qbx_forced_limit=qbx_forced_limit))
                        -
                        op_map(sym.S(knl, density, lam=lam2,
                            qbx_forced_limit=qbx_forced_limit))))
        else:
            return (
                    op_map(sym.S(knl, density, lam=self.lambdas[i-1],
                        qbx_forced_limit=qbx_forced_limit)))

    def representation(self, unknown):
        """Return (u, v) in a :mod:`numpy` object array.
        """
        sig1, sig2 = unknown
        S_G = partial(self.S_G, qbx_forced_limit=None)  # noqa: N806
        laplacian = partial(sym.laplace, 2)

        def u(op_map=None):
            return S_G(1, sig1, op_map=op_map) + S_G(0, sig2, op_map=op_map)

        return sym.make_obj_array([
            u(),
            -u(op_map=laplacian) + self.b*u()
            ])

    def operator(self, unknown):
        sig1, sig2 = unknown
        lam1, lam2 = self.lambdas
        S_G = partial(self.S_G, qbx_forced_limit=1)  # noqa: N806

        c = self.c

        def Sn_G(i, density):  # noqa
            return self.S_G(i, density,
                        qbx_forced_limit="avg",
                        op_map=partial(sym.normal_derivative, 2))

        d = sym.make_obj_array([
            0.5*sig1,
            0.5*lam2**2*sig1 - 0.5*sig2
            ])
        a = sym.make_obj_array([
            # A11
            Sn_G(1, sig1) + c*S_G(1, sig1)
            # A12
            + Sn_G(0, sig2) + c*S_G(0, sig2),

            # A21
            lam2**2*Sn_G(1, sig1)
            # A22
            - Sn_G(1, sig2) + lam1**2*Sn_G(0, sig2)
            ])

        return d+a
