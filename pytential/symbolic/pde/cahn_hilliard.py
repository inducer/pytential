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
    def __init__(self, lambda1, lambda2, c):
        self.lambdas = (lambda1, lambda2)
        self.c = c

    def make_unknown(self, name):
        return sym.make_sym_vector(name, 2)

    def S_G(self, i, density, qbx_forced_limit, op_map=None):
        if op_map is None:
            op_map = lambda x: x  # noqa: E731

        from sumpy.kernel import HelmholtzKernel
        hhk = HelmholtzKernel(2, allow_evanescent=True)
        hhk_scaling = 1j/4

        if i == 0:
            lam1, lam2 = self.lambdas
            return (
                    # FIXME: Verify scaling
                    -1/(2*np.pi*(lam1**2-lam2**2)) / hhk_scaling
                    * (
                        op_map(sym.S(hhk, density, k=1j*lam1,
                            qbx_forced_limit=qbx_forced_limit))
                        - op_map(sym.S(hhk, density, k=1j*lam2,
                            qbx_forced_limit=qbx_forced_limit))))
        else:
            return (
                    # FIXME: Verify scaling

                    -1/(2*np.pi) / hhk_scaling
                    * op_map(sym.S(hhk, density, k=1j*self.lambdas[i-1],
                        qbx_forced_limit=qbx_forced_limit)))

    def representation(self, unknown):
        sig1, sig2 = unknown
        S_G = partial(self.S_G, qbx_forced_limit=None)

        return S_G(1, sig1) + S_G(0, sig2)

    def operator(self, unknown):
        sig1, sig2 = unknown
        lam1, lam2 = self.lambdas
        S_G = partial(self.S_G, qbx_forced_limit=1)

        c = self.c

        def Sn_G(i, density):
            return self.S_G(i, density,
                        qbx_forced_limit="avg",
                        op_map=partial(sym.normal_derivative, 2))

        from pytools import obj_array

        d = obj_array.new_1d([
            0.5*sig1,
            0.5*lam2**2*sig1 - 0.5*sig2
            ])
        a = obj_array.new_1d([
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
