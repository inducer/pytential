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

import numpy as np

from pytential import sym
from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative,
    TargetPointMultiplier, LaplaceKernel)
from pytential.symbolic.stokes import (StressletWrapperBase, StokesletWrapperBase,
        _MU_SYM_DEFAULT)
from sumpy.symbolic import SpatialConstant


_NU_SYM_DEFAULT = SpatialConstant("nu")


class StressletWrapperYoshida(StressletWrapperBase):
    """Stresslet Wrapper using Yoshida et al's method [1] which uses Laplace
    derivatives.

    [1] Yoshida, K. I., Nishimura, N., & Kobayashi, S. (2001). Application of
        fast multipole Galerkin boundary integral equation method to elastostatic
        crack problems in 3D.
        International Journal for Numerical Methods in Engineering, 50(3), 525-547.
    """

    def __init__(self, dim=None, mu_sym=_MU_SYM_DEFAULT, nu_sym=_NU_SYM_DEFAULT):
        self.dim = dim
        if dim != 3:
            raise ValueError("unsupported dimension given to "
                "StressletWrapperYoshida")
        self.kernel = LaplaceKernel(dim=3)
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):
        return self.apply_stokeslet_and_stresslet([0]*self.dim,
            density_vec_sym, dir_vec_sym, qbx_forced_limit, 0, 1,
            extra_deriv_dirs)

    def apply_stokeslet_and_stresslet(self, stokeslet_density_vec_sym,
            stresslet_density_vec_sym, dir_vec_sym,
            qbx_forced_limit, stokeslet_weight, stresslet_weight,
            extra_deriv_dirs=()):

        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)
        stokeslet_weight *= -1

        def C(i, j, k, l):   # noqa: E741
            res = 0
            if i == j and k == l:
                res += lam
            if i == k and j == l:
                res += mu
            if i == l and j == k:
                res += mu
            return res * stresslet_weight

        def add_extra_deriv_dirs(target_kernel):
            for deriv_dir in extra_deriv_dirs:
                target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)
            return target_kernel

        def P(i, j, int_g):
            int_g = int_g.copy(target_kernel=add_extra_deriv_dirs(
                int_g.target_kernel))
            res = -int_g.copy(target_kernel=TargetPointMultiplier(j,
                    AxisTargetDerivative(i, int_g.target_kernel)))
            if i == j:
                res += (3 - 4*nu)*int_g
            return res / (4*mu*(1 - nu))

        def Q(i, int_g):
            res = int_g.copy(target_kernel=add_extra_deriv_dirs(
                AxisTargetDerivative(i, int_g.target_kernel)))
            return res / (4*mu*(1 - nu))

        sym_expr = np.zeros((3,), dtype=object)

        kernel = self.kernel
        source = [sym.NodeCoordinateComponent(d) for d in range(3)]
        normal = dir_vec_sym
        sigma = stresslet_density_vec_sym

        source_kernels = [None]*4
        for i in range(3):
            source_kernels[i] = AxisSourceDerivative(i, kernel)
        source_kernels[3] = kernel

        for i in range(3):
            for k in range(3):
                densities = [0]*4
                for l in range(3):   # noqa: E741
                    for j in range(3):
                        for m in range(3):
                            densities[l] += C(k, l, m, j)*normal[m]*sigma[j]
                densities[3] += stokeslet_weight * stokeslet_density_vec_sym[k]
                int_g = sym.IntG(target_kernel=kernel,
                    source_kernels=tuple(source_kernels),
                    densities=tuple(densities),
                    qbx_forced_limit=qbx_forced_limit)
                sym_expr[i] += P(i, k, int_g)

            densities = [0]*4
            for k in range(3):
                for m in range(3):
                    for j in range(3):
                        for l in range(3):   # noqa: E741
                            densities[l] += \
                                    C(k, l, m, j)*normal[m]*sigma[j]*source[k]
                            if k == l:
                                densities[3] += \
                                        C(k, l, m, j)*normal[m]*sigma[j]
                densities[3] += stokeslet_weight * source[k] \
                        * stokeslet_density_vec_sym[k]
            int_g = sym.IntG(target_kernel=kernel,
                source_kernels=tuple(source_kernels),
                densities=tuple(densities),
                qbx_forced_limit=qbx_forced_limit)
            sym_expr[i] += Q(i, int_g)

        return sym_expr


class StokesletWrapperYoshida(StokesletWrapperBase):
    """Stokeslet Wrapper using Yoshida et al's method [1] which uses Laplace
    derivatives.

    [1] Yoshida, K. I., Nishimura, N., & Kobayashi, S. (2001). Application of
        fast multipole Galerkin boundary integral equation method to elastostatic
        crack problems in 3D.
        International Journal for Numerical Methods in Engineering, 50(3), 525-547.
    """

    def __init__(self, dim=None, mu_sym=_MU_SYM_DEFAULT, nu_sym=_NU_SYM_DEFAULT):
        self.dim = dim
        if dim != 3:
            raise ValueError("unsupported dimension given to "
                "StokesletWrapperYoshida")
        self.kernel = LaplaceKernel(dim=3)
        self.mu = mu_sym
        self.nu = nu_sym
        self.stresslet = StressletWrapperYoshida(3, self.mu, self.nu)

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
        return self.stresslet.apply_stokeslet_and_stresslet(density_vec_sym,
            [0]*self.dim, [0]*self.dim, qbx_forced_limit, 1, 0,
            extra_deriv_dirs)
