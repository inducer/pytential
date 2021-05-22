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
from pytential.symbolic.pde.system_utils import merge_int_g_exprs
from sumpy.kernel import (LineOfCompressionKernel,
    AxisTargetDerivative, AxisSourceDerivative, TargetPointMultiplier)
from pymbolic import var
from pytential.symbolic.stokes import HebekerExteriorStokesOperator, StokesOperator
from pytential.symbolic.primitives import NodeCoordinateComponent


class MindlinOperator(StokesOperator):
    """Representation for 3D Stokes Flow based on [Hebeker1986]_.

    Inherits from :class:`StokesOperator`.

    .. [Hebeker1986] F. C. Hebeker, *Efficient Boundary Element Methods for
        Three-Dimensional Exterior Viscous Flow*, Numerical Methods for
        Partial Differential Equations, Vol. 2, 1986,
        `DOI <https://doi.org/10.1002/num.1690020404>`__.

    .. automethod:: __init__
    """

    def __init__(self, *, method="biharmonic", mu_sym=var("mu"), nu_sym=var("nu")):
        if method not in ["biharmonic", "laplace"]:
            raise ValueError(f"invalid method: {method}."
                    "Needs to be one of laplace, biharmonic")

        self.method = method
        self.mu = mu_sym
        self.nu = nu_sym
        self._stokes = HebekerExteriorStokesOperator(eta=0, method=method,
                mu_sym=mu_sym, nu_sym=nu_sym)
        self._stokes_a = HebekerExteriorStokesOperator(eta=0, method=method,
                mu_sym=mu_sym + 4*nu_sym, nu_sym=-nu_sym)
        self.dim = 3
        self.compression_knl = LineOfCompressionKernel(self.dim, 2, mu_sym, nu_sym)

    def K(self, sigma, normal, qbx_forced_limit):
        return merge_int_g_exprs(self._stokes.stresslet.apply(sigma, normal,
                qbx_forced_limit=qbx_forced_limit))

    def A(self, sigma, normal, qbx_forced_limit):
        result = -self._stokes_a.stresslet.apply(sigma, normal,
                qbx_forced_limit=qbx_forced_limit)
        new_density = sum(a*b for a, b in zip(sigma, normal))
        for i in range(self.dim):
            result[i] += 2*self._create_int_g(self._stokes.laplace_kernel, [], [i],
                    new_density, qbx_forced_limit)
        return result

    def B(self, sigma, normal, qbx_forced_limit):

        def create_int_g(density, source_dirs, target_dir):
            knl = self.compression_knl
            for source_dir in source_dirs:
                knl = AxisSourceDerivative(source_dir, knl)
            knl = AxisTargetDerivative(target_dir, knl)
            kwargs = {}
            args = [arg.loopy_arg.name for arg in knl.get_args()]
            for arg in args:
                kwargs[arg] = var(arg)

            return sym.int_g_vec(knl, density, qbx_forced_limit=qbx_forced_limit,
                    **kwargs)

        sym_expr = np.zeros((self.dim,), dtype=object)
        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)
        sigma_normal_product = sum(a*b for a, b in zip(sigma, normal))

        for i in range(self.dim):
            val = self._create_int_g(sigma[0]*normal[0], [0, 0], [i])
            val += self._create_int_g(sigma[1]*normal[1], [1, 1], [i])
            val -= self._create_int_g(sigma[2]*normal[2], [2, 2], [i])
            val += self._create_int_g(sigma[0]*normal[1]+sigma[1]*normal[0],
                                                          [0, 1], [i])
            val *= 2*mu
            val -= 2*lam*create_int_g(sigma_normal_product, [2, 2], i)
            if i == 2:
                val *= -1
            sym_expr[i] = val

        return sym_expr

    def C(self, sigma, normal, qbx_forced_limit):
        result = np.zeros((self.dim,), dtype=object)
        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)
        alpha = (lam + mu)/(lam + 2*mu)
        y = [NodeCoordinateComponent(i) for i in range(self.dim)]
        sigma_normal_product = sum(a*b for a, b in zip(sigma, normal))

        def get_laplace_int_g(density, source_deriv_dirs, target_deriv_dirs=[]):
            int_g = self._create_int_g(self._stokes.laplace_kernel,
                source_deriv_dirs, target_deriv_dirs, density,
                qbx_forced_limit=qbx_forced_limit)
            return int_g.copy(
                target_kernel=TargetPointMultiplier(2, int_g.target_kernel))

        for i in range(self.dim):
            # phi_c  in Gimbutas et, al.
            temp = get_laplace_int_g(y[2]*sigma[0]*normal[0], [0, 0], [i])
            temp += get_laplace_int_g(y[2]*sigma[1]*normal[1], [1, 1], [i])
            temp += get_laplace_int_g(y[2]*sigma[2]*normal[2], [2, 2], [i])
            temp += get_laplace_int_g(y[2]*(sigma[0]*normal[1] + sigma[1]*normal[0]),
                                                               [0, 1], [i])
            temp -= get_laplace_int_g(y[2]*(sigma[0]*normal[2] + sigma[2]*normal[0]),
                                                               [0, 2], [i])
            temp -= get_laplace_int_g(y[2]*(sigma[1]*normal[2] + sigma[2]*normal[1]),
                                                               [1, 2], [i])
            temp *= -2*alpha*mu
            result[i] += temp

            # G in Gimbutas et, al.
            temp = get_laplace_int_g(
                y[2]*mu*(sigma[0]*normal[2] + sigma[2]*normal[0]), [0], [i])
            temp += get_laplace_int_g(
                y[2]*mu*(sigma[0]*normal[2] + sigma[2]*normal[0]), [1], [i])
            temp += get_laplace_int_g(
                y[2]*(mu*-2*sigma[2]*normal[2] - lam*sigma_normal_product), [2], [i])
            temp *= (2*alpha - 2)
            result[i] += temp

        # H in Gimubtas et, al.
        temp = get_laplace_int_g(sigma[0]*normal[0], [0, 0])
        temp += get_laplace_int_g(sigma[1]*normal[1], [1, 1])
        temp += get_laplace_int_g(sigma[2]*normal[2], [2, 2])
        temp += get_laplace_int_g(sigma[0]*normal[1] + sigma[1]*normal[0], [0, 1])
        temp -= get_laplace_int_g(sigma[0]*normal[2] + sigma[2]*normal[0], [0, 2])
        temp -= get_laplace_int_g(sigma[1]*normal[2] + sigma[2]*normal[1], [1, 2])

        result[2] -= -2*(2 - alpha)*mu*temp

        return result

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        # A and C are both derivatives of Biharmonic Green's function
        # TODO: make merge_int_g_exprs smart enough to merge two different
        # kernels into two separate IntGs.
        result = self.A(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)
        result += self.C(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)
        result = merge_int_g_exprs(result, base_kernel=self._stokes.base_kernel)

        result += merge_int_g_exprs(self.B(sigma, normal=normal,
            qbx_forced_limit=qbx_forced_limit), base_kernel=self.compression_knl)
        return result

    def _create_int_g(self, knl, source_deriv_dirs, target_deriv_dirs, density,
            **kwargs):
        for deriv_dir in source_deriv_dirs:
            knl = AxisSourceDerivative(deriv_dir, knl)

        for deriv_dir in target_deriv_dirs:
            knl = AxisTargetDerivative(deriv_dir, knl)
            if deriv_dir == 2:
                density *= -1

        args = [arg.loopy_arg.name for arg in knl.get_args()]
        for arg in args:
            kwargs[arg] = var(arg)

        res = sym.S(knl, density, **kwargs)
        return res
