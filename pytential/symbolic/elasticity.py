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
    AxisTargetDerivative, AxisSourceDerivative, TargetPointMultiplier,
    LaplaceKernel, BiharmonicKernel)
from pymbolic import var
from pytential.symbolic.stokes import (
    StokesletWrapperTornberg, StressletWrapper, StokesletWrapper,
    StokesletWrapperBase)
from pytential.symbolic.primitives import NodeCoordinateComponent


class KelvinOperator:

    def __init__(self, method, mu_sym, nu_sym):
        if nu_sym == 0.5:
            raise ValueError("poisson's ratio cannot be 0.5")

        self.dim = 3
        self.mu = mu_sym
        self.nu = nu_sym

        if method == "laplace":
            self.stresslet = StressletWrapperYoshida(dim=self.dim,
                    mu_sym=mu_sym, nu_sym=nu_sym)
            self.stokeslet = StokesletWrapperTornberg(dim=self.dim,
                    mu_sym=mu_sym, nu_sym=nu_sym)
        elif method == "biharmonic":
            self.stresslet = StressletWrapper(dim=self.dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
            self.stokeslet = StokesletWrapper(dim=self.dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
        else:
            raise ValueError(f"invalid method: {method}."
                    "Needs to be one of naive, laplace, biharmonic")

        if method == "biharmonic":
            self.base_kernel = BiharmonicKernel(dim=self.dim)
        else:
            self.base_kernel = None

        self.laplace_kernel = LaplaceKernel(dim=self.dim)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        return merge_int_g_exprs(self.stresslet.apply(sigma, normal,
            qbx_forced_limit=qbx_forced_limit),
            base_kernel=self.base_kernel)


class MindlinOperator:
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
        self.free_space_op = KelvinOperator(method=method, mu_sym=mu_sym,
                nu_sym=nu_sym)
        self.modified_free_space_op = KelvinOperator(
                method=method, mu_sym=mu_sym + 4*nu_sym, nu_sym=-nu_sym)
        self.dim = 3
        self.compression_knl = LineOfCompressionKernel(self.dim, 2, mu_sym, nu_sym)

    def K(self, sigma, normal, qbx_forced_limit):
        return merge_int_g_exprs(self.free_space_op.stresslet.apply(sigma, normal,
                qbx_forced_limit=qbx_forced_limit))

    def A(self, sigma, normal, qbx_forced_limit):
        result = -self.modified_free_space_op.stresslet.apply(sigma, normal,
            qbx_forced_limit=qbx_forced_limit)

        new_density = sum(a*b for a, b in zip(sigma, normal))
        int_g = sym.S(self.free_space_op.laplace_kernel, new_density,
            qbx_forced_limit=qbx_forced_limit)

        for i in range(self.dim):
            temp = 2*int_g.copy(
                    target_kernel=AxisTargetDerivative(i, int_g.target_kernel))
            if i == 2:
                temp *= -1
            result[i] += temp
        return result

    def _create_int_g(self, knl, source_deriv_dirs, target_deriv_dirs, density,
            **kwargs):

        for deriv_dir in target_deriv_dirs:
            knl = AxisTargetDerivative(deriv_dir, knl)
            if deriv_dir == 2:
                density *= -1

        args = [arg.loopy_arg.name for arg in knl.get_args()]
        for arg in args:
            kwargs[arg] = var(arg)

        res = sym.S(knl, density, **kwargs)
        return res

    def B(self, sigma, normal, qbx_forced_limit):
        sym_expr = np.zeros((self.dim,), dtype=object)
        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)
        sigma_normal_product = sum(a*b for a, b in zip(sigma, normal))

        source_kernel_dirs = [[0, 0], [1, 1], [2, 2], [0, 1]]
        densities = [
            sigma[0]*normal[0]*2*mu,
            sigma[1]*normal[1]*2*mu,
            -sigma[2]*normal[2]*2*mu - 2*lam*sigma_normal_product,
            (sigma[0]*normal[1] + sigma[1]*normal[0])*2*mu,
        ]
        source_kernels = [
            AxisSourceDerivative(a, AxisSourceDerivative(b, self.compression_knl))
            for a, b in source_kernel_dirs
        ]

        kwargs = {"qbx_forced_limit": qbx_forced_limit}
        args = [arg.loopy_arg.name for arg in self.compression_knl.get_args()]
        for arg in args:
            kwargs[arg] = var(arg)

        int_g = sym.IntG(source_kernels=tuple(source_kernels),
                target_kernel=self.compression_knl, densities=tuple(densities),
                **kwargs)

        for i in range(self.dim):
            sym_expr[i] = int_g.copy(target_kernel=AxisTargetDerivative(
                i, int_g.target_kernel))

        return sym_expr

    def C(self, sigma, normal, qbx_forced_limit):
        result = np.zeros((self.dim,), dtype=object)
        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)
        alpha = (lam + mu)/(lam + 2*mu)
        y = [NodeCoordinateComponent(i) for i in range(self.dim)]
        sigma_normal_product = sum(a*b for a, b in zip(sigma, normal))

        laplace_kernel = self.free_space_op.laplace_kernel
        densities = []
        source_kernels = []

        # phi_c  in Gimbutas et, al.
        densities.extend([
            -2*alpha*mu*y[2]*sigma[0]*normal[0],
            -2*alpha*mu*y[2]*sigma[1]*normal[1],
            -2*alpha*mu*y[2]*sigma[2]*normal[2],
            -2*alpha*mu*y[2]*(sigma[0]*normal[1] + sigma[1]*normal[0]),
            +2*alpha*mu*y[2]*(sigma[0]*normal[2] + sigma[2]*normal[0]),
            +2*alpha*mu*y[2]*(sigma[1]*normal[2] + sigma[2]*normal[1]),
        ])
        source_kernel_dirs = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]
        source_kernels.extend([
            AxisSourceDerivative(a, AxisSourceDerivative(b, laplace_kernel))
            for a, b in source_kernel_dirs
        ])

        # G in Gimbutas et, al.
        densities.extend([
            (2*alpha - 2)*y[2]*mu*(sigma[0]*normal[2] + sigma[2]*normal[0]),
            (2*alpha - 2)*y[2]*mu*(sigma[1]*normal[2] + sigma[2]*normal[1]),
            (2*alpha - 2)*y[2]*(mu*-2*sigma[2]*normal[2]
                - lam*sigma_normal_product),
        ])
        source_kernels.extend(
            [AxisSourceDerivative(i, laplace_kernel) for i in range(3)])

        int_g = sym.IntG(source_kernels=tuple(source_kernels),
                target_kernel=laplace_kernel, densities=tuple(densities),
                qbx_forced_limit=qbx_forced_limit)

        for i in range(self.dim):
            result[i] = int_g.copy(target_kernel=AxisTargetDerivative(
                i, int_g.target_kernel))

            if i == 2:
                # Target derivative w.r.t x[2] is flipped due to target image
                result[i] *= -1

        # H in Gimubtas et, al.
        densities = [
            (-2)*(2 - alpha)*mu*sigma[0]*normal[0],
            (-2)*(2 - alpha)*mu*sigma[1]*normal[1],
            (-2)*(2 - alpha)*mu*sigma[2]*normal[2],
            (-2)*(2 - alpha)*mu*(sigma[0]*normal[1] + sigma[1]*normal[0]),
            (+2)*(2 - alpha)*mu*(sigma[0]*normal[2] + sigma[2]*normal[0]),
            (+2)*(2 - alpha)*mu*(sigma[1]*normal[2] + sigma[2]*normal[1]),
        ]
        source_kernel_dirs = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]
        source_kernels = [
            AxisSourceDerivative(a, AxisSourceDerivative(b, laplace_kernel))
            for a, b in source_kernel_dirs
        ]
        H = sym.IntG(source_kernels=tuple(source_kernels),
                target_kernel=laplace_kernel, densities=tuple(densities),
                qbx_forced_limit=qbx_forced_limit)
        result[2] -= H

        return result

    def free_space_operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        return self.free_space_op.operator(sigma=sigma, normal=normal,
            qbx_forced_limit=qbx_forced_limit)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        resultA = self.A(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)
        resultB = self.C(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)
        resultC = self.B(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)

        if self.method == "biharmonic":
            # A and C are both derivatives of Biharmonic Green's function
            # TODO: make merge_int_g_exprs smart enough to merge two different
            # kernels into two separate IntGs.
            result = merge_int_g_exprs(resultA + resultB,
                base_kernel=self.free_space_op.base_kernel)
            result += merge_int_g_exprs(resultC, base_kernel=self.compression_knl)
            return result
        else:
            return resultA + resultB + resultC

    def get_density_var(self, name="sigma"):
        """
        :returns: a symbolic vector corresponding to the density.
        """
        return sym.make_sym_vector(name, self.dim)


class StressletWrapperYoshida(StokesletWrapperBase):
    """Stresslet Wrapper using Yoshida et al's method [1] which uses Laplace
    derivatives.

    [1] Yoshida, K. I., Nishimura, N., & Kobayashi, S. (2001). Application of
        fast multipole Galerkin boundary integral equation method to elastostatic
        crack problems in 3D.
        International Journal for Numerical Methods in Engineering, 50(3), 525-547.
    """

    def __init__(self, dim=None, mu_sym=var("mu"), nu_sym=0.5):
        self.dim = dim
        if dim != 3:
            raise ValueError("unsupported dimension given to "
                "StressletWrapperYoshida")
        self.kernel = LaplaceKernel(dim=self.dim)
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit):

        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)

        def C(i, j, k, l):   # noqa: E741
            res = 0
            if i == j and k == l:
                res += lam
            if i == k and j == l:
                res += mu
            if i == l and j == k:
                res += mu
            return res

        def P(i, j, int_g):
            res = -int_g.copy(target_kernel=TargetPointMultiplier(j,
                AxisTargetDerivative(i, int_g.target_kernel)))
            if i == j:
                res += (3 - 4*nu)*int_g
            return res / (4*mu*(1 - nu))

        def Q(i, int_g):
            res = int_g.copy(target_kernel=AxisTargetDerivative(i,
                int_g.target_kernel))
            return res / (4*mu*(1 - nu))

        sym_expr = np.zeros((self.dim,), dtype=object)

        kernel = self.kernel
        source = [sym.NodeCoordinateComponent(d) for d in range(self.dim)]
        normal = dir_vec_sym
        sigma = density_vec_sym

        for i in range(3):
            for k in range(3):
                source_kernels = [None]*3
                densities = [0]*3
                for l in range(3):   # noqa: E741
                    source_kernels[l] = AxisSourceDerivative(l, kernel)
                    for j in range(3):
                        for m in range(3):
                            densities[l] += C(k, l, m, j)*normal[m]*sigma[j]
                int_g = sym.IntG(target_kernel=kernel,
                    source_kernels=tuple(source_kernels),
                    densities=tuple(densities),
                    qbx_forced_limit=qbx_forced_limit)
                sym_expr[i] += P(i, j, int_g)

            source_kernels = [None]*4
            densities = [0]*4
            for l in range(3):   # noqa: E741
                source_kernels[l] = AxisSourceDerivative(l, kernel)
            source_kernels[3] = kernel

            for k in range(3):
                for m in range(3):
                    for j in range(3):
                        for l in range(3):   # noqa: E741
                            densities[l] += \
                                    C(k, l, m, j)*normal[m]*sigma[j]*source[k]
                            if k == l:
                                densities[3] += \
                                        C(k, l, m, j)*normal[m]*sigma[j]
            int_g = sym.IntG(target_kernel=kernel,
                source_kernels=tuple(source_kernels),
                densities=tuple(densities),
                qbx_forced_limit=qbx_forced_limit)
            sym_expr[i] += Q(i, int_g)

        return sym_expr
