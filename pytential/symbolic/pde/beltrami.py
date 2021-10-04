__copyright__ = "Copyright (C) 2021 Alexandru Fikl"

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
.. autoclass:: LaplaceBeltramiOperator
.. autoclass:: YukawaBeltramiOperator
.. autoclass:: HelmholtzbeltramiOperator
"""

from functools import partial
from typing import Any, Dict, Optional

import numpy as np

from pytential import sym
from sumpy.kernel import Kernel


# {{{ beltrami operator

class BeltramiOperator:
    def __init__(self, kernel: Kernel, *,
            precond: str = "left",
            kernel_arguments: Optional[Dict[str, Any]] = None) -> None:
        if kernel_arguments is None:
            kernel_arguments = {}

        if precond not in ("left", "right"):
            raise ValueError(f"unknown preconditioning '{precond}'")

        self.kernel = kernel
        self.precond = precond
        self.kernel_arguments = kernel_arguments

    @property
    def dim(self):
        return self.kernel.dim

    @property
    def dtype(self):
        return np.dtype(
                np.complex128 if self.kernel.is_complex_valued
                else np.float64)

    def get_density_var(self, name: str = "sigma"):
        return sym.var(name)

    def prepare_solution(self, sigma: sym.var) -> sym.var:
        S = partial(sym.S, self.kernel,     # noqa: N806
                qbx_forced_limit=+1, **self.kernel_arguments)

        if self.precond == "left":
            return S(sigma)
        else:
            return S(S(sigma))

    def prepare_rhs(self, b: sym.var) -> sym.var:
        S = partial(sym.S, self.kernel,     # noqa: N806
                qbx_forced_limit=+1, **self.kernel_arguments)

        if self.precond == "left":
            return S(b)
        else:
            return b

    def _get_left_operator(self, sigma: sym.var, kappa: sym.var, **kwargs: Any):
        from sumpy.kernel import LaplaceKernel
        if isinstance(self.kernel, LaplaceKernel):
            raise TypeError(
                    f"{type(self).__name__} does not support the Laplace "
                    "kernel, use LaplaceBeltramiOperator instead")

        context = self.kernel_arguments.copy()
        context.update(kwargs)
        kappa = (self.dim - 1) * kappa

        knl = self.kernel
        lknl = LaplaceKernel(self.dim)

        # {{{ layer potentials

        # laplace
        S0 = partial(sym.S, lknl, qbx_forced_limit=+1, **kwargs)        # noqa: N806
        D0 = partial(sym.D, lknl, qbx_forced_limit="avg", **kwargs)     # noqa: N806
        Dp0 = partial(sym.Dp, lknl, qbx_forced_limit="avg", **kwargs)   # noqa: N806

        # base
        S = partial(sym.S, knl, qbx_forced_limit=+1, **context)         # noqa: N806
        Sp = partial(sym.Sp, knl, qbx_forced_limit="avg", **context)    # noqa: N806
        Spp = partial(sym.Spp, knl, qbx_forced_limit="avg", **context)  # noqa: N806

        # }}}

        # {{{ operator

        # similar to 6.2 in [ONeil2017]
        op = (
                sigma / 4
                - D0(D0(sigma))
                + S(kappa * Sp(sigma))
                + S(Spp(sigma) + Dp0(sigma))
                - S(Dp0(sigma))
                + S0(Dp0(sigma))
                )

        # }}}

        return op

    def _get_right_operator(self, sigma: sym.var, kappa: sym.var, **kwargs: Any):
        raise NotImplementedError(
                f"right preconditioning for {type(self.kernel).__name__}")

    def operator(self,
            sigma: sym.var,
            mean_curvature: Optional[sym.var] = None,
            **kwargs) -> sym.var:
        if mean_curvature is None:
            mean_curvature = sym.mean_curvature(self.dim)

        if self.precond == "left":
            return self._get_left_operator(sigma, mean_curvature, **kwargs)
        else:
            return self._get_right_operator(sigma, mean_curvature, **kwargs)

# }}}


# {{{ Laplace-Beltrami operator

class LaplaceBeltramiOperator(BeltramiOperator):
    def __init__(self, dim: int, precond: str = "left") -> None:
        from sumpy.kernel import LaplaceKernel
        super().__init__(kernel=LaplaceKernel(dim), precond=precond)

    def _get_left_operator(self, sigma: sym.var, kappa: sym.var, **kwargs: Any):
        knl = self.kernel
        context = self.kernel_arguments.copy()
        context.update(kwargs)
        kappa = (self.dim - 1) * kappa

        # {{{ layer potentials

        S = partial(sym.S, knl, qbx_forced_limit=+1, **context)          # noqa: N806
        Sp = partial(sym.Sp, knl, qbx_forced_limit="avg", **context)     # noqa: N806
        Spp = partial(sym.Spp, knl, qbx_forced_limit="avg", **context)   # noqa: N806
        D = partial(sym.D, knl, qbx_forced_limit="avg", **context)       # noqa: N806
        Dp = partial(sym.Dp, knl, qbx_forced_limit="avg", **context)     # noqa: N806

        # }}}

        # {{{ operator

        def W(operand: sym.Expression) -> sym.Expression:                # noqa: N802
            return sym.Ones() * sym.integral(self.dim, self.dim - 1, operand)

        # NOTE: based on Lemma 3.1 in [ONeil2017], but doing :math:`-\Delta_\Gamma`
        op = (
                sigma / 4
                - D(D(sigma))
                + S(Spp(sigma) + Dp(sigma))
                + S(kappa * Sp(sigma))
                - S(W(S(sigma)))
                )

        # }}}

        return op

    def _get_right_operator(self, sigma: sym.var, kappa: sym.var, **kwargs: Any):
        knl = self.kernel
        context = self.kernel_arguments.copy()
        context.update(kwargs)
        kappa = (self.dim - 1) * kappa

        # {{{ layer potentials

        S = partial(sym.S, knl, qbx_forced_limit=+1, **context)          # noqa: N806
        Sp = partial(sym.Sp, knl, qbx_forced_limit="avg", **context)     # noqa: N806
        Spp = partial(sym.Spp, knl, qbx_forced_limit="avg", **context)   # noqa: N806
        Dp = partial(sym.Dp, knl, qbx_forced_limit="avg", **context)     # noqa: N806

        # }}}

        # {{{ operator

        def W(operand: sym.Expression) -> sym.Expression:                # noqa: N802
            return sym.Ones() * sym.integral(self.dim, self.dim - 1, operand)

        # NOTE: based on Lemma 3.2 in [ONeil2017], but doing :math:`-\Delta_\Gamma`
        op = (
                sigma / 4
                - Sp(Sp(sigma))
                + (Spp(S(sigma)) + Dp(S(sigma)))
                + kappa * Sp(S(sigma))
                + W(S(S(sigma)))
                )

        # }}}

        return op

# }}}


# {{{ Yukawa-Beltrami operator

class YukawaBeltramiOperator(BeltramiOperator):
    def __init__(self, dim: int,
            precond: str = "left",
            yukawa_k_name: str = "k") -> None:
        from sumpy.kernel import YukawaKernel
        super().__init__(
                kernel=YukawaKernel(dim, yukawa_lambda_name=yukawa_k_name),
                precond=precond,
                kernel_arguments={yukawa_k_name: sym.var(yukawa_k_name)}
                )

# }}}


# {{{ Helmholtz-Beltrami operator

class HelmholtzBeltramiOperator(BeltramiOperator):
    def __init__(self, dim: int,
            precond: str = "left",
            helmholtz_k_name: str = "k") -> None:
        from sumpy.kernel import HelmholtzKernel
        super().__init__(
                kernel=HelmholtzKernel(dim, helmholtz_k_name=helmholtz_k_name),
                precond=precond,
                kernel_arguments={helmholtz_k_name: sym.var(helmholtz_k_name)}
                )

# }}}
