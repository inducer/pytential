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
.. autoclass:: BeltramiOperator
.. autoclass:: LaplaceBeltramiOperator
.. autoclass:: YukawaBeltramiOperator
.. autoclass:: HelmholtzBeltramiOperator
"""

from functools import partial
from typing import Any

import numpy as np

from pytential import sym
from sumpy.kernel import Kernel


# {{{ beltrami operator

class BeltramiOperator:
    """Beltrami-type operators on closed surfaces.

    The construction of the operators is based on [ONeil2018]_ and takes
    any scalar PDE kernel. However, specific kernels may require additional
    work to allow for unique solutions. For example, the Laplace-Beltrami
    equation is only unique up to a constant, so using
    :class:`LaplaceBeltramiOperator` is recommended to take this into account.
    In general, the boundary integral equation can be constructed as

    .. code:: python

        beltrami = BeltramiOperator({...})
        sigma = beltrami.get_density_var("sigma")
        bc = beltrami.get_density_var("bc")

        rhs = beltrami.prepare_rhs(bc)
        solution = beltrami.prepare_solution(sigma)
        op = beltrami.operator(sigma)

    where :meth:`prepare_solution` is required to recover the solution from the
    density due to the different types of preconditioning.

    Note that a naive implementation of these operators is inefficient
    and can even be inaccurate (due to a subtraction of hypersingular operators).

    .. [ONeil2018] M. O'Neil, *Second-Kind Integral Equations for the
        Laplace-Beltrami Problem on Surfaces in Three Dimensions*,
        Advances in Computational Mathematics, Vol. 44, pp. 1385-1409, 2018,
        `DOI <http://dx.doi.org/10.1007/s10444-018-9587-7>`__.

    .. attribute:: dim

    .. automethod:: __init__

    .. automethod:: get_density_var
    .. automethod:: prepare_solution
    .. automethod:: prepare_rhs
    .. automethod:: operator
    """

    def __init__(self, kernel: Kernel, *,
            dim: int | None = None,
            precond: str = "left",
            kernel_arguments: dict[str, Any] | None = None) -> None:
        if dim is None:
            dim = kernel.dim - 1

        if precond not in ("left", "right"):
            raise ValueError(f"unknown preconditioning '{precond}'")

        if kernel_arguments is None:
            kernel_arguments = {}

        self.dim = dim
        self.kernel = kernel
        self.precond = precond
        self.kernel_arguments = kernel_arguments

    @property
    def ambient_dim(self):
        return self.kernel.dim

    @property
    def dtype(self):
        return np.dtype(
                np.complex128 if self.kernel.is_complex_valued
                else np.float64)

    def get_density_var(self, name: str = "sigma") -> sym.var:
        """
        :returns: a symbolic expression for the density.
        """
        return sym.var(name)

    def prepare_solution(self, sigma: sym.var) -> sym.var:
        """
        :returns: an expression for the solution to the original Beltrami
            equation based on the density *sigma* and the type of
            preconditioning used in the operator.
        """
        S = partial(sym.S, self.kernel,
                qbx_forced_limit=+1,
                kernel_arguments=self.kernel_arguments)

        if self.precond == "left":
            return S(sigma)
        else:
            return S(S(sigma))

    def prepare_rhs(self, b: sym.var) -> sym.var:
        """
        :returns: a modified expression for the right-hands ide *b* based on
            the preconditiong used in the operator.
        """
        S = partial(sym.S, self.kernel,
                qbx_forced_limit=+1,
                kernel_arguments=self.kernel_arguments)

        if self.precond == "left":
            return S(b)
        else:
            return b

    def operator(self,
            sigma: sym.var,
            mean_curvature: sym.var | None = None,
            **kwargs) -> sym.var:
        """
        :arg mean_curvature: an expression for the mean curvature that can be
            used in the construction of the operator.
        :returns: a Fredholm integral equation of the second kind for the
            Beltrami PDE with the unknown density *sigma*.
        """
        from sumpy.kernel import LaplaceKernel
        if isinstance(self.kernel, LaplaceKernel):
            raise TypeError(
                    f"{type(self).__name__} does not support the Laplace "
                    "kernel, use LaplaceBeltramiOperator instead")

        if mean_curvature is None:
            mean_curvature = sym.mean_curvature(self.ambient_dim, dim=self.dim)

        kappa = self.dim * mean_curvature
        context = self.kernel_arguments.copy()
        context.update(kwargs)

        knl = self.kernel
        lknl = LaplaceKernel(knl.dim)

        # {{{ layer potentials

        # laplace
        S0 = partial(sym.S, lknl, qbx_forced_limit=+1, kernel_arguments=kwargs)
        D0 = partial(sym.D, lknl, qbx_forced_limit="avg", kernel_arguments=kwargs)
        Sp0 = partial(sym.Sp, lknl, qbx_forced_limit="avg", kernel_arguments=kwargs)
        Dp0 = partial(sym.Dp, lknl, qbx_forced_limit="avg", kernel_arguments=kwargs)

        # base
        S = partial(sym.S, knl, qbx_forced_limit=+1, kernel_arguments=context)
        Sp = partial(sym.Sp, knl, qbx_forced_limit="avg", kernel_arguments=context)
        Spp = partial(sym.Spp, knl, qbx_forced_limit="avg", kernel_arguments=context)

        # }}}

        if self.precond == "left":
            # similar to 6.2 in [ONeil2018]
            op = (
                    sigma / 4
                    - D0(D0(sigma))
                    + S(kappa * Sp(sigma))
                    + S(Spp(sigma) + Dp0(sigma))
                    - S(Dp0(sigma))
                    + S0(Dp0(sigma))
                    )
        else:
            # similar to 6.2 in [ONeil2018]
            op = (
                    sigma / 4
                    - Sp0(Sp0(sigma))
                    + (Spp(S(sigma)) + Dp0(S(sigma)))
                    - Dp0(S(sigma) - S0(sigma))
                    + kappa * Sp(S(sigma))
                    )

        return op

# }}}


# {{{ Laplace-Beltrami operator

class LaplaceBeltramiOperator(BeltramiOperator):
    r"""Laplace-Beltrami operator on a closed surface :math:`\Sigma`

    .. math::

        -\Delta_\Sigma u = b

    Inherits from :class:`BeltramiOperator`.

    .. automethod:: __init__
    """

    def __init__(self, ambient_dim, *,
            dim: int | None = None,
            precond: str = "left") -> None:
        from sumpy.kernel import LaplaceKernel
        super().__init__(
                LaplaceKernel(ambient_dim),
                dim=dim,
                precond=precond)

    def operator(self,
            sigma: sym.var,
            mean_curvature: sym.var | None = None,
            **kwargs) -> sym.var:
        """
        :arg mean_curvature: an expression for the mean curvature that can be
            used in the construction of the operator.
        :returns: a Fredholm integral equation of the second kind for the
            Laplace-Beltrami PDE with the unknown density *sigma*.
        """
        if mean_curvature is None:
            mean_curvature = sym.mean_curvature(self.ambient_dim, dim=self.dim)

        kappa = self.dim * mean_curvature
        context = self.kernel_arguments.copy()
        context.update(kwargs)

        knl = self.kernel

        # {{{ layer potentials

        S = partial(sym.S, knl, qbx_forced_limit=+1, kernel_arguments=context)
        Sp = partial(sym.Sp, knl, qbx_forced_limit="avg", kernel_arguments=context)
        Spp = partial(sym.Spp, knl, qbx_forced_limit="avg", kernel_arguments=context)
        D = partial(sym.D, knl, qbx_forced_limit="avg", kernel_arguments=context)
        Dp = partial(sym.Dp, knl, qbx_forced_limit="avg", kernel_arguments=context)

        def Wl(operand: sym.Expression) -> sym.Expression:
            return sym.Ones() * sym.integral(self.ambient_dim, self.dim, operand)

        def Wr(operand: sym.Expression) -> sym.Expression:
            return sym.Ones() * sym.integral(self.ambient_dim, self.dim, operand)

        # }}}

        if self.precond == "left":
            # NOTE: based on Lemma 3.1 in [ONeil2018] for :math:`-\Delta_\Gamma`
            op = (
                    sigma / 4
                    - D(D(sigma))
                    + S(Spp(sigma) + Dp(sigma))
                    + S(kappa * Sp(sigma))
                    - S(Wl(S(sigma)))
                    )
        else:
            # NOTE: based on Lemma 3.2 in [ONeil2018] for :math:`-\Delta_\Gamma`
            op = (
                    sigma / 4
                    - Sp(Sp(sigma))
                    + (Spp(S(sigma)) + Dp(S(sigma)))
                    + kappa * Sp(S(sigma))
                    + Wr(S(S(sigma)))
                    )

        return op

# }}}


# {{{ Yukawa-Beltrami operator

class YukawaBeltramiOperator(BeltramiOperator):
    r"""Yukawa-Beltrami operator on a closed surface :math:`\Sigma`.

    .. math::

        -\Delta_\Sigma u + k^2 u = b.

    Inherits from :class:`BeltramiOperator`.

    .. automethod:: __init__
    """

    def __init__(self, ambient_dim: int, *,
            dim: int | None = None,
            precond: str = "left",
            yukawa_k_name: str = "k") -> None:
        from sumpy.kernel import YukawaKernel
        super().__init__(
                YukawaKernel(ambient_dim, yukawa_lambda_name=yukawa_k_name),
                dim=dim,
                precond=precond,
                kernel_arguments={yukawa_k_name: sym.var(yukawa_k_name)}
                )

# }}}


# {{{ Helmholtz-Beltrami operator

class HelmholtzBeltramiOperator(BeltramiOperator):
    r"""Helmholtz-Beltrami operator on a closed surface :math:`\Sigma`

    .. math::

        -\Delta_\Sigma u - k^2 u = b

    Inherits from :class:`BeltramiOperator`.

    .. automethod:: __init__
    """

    def __init__(self, ambient_dim: int, *,
            dim: int | None = None,
            precond: str = "left",
            helmholtz_k_name: str = "k") -> None:
        from sumpy.kernel import HelmholtzKernel
        super().__init__(
                HelmholtzKernel(ambient_dim, helmholtz_k_name=helmholtz_k_name),
                dim=dim,
                precond=precond,
                kernel_arguments={helmholtz_k_name: sym.var(helmholtz_k_name)}
                )

# }}}
