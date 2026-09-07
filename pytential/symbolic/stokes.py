from __future__ import annotations


__copyright__ = "Copyright (C) 2017 Natalie Beams"

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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from pymbolic.typing import ArithmeticExpression
from pytools.obj_array import ObjectArray1D, from_numpy
from sumpy.kernel import (
    AxisTargetDerivative,
    LaplaceKernel,
    StokesletSystemKernel,
    StressletSystemKernel,
)

from pytential import sym


if TYPE_CHECKING:
    from pytential.symbolic.primitives import QBXForcedLimit, Side

__doc__ = """
.. autoclass:: StokesletWrapper
.. autoclass:: StressletWrapper

.. autoclass:: StokesOperator
.. autoclass:: HsiaoKressExteriorStokesOperator
.. autoclass:: HebekerExteriorStokesOperator
"""

Vector = ObjectArray1D[ArithmeticExpression]


# {{{ StokesletWrapper

class StokesletWrapper:
    """Wrapper class for the :class:`~sumpy.kernel.StokesletSystemKernel` kernel.

    This class is meant to shield the user from the messiness of writing out
    every term in the expansion of the double-indexed Stokeslet kernel applied
    to the density vector.

    The :meth:`apply` function returns the integral expressions needed for the
    vector velocity resulting from convolution with the vector density, and is
    meant to work similarly to calling :func:`~pytential.symbolic.primitives.S`
    (which returns a :class:`~pytential.symbolic.primitives.IntG`).

    Similar functions are available for other useful variables related to
    the flow: :meth:`apply_pressure`, :meth:`apply_derivative` (target derivative),
    :meth:`apply_stress` (applies the symmetric viscous stress tensor in
    the requested direction).

    .. autoattribute:: dim

    .. automethod:: __init__
    .. automethod:: apply
    .. automethod:: apply_pressure
    .. automethod:: apply_derivative
    .. automethod:: apply_stress
    """

    dim: int

    stokeslet: StokesletSystemKernel
    stresslet: StressletSystemKernel

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.stokeslet = StokesletSystemKernel(dim)
        self.stresslet = StressletSystemKernel(dim)

    def apply(
            self,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        """Symbolic expressions for integrating the Stokeslet kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the dyadic Stokeslet kernel with the
        *density_vec_sym*.

        :arg mu_sym: a symbolic variable for the viscosity.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        if mu_sym is None:
            mu_sym = sym.SpatialConstant("mu")

        from itertools import product
        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp, i in product(range(self.dim), repeat=2):
            sym_expr[comp] += sym.int_g_vec(
                self.stokeslet[comp, i],
                density_vec_sym[i],
                qbx_forced_limit=qbx_forced_limit,
                mu=mu_sym,
            )

        return from_numpy(sym_expr, ArithmeticExpression)

    def apply_pressure(
            self,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ArithmeticExpression:
        """Symbolic expression for pressure field associated with the Stokeslet."""

        kernel = LaplaceKernel(dim=self.dim)

        sym_expr = 0
        for i in range(self.dim):
            sym_expr += sym.int_g_vec(
                AxisTargetDerivative(i, kernel),
                density_vec_sym[i],
                qbx_forced_limit=qbx_forced_limit)

        return sym_expr

    def apply_derivative(
            self,
            deriv_dir: int,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        """Symbolic derivative of velocity from Stokeslet.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        dyadic Stokeslet kernel with variable *density_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative.
        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg mu_sym: a symbolic variable for the viscosity.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        if mu_sym is None:
            mu_sym = sym.SpatialConstant("mu")

        from itertools import product
        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp, i in product(range(self.dim), repeat=2):
            sym_expr[comp] += sym.int_g_vec(
                AxisTargetDerivative(deriv_dir, self.stokeslet[comp, i]),
                density_vec_sym[i],
                qbx_forced_limit=qbx_forced_limit,
                mu=mu_sym,
            )

        return from_numpy(sym_expr, ArithmeticExpression)

    def apply_stress(
            self,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            dir_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        r"""Symbolic expression for viscous stress applied to a direction.

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress

        .. math::

            -p \delta_{ij} + \mu (\nabla_i u_j + \nabla_j u_i)

        applied in the direction of *dir_vec_sym*.

        Note that this computation is very similar to computing a double-layer
        potential with the Stresslet kernel in :class:`StressletWrapper`. The
        difference is that here the direction vector is applied at the target
        points, while in the Stresslet the direction is applied at the source
        points.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector for the application direction.
        :arg mu_sym: a symbolic variable for the viscosity.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        if mu_sym is None:
            mu_sym = sym.SpatialConstant("mu")

        from itertools import product
        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp, i, j in product(range(self.dim), repeat=3):
            sym_expr[comp] += dir_vec_sym[i] * sym.int_g_vec(
                self.stresslet[comp, i, j],
                density_vec_sym[j],
                qbx_forced_limit=qbx_forced_limit,
                mu=mu_sym,
            )

        return from_numpy(sym_expr, ArithmeticExpression)

# }}}


# {{{ StressletWrapper

class StressletWrapper:
    """Wrapper class for the :class:`~sumpy.kernel.StressletSystemKernel` kernel.

    This class is meant to shield the user from the messiness of writing out
    every term in the expansion of the triple-indexed Stresslet kernel applied
    to both a normal vector and the density vector. It provides the same
    functionality as :class:`StokesletWrapper`.

    .. autoattribute:: dim

    .. automethod:: __init__
    .. automethod:: apply
    .. automethod:: apply_pressure
    .. automethod:: apply_derivative
    .. automethod:: apply_stress
    """

    dim: int
    stresslet: StressletSystemKernel

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.stresslet = StressletSystemKernel(dim)

    def apply(
            self,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            dir_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        """Symbolic expressions for integrating the Stresslet kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the triadic Stresslet kernel with
        *density_vec_sym* and source direction vectors *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector variable for the direction vector.
        :arg mu_sym: a symbolic variable for the viscosity.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        if mu_sym is None:
            mu_sym = sym.SpatialConstant("mu")

        from itertools import product
        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp, i, j in product(range(self.dim), repeat=3):
            sym_expr[comp] += sym.int_g_vec(
                self.stresslet[comp, i, j],
                dir_vec_sym[i] * density_vec_sym[j],
                qbx_forced_limit=qbx_forced_limit,
                mu=mu_sym
            )

        return from_numpy(sym_expr, ArithmeticExpression)

    def apply_pressure(
            self,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            dir_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ArithmeticExpression:
        """Symbolic expression for pressure field associated with the Stresslet."""
        if mu_sym is None:
            mu_sym = sym.SpatialConstant("mu")

        from itertools import product
        kernel = LaplaceKernel(dim=self.dim)

        sym_expr = 0
        for i, j in product(range(self.dim), repeat=2):
            sym_expr += 2 * mu_sym * sym.int_g_vec(
                AxisTargetDerivative(i, AxisTargetDerivative(j, kernel)),
                density_vec_sym[i] * dir_vec_sym[j],
                qbx_forced_limit=qbx_forced_limit,
            )

        return sym_expr

    def apply_derivative(
            self,
            deriv_dir: int,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            dir_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        """Symbolic derivative of velocity from Stresslet.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        triadic Stresslet kernel with the *density_vec_sym* and source
        direction vectors *dir_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative.
        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector variable for the normal direction.
        :arg mu_sym: a symbolic variable for the viscosity.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        if mu_sym is None:
            mu_sym = sym.SpatialConstant("mu")

        from itertools import product
        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp, i, j in product(range(self.dim), repeat=3):
            sym_expr[comp] += sym.int_g_vec(
                AxisTargetDerivative(deriv_dir, self.stresslet[comp, i, j]),
                dir_vec_sym[i] * density_vec_sym[j],
                qbx_forced_limit=qbx_forced_limit,
                mu=mu_sym
            )

        return from_numpy(sym_expr, ArithmeticExpression)

    def apply_stress(
            self,
            density_vec_sym: ObjectArray1D[ArithmeticExpression],
            normal_vec_sym: ObjectArray1D[ArithmeticExpression],
            dir_vec_sym: ObjectArray1D[ArithmeticExpression],
            *,
            mu_sym: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        r"""Symbolic expression for viscous stress applied to a direction.

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress

        .. math::

            -p \delta_{ij} + \mu (\nabla_i u_j + \nabla_j u_i)

        applied in the direction of *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg normal_vec_sym: a symbolic vector variable for the normal vectors
            (outward facing normals at source locations).
        :arg dir_vec_sym: a symbolic vector for the application direction.
        :arg mu_sym: a symbolic variable for the viscosity.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        if mu_sym is None:
            mu_sym = sym.SpatialConstant("mu")

        # velocity velocity gradient
        sym_grad_matrix = np.empty((self.dim, self.dim), dtype=object)
        for i in range(self.dim):
            sym_grad_matrix[:, i] = self.apply_derivative(
                i, density_vec_sym, normal_vec_sym,
                qbx_forced_limit=qbx_forced_limit,
                mu_sym=mu_sym
            )

        # compute stress
        sym_expr = np.zeros((self.dim,), dtype=object)
        for comp in range(self.dim):
            sym_expr[comp] = -dir_vec_sym[comp] * self.apply_pressure(
                density_vec_sym, normal_vec_sym,
                qbx_forced_limit=qbx_forced_limit,
                mu_sym=mu_sym,
            )

            for i in range(self.dim):
                sym_expr[comp] += mu_sym * dir_vec_sym[i] * (
                        sym_grad_matrix[comp, i]
                        + sym_grad_matrix[i, comp])

        return from_numpy(sym_expr, ArithmeticExpression)

# }}}


# {{{ base Stokes operator

class StokesOperator(ABC):
    """
    .. autoattribute:: side
    .. autoattribute:: ambient_dim
    .. autoproperty:: dim

    .. automethod:: __init__
    .. automethod:: get_density_var
    .. automethod:: prepare_rhs
    .. automethod:: operator

    .. automethod:: velocity
    .. automethod:: pressure
    """

    ambient_dim: int
    side: Side

    stokeslet: StokesletWrapper
    stresslet: StressletWrapper

    def __init__(self, ambient_dim: int, side: Side) -> None:
        """
        :arg ambient_dim: dimension of the ambient space.
        :arg side: :math:`+1` for exterior or :math:`-1` for interior.
        """

        if side not in [+1, -1]:
            raise ValueError(f"invalid evaluation side: {side}")

        self.ambient_dim = ambient_dim
        self.side = side

        self.stokeslet = StokesletWrapper(self.ambient_dim)
        self.stresslet = StressletWrapper(self.ambient_dim)

    @property
    def dim(self) -> int:
        return self.ambient_dim - 1

    def get_density_var(
            self, name: str = "sigma",
        ) -> ObjectArray1D[ArithmeticExpression]:
        """
        :returns: a symbolic vector corresponding to the density.
        """
        return sym.make_sym_vector(name, self.ambient_dim)

    def prepare_rhs(
            self,
            b: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        """
        :returns: a (potentially) modified right-hand side *b* that matches
            requirements of the representation.
        """
        return b

    @abstractmethod
    def operator(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        """
        :returns: the integral operator that should be solved to obtain the
            density *sigma*.
        """

    @abstractmethod
    def velocity(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        """
        :returns: a representation of the velocity field in the Stokes flow.
        """

    @abstractmethod
    def pressure(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ArithmeticExpression:
        """
        :returns: a representation of the pressure in the Stokes flow.
        """

# }}}


# {{{ exterior Stokes flow

class HsiaoKressExteriorStokesOperator(StokesOperator):
    """Representation for 2D Stokes Flow based on [HsiaoKress1985]_.

    Inherits from :class:`StokesOperator`.

    .. [HsiaoKress1985] G. C. Hsiao and R. Kress, *On an Integral Equation for
        the Two-Dimensional Exterior Stokes Problem*,
        Applied Numerical Mathematics, Vol. 1, 1985,
        `DOI <https://doi.org/10.1016/0168-9274(85)90029-7>`__.

    .. automethod:: __init__
    """

    omega: ObjectArray1D[ArithmeticExpression]
    alpha: float
    eta: float

    def __init__(
            self,
            *,
            omega: ObjectArray1D[ArithmeticExpression],
            alpha: float | None = None,
            eta: float | None = None,
        ) -> None:
        r"""
        :arg omega: farfield behaviour of the velocity field, as defined
            by :math:`A` in [HsiaoKress1985]_ Equation 2.3.
        :arg alpha: real parameter :math:`\alpha > 0`.
        :arg eta: real parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning.
        """
        super().__init__(ambient_dim=2, side=+1)

        # NOTE: in [hsiao-kress], there is an analysis on a circle, which
        # recommends values in
        #   1/2 <= alpha <= 2 and max(1/alpha, 1) <= eta <= min(2, 2/alpha)
        # so we choose alpha = eta = 1, which seems to be in line with some
        # of the presented numerical results too.

        if alpha is None:
            alpha = 1.0

        if eta is None:
            eta = 1.0

        self.omega = omega
        self.alpha = alpha
        self.eta = eta

    def _farfield(
            self,
            *,
            mu: ArithmeticExpression | None,
            qbx_forced_limit: QBXForcedLimit | None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        if mu is None:
            mu = sym.SpatialConstant("mu")

        length = sym.integral(self.ambient_dim, self.dim, 1)
        return self.stokeslet.apply(
                -self.omega / length,
                mu_sym=mu,
                qbx_forced_limit=qbx_forced_limit)

    def _operator(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None,
            qbx_forced_limit: QBXForcedLimit | None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        if mu is None:
            mu = sym.SpatialConstant("mu")

        slp_qbx_forced_limit = qbx_forced_limit
        if slp_qbx_forced_limit == "avg":
            slp_qbx_forced_limit = +1

        # NOTE: we set a dofdesc here to force the evaluation of this integral
        # on the source instead of the target when using automatic tagging
        # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
        dd = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)

        int_sigma = sym.integral(self.ambient_dim, self.dim, sigma, dofdesc=dd)
        meanless_sigma = sym.cse(sigma - sym.mean(self.ambient_dim, self.dim, sigma))

        op_k = self.stresslet.apply(
                sigma, normal,
                mu_sym=mu,
                qbx_forced_limit=qbx_forced_limit)
        op_s = (
                self.alpha / (2.0 * np.pi) * int_sigma
                - self.stokeslet.apply(
                    meanless_sigma,
                    mu_sym=mu,
                    qbx_forced_limit=slp_qbx_forced_limit)
                )

        return op_k + self.eta * op_s

    @override
    def prepare_rhs(
            self,
            b: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        return b + self._farfield(mu=mu, qbx_forced_limit=+1)

    @override
    def operator(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        # NOTE: H. K. 1985 Equation 2.18
        return (
            -0.5 * self.side * sigma
            - self._operator(sigma, normal, mu=mu, qbx_forced_limit="avg")
        )

    @override
    def velocity(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        # NOTE: H. K. 1985 Equation 2.16
        return (
            -self._farfield(mu=mu, qbx_forced_limit=qbx_forced_limit)
            - self._operator(sigma, normal, mu=mu, qbx_forced_limit=qbx_forced_limit)
        )

    @override
    def pressure(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ArithmeticExpression:
        # FIXME: H. K. 1985 Equation 2.17
        raise NotImplementedError


class HebekerExteriorStokesOperator(StokesOperator):
    """Representation for 3D Stokes Flow based on [Hebeker1986]_.

    Inherits from :class:`StokesOperator`.

    .. [Hebeker1986] F. C. Hebeker, *Efficient Boundary Element Methods for
        Three-Dimensional Exterior Viscous Flow*, Numerical Methods for
        Partial Differential Equations, Vol. 2, 1986,
        `DOI <https://doi.org/10.1002/num.1690020404>`__.

    .. automethod:: __init__
    """

    eta: float

    def __init__(self, *, eta: float | None = None) -> None:
        r"""
        :arg eta: a parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning of the operator.
        """

        super().__init__(ambient_dim=3, side=+1)

        # NOTE: eta is chosen here based on H. 1986 Figure 1, which is
        # based on solving on the unit sphere
        if eta is None:
            eta = 0.75

        self.eta = eta

    def _operator(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None,
            qbx_forced_limit: QBXForcedLimit | None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        if mu is None:
            mu = sym.SpatialConstant("mu")

        slp_qbx_forced_limit = qbx_forced_limit
        if slp_qbx_forced_limit == "avg":
            slp_qbx_forced_limit = self.side

        op_w = self.stresslet.apply(
                sigma, normal,
                mu_sym=mu,
                qbx_forced_limit=qbx_forced_limit)
        op_v = self.stokeslet.apply(
                sigma,
                mu_sym=mu,
                qbx_forced_limit=slp_qbx_forced_limit)

        return op_w + self.eta * op_v

    @override
    def operator(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        # NOTE: H. 1986 Equation 17
        return (
            -0.5 * self.side * sigma
            - self._operator(sigma, normal, mu=mu, qbx_forced_limit="avg")
        )

    @override
    def velocity(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ObjectArray1D[ArithmeticExpression]:
        # NOTE: H. 1986 Equation 16
        return -self._operator(sigma, normal, mu=mu, qbx_forced_limit=qbx_forced_limit)

    @override
    def pressure(
            self,
            sigma: ObjectArray1D[ArithmeticExpression],
            normal: ObjectArray1D[ArithmeticExpression],
            *,
            mu: ArithmeticExpression | None = None,
            qbx_forced_limit: QBXForcedLimit | None = None,
        ) -> ArithmeticExpression:
        # FIXME: not given in H. 1986, but should be easy to derive using the
        # equivalent single-/double-layer pressure kernels
        raise NotImplementedError

# }}}
