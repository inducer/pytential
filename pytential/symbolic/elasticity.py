from __future__ import annotations


__copyright__ = """
Copyright (C) 2017 Natalie Beams
Copyright (C) 2022 Isuru Fernando
"""

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
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from typing_extensions import override

from pytools import obj_array
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    BiharmonicKernel,
    ElasticityKernel,
    Kernel,
    LaplaceKernel,
    StokesletKernel,
    StressletKernel,
    TargetPointMultiplier,
)
from sumpy.symbolic import SpatialConstant

from pytential import sym
from pytential.symbolic.pde.system_utils import rewrite_using_base_kernel


if TYPE_CHECKING:
    from pymbolic.typing import ArithmeticExpression
    from pytools.obj_array import ObjectArray1D

    from pytential.symbolic.primitives import QBXForcedLimit

__doc__ = """
.. autoclass:: VectorExpression

.. autoclass:: Method
    :members:

.. autofunction:: make_elasticity_wrapper
.. autofunction:: make_elasticity_double_layer_wrapper

.. autoclass:: ElasticityWrapperBase
.. autoclass:: ElasticityDoubleLayerWrapperBase

.. autoclass:: ElasticityWrapperNaive
.. autoclass:: ElasticityDoubleLayerWrapperNaive

.. autoclass:: ElasticityWrapperBiharmonic
.. autoclass:: ElasticityDoubleLayerWrapperBiharmonic

.. autoclass:: ElasticityWrapperYoshida
.. autoclass:: ElasticityDoubleLayerWrapperYoshida
"""

VectorExpression: TypeAlias = "ObjectArray1D[ArithmeticExpression]"


# {{{ ElasiticityWrapper ABCs

@dataclass
class ElasticityWrapperBase(ABC):
    """Wrapper class for the :class:`~sumpy.kernel.ElasticityKernel` kernel.

    This class is meant to shield the user from the messiness of writing
    out every term in the expansion of the double-indexed elasticity kernel
    applied to a density vector.

    The :meth:`apply` function returns the integral expressions needed for
    the vector displacement resulting from convolution with the vector density,
    and is meant to work similarly to calling
    :func:`~pytential.symbolic.primitives.S` (which is
    :class:`~pytential.symbolic.primitives.IntG`).

    Similar functions are available for other useful things related to
    the flow: :meth:`apply_derivative` (target derivative).

    .. autoattribute:: dim
    .. autoattribute:: mu
    .. autoattribute:: nu

    .. automethod:: apply
    .. automethod:: apply_derivative
    """

    dim: int
    """Ambient dimension of the representation."""
    mu: float | SpatialConstant
    r"""Expression or value for the shear modulus :math:`\mu`."""
    nu: float | SpatialConstant
    r"""Expression or value for Poisson's ratio :math:`\nu`."""

    @abstractmethod
    def apply(self,
              density_vec_sym: VectorExpression,
              qbx_forced_limit: QBXForcedLimit,
              extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:
        """Symbolic expressions for the elasticity single-layer potential.

        This constructs an object array of symbolic expressions for the vector
        resulting from integrating the dyadic elasticity kernel with the density
        *density_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        :arg extra_deriv_dirs: adds target derivatives to all the integral
            objects with the given derivative axis.
        """

    def apply_derivative(self,
                         deriv_dir: int,
                         density_vec_sym: VectorExpression,
                         qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
        """Symbolic derivative of the elasticity single-layer kernel.

        This constructs an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        dyadic elasticity kernel with the density *density_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative,
            i.e. the directions *x*, *y*, *z* correspond to the integers *0*,
            *1*, *2*.
        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        return self.apply(density_vec_sym, qbx_forced_limit, (deriv_dir,))


@dataclass
class ElasticityDoubleLayerWrapperBase(ABC):
    """Wrapper class for the double layer of
    :class:`~sumpy.kernel.ElasticityKernel` kernel.

    This class is meant to shield the user from the messiness of writing
    out every term in the expansion of the triple-indexed elasticity double-layer
    kernel applied to both a normal vector and the density vector.

    The :meth:`apply` function returns the integral expressions needed for
    convolving the kernel with a vector density, and is meant to work
    similarly to :func:`~pytential.symbolic.primitives.D` (which is
    :class:`~pytential.symbolic.primitives.IntG`).

    Similar functions are available for other useful things related to
    the flow: :meth:`apply_derivative` (target derivative).

    .. autoattribute:: dim
    .. autoattribute:: mu
    .. autoattribute:: nu

    .. automethod:: apply
    .. automethod:: apply_derivative
    """

    dim: int
    """Ambient dimension of the representation."""
    mu: float | SpatialConstant
    r"""Expression or value for the shear modulus :math:`\mu`."""
    nu: float | SpatialConstant
    r"""Expression or value for Poisson's ration :math:`\nu`."""

    @abstractmethod
    def apply(self,
              density_vec_sym: VectorExpression,
              dir_vec_sym: VectorExpression,
              qbx_forced_limit: QBXForcedLimit,
              extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:
        """Symbolic expressions for the elasticity double-layer potential.

        This constructs an object array of symbolic expressions for the vector
        resulting from integrating the triadic kernel with the density
        *density_vec_sym* and the source direction vector *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector variable for the direction vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        :arg extra_deriv_dirs: adds target derivatives to all the integral
            objects with the given derivative axis.
        """

    def apply_derivative(self,
                         deriv_dir: int,
                         density_vec_sym: VectorExpression,
                         dir_vec_sym: VectorExpression,
                         qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
        """Symbolic derivative of the elasticity double-layer potential.

        This constructs an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        triadic elasticity kernel with variable *density_vec_sym* and the
        source direction *dir_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative,
            i.e. the directions *x*, *y*, *z* correspond to the integers *0*,
            *1*, *2*.
        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector variable for the normal direction.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        return self.apply(density_vec_sym, dir_vec_sym, qbx_forced_limit, (deriv_dir,))

# }}}


# {{{ Naive and Biharmonic impl

def _create_int_g(knl: Kernel,
                  deriv_dirs: tuple[int, ...],
                  density: ArithmeticExpression,
                  *,
                  qbx_forced_limit: QBXForcedLimit,
                  **kwargs: float | SpatialConstant) -> ArithmeticExpression:
    for deriv_dir in deriv_dirs:
        knl = AxisTargetDerivative(deriv_dir, knl)

    kernel_arg_names = {
        karg.loopy_arg.name
        for karg in (*knl.get_args(), *knl.get_source_args())}

    # When the kernel is Laplace, mu and nu are not kernel arguments
    # Also when nu==0.5 is not a kernel argument to StokesletKernel
    for var_name in ("mu", "nu"):
        if var_name not in kernel_arg_names:
            kwargs.pop(var_name)

    return sym.int_g_vec(
        knl, density, qbx_forced_limit,
        kernel_arguments=cast("dict[str, ArithmeticExpression]", kwargs),
    )


@dataclass
class _ElasticityWrapperNaiveOrBiharmonic:
    dim: int
    mu: float | SpatialConstant
    nu: float | SpatialConstant
    base_kernel: Kernel | None

    def __post_init__(self):
        if not (self.dim == 3 or self.dim == 2):
            raise ValueError(
                f"unsupported dimension given to {type(self).__name__!r}: {self.dim}"
            )

    @cached_property
    def kernel_dict(self) -> dict[tuple[int, int], Kernel]:
        d: dict[tuple[int, int], Kernel] = {}

        # The dictionary allows us to exploit symmetry -- that :math:`T_{01}`
        # is identical to :math:`T_{10}` -- and avoid creating multiple
        # expansions for the same kernel in a different ordering.
        for i in range(self.dim):
            for j in range(i, self.dim):
                if self.nu == 0.5:
                    d[i, j] = StokesletKernel(dim=self.dim, icomp=i, jcomp=j)
                else:
                    d[i, j] = ElasticityKernel(dim=self.dim, icomp=i, jcomp=j)
                d[j, i] = d[i, j]

        return d

    def _get_int_g(self,
                   idx: tuple[int, int],
                   density_sym: ArithmeticExpression,
                   dir_vec_sym: VectorExpression,
                   qbx_forced_limit: QBXForcedLimit,
                   deriv_dirs: tuple[int, ...]) -> ArithmeticExpression:
        """
        Returns the convolution of the elasticity kernel given by `idx`
        and its derivatives.
        """
        res = _create_int_g(
            self.kernel_dict[idx],
            deriv_dirs,
            density=density_sym*dir_vec_sym[idx[-1]],
            qbx_forced_limit=qbx_forced_limit,
            mu=self.mu,
            nu=self.nu)

        return res / (2 * (1 - self.nu))

    def apply(self,
              density_vec_sym: VectorExpression,
              qbx_forced_limit: QBXForcedLimit,
              extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:

        sym_expr = np.zeros((self.dim,), dtype=object)
        dir_vec_sym = obj_array.new_1d([1] * self.dim)

        # For stokeslet, there's no direction vector involved
        # passing a list of ones instead to remove its usage.
        for comp in range(self.dim):
            for i in range(self.dim):
                sym_expr[comp] += self._get_int_g(
                    (comp, i),
                    density_vec_sym[i],
                    dir_vec_sym,
                    qbx_forced_limit,
                    deriv_dirs=extra_deriv_dirs)

        return np.array(rewrite_using_base_kernel(
            sym_expr,
            base_kernel=self.base_kernel))


class ElasticityWrapperNaive(_ElasticityWrapperNaiveOrBiharmonic,
                             ElasticityWrapperBase):
    r"""Elasticity single-layer wrapper based on the standard elasticity kernel.

    This representation uses the elasticity kernel denoted by
    :attr:`Method.Naive`.
    """

    def __init__(self,
                 dim: int,
                 mu: float | SpatialConstant,
                 nu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, nu=nu, base_kernel=None)
        ElasticityWrapperBase.__init__(self, dim=dim, mu=mu, nu=nu)


class ElasticityWrapperBiharmonic(_ElasticityWrapperNaiveOrBiharmonic,
                                  ElasticityWrapperBase):
    r"""Elasticity single-layer wrapper based on the biharmonic kernel.

    This representation uses the biharmonic kernel denoted by
    :attr:`Method.Biharmonic`.
    """

    def __init__(self,
                 dim: int,
                 mu: float | SpatialConstant,
                 nu: float | SpatialConstant) -> None:
        super().__init__(
                dim=dim, mu=mu, nu=nu,
                base_kernel=BiharmonicKernel(dim))
        ElasticityWrapperBase.__init__(self, dim=dim, mu=mu, nu=nu)


# }}}


# {{{ ElasticityDoubleLayerWrapper Naive and Biharmonic impl

# NOTE: this is used in the `kernel_dict` to store the Laplace kernel
LAPLACIAN_INDEX = (-1, -1, -1)


@dataclass
class _ElasticityDoubleLayerWrapperNaiveOrBiharmonic:
    dim: int
    mu: float | SpatialConstant
    nu: float | SpatialConstant
    base_kernel: Kernel | None

    def __post_init__(self):
        if not (self.dim == 3 or self.dim == 2):
            raise ValueError("unsupported dimension given to "
                             f"ElasticityDoubleLayerWrapper: {self.dim}")

    @cached_property
    def kernel_dict(self) -> dict[tuple[int, int, int], Kernel]:
        d: dict[tuple[int, int, int], Kernel] = {}

        for i in range(self.dim):
            for j in range(i, self.dim):
                for k in range(j, self.dim):
                    d[i, j, k] = (
                        StressletKernel(dim=self.dim, icomp=i, jcomp=j, kcomp=k))

        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{012}` is identical to :math:`T_{120}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if (i, j, k) in d:
                        continue
                    s = cast("tuple[int, int, int]", tuple(sorted([i, j, k])))
                    d[i, j, k] = d[s]

        # For elasticity (nu != 0.5), we need the laplacian of the
        # BiharmonicKernel which is the LaplaceKernel.
        if self.nu != 0.5:
            d[LAPLACIAN_INDEX] = LaplaceKernel(self.dim)

        return d

    def _get_int_g(self,
                   idx: tuple[int, int, int],
                   density_sym: ArithmeticExpression,
                   dir_vec_sym: VectorExpression,
                   qbx_forced_limit: QBXForcedLimit,
                   deriv_dirs: tuple[int, ...]) -> ArithmeticExpression:
        """
        Returns the convolution of the double layer of the elasticity kernel
        given by `idx` and its derivatives.
        """

        nu = self.nu
        kernel_indices = [idx]
        dir_vec_indices = [idx[-1]]
        coeffs = [1]

        kernel_indices = [idx, LAPLACIAN_INDEX, LAPLACIAN_INDEX, LAPLACIAN_INDEX]
        dir_vec_indices = [idx[-1], idx[1], idx[0], idx[2]]
        coeffs = [1, (1 - 2*nu)/self.dim, -(1 - 2*nu)/self.dim, -(1 - 2*nu)]
        extra_deriv_dirs_vec = [[], [idx[0]], [idx[1]], [idx[2]]]
        if idx[0] != idx[1]:
            coeffs[-1] = 0

        result = 0
        for kernel_idx, dir_vec_idx, coeff, extra_deriv_dirs in zip(
                    kernel_indices,
                    dir_vec_indices,
                    coeffs,
                    extra_deriv_dirs_vec, strict=True):
            if coeff == 0:
                continue

            knl = self.kernel_dict[kernel_idx]
            result += coeff * _create_int_g(
                    knl,
                    (*deriv_dirs, *extra_deriv_dirs),
                    density=density_sym*dir_vec_sym[dir_vec_idx],
                    qbx_forced_limit=qbx_forced_limit,
                    mu=self.mu,
                    nu=self.nu)

        return result/(2*(1 - nu))

    def apply(self,
              density_vec_sym: VectorExpression,
              dir_vec_sym: VectorExpression,
              qbx_forced_limit: QBXForcedLimit,
              extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:

        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += self._get_int_g((comp, i, j),
                        density_vec_sym[i], dir_vec_sym,
                        qbx_forced_limit, deriv_dirs=extra_deriv_dirs)

        return np.array(rewrite_using_base_kernel(
            sym_expr,
            base_kernel=self.base_kernel))

    def apply_single_and_double_layer(
            self,
            stokeslet_density_vec_sym: VectorExpression,
            stresslet_density_vec_sym: VectorExpression,
            dir_vec_sym: VectorExpression,
            qbx_forced_limit: QBXForcedLimit,
            stokeslet_weight: ArithmeticExpression,
            stresslet_weight: ArithmeticExpression,
            extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:

        stokeslet_obj = _ElasticityWrapperNaiveOrBiharmonic(
                dim=self.dim,
                mu=self.mu, nu=self.nu,
                base_kernel=self.base_kernel)

        sym_expr = 0
        if stresslet_weight != 0:
            sym_expr += stresslet_weight * self.apply(
                stresslet_density_vec_sym, dir_vec_sym,
                qbx_forced_limit, extra_deriv_dirs)

        if stokeslet_weight != 0:
            sym_expr += stokeslet_weight * stokeslet_obj.apply(
                stokeslet_density_vec_sym,
                qbx_forced_limit, extra_deriv_dirs)

        return sym_expr


class ElasticityDoubleLayerWrapperNaive(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        ElasticityDoubleLayerWrapperBase):
    r"""Elasticity double-layer wrapper based on the standard elasticity kernel.

    This representation uses the elasticity kernel denoted by
    :attr:`Method.Naive`.
    """

    def __init__(self,
                 dim: int,
                 mu: float | SpatialConstant,
                 nu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, nu=nu, base_kernel=None)
        ElasticityDoubleLayerWrapperBase.__init__(
            self, dim=dim, mu=mu, nu=nu)


class ElasticityDoubleLayerWrapperBiharmonic(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        ElasticityDoubleLayerWrapperBase):
    r"""Elasticity double-layer wrapper based on the biharmonic kernel.

    This representation uses the biharmonic kernel denoted by
    :attr:`Method.Biharmonic`.
    """

    def __init__(self,
                 dim: int,
                 mu: float | SpatialConstant,
                 nu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, nu=nu, base_kernel=BiharmonicKernel(dim))
        ElasticityDoubleLayerWrapperBase.__init__(
            self, dim=dim, mu=mu, nu=nu)

# }}}


# {{{ dispatch function

class Method(Enum):
    """Method to represent the elasticity kernel."""
    Naive = 1
    """The naive representation is given by the standard tensor expression of
    kernel using e.g. :class:`~sumpy.kernel.ElasticityKernel`.
    """
    Laplace = 2
    """A representation of the elasticity kernel in terms of the Laplace kernel."""
    Biharmonic = 3
    """A representation of the elasticity kernel in terms of the biharmonic kernel.
    """


def make_elasticity_wrapper(
        dim: int,
        mu: float | str | SpatialConstant = "mu",
        nu: float | str | SpatialConstant = "nu",
        method: Method = Method.Naive) -> ElasticityWrapperBase:
    """Creates an appropriate :class:`ElasticityWrapperBase` object.

    :param dim: ambient dimension of the representation.
    :param mu: expression or value for the shear modulus.
    :param nu: expression or value for Poisson's ratio. If this is set to
        *0.5* exactly, an appropriate Stokes object is created instead.
    :param method: method to use - defaults to the :attr:`Method.Naive`.
    """
    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if isinstance(nu, str):
        nu = SpatialConstant(nu)

    if nu == 0.5:
        from pytential.symbolic.stokes import make_stokeslet_wrapper
        return make_stokeslet_wrapper(dim=dim, mu=mu, method=method)

    if method == Method.Naive:
        return ElasticityWrapperNaive(dim=dim, mu=mu, nu=nu)
    elif method == Method.Biharmonic:
        return ElasticityWrapperBiharmonic(dim=dim, mu=mu, nu=nu)
    elif method == Method.Laplace:
        return ElasticityWrapperYoshida(dim=dim, mu=mu, nu=nu)
    else:
        raise ValueError(f"invalid 'method': {method}")


def make_elasticity_double_layer_wrapper(
        dim: int,
        mu: float | str | SpatialConstant = "mu",
        nu: float | str | SpatialConstant = "nu",
        method: Method = Method.Naive) -> ElasticityDoubleLayerWrapperBase:
    """Creates an appropriate :class:`ElasticityDoubleLayerWrapperBase` object.

    :param dim: ambient dimension of the representation.
    :param mu: expression or value for the shear modulus.
    :param nu: expression or value for Poisson's ratio. If this is set to
        *0.5* exactly, an appropriate Stokes object is created instead.
    :param method: method to use - defaults to the :attr:`Method.Naive`.
    """
    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if isinstance(nu, str):
        nu = SpatialConstant(nu)

    if nu == 0.5:
        from pytential.symbolic.stokes import make_stresslet_wrapper
        return make_stresslet_wrapper(dim, mu=mu, method=method)

    if method == Method.Naive:
        return ElasticityDoubleLayerWrapperNaive(dim=dim, mu=mu, nu=nu)
    elif method == Method.Biharmonic:
        return ElasticityDoubleLayerWrapperBiharmonic(dim=dim, mu=mu, nu=nu)
    elif method == Method.Laplace:
        return ElasticityDoubleLayerWrapperYoshida(dim=dim, mu=mu, nu=nu)
    else:
        raise ValueError(f"invalid 'method': {method}")


# }}}


# {{{ Yoshida

@dataclass
class ElasticityDoubleLayerWrapperYoshida(ElasticityDoubleLayerWrapperBase):
    r"""Elasticity double-layer wrapper based on [Yoshida2001]_.

    This representation uses the Laplace kernel denoted by :attr:`Method.Laplace`
    and can only be used in 3D.

    .. [Yoshida2001] K.-I. Yoshida, N. Nishimura, S. Kobayashi,
        *Application of Fast Multipole Galerkin Boundary Integral Equation Method
        to Elastostatic Crack Problems in 3D*,
        International Journal for Numerical Methods in Engineering, Vol. 50, pp.
        525--547, 2001,
        `DOI <https://doi.org/10.1002/1097-0207(20010130)50:3%3C525::aid-nme34%3E3.0.co;2-4>`__.
    """

    def __post_init__(self):
        if not self.dim == 3:
            raise ValueError("unsupported dimension given to "
                             f"ElasticityDoubleLayerWrapperYoshida: {self.dim}")

    @cached_property
    def laplace_kernel(self):
        return LaplaceKernel(dim=3)

    @override
    def apply(self,
              density_vec_sym: VectorExpression,
              dir_vec_sym: VectorExpression,
              qbx_forced_limit: QBXForcedLimit,
              extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:
        return self.apply_single_and_double_layer(
            obj_array.new_1d([0]*self.dim),
            density_vec_sym,
            dir_vec_sym,
            qbx_forced_limit,
            0, 1,
            extra_deriv_dirs)

    def apply_single_and_double_layer(
            self,
            stokeslet_density_vec_sym: VectorExpression,
            stresslet_density_vec_sym: VectorExpression,
            dir_vec_sym: VectorExpression,
            qbx_forced_limit: QBXForcedLimit,
            stokeslet_weight: ArithmeticExpression,
            stresslet_weight: ArithmeticExpression,
            extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:

        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)
        stokeslet_weight *= -1

        def C(i: int, j: int, k: int, l: int) -> ArithmeticExpression:   # noqa: E741
            res = 0
            if i == j and k == l:
                res += lam
            if i == k and j == l:
                res += mu
            if i == l and j == k:
                res += mu

            return res * stresslet_weight

        def add_extra_deriv_dirs(target_kernel: Kernel) -> Kernel:
            for deriv_dir in extra_deriv_dirs:
                target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

            return target_kernel

        def P(i: int, j: int, int_g: sym.IntG) -> ArithmeticExpression:
            target_kernel = add_extra_deriv_dirs(
                TargetPointMultiplier(j, AxisTargetDerivative(i, int_g.target_kernel))
            )

            res = -int_g.copy(target_kernel=target_kernel)
            if i == j:
                res += (3 - 4*nu)*int_g.copy(
                    target_kernel=add_extra_deriv_dirs(int_g.target_kernel))

            return res / (4*mu*(1 - nu))

        def Q(i: int, int_g: sym.IntG) -> ArithmeticExpression:
            target_kernel = add_extra_deriv_dirs(
                AxisTargetDerivative(i, int_g.target_kernel)
            )
            res = int_g.copy(target_kernel=target_kernel)

            return res / (4*mu*(1 - nu))

        sym_expr = np.zeros((3,), dtype=object)

        source = sym.nodes(3).as_vector()
        normal = dir_vec_sym
        sigma = stresslet_density_vec_sym

        kernel = self.laplace_kernel
        source_kernels: tuple[Kernel, ...] = tuple(
            [AxisSourceDerivative(i, kernel) for i in range(3)]
            + [kernel])

        for i in range(3):
            for k in range(3):
                densities = obj_array.new_1d([0] * len(source_kernels))
                for l in range(3):   # noqa: E741
                    for j in range(3):
                        for m in range(3):
                            densities[l] += C(k, l, m, j)*normal[m]*sigma[j]
                densities[3] += stokeslet_weight * stokeslet_density_vec_sym[k]

                int_g = sym.IntG(
                    target_kernel=kernel,
                    source_kernels=tuple(source_kernels),
                    densities=densities,
                    source=sym.DEFAULT_DOFDESC,
                    target=sym.DEFAULT_DOFDESC,
                    qbx_forced_limit=qbx_forced_limit)
                sym_expr[i] += P(i, k, int_g)

            densities = obj_array.new_1d([0] * len(source_kernels))
            for k in range(3):
                for m in range(3):
                    for j in range(3):
                        for l in range(3):   # noqa: E741
                            densities[l] += C(k, l, m, j)*normal[m]*sigma[j]*source[k]
                            if k == l:
                                densities[3] += C(k, l, m, j)*normal[m]*sigma[j]
                densities[3] += (
                        stokeslet_weight * source[k] * stokeslet_density_vec_sym[k])

            int_g = sym.IntG(
                target_kernel=kernel,
                source_kernels=tuple(source_kernels),
                densities=densities,
                source=sym.DEFAULT_DOFDESC,
                target=sym.DEFAULT_DOFDESC,
                qbx_forced_limit=qbx_forced_limit)
            sym_expr[i] += Q(i, int_g)

        return sym_expr


@dataclass
class ElasticityWrapperYoshida(ElasticityWrapperBase):
    r"""Elasticity single-layer wrapper based on [Yoshida2001]_.

    This representation uses the Laplace kernel denoted by :attr:`Method.Laplace`
    and can only be used in 3D.
    """

    def __post_init__(self):
        if not self.dim == 3:
            raise ValueError("unsupported dimension given to "
                             f"ElasticityDoubleLayerWrapperYoshida: {self.dim}")

    @cached_property
    def stresslet(self):
        return ElasticityDoubleLayerWrapperYoshida(3, self.mu, self.nu)

    @override
    def apply(self,
              density_vec_sym: VectorExpression,
              qbx_forced_limit: QBXForcedLimit,
              extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:
        return self.stresslet.apply_single_and_double_layer(
            density_vec_sym,
            obj_array.new_1d([0]*self.dim),
            obj_array.new_1d([0]*self.dim),
            qbx_forced_limit,
            1, 0,
            extra_deriv_dirs)

# }}}
