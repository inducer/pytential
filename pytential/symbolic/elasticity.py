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

import numpy as np
from sumpy.kernel import (AxisSourceDerivative, AxisTargetDerivative,
                          BiharmonicKernel, ElasticityKernel, Kernel,
                          LaplaceKernel, StokesletKernel, StressletKernel,
                          TargetPointMultiplier)
from sumpy.symbolic import SpatialConstant

from pytential import sym
from pytential.symbolic.pde.system_utils import rewrite_using_base_kernel
from pytential.symbolic.typing import ExpressionT

__doc__ = """
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


# {{{ ElasiticityWrapper ABCs

# It is OK if these "escape" into pytential expressions because mappers will
# use the MRO to dispatch them to `map_variable`.
_MU_SYM_DEFAULT = SpatialConstant("mu")
_NU_SYM_DEFAULT = SpatialConstant("nu")


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
    mu: ExpressionT
    r"""Expression or value for the shear modulus :math:`\mu`."""
    nu: ExpressionT
    r"""Expression or value for Poisson's ratio :math:`\nu`."""

    @abstractmethod
    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
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

    def apply_derivative(self, deriv_dir, density_vec_sym, qbx_forced_limit):
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
    mu: ExpressionT
    r"""Expression or value for the shear modulus :math:`\mu`."""
    nu: ExpressionT
    r"""Expression or value for Poisson's ration :math:`\nu`."""

    @abstractmethod
    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):
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
        raise NotImplementedError

    def apply_derivative(self, deriv_dir, density_vec_sym, dir_vec_sym,
            qbx_forced_limit):
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
        return self.apply(density_vec_sym, dir_vec_sym, qbx_forced_limit,
                (deriv_dir,))

# }}}


# {{{ Naive and Biharmonic impl

def _create_int_g(knl, deriv_dirs, density, **kwargs):
    for deriv_dir in deriv_dirs:
        knl = AxisTargetDerivative(deriv_dir, knl)

    kernel_arg_names = {karg.loopy_arg.name
            for karg in (knl.get_args() + knl.get_source_args())}

    # When the kernel is Laplace, mu and nu are not kernel arguments
    # Also when nu==0.5, it's not a kernel argument to StokesletKernel
    for var_name in ["mu", "nu"]:
        if var_name not in kernel_arg_names:
            kwargs.pop(var_name)

    res = sym.int_g_vec(knl, density, **kwargs)
    return res


@dataclass
class _ElasticityWrapperNaiveOrBiharmonic:
    dim: int
    mu: ExpressionT
    nu: ExpressionT
    base_kernel: Kernel

    def __post_init__(self):
        if not (self.dim == 3 or self.dim == 2):
            raise ValueError(
                    f"unsupported dimension given to ElasticityWrapper: {self.dim}")

    @cached_property
    def kernel_dict(self):
        d = {}
        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{01}` is identical to :math:`T_{10}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.
        for i in range(self.dim):
            for j in range(i, self.dim):
                if self.nu == 0.5:
                    d[(i, j)] = StokesletKernel(dim=self.dim, icomp=i,
                        jcomp=j)
                else:
                    d[(i, j)] = ElasticityKernel(dim=self.dim, icomp=i,
                        jcomp=j)
                d[(j, i)] = d[(i, j)]

        return d

    def _get_int_g(self, idx, density_sym, dir_vec_sym, qbx_forced_limit,
            deriv_dirs):
        """
        Returns the convolution of the elasticity kernel given by `idx`
        and its derivatives.
        """
        res = _create_int_g(self.kernel_dict[idx], deriv_dirs,
                    density=density_sym*dir_vec_sym[idx[-1]],
                    qbx_forced_limit=qbx_forced_limit, mu=self.mu,
                    nu=self.nu)/(2*(1-self.nu))
        return res

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):

        sym_expr = np.zeros((self.dim,), dtype=object)

        # For stokeslet, there's no direction vector involved
        # passing a list of ones instead to remove its usage.
        for comp in range(self.dim):
            for i in range(self.dim):
                sym_expr[comp] += self._get_int_g((comp, i),
                        density_vec_sym[i], [1]*self.dim,
                        qbx_forced_limit, deriv_dirs=extra_deriv_dirs)

        return np.array(rewrite_using_base_kernel(sym_expr,
            base_kernel=self.base_kernel))


class ElasticityWrapperNaive(_ElasticityWrapperNaiveOrBiharmonic,
                             ElasticityWrapperBase):
    r"""Elasticity single-layer wrapper based on the standard elasticity kernel.

    This representation uses the elasticity kernel denoted by
    :attr:`Method.Naive`.
    """

    def __init__(self, dim, mu, nu):
        super().__init__(dim=dim, mu=mu, nu=nu, base_kernel=None)
        ElasticityWrapperBase.__init__(self, dim=dim, mu=mu, nu=nu)


class ElasticityWrapperBiharmonic(_ElasticityWrapperNaiveOrBiharmonic,
                                  ElasticityWrapperBase):
    r"""Elasticity single-layer wrapper based on the biharmonic kernel.

    This representation uses the biharmonic kernel denoted by
    :attr:`Method.Biharmonic`.
    """

    def __init__(self, dim, mu, nu):
        super().__init__(dim=dim, mu=mu, nu=nu,
                base_kernel=BiharmonicKernel(dim))
        ElasticityWrapperBase.__init__(self, dim=dim, mu=mu, nu=nu)


# }}}


# {{{ ElasticityDoubleLayerWrapper Naive and Biharmonic impl

@dataclass
class _ElasticityDoubleLayerWrapperNaiveOrBiharmonic:
    dim: int
    mu: ExpressionT
    nu: ExpressionT
    base_kernel: Kernel

    def __post_init__(self):
        if not (self.dim == 3 or self.dim == 2):
            raise ValueError("unsupported dimension given to "
                             f"ElasticityDoubleLayerWrapper: {self.dim}")

    @cached_property
    def kernel_dict(self):
        d = {}

        for i in range(self.dim):
            for j in range(i, self.dim):
                for k in range(j, self.dim):
                    d[(i, j, k)] = StressletKernel(dim=self.dim, icomp=i,
                            jcomp=j, kcomp=k)

        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{012}` is identical to :math:`T_{120}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if (i, j, k) in d:
                        continue
                    s = tuple(sorted([i, j, k]))
                    d[(i, j, k)] = d[s]

        # For elasticity (nu != 0.5), we need the laplacian of the
        # BiharmonicKernel which is the LaplaceKernel.
        if self.nu != 0.5:
            d["laplacian"] = LaplaceKernel(self.dim)

        return d

    def _get_int_g(self, idx, density_sym, dir_vec_sym, qbx_forced_limit,
            deriv_dirs):
        """
        Returns the convolution of the double layer of the elasticity kernel
        given by `idx` and its derivatives.
        """

        nu = self.nu
        kernel_indices = [idx]
        dir_vec_indices = [idx[-1]]
        coeffs = [1]
        extra_deriv_dirs_vec = [[]]

        kernel_indices = [idx, "laplacian", "laplacian", "laplacian"]
        dir_vec_indices = [idx[-1], idx[1], idx[0], idx[2]]
        coeffs = [1, (1 - 2*nu)/self.dim, -(1 - 2*nu)/self.dim, -(1 - 2*nu)]
        extra_deriv_dirs_vec = [[], [idx[0]], [idx[1]], [idx[2]]]
        if idx[0] != idx[1]:
            coeffs[-1] = 0

        result = 0
        for kernel_idx, dir_vec_idx, coeff, extra_deriv_dirs in \
                zip(kernel_indices, dir_vec_indices, coeffs,
                        extra_deriv_dirs_vec):
            if coeff == 0:
                continue
            knl = self.kernel_dict[kernel_idx]
            result += _create_int_g(knl, tuple(deriv_dirs) + tuple(extra_deriv_dirs),
                    density=density_sym*dir_vec_sym[dir_vec_idx],
                    qbx_forced_limit=qbx_forced_limit, mu=self.mu, nu=self.nu) * \
                            coeff
        return result/(2*(1 - nu))

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):

        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += self._get_int_g((comp, i, j),
                        density_vec_sym[i], dir_vec_sym,
                        qbx_forced_limit, deriv_dirs=extra_deriv_dirs)

        return np.array(rewrite_using_base_kernel(sym_expr,
            base_kernel=self.base_kernel))

    def apply_single_and_double_layer(self, stokeslet_density_vec_sym,
            stresslet_density_vec_sym, dir_vec_sym,
            qbx_forced_limit, stokeslet_weight, stresslet_weight,
            extra_deriv_dirs=()):

        stokeslet_obj = _ElasticityWrapperNaiveOrBiharmonic(dim=self.dim,
                mu=self.mu, nu=self.nu, base_kernel=self.base_kernel)

        sym_expr = 0
        if stresslet_weight != 0:
            sym_expr += self.apply(stresslet_density_vec_sym, dir_vec_sym,
                qbx_forced_limit, extra_deriv_dirs) * stresslet_weight
        if stokeslet_weight != 0:
            sym_expr += stokeslet_obj.apply(stokeslet_density_vec_sym,
                qbx_forced_limit, extra_deriv_dirs) * stokeslet_weight

        return sym_expr


class ElasticityDoubleLayerWrapperNaive(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        ElasticityDoubleLayerWrapperBase):
    r"""Elasticity double-layer wrapper based on the standard elasticity kernel.

    This representation uses the elasticity kernel denoted by
    :attr:`Method.Naive`.
    """

    def __init__(self, dim, mu, nu):
        super().__init__(dim=dim, mu=mu, nu=nu,
                base_kernel=None)
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim,
            mu=mu, nu=nu)


class ElasticityDoubleLayerWrapperBiharmonic(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        ElasticityDoubleLayerWrapperBase):
    r"""Elasticity double-layer wrapper based on the biharmonic kernel.

    This representation uses the biharmonic kernel denoted by
    :attr:`Method.Biharmonic`.
    """

    def __init__(self, dim, mu, nu):
        super().__init__(dim=dim, mu=mu, nu=nu,
                base_kernel=BiharmonicKernel(dim))
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim,
            mu=mu, nu=nu)

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
        mu: ExpressionT = _MU_SYM_DEFAULT,
        nu: ExpressionT = _NU_SYM_DEFAULT,
        method: Method = Method.Naive) -> ElasticityWrapperBase:
    """Creates an appropriate :class:`ElasticityWrapperBase` object.

    :param dim: ambient dimension of the representation.
    :param mu: expression or value for the shear modulus.
    :param nu: expression or value for Poisson's ratio. If this is set to
        *0.5* exactly, an appropriate Stokes object is created instead.
    :param method: method to use - defaults to the :attr:`Method.Naive`.
    """
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
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")


def make_elasticity_double_layer_wrapper(
        dim: int,
        mu: ExpressionT = _MU_SYM_DEFAULT,
        nu: ExpressionT = _NU_SYM_DEFAULT,
        method: Method = Method.Naive) -> ElasticityDoubleLayerWrapperBase:
    """Creates an appropriate :class:`ElasticityDoubleLayerWrapperBase` object.

    :param dim: ambient dimension of the representation.
    :param mu: expression or value for the shear modulus.
    :param nu: expression or value for Poisson's ratio. If this is set to
        *0.5* exactly, an appropriate Stokes object is created instead.
    :param method: method to use - defaults to the :attr:`Method.Naive`.
    """
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
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")


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

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):
        return self.apply_single_and_double_layer([0]*self.dim,
            density_vec_sym, dir_vec_sym, qbx_forced_limit, 0, 1,
            extra_deriv_dirs)

    def apply_single_and_double_layer(self, stokeslet_density_vec_sym,
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
            res = -int_g.copy(target_kernel=add_extra_deriv_dirs(
                TargetPointMultiplier(j,
                    AxisTargetDerivative(i, int_g.target_kernel))))
            if i == j:
                res += (3 - 4*nu)*int_g.copy(
                    target_kernel=add_extra_deriv_dirs(int_g.target_kernel))
            return res / (4*mu*(1 - nu))

        def Q(i, int_g):
            res = int_g.copy(target_kernel=add_extra_deriv_dirs(
                AxisTargetDerivative(i, int_g.target_kernel)))
            return res / (4*mu*(1 - nu))

        sym_expr = np.zeros((3,), dtype=object)

        kernel = self.laplace_kernel
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

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
        return self.stresslet.apply_single_and_double_layer(density_vec_sym,
            [0]*self.dim, [0]*self.dim, qbx_forced_limit, 1, 0,
            extra_deriv_dirs)

# }}}
