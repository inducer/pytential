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

import numpy as np

from pytential import sym
from pytential.symbolic.pde.systems import (rewrite_using_base_kernel,
    merge_int_g_exprs)
from pytential.symbolic.typing import ExpressionT
from sumpy.kernel import (StressletKernel, LaplaceKernel, StokesletKernel,
    ElasticityKernel, BiharmonicKernel, Kernel, LineOfCompressionKernel,
    AxisTargetDerivative, AxisSourceDerivative, TargetPointMultiplier)
from sumpy.symbolic import SpatialConstant
import pymbolic

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from enum import Enum

__doc__ = """
.. autoclass:: ElasticityWrapperBase
.. autoclass:: ElasticityDoubleLayerWrapperBase
.. autoclass:: Method

.. automethod:: pytential.symbolic.elasticity.make_elasticity_wrapper
.. automethod:: pytential.symbolic.elasticity.make_elasticity_double_layer_wrapper
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
    out every term in the expansion of the double-indexed Elasticity kernel
    applied to the density vector.  The object is created
    to do some of the set-up and bookkeeping once, rather than every
    time we want to create a symbolic expression based on the kernel -- say,
    once when we solve for the density, and once when we want a symbolic
    representation for the solution, for example.

    The :meth:`apply` function returns the integral expressions needed for
    the vector velocity resulting from convolution with the vector density,
    and is meant to work similarly to calling
    :func:`~pytential.symbolic.primitives.S` (which is
    :class:`~pytential.symbolic.primitives.IntG`).

    Similar functions are available for other useful things related to
    the flow: :meth:`apply_derivative` (target derivative).

    .. automethod:: apply
    .. automethod:: apply_derivative
    """
    dim: int
    mu: ExpressionT
    nu: ExpressionT

    @abstractmethod
    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
        """Symbolic expressions for integrating Elasticity kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the dyadic Elasticity kernel with
        variable *density_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        :arg extra_deriv_dirs: adds target derivatives to all the integral
            objects with the given derivative axis.
        """

    def apply_derivative(self, deriv_dir, density_vec_sym, qbx_forced_limit):
        """Symbolic derivative of velocity from Elasticity kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        dyadic Elasticity kernel with variable *density_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative.
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
    out every term in the expansion of the triple-indexed Stresslet
    kernel applied to both a normal vector and the density vector.
    The object is created to do some of the set-up and bookkeeping once,
    rather than every time we want to create a symbolic expression based
    on the kernel -- say, once when we solve for the density, and once when
    we want a symbolic representation for the solution, for example.

    The :meth:`apply` function returns the integral expressions needed for
    convolving the kernel with a vector density, and is meant to work
    similarly to :func:`~pytential.symbolic.primitives.S` (which is
    :class:`~pytential.symbolic.primitives.IntG`).

    Similar functions are available for other useful things related to
    the flow: :meth:`apply_derivative` (target derivative).

    .. automethod:: apply
    .. automethod:: apply_derivative
    """
    dim: int
    mu: ExpressionT
    nu: ExpressionT

    @abstractmethod
    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):
        """Symbolic expressions for integrating Stresslet kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the dyadic Stresslet kernel with
        variable *density_vec_sym* and source direction vectors *dir_vec_sym*.

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
        """Symbolic derivative of velocity from Elasticity kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        dyadic Elasticity kernel with variable *density_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative.
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
    def __init__(self, dim, mu, nu):
        super().__init__(dim=dim, mu=mu, nu=nu, base_kernel=None)
        ElasticityWrapperBase.__init__(self, dim=dim, mu=mu, nu=nu)


class ElasticityWrapperBiharmonic(_ElasticityWrapperNaiveOrBiharmonic,
                                  ElasticityWrapperBase):
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
    def __init__(self, dim, mu, nu):
        super().__init__(dim=dim, mu=mu, nu=nu,
                base_kernel=None)
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim,
            mu=mu, nu=nu)


class ElasticityDoubleLayerWrapperBiharmonic(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        ElasticityDoubleLayerWrapperBase):
    def __init__(self, dim, mu, nu):
        super().__init__(dim=dim, mu=mu, nu=nu,
                base_kernel=BiharmonicKernel(dim))
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim,
            mu=mu, nu=nu)

# }}}


# {{{ dispatch function

class Method(Enum):
    """Method to use in Elasticity/Stokes problem.
    """
    naive = 1
    laplace = 2
    biharmonic = 3
    laplace_slow = 4


def make_elasticity_wrapper(
        dim: int,
        mu: ExpressionT = _MU_SYM_DEFAULT,
        nu: ExpressionT = _NU_SYM_DEFAULT,
        method: Method = Method.naive) -> ElasticityWrapperBase:
    """Creates a :class:`ElasticityWrapperBase` object depending on the input
    values.

    :param: dim: dimension
    :param: mu: viscosity symbol, defaults to a variable named "mu"
    :param: nu: poisson ratio symbol, defaults to a variable named "nu"
    :param: method: method to use, defaults to the *Method* enum value naive.

    :return: a :class:`ElasticityWrapperBase` object
    """

    if nu == 0.5:
        from pytential.symbolic.stokes import StokesletWrapper
        return StokesletWrapper(dim=dim, mu=mu, method=method)
    if method == Method.naive:
        return ElasticityWrapperNaive(dim=dim, mu=mu, nu=nu)
    elif method == Method.biharmonic:
        return ElasticityWrapperBiharmonic(dim=dim, mu=mu, nu=nu)
    elif method == Method.laplace:
        if nu == 0.5:
            from pytential.symbolic.stokes import StokesletWrapperTornberg
            return StokesletWrapperTornberg(dim=dim,
                mu=mu, nu=nu)
        else:
            return ElasticityWrapperYoshida(dim=dim,
                mu=mu, nu=nu)
    elif method == Method.laplace_slow:
        if nu == 0.5:
            raise ValueError("invalid value of nu=0.5 for method laplace_slow")
        else:
            return ElasticityWrapperFu(dim=dim,
                mu=mu, nu=nu)
    else:
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")


def make_elasticity_double_layer_wrapper(
        dim: int,
        mu: ExpressionT = _MU_SYM_DEFAULT,
        nu: ExpressionT = _NU_SYM_DEFAULT,
        method: Method = Method.naive) -> ElasticityDoubleLayerWrapperBase:
    """Creates a :class:`ElasticityDoubleLayerWrapperBase` object depending on the
    input values.

    :param: dim: dimension
    :param: mu: viscosity symbol, defaults to a variable named "mu"
    :param: nu: poisson ratio symbol, defaults to a variable named "nu"
    :param: method: method to use, defaults to the *Method* enum value naive.

    :return: a :class:`ElasticityDoubleLayerWrapperBase` object
    """
    if nu == 0.5:
        from pytential.symbolic.stokes import StressletWrapper
        return StressletWrapper(dim=dim, mu=mu, method=method)
    if method == Method.naive:
        return ElasticityDoubleLayerWrapperNaive(dim=dim, mu=mu,
            nu=nu)
    elif method == Method.biharmonic:
        return ElasticityDoubleLayerWrapperBiharmonic(dim=dim, mu=mu,
            nu=nu)
    elif method == Method.laplace:
        if nu == 0.5:
            from pytential.symbolic.stokes import StressletWrapperTornberg
            return StressletWrapperTornberg(dim=dim,
                mu=mu, nu=nu)
        else:
            return ElasticityDoubleLayerWrapperYoshida(dim=dim,
                mu=mu, nu=nu)
    elif method == Method.laplace_slow:
        if nu == 0.5:
            raise ValueError("invalid value of nu=0.5 for method laplace_slow")
        else:
            return ElasticityDoubleLayerWrapperFu(dim=dim,
                mu=mu, nu=nu)
    else:
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")


# }}}


# {{{ Yoshida

@dataclass
class ElasticityDoubleLayerWrapperYoshida(ElasticityDoubleLayerWrapperBase):
    r"""ElasticityDoubleLayer Wrapper using Yoshida et al's method [1] which uses
    Laplace derivatives.

    [1] Yoshida, K. I., Nishimura, N., & Kobayashi, S. (2001). Application of
        fast multipole Galerkin boundary integral equation method to elastostatic
        crack problems in 3D.
        International Journal for Numerical Methods in Engineering, 50(3), 525-547.
        `DOI <https://doi.org/10.1002/1097-0207(20010130)50:3\<525::AID-NME34\>3.0.CO;2-4>`__  # noqa
    """
    dim: int
    mu: ExpressionT
    nu: ExpressionT

    def __post_init__(self):
        if not self.dim == 3:
            raise ValueError("unsupported dimension given to "
                             "ElasticityDoubleLayerWrapperYoshida: {self.dim}")

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
        source = sym.nodes(3).as_vector()
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
    r"""Elasticity single layer using Yoshida et al's method [1] which uses Laplace
    derivatives.

    [1] Yoshida, K. I., Nishimura, N., & Kobayashi, S. (2001). Application of
        fast multipole Galerkin boundary integral equation method to elastostatic
        crack problems in 3D.
        International Journal for Numerical Methods in Engineering, 50(3), 525-547.
        `DOI <https://doi.org/10.1002/1097-0207(20010130)50:3\<525::AID-NME34\>3.0.CO;2-4>`__  # noqa
    """
    dim: int
    mu: ExpressionT
    nu: ExpressionT

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

# {{{ Fu

@dataclass
class ElasticityDoubleLayerWrapperFu(ElasticityDoubleLayerWrapperBase):
    r"""ElasticityDoubleLayer Wrapper using Fu et al's method [1] which uses
    Laplace derivatives.

    [1] Fu, Y., Klimkowski, K. J., Rodin, G. J., Berger, E., Browne, J. C.,
        Singer, J. K., ... & Vemaganti, K. S. (1998). A fast solution method for
        three‐dimensional many‐particle problems of linear elasticity.
        International Journal for Numerical Methods in Engineering, 42(7), 1215-1229.
    """
    dim: int
    mu: ExpressionT
    nu: ExpressionT

    def __post_init__(self):
        if not self.dim == 3:
            raise ValueError("unsupported dimension given to "
                             "ElasticityDoubleLayerWrapperFu: {self.dim}")

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
        stokeslet_weight *= -1

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

        def R(i, j, p, int_g):
            res = int_g.copy(target_kernel=add_extra_deriv_dirs(
                TargetPointMultiplier(j, AxisTargetDerivative(i,
                    AxisTargetDerivative(p, int_g.target_kernel)))))
            if j == p:
                res += (1 - 2*nu)*int_g.copy(target_kernel=add_extra_deriv_dirs(
                    AxisTargetDerivative(i, int_g.target_kernel)))
            if i == j:
                res -= (1 - 2*nu)*int_g.copy(target_kernel=add_extra_deriv_dirs(
                    AxisTargetDerivative(p, int_g.target_kernel)))
            if i == p:
                res -= 2*(1 - nu)*int_g.copy(target_kernel=add_extra_deriv_dirs(
                    AxisTargetDerivative(j, int_g.target_kernel)))
            return res / (2*mu*(1 - nu))

        def S(i, p, int_g):
            res = int_g.copy(target_kernel=add_extra_deriv_dirs(
                AxisTargetDerivative(i,
                    AxisTargetDerivative(p, int_g.target_kernel))))
            return res / (-2*mu*(1 - nu))

        sym_expr = np.zeros((3,), dtype=object)

        kernel = self.laplace_kernel
        source = sym.nodes(3).as_vector()
        normal = dir_vec_sym
        sigma = stresslet_density_vec_sym

        for i in range(3):
            for j in range(3):
                density = stokeslet_weight * stokeslet_density_vec_sym[j]
                int_g = sym.IntG(target_kernel=kernel,
                        source_kernels=(kernel,),
                        densities=(density,),
                        qbx_forced_limit=qbx_forced_limit)
                sym_expr[i] += P(i, j, int_g)

            density = sum(stokeslet_weight
                * stokeslet_density_vec_sym[j] * source[j] for j in range(3))
            int_g = sym.IntG(target_kernel=kernel,
                source_kernels=(kernel,),
                densities=(density,),
                qbx_forced_limit=qbx_forced_limit)
            sym_expr[i] += Q(i, int_g)

            for j in range(3):
                for p in range(3):
                    density = stresslet_weight * normal[p] * sigma[j]
                    int_g = sym.IntG(target_kernel=kernel,
                        source_kernels=(kernel,),
                        densities=(density,),
                        qbx_forced_limit=qbx_forced_limit)
                    sym_expr[i] += R(i, j, p, int_g)

            for p in range(3):
                density = sum(stresslet_weight * normal[p] * sigma[j] * source[j]
                              for j in range(3))
                int_g = sym.IntG(target_kernel=kernel,
                    source_kernels=(kernel,),
                    densities=(density,),
                    qbx_forced_limit=qbx_forced_limit)
                sym_expr[i] += S(i, p, int_g)

        return sym_expr


@dataclass
class ElasticityWrapperFu(ElasticityWrapperBase):
    r"""Elasticity single layer using Fu et al's method [1] which uses
    Laplace derivatives.

    [1] Fu, Y., Klimkowski, K. J., Rodin, G. J., Berger, E., Browne, J. C.,
        Singer, J. K., ... & Vemaganti, K. S. (1998). A fast solution method for
        three‐dimensional many‐particle problems of linear elasticity.
        International Journal for Numerical Methods in Engineering, 42(7), 1215-1229.
    """
    dim: int
    mu: ExpressionT
    nu: ExpressionT

    def __post_init__(self):
        if not self.dim == 3:
            raise ValueError("unsupported dimension given to "
                             f"ElasticityDoubleLayerWrapperFu: {self.dim}")

    @cached_property
    def stresslet(self):
        return ElasticityDoubleLayerWrapperFu(3, self.mu, self.nu)

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
        return self.stresslet.apply_single_and_double_layer(density_vec_sym,
            [0]*self.dim, [0]*self.dim, qbx_forced_limit, 1, 0,
            extra_deriv_dirs)

# }}}


# {{{ Kelvin operator

class ElasticityOperator:
    """
    .. automethod:: __init__
    .. automethod:: get_density_var
    .. automethod:: operator
    """

    def __init__(
            self,
            dim: int,
            mu: ExpressionT = _MU_SYM_DEFAULT,
            nu: ExpressionT = _NU_SYM_DEFAULT,
            method: Method = Method.naive):

        self.dim = dim
        self.mu = mu
        self.nu = nu
        self.method = method

    def get_density_var(self, name="sigma"):
        """
        :returns: a (potentially) modified right-hand side *b* that matches
            requirements of the representation.
        """
        return sym.make_sym_vector(name, self.dim)

    @abstractmethod
    def operator(self, sigma):
        """
        :returns: the integral operator that should be solved to obtain the
            density *sigma*.
        """
        raise NotImplementedError


class KelvinOperator(ElasticityOperator):
    """Representation for free space Green's function for elasticity commonly
    known as the Kelvin solution [1] given by Lord Kelvin.
    [1] Gimbutas, Z., & Greengard, L. (2016). A fast multipole method for the
        evaluation of elastostatic fields in a half-space with zero normal stress.
        Advances in Computational Mathematics, 42(1), 175-198.
    .. automethod:: __init__
    .. automethod:: operator
    """

    def __init__(
            self,
            mu: ExpressionT = _MU_SYM_DEFAULT,
            nu: ExpressionT = _NU_SYM_DEFAULT,
            method: Method = Method.naive) -> ElasticityWrapperBase:

        dim = 3
        super().__init__(dim=dim, method=method, mu=mu, nu=nu)
        self.double_layer_op = make_elasticity_double_layer_wrapper(
                dim=dim, mu=mu, nu=nu, method=method)
        self.single_layer_op = make_elasticity_wrapper(
                dim=dim, mu=mu, nu=nu, method=method)
        self.laplace_kernel = LaplaceKernel(3)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        return self.double_layer_op.apply(sigma, normal,
            qbx_forced_limit=qbx_forced_limit)

# }}}


# {{{ Mindlin operator

class MindlinOperator(ElasticityOperator):
    """Representation for elasticity in a half-space with zero normal stress which
    is based on Mindlin's explicit solution. See [1] and [2].
    [1] Mindlin, R. D. (1936). Force at a point in the interior of a semi‐infinite
        solid. Physics, 7(5), 195-202.
    [2] Gimbutas, Z., & Greengard, L. (2016). A fast multipole method for the
        evaluation of elastostatic fields in a half-space with zero normal stress.
        Advances in Computational Mathematics, 42(1), 175-198.
    .. automethod:: __init__
    .. automethod:: operator
    .. automethod:: free_space_operator
    .. automethod:: get_density_var
    """

    def __init__(self, *,
            method: Method = Method.biharmonic,
            mu: ExpressionT = _MU_SYM_DEFAULT,
            nu: ExpressionT = _NU_SYM_DEFAULT,
            line_of_compression_tol: float = 0.0):

        super().__init__(dim=3, method=method, mu=mu, nu=nu)
        self.free_space_op = KelvinOperator(method=method, mu=mu,
                nu=nu)
        self.modified_free_space_op = KelvinOperator(
                method=method, mu=mu + 4*nu, nu=-nu)
        self.compression_knl = LineOfCompressionKernel(3, 2, mu, nu)

    def K(self, sigma, normal, qbx_forced_limit):
        return merge_int_g_exprs(self.free_space_op.double_layer_op.apply(
            sigma, normal, qbx_forced_limit=qbx_forced_limit))

    def A(self, sigma, normal, qbx_forced_limit):
        result = -self.modified_free_space_op.double_layer_op.apply(
            sigma, normal, qbx_forced_limit=qbx_forced_limit)

        new_density = sum(a*b for a, b in zip(sigma, normal))
        int_g = sym.S(self.free_space_op.laplace_kernel, new_density,
            qbx_forced_limit=qbx_forced_limit)

        for i in range(3):
            temp = 2*int_g.copy(
                    target_kernel=AxisTargetDerivative(i, int_g.target_kernel))
            if i == 2:
                temp *= -1
            result[i] += temp
        return result

    def B(self, sigma, normal, qbx_forced_limit):
        sym_expr = np.zeros((3,), dtype=object)
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
            kwargs[arg] = pymbolic.var(arg)

        int_g = sym.IntG(source_kernels=tuple(source_kernels),
                target_kernel=self.compression_knl, densities=tuple(densities),
                **kwargs)

        for i in range(3):
            if self.method == Method.naive:
                source_kernels = [AxisSourceDerivative(i, knl) for knl in
                    int_g.source_kernels]
                densities = [(-1)*density for density in int_g.densities]
                sym_expr[i] = int_g.copy(source_kernels=tuple(source_kernels),
                    densities=tuple(densities))
            else:
                sym_expr[i] = int_g.copy(target_kernel=AxisTargetDerivative(
                    i, int_g.target_kernel))

        return sym_expr

    def C(self, sigma, normal, qbx_forced_limit):
        result = np.zeros((3,), dtype=object)
        mu = self.mu
        nu = self.nu
        lam = 2*nu*mu/(1-2*nu)
        alpha = (lam + mu)/(lam + 2*mu)
        y = sym.nodes(3).as_vector()
        sigma_normal_product = sum(a*b for a, b in zip(sigma, normal))

        laplace_kernel = self.free_space_op.laplace_kernel

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

        # phi_c  in Gimbutas et, al.
        densities = [
            -2*alpha*mu*y[2]*sigma[0]*normal[0],
            -2*alpha*mu*y[2]*sigma[1]*normal[1],
            -2*alpha*mu*y[2]*sigma[2]*normal[2],
            -2*alpha*mu*y[2]*(sigma[0]*normal[1] + sigma[1]*normal[0]),
            +2*alpha*mu*y[2]*(sigma[0]*normal[2] + sigma[2]*normal[0]),
            +2*alpha*mu*y[2]*(sigma[1]*normal[2] + sigma[2]*normal[1]),
        ]
        source_kernel_dirs = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]
        source_kernels = [
            AxisSourceDerivative(a, AxisSourceDerivative(b, laplace_kernel))
            for a, b in source_kernel_dirs
        ]

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

        for i in range(3):
            if self.method == Method.naive:
                source_kernels = [AxisSourceDerivative(i, knl) for knl in
                    int_g.source_kernels]
                densities = [(-1)*density for density in int_g.densities]

                if i == 2:
                    # Target derivative w.r.t x[2] is flipped due to target image
                    densities[2] *= -1
                    # Subtract H
                    source_kernels = source_kernels + list(H.source_kernels)
                    densities = densities + [(-1)*d for d in H.densities]

                result[i] = int_g.copy(source_kernels=tuple(source_kernels),
                    densities=tuple(densities))
            else:
                result[i] = int_g.copy(target_kernel=AxisTargetDerivative(
                    i, int_g.target_kernel))
                if i == 2:
                    # Target derivative w.r.t x[2] is flipped due to target image
                    result[2] *= -1
                    # Subtract H
                    result[2] -= H

        return result

    def free_space_operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        return self.free_space_op.operator(sigma=sigma, normal=normal,
            qbx_forced_limit=qbx_forced_limit)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        resultA = self.A(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)
        resultC = self.C(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)
        resultB = self.B(sigma, normal=normal, qbx_forced_limit=qbx_forced_limit)

        if self.method == Method.biharmonic:
            # A and C are both derivatives of Biharmonic Green's function
            # TODO: make merge_int_g_exprs smart enough to merge two different
            # kernels into two separate IntGs.
            result = rewrite_using_base_kernel(resultA + resultC,
                base_kernel=self.free_space_op.double_layer_op.base_kernel)
            result += rewrite_using_base_kernel(resultB,
                base_kernel=self.compression_knl)
            return result
        else:
            return resultA + resultB + resultC

    def get_density_var(self, name="sigma"):
        """
        :returns: a symbolic vector corresponding to the density.
        """
        return sym.make_sym_vector(name, 3)

# }}}
