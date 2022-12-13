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
from pytential.symbolic.pde.system_utils import rewrite_using_base_kernel
from sumpy.kernel import (StressletKernel, LaplaceKernel, StokesletKernel,
    ElasticityKernel, BiharmonicKernel,
    AxisTargetDerivative, AxisSourceDerivative, TargetPointMultiplier)
from sumpy.symbolic import SpatialConstant
from abc import ABC, abstractmethod
from pytential.symbolic.typing import ExpressionT

__doc__ = """
.. autoclass:: ElasticityWrapperBase
.. autoclass:: ElasticityDoubleLayerWrapperBase

.. automethod:: pytential.symbolic.elasticity.create_elasticity_wrapper
.. automethod:: pytential.symbolic.elasticity.create_elasticity_double_layer_wrapper
"""


# {{{ ElasiticityWrapper ABCs

_MU_SYM_DEFAULT = SpatialConstant("mu")
_NU_SYM_DEFAULT = SpatialConstant("nu")


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
    def __init__(self, dim, mu_sym, nu_sym):
        self.dim = dim
        self.mu = mu_sym
        self.nu = nu_sym

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
    def __init__(self, dim, mu_sym, nu_sym):
        self.dim = dim
        self.mu = mu_sym
        self.nu = nu_sym

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

    kernel_arg_names = set(karg.loopy_arg.name
            for karg in (knl.get_args() + knl.get_source_args()))

    # When the kernel is Laplace, mu and nu are not kernel arguments
    # Also when nu==0.5, it's not a kernel argument to StokesletKernel
    for var_name in ["mu", "nu"]:
        if var_name not in kernel_arg_names:
            kwargs.pop(var_name)

    res = sym.int_g_vec(knl, density, **kwargs)
    return res


class _ElasticityWrapperNaiveOrBiharmonic:
    def __init__(self, dim, mu_sym, nu_sym, base_kernel):
        self.dim = dim
        self.mu = mu_sym
        self.nu = nu_sym

        if not (dim == 3 or dim == 2):
            raise ValueError(
                    f"unsupported dimension given to ElasticityWrapper: {dim}")

        self.base_kernel = base_kernel

        self.kernel_dict = {}

        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{01}` is identical to :math:`T_{10}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.
        for i in range(dim):
            for j in range(i, dim):
                if nu_sym == 0.5:
                    self.kernel_dict[(i, j)] = StokesletKernel(dim=dim, icomp=i,
                        jcomp=j)
                else:
                    self.kernel_dict[(i, j)] = ElasticityKernel(dim=dim, icomp=i,
                        jcomp=j)
                self.kernel_dict[(j, i)] = self.kernel_dict[(i, j)]

    def _get_int_g(self, idx, density_sym, dir_vec_sym, qbx_forced_limit,
            deriv_dirs):
        """
        Returns the Integral of the elasticity kernel given by `idx`
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
    def __init__(self, dim, mu_sym, nu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=nu_sym, base_kernel=None)
        ElasticityWrapperBase.__init__(self, dim=dim, mu_sym=mu_sym, nu_sym=nu_sym)


class ElasticityWrapperBiharmonic(_ElasticityWrapperNaiveOrBiharmonic,
                                  ElasticityWrapperBase):
    def __init__(self, dim, mu_sym, nu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=nu_sym,
                base_kernel=BiharmonicKernel(dim))
        ElasticityWrapperBase.__init__(self, dim=dim, mu_sym=mu_sym, nu_sym=nu_sym)


# }}}


# {{{ ElasticityDoubleLayerWrapper Naive and Biharmonic impl

class _ElasticityDoubleLayerWrapperNaiveOrBiharmonic:

    def __init__(self, dim, mu_sym, nu_sym, base_kernel):
        self.dim = dim
        self.mu = mu_sym
        self.nu = nu_sym

        if not (dim == 3 or dim == 2):
            raise ValueError("unsupported dimension given to "
                             f"ElasticityDoubleLayerWrapper: {dim}")

        self.base_kernel = base_kernel

        self.kernel_dict = {}

        for i in range(dim):
            for j in range(i, dim):
                for k in range(j, dim):
                    self.kernel_dict[(i, j, k)] = StressletKernel(dim=dim, icomp=i,
                            jcomp=j, kcomp=k)

        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{012}` is identical to :math:`T_{120}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    if (i, j, k) in self.kernel_dict:
                        continue
                    s = tuple(sorted([i, j, k]))
                    self.kernel_dict[(i, j, k)] = self.kernel_dict[s]

        # For elasticity (nu != 0.5), we need the LaplaceKernel
        if nu_sym != 0.5:
            self.kernel_dict["laplace"] = LaplaceKernel(self.dim)

    def _get_int_g(self, idx, density_sym, dir_vec_sym, qbx_forced_limit,
            deriv_dirs):
        """
        Returns the Integral of the Stresslet kernel given by `idx`
        and its derivatives.
        """

        nu = self.nu
        kernel_indices = [idx]
        dir_vec_indices = [idx[-1]]
        coeffs = [1]
        extra_deriv_dirs_vec = [[]]

        kernel_indices = [idx, "laplace", "laplace", "laplace"]
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
                mu_sym=self.mu, nu_sym=self.nu, base_kernel=self.base_kernel)

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
    def __init__(self, dim, mu_sym, nu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=nu_sym,
                base_kernel=None)
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim,
            mu_sym=mu_sym, nu_sym=nu_sym)


class ElasticityDoubleLayerWrapperBiharmonic(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        ElasticityDoubleLayerWrapperBase):
    def __init__(self, dim, mu_sym, nu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=nu_sym,
                base_kernel=BiharmonicKernel(dim))
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim,
            mu_sym=mu_sym, nu_sym=nu_sym)

# }}}


# {{{ dispatch function

def create_elasticity_wrapper(
        dim: int,
        mu_sym: ExpressionT = _MU_SYM_DEFAULT,
        nu_sym: ExpressionT = _NU_SYM_DEFAULT,
        method: str = "naive") -> ElasticityWrapperBase:
    """Creates a :class:`ElasticityWrapperBase` object depending on the input
    values.

    :param: dim: dimension
    :param: mu_sym: viscosity symbol, defaults to "mu"
    :param: nu_sym: poisson ratio symbol, defaults to "nu"
    :param: method: method to use, defaults to "naive".
        One of ("naive", "laplace", "biharmonic")

    :return: a :class:`ElasticityWrapperBase` object
    """

    if nu_sym == 0.5:
        from pytential.symbolic.stokes import StokesletWrapper
        return StokesletWrapper(dim=dim, mu_sym=mu_sym, method=method)
    if method == "naive":
        return ElasticityWrapperNaive(dim=dim, mu_sym=mu_sym, nu_sym=nu_sym)
    elif method == "biharmonic":
        return ElasticityWrapperBiharmonic(dim=dim, mu_sym=mu_sym, nu_sym=nu_sym)
    elif method == "laplace":
        if nu_sym == 0.5:
            from pytential.symbolic.stokes import StokesletWrapperTornberg
            return StokesletWrapperTornberg(dim=dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
        else:
            return ElasticityWrapperYoshida(dim=dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
    else:
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")


def create_elasticity_double_layer_wrapper(
        dim: int,
        mu_sym: ExpressionT = _MU_SYM_DEFAULT,
        nu_sym: ExpressionT = _NU_SYM_DEFAULT,
        method: str = "naive") -> ElasticityDoubleLayerWrapperBase:
    """Creates a :class:`ElasticityDoubleLayerWrapperBase` object depending on the
    input values.

    :param: dim: dimension
    :param: mu_sym: viscosity symbol, defaults to "mu"
    :param: nu_sym: poisson ratio symbol, defaults to "nu"
    :param: method: method to use, defaults to "naive".
        One of ("naive", "laplace", "biharmonic")

    :return: a :class:`ElasticityDoubleLayerWrapperBase` object
    """
    if nu_sym == 0.5:
        from pytential.symbolic.stokes import StressletWrapper
        return StressletWrapper(dim=dim, mu_sym=mu_sym, method=method)
    if method == "naive":
        return ElasticityDoubleLayerWrapperNaive(dim=dim, mu_sym=mu_sym,
            nu_sym=nu_sym)
    elif method == "biharmonic":
        return ElasticityDoubleLayerWrapperBiharmonic(dim=dim, mu_sym=mu_sym,
            nu_sym=nu_sym)
    elif method == "laplace":
        if nu_sym == 0.5:
            from pytential.symbolic.stokes import StressletWrapperTornberg
            return StressletWrapperTornberg(dim=dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
        else:
            return ElasticityDoubleLayerWrapperYoshida(dim=dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
    else:
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")


# }}}


# {{{ Yoshida

class ElasticityDoubleLayerWrapperYoshida(ElasticityDoubleLayerWrapperBase):
    """ElasticityDoubleLayer Wrapper using Yoshida et al's method [1] which uses
    Laplace derivatives.

    [1] Yoshida, K. I., Nishimura, N., & Kobayashi, S. (2001). Application of
        fast multipole Galerkin boundary integral equation method to elastostatic
        crack problems in 3D.
        International Journal for Numerical Methods in Engineering, 50(3), 525-547.
        https://doi.org/10.1002/1097-0207(20010130)50:3<525::AID-NME34>3.0.CO;2-4
    """

    def __init__(self, dim=None, mu_sym=_MU_SYM_DEFAULT, nu_sym=_NU_SYM_DEFAULT):
        self.dim = dim
        if dim != 3:
            raise ValueError("unsupported dimension given to "
                             "ElasticityDoubleLayerWrapperYoshida: {dim}")
        self.kernel = LaplaceKernel(dim=3)
        self.mu = mu_sym
        self.nu = nu_sym

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


class ElasticityWrapperYoshida(ElasticityWrapperBase):
    """Elasticity single layer using Yoshida et al's method [1] which uses Laplace
    derivatives.

    [1] Yoshida, K. I., Nishimura, N., & Kobayashi, S. (2001). Application of
        fast multipole Galerkin boundary integral equation method to elastostatic
        crack problems in 3D.
        International Journal for Numerical Methods in Engineering, 50(3), 525-547.
        https://doi.org/10.1002/1097-0207(20010130)50:3<525::AID-NME34>3.0.CO;2-4
    """

    def __init__(self, dim=None, mu_sym=_MU_SYM_DEFAULT, nu_sym=_NU_SYM_DEFAULT):
        self.dim = dim
        if dim != 3:
            raise ValueError("unsupported dimension given to "
                             "ElasticityWrapperYoshida: {dim}")
        self.kernel = LaplaceKernel(dim=3)
        self.mu = mu_sym
        self.nu = nu_sym
        self.stresslet = ElasticityDoubleLayerWrapperYoshida(3, self.mu, self.nu)

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
        return self.stresslet.apply_single_and_double_layer(density_vec_sym,
            [0]*self.dim, [0]*self.dim, qbx_forced_limit, 1, 0,
            extra_deriv_dirs)

# }}}
