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

import numpy as np

from pytential import sym
from pytential.symbolic.pde.system_utils import merge_int_g_exprs
from sumpy.kernel import (StressletKernel, LaplaceKernel,
    ElasticityKernel, BiharmonicKernel,
    AxisTargetDerivative, AxisSourceDerivative, TargetPointMultiplier)
from pymbolic import var

__doc__ = """
.. autoclass:: StokesletWrapper
.. autoclass:: StressletWrapper

.. autoclass:: StokesOperator
.. autoclass:: HsiaoKressExteriorStokesOperator
.. autoclass:: HebekerExteriorStokesOperator
"""


# {{{ StokesletWrapper

class StokesletWrapperBase:
    """Wrapper class for the :class:`~sumpy.kernel.StokesletKernel` kernel.

    This class is meant to shield the user from the messiness of writing
    out every term in the expansion of the double-indexed Stokeslet kernel
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
    the flow: :meth:`apply_pressure`, :meth:`apply_derivative` (target derivative),
    :meth:`apply_stress` (applies symmetric viscous stress tensor in
    the requested direction).

    .. automethod:: apply
    .. automethod:: apply_pressure
    .. automethod:: apply_derivative
    .. automethod:: apply_stress
    """
    def __init__(self, dim, mu_sym, nu_sym):
        self.dim = dim
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=[]):
        """Symbolic expressions for integrating Stokeslet kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the dyadic Stokeslet kernel with
        variable *density_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        raise NotImplementedError

    def apply_pressure(self, density_vec_sym, qbx_forced_limit):
        """Symbolic expression for pressure field associated with the Stokeslet."""
        from pytential.symbolic.mappers import DerivativeTaker
        kernel = LaplaceKernel(dim=self.dim)
        sym_expr = 0

        for i in range(self.dim):
            sym_expr += (DerivativeTaker(i).map_int_g(
                         sym.S(kernel, density_vec_sym[i],
                         qbx_forced_limit=qbx_forced_limit)))

        return merge_int_g_exprs(sym_expr)

    def apply_derivative(self, deriv_dir, density_vec_sym, qbx_forced_limit):
        """Symbolic derivative of velocity from Stokeslet.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        dyadic Stokeslet kernel with variable *density_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative.
        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        return self.apply(density_vec_sym, qbx_forced_limit, [deriv_dir])

    def apply_stress(self, density_vec_sym, dir_vec_sym, qbx_forced_limit):
        r"""Symbolic expression for viscous stress applied to a direction.

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress

        .. math::

            -p \delta_{ij} + \mu (\nabla_i u_j + \nabla_j u_i)

        applied in the direction of *dir_vec_sym*.

        Note that this computation is very similar to computing
        a double-layer potential with the Stresslet kernel in
        :class:`StressletWrapper`. The difference is that here the direction
        vector is applied at the target points, while in the Stresslet the
        direction is applied at the source points.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector for the application direction.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        raise NotImplementedError


class StressletWrapperBase:
    """Wrapper class for the :class:`~sumpy.kernel.StressletKernel` kernel.

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
    the flow: :meth:`apply_pressure`, :meth:`apply_derivative` (target derivative),
    :meth:`apply_stress` (applies symmetric viscous stress tensor in
    the requested direction).

    .. automethod:: apply
    .. automethod:: apply_pressure
    .. automethod:: apply_derivative
    .. automethod:: apply_stress
    """
    def __init__(self, dim, mu_sym, nu_sym):
        self.dim = dim
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit, extra_deriv_dirs=[]):
        """Symbolic expressions for integrating Stresslet kernel.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the dyadic Stresslet kernel with
        variable *density_vec_sym* and source direction vectors *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector variable for the direction vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        raise NotImplementedError

    def apply_pressure(self, density_vec_sym, dir_vec_sym, qbx_forced_limit):
        """Symbolic expression for pressure field associated with the Stresslet."""
        import itertools
        from pytential.symbolic.mappers import DerivativeTaker
        kernel = LaplaceKernel(dim=self.dim)

        factor = (2. * self.mu)

        sym_expr = 0

        for i, j in itertools.product(range(self.dim), range(self.dim)):
            sym_expr += factor * DerivativeTaker(i).map_int_g(
                                   DerivativeTaker(j).map_int_g(
                                       sym.int_g_vec(kernel,
                                             density_vec_sym[i] * dir_vec_sym[j],
                                             qbx_forced_limit=qbx_forced_limit)))

        return merge_int_g_exprs(sym_expr)

    def apply_stress(self, density_vec_sym, normal_vec_sym, dir_vec_sym,
                        qbx_forced_limit):
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
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        raise NotImplementedError


def _create_int_g(knl, deriv_dirs, density, **kwargs):
    for deriv_dir in deriv_dirs:
        knl = AxisTargetDerivative(deriv_dir, knl)

    args = [arg.loopy_arg.name for arg in knl.get_args()]
    for arg in args:
        kwargs[arg] = var(arg)

    res = sym.S(knl, density, **kwargs)
    return res


class StokesletWrapper(StokesletWrapperBase):
    def __init__(self, dim=None, mu_sym=var("mu"), nu_sym=0.5):
        super().__init__(dim, mu_sym, nu_sym)
        if not (dim == 3 or dim == 2):
            raise ValueError("unsupported dimension given to StokesletWrapper")

        self.kernel_dict = {}

        for i in range(dim):
            for j in range(i, dim):
                self.kernel_dict[(i, j)] = ElasticityKernel(dim=dim, icomp=i,
                    jcomp=j, viscosity_mu=mu_sym, poisson_ratio=nu_sym)

        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{01}` is identical to :math:`T_{10}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.
        for i in range(dim):
            for j in range(i):
                self.kernel_dict[(i, j)] = self.kernel_dict[(j, i)]

    def get_int_g(self, idx, density_sym, dir_vec_sym, qbx_forced_limit,
            deriv_dirs):
        """
        Returns the Integral of the Stokeslet/Stresslet kernel given by `idx`
        and its derivatives.
        """
        return _create_int_g(self.kernel_dict[idx], deriv_dirs,
                    density=density_sym*dir_vec_sym[idx[-1]],
                    qbx_forced_limit=qbx_forced_limit)/(2*(1-self.nu))

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=[]):

        sym_expr = np.zeros((self.dim,), dtype=object)

        # For stokeslet, there's no direction vector involved
        # passing a list of ones instead to remove its usage.
        for comp in range(self.dim):
            for i in range(self.dim):
                sym_expr[comp] += self.get_int_g((comp, i),
                        density_vec_sym[i], [1]*self.dim,
                        qbx_forced_limit, deriv_dirs=extra_deriv_dirs)

        return sym_expr

    def apply_stress(self, density_vec_sym, dir_vec_sym, qbx_forced_limit):

        sym_expr = np.zeros((self.dim,), dtype=object)
        stresslet_obj = StressletWrapper(dim=self.dim,
                                         mu_sym=self.mu, nu_sym=self.nu)

        # For stokeslet, there's no direction vector involved
        # passing a list of ones instead to remove its usage.
        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += dir_vec_sym[i] * \
                        stresslet_obj.get_int_g((comp, i, j),
                        density_vec_sym[j], [1]*self.dim,
                        qbx_forced_limit, deriv_dirs=[])

        return sym_expr

# }}}


# {{{ StressletWrapper

class StressletWrapper(StressletWrapperBase):
    def __init__(self, dim=None, mu_sym=var("mu"), nu_sym=0.5):
        super().__init__(dim, mu_sym, nu_sym)
        if not (dim == 3 or dim == 2):
            raise ValueError("unsupported dimension given to StokesletWrapper")

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
        self.kernel_dict['laplace'] = LaplaceKernel(self.dim)

    def get_int_g(self, idx, density_sym, dir_vec_sym, qbx_forced_limit,
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

        kernel_indices = [idx, 'laplace', 'laplace', 'laplace']
        dir_vec_indices = [idx[-1], idx[1], idx[0], idx[2]]
        coeffs = [1, (1 - 2*nu)/self.dim, -(1 - 2*nu)/self.dim, -(1 - 2*nu)]
        extra_deriv_dirs_vec = [[], [idx[0]], [idx[1]], [idx[2]]]
        if idx[0] != idx[1]:
            coeffs[-1] = 0

        result = 0
        for kernel_idx, dir_vec_idx, coeff, extra_deriv_dirs in \
                zip(kernel_indices, dir_vec_indices, coeffs,
                        extra_deriv_dirs_vec):
            knl = self.kernel_dict[kernel_idx]
            result += _create_int_g(knl, deriv_dirs + extra_deriv_dirs,
                    density=density_sym*dir_vec_sym[dir_vec_idx],
                    qbx_forced_limit=qbx_forced_limit) * coeff
        return result/(2*(1 - nu))

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=[]):

        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += self.get_int_g((comp, i, j),
                        density_vec_sym[i], dir_vec_sym,
                        qbx_forced_limit, deriv_dirs=extra_deriv_dirs)

        return sym_expr

    def apply_stokeslet_and_stresslet(self, stokeslet_density_vec_sym,
            stresslet_density_vec_sym, dir_vec_sym,
            qbx_forced_limit, stokeslet_weight, stresslet_weight,
            extra_deriv_dirs=[]):

        stokeslet_obj = StokesletWrapper(dim=self.dim,
                                         mu_sym=self.mu, nu_sym=self.nu)

        sym_expr = self.apply(stresslet_density_vec_sym, dir_vec_sym,
                qbx_forced_limit, extra_deriv_dirs) * stresslet_weight
        sym_expr += stokeslet_obj.apply(stokeslet_density_vec_sym,
                qbx_forced_limit, extra_deriv_dirs) * stokeslet_weight

        return sym_expr


    def apply_stress(self, density_vec_sym, normal_vec_sym, dir_vec_sym,
                        qbx_forced_limit):

        sym_expr = np.empty((self.dim,), dtype=object)

        # Build velocity derivative matrix
        sym_grad_matrix = np.empty((self.dim, self.dim), dtype=object)
        for i in range(self.dim):
            sym_grad_matrix[:, i] = self.apply_derivative(i, density_vec_sym,
                                     normal_vec_sym, qbx_forced_limit)

        for comp in range(self.dim):

            # First, add the pressure term:
            sym_expr[comp] = - dir_vec_sym[comp] * self.apply_pressure(
                                            density_vec_sym, normal_vec_sym,
                                            qbx_forced_limit)

            # Now add the velocity derivative components
            for j in range(self.dim):
                sym_expr[comp] = sym_expr[comp] + (
                                    dir_vec_sym[j] * self.mu * (
                                        sym_grad_matrix[comp][j]
                                        + sym_grad_matrix[j][comp])
                                        )

        return sym_expr

# }}}


# {{{ Stokeslet/Stresslet using Laplace

class StressletWrapperTornberg(StressletWrapperBase):
    """A Stresslet wrapper using Tornberg and Greengard's method which
    uses Laplace derivatives.

    [1] Tornberg, A. K., & Greengard, L. (2008). A fast multipole method for the
        three-dimensional Stokes equations.
        Journal of Computational Physics, 227(3), 1613-1619.
    """
    def __init__(self, dim=None, mu_sym=var("mu"), nu_sym=0.5):
        self.dim = dim
        if dim != 3:
            raise ValueError("unsupported dimension given to "
                             "StressletWrapperTornberg")
        if nu_sym != 0.5:
            raise ValueError("nu != 0.5 is not supported")
        self.kernel = LaplaceKernel(dim=self.dim)
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=[]):
        return self.apply_stokeslet_and_stresslet([0]*self.dim,
            density_vec_sym, dir_vec_sym, qbx_forced_limit, 0, 1, extra_deriv_dirs)

    def apply_stokeslet_and_stresslet(self, stokeslet_density_vec_sym,
            stresslet_density_vec_sym, dir_vec_sym,
            qbx_forced_limit, stokeslet_weight, stresslet_weight,
            extra_deriv_dirs):

        sym_expr = np.zeros((self.dim,), dtype=object)

        source = [sym.NodeCoordinateComponent(d) for d in range(self.dim)]
        common_source_kernels = [AxisSourceDerivative(k, self.kernel) for
                k in range(self.dim)]
        common_source_kernels.append(self.kernel)

        stokeslet_weight *= 3.0/6
        stresslet_weight *= -0.5*self.mu**(-1)

        for i in range(self.dim):
            for j in range(self.dim):
                densities = [stresslet_weight*(
                    stresslet_density_vec_sym[k] * dir_vec_sym[j]
                    + stresslet_density_vec_sym[j] * dir_vec_sym[k]) \
                            for k in range(self.dim)]
                densities.append(stokeslet_weight*stokeslet_density_vec_sym[j])
                target_kernel = TargetPointMultiplier(j,
                        AxisTargetDerivative(i, self.kernel))
                for deriv_dir in extra_deriv_dirs:
                    target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)
                sym_expr[i] -= sym.IntG(target_kernel=target_kernel,
                    source_kernels=tuple(common_source_kernels),
                    densities=tuple(densities),
                    qbx_forced_limit=qbx_forced_limit)

                if i == j:
                    sym_expr[i] += sym.IntG(target_kernel=self.kernel,
                        source_kernels=common_source_kernels,
                        densities=densities,
                        qbx_forced_limit=qbx_forced_limit)

            common_density0 = sum(source[k] * stresslet_density_vec_sym[k] for
                    k in range(self.dim))
            common_density1 = sum(source[k] * dir_vec_sym[k] for
                    k in range(self.dim))
            common_density2 = sum(source[k] * stokeslet_density_vec_sym[k] for
                    k in range(self.dim))
            densities = [stresslet_weight*(common_density0 * dir_vec_sym[k]
                    + common_density1 * stresslet_density_vec_sym[k]) for
                    k in range(self.dim)]
            densities.append(common_density2)

            target_kernel = AxisTargetDerivative(i, self.kernel)
            for deriv_dir in extra_deriv_dirs:
                target_kernel = AxisTargetDerivative(deriv_dir, targer_kernel)
            sym_expr[i] += sym.IntG(target_kernel=target_kernel,
                source_kernels=tuple(common_source_kernels),
                densities=tuple(densities),
                qbx_forced_limit=qbx_forced_limit)

        return sym_expr


class StokesletWrapperTornberg(StokesletWrapperBase):
    """A Stresslet wrapper using Tornberg and Greengard's method which
    uses Laplace derivatives.

    [1] Tornberg, A. K., & Greengard, L. (2008). A fast multipole method for the
        three-dimensional Stokes equations.
        Journal of Computational Physics, 227(3), 1613-1619.
    """

    def __init__(self, dim=None, mu_sym=var("mu"), nu_sym=0.5):
        self.dim = dim
        if dim != 3:
            raise ValueError("unsupported dimension given to "
                             "StokesletWrapperTornberg")
        if nu_sym != 0.5:
            raise ValueError("nu != 0.5 is not supported")
        self.kernel = LaplaceKernel(dim=self.dim)
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=[]):
        stresslet = StressletWrapperTornberg(dim=3, self.mu, self.nu)
        return stresslet.apply_stokeslet_and_stresslet(density_vec_sym,
            [0]*self.dim, [0]*self.dim, qbx_forced_limit, 1, 0,
            extra_deriv_dirs)


# }}}


# {{{ base Stokes operator

class StokesOperator:
    """
    .. attribute:: ambient_dim
    .. attribute:: side

    .. automethod:: __init__
    .. automethod:: get_density_var
    .. automethod:: prepare_rhs
    .. automethod:: operator

    .. automethod:: velocity
    .. automethod:: pressure
    """

    def __init__(self, ambient_dim, side, method, mu_sym, nu_sym):
        """
        :arg ambient_dim: dimension of the ambient space.
        :arg side: :math:`+1` for exterior or :math:`-1` for interior.
        """
        from pytential.symbolic.elasticity import (StressletWrapperYoshida,
            StokesletWrapperYoshida)

        if side not in [+1, -1]:
            raise ValueError(f"invalid evaluation side: {side}")

        self.ambient_dim = ambient_dim
        self.side = side
        self.mu = mu_sym
        self.nu = nu_sym

        if method == "laplace":
            if nu_sym == 0.5:
                self.stresslet = StressletWrapperTornberg(dim=self.ambient_dim,
                    mu_sym=mu_sym, nu_sym=nu_sym)
                self.stokeslet = StokesletWrapperTornberg(dim=self.ambient_dim,
                    mu_sym=mu_sym, nu_sym=nu_sym)
            else:
                self.stresslet = StressletWrapperYoshida(dim=self.ambient_dim,
                    mu_sym=mu_sym, nu_sym=nu_sym)
                self.stokeslet = StokesletWrapperYoshida(dim=self.ambient_dim,
                    mu_sym=mu_sym, nu_sym=nu_sym)
        elif method == "biharmonic" or method == "naive":
            self.stresslet = StressletWrapper(dim=self.ambient_dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
            self.stokeslet = StokesletWrapper(dim=self.ambient_dim,
                mu_sym=mu_sym, nu_sym=nu_sym)
        else:
            raise ValueError(f"invalid method: {method}."
                    "Needs to be one of naive, laplace, biharmonic")

        if method == "biharmonic":
            self.base_kernel = BiharmonicKernel(dim=ambient_dim)
        else:
            self.base_kernel = None

    @property
    def dim(self):
        return self.ambient_dim - 1

    def get_density_var(self, name="sigma"):
        """
        :returns: a symbolic vector corresponding to the density.
        """
        return sym.make_sym_vector(name, self.ambient_dim)

    def prepare_rhs(self, b):
        """
        :returns: a (potentially) modified right-hand side *b* that matches
            requirements of the representation.
        """
        return b

    def operator(self, sigma):
        """
        :returns: the integral operator that should be solved to obtain the
            density *sigma*.
        """
        raise NotImplementedError

    def velocity(self, sigma, *, normal, qbx_forced_limit=None):
        """
        :returns: a representation of the velocity field in the Stokes flow.
        """
        raise NotImplementedError

    def pressure(self, sigma, *, normal, qbx_forced_limit=None):
        """
        :returns: a representation of the pressure in the Stokes flow.
        """
        raise NotImplementedError

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

    def __init__(self, *, omega, alpha=None, eta=None, method="biharmonic",
            mu_sym=var("mu"), nu_sym=0.5):
        r"""
        :arg omega: farfield behaviour of the velocity field, as defined
            by :math:`A` in [HsiaoKress1985]_ Equation 2.3.
        :arg alpha: real parameter :math:`\alpha > 0`.
        :arg eta: real parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning.
        """
        super().__init__(ambient_dim=2, side=+1, method=method,
                mu_sym=mu_sym, nu_sym=nu_sym)

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

    def _farfield(self, qbx_forced_limit):
        source_dofdesc = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)
        length = sym.integral(self.ambient_dim, self.dim, 1, dofdesc=source_dofdesc)
        return self.stresslet.apply(
                -self.omega / length, [0]*self.dim, [0]*self.dim,
                qbx_forced_limit, 1, 0)

    def _operator(self, sigma, normal, qbx_forced_limit):
        slp_qbx_forced_limit = qbx_forced_limit
        if slp_qbx_forced_limit == "avg":
            slp_qbx_forced_limit = "avg"

        # NOTE: we set a dofdesc here to force the evaluation of this integral
        # on the source instead of the target when using automatic tagging
        # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
        dd = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)
        int_sigma = sym.integral(self.ambient_dim, self.dim, sigma, dofdesc=dd)

        meanless_sigma = sym.cse(sigma - sym.mean(self.ambient_dim,
            self.dim, sigma, dofdesc=dd))

        result = self.eta * self.alpha / (2.0 * np.pi) * int_sigma
        result += self.stresslet.apply_stokeslet_and_stresslet(meanless_sigma,
                sigma, normal, qbx_forced_limit, -self.eta, 1)

        return result

    def prepare_rhs(self, b):
        return merge_int_g_exprs(b + self._farfield(qbx_forced_limit=+1),
                base_kernel=self.base_kernel)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        # NOTE: H. K. 1985 Equation 2.18
        return merge_int_g_exprs(-0.5 * self.side * sigma - self._operator(
            sigma, normal, qbx_forced_limit), base_kernel=self.base_kernel)

    def velocity(self, sigma, *, normal, qbx_forced_limit=2):
        # NOTE: H. K. 1985 Equation 2.16
        return merge_int_g_exprs(
                -self._farfield(qbx_forced_limit)
                - self._operator(sigma, normal, qbx_forced_limit),
                base_kernel=self.base_kernel)

    def pressure(self, sigma, *, normal, qbx_forced_limit=2):
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

    def __init__(self, *, eta=None, method="laplace", mu_sym=var("mu"), nu_sym=0.5):
        r"""
        :arg eta: a parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning of the operator.
        """

        super().__init__(ambient_dim=3, side=+1, method=method,
                mu_sym=mu_sym, nu_sym=nu_sym)

        # NOTE: eta is chosen here based on H. 1986 Figure 1, which is
        # based on solving on the unit sphere
        if eta is None:
            eta = 0.75

        self.eta = eta
        self.laplace_kernel = LaplaceKernel(3)

    def _operator(self, sigma, normal, qbx_forced_limit):
        return self.stresslet.apply_stokeslet_and_stresslet(sigma,
                sigma, normal, qbx_forced_limit, self.eta, 1)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        # NOTE: H. 1986 Equation 17
        return merge_int_g_exprs(-0.5 * self.side * sigma - self._operator(sigma,
            normal, qbx_forced_limit), base_kernel=self.base_kernel, verbose=True)

    def velocity(self, sigma, *, normal, qbx_forced_limit=2):
        # NOTE: H. 1986 Equation 16
        return merge_int_g_exprs(-self._operator(sigma, normal,
            qbx_forced_limit), base_kernel=self.base_kernel)

    def pressure(self, sigma, *, normal, qbx_forced_limit=2):
        # FIXME: not given in H. 1986, but should be easy to derive using the
        # equivalent single-/double-layer pressure kernels
        raise NotImplementedError

# }}}
