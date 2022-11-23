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
from sumpy.kernel import (LaplaceKernel, BiharmonicKernel,
    AxisTargetDerivative, AxisSourceDerivative, TargetPointMultiplier)
from pytential.symbolic.elasticity import (ElasticityWrapperBase,
    ElasticityDoubleLayerWrapperBase,
    _ElasticityWrapperNaiveOrBiharmonic,
    _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
    _MU_SYM_DEFAULT)
from abc import abstractmethod

__doc__ = """

.. autoclass:: StokesletWrapperBase
.. autoclass:: StressletWrapperBase
.. automethod:: pytential.symbolic.stokes.StokesletWrapper
.. automethod:: pytential.symbolic.stokes.StressletWrapper

.. autoclass:: StokesOperator
.. autoclass:: HsiaoKressExteriorStokesOperator
.. autoclass:: HebekerExteriorStokesOperator
"""


# {{{ StokesletWrapper/StressletWrapper base classes

class StokesletWrapperBase(ElasticityWrapperBase):
    """Wrapper class for the :class:`~sumpy.kernel.StokesletKernel` kernel.

    In addition to the methods in
    :class:`pytential.symbolic.elasticity.ElasticityWrapperBase`, this class
    also provides :meth:`apply_stress` which applies symmetric viscous stress tensor
    in the requested direction and :meth:`apply_pressure`.

    .. automethod:: apply
    .. automethod:: apply_pressure
    .. automethod:: apply_derivative
    .. automethod:: apply_stress
    """
    def __init__(self, dim, mu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=0.5)

    def apply_pressure(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
        """Symbolic expression for pressure field associated with the Stokeslet."""
        # Pressure representation doesn't differ depending on the implementation
        # and is implemented in base class here.
        lknl = LaplaceKernel(dim=self.dim)

        sym_expr = 0
        for i in range(self.dim):
            deriv_dirs = tuple(extra_deriv_dirs) + (i,)
            knl = lknl
            for deriv_dir in deriv_dirs:
                knl = AxisTargetDerivative(deriv_dir, knl)
            sym_expr += sym.int_g_vec(knl, density_vec_sym[i],
                                      qbx_forced_limit=qbx_forced_limit)
        return sym_expr

    def apply_stress(self, density_vec_sym, dir_vec_sym, qbx_forced_limit):
        r"""Symbolic expression for viscous stress applied to a direction.

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress

        .. math::

            -p \delta_{ij} + \mu (\nabla_i u_j + \nabla_j u_i)

        applied in the direction of *dir_vec_sym*.

        Note that this computation is very similar to computing
        a double-layer potential with the Stresslet kernel in
        :class:`StressletWrapperBase`. The difference is that here the direction
        vector is applied at the target points, while in the Stresslet the
        direction is applied at the source points.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector for the application direction.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        raise NotImplementedError


class StressletWrapperBase(ElasticityDoubleLayerWrapperBase):
    """Wrapper class for the :class:`~sumpy.kernel.StressletKernel` kernel.

    In addition to the methods in
    :class:`pytential.symbolic.elasticity.ElasticityDoubleLayerWrapperBase`, this
    class also provides :meth:`apply_stress` which applies symmetric viscous stress
    tensor in the requested direction and :meth:`apply_pressure`.

    .. automethod:: apply
    .. automethod:: apply_pressure
    .. automethod:: apply_derivative
    .. automethod:: apply_stress
    """
    def __init__(self, dim, mu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=0.5)

    def apply_pressure(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
                       extra_deriv_dirs=()):
        """Symbolic expression for pressure field associated with the Stresslet.
        """
        # Pressure representation doesn't differ depending on the implementation
        # and is implemented in base class here.

        import itertools
        lknl = LaplaceKernel(dim=self.dim)

        factor = (2. * self.mu)

        sym_expr = 0

        for i, j in itertools.product(range(self.dim), range(self.dim)):
            deriv_dirs = tuple(extra_deriv_dirs) + (i, j)
            knl = lknl
            for deriv_dir in deriv_dirs:
                knl = AxisTargetDerivative(deriv_dir, knl)
            sym_expr += factor * sym.int_g_vec(knl,
                                             density_vec_sym[i] * dir_vec_sym[j],
                                             qbx_forced_limit=qbx_forced_limit)

        return sym_expr

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

# }}}


# {{{ Stokeslet/StressletWrapper Naive and Biharmonic

class _StokesletWrapperNaiveOrBiharmonic(_ElasticityWrapperNaiveOrBiharmonic,
                                         StokesletWrapperBase):

    def apply_pressure(self, density_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):
        sym_expr = super().apply_pressure(density_vec_sym, qbx_forced_limit,
                                          extra_deriv_dirs=extra_deriv_dirs)
        res, = rewrite_using_base_kernel([sym_expr], base_kernel=self.base_kernel)
        return res

    def apply_stress(self, density_vec_sym, dir_vec_sym, qbx_forced_limit):

        sym_expr = np.zeros((self.dim,), dtype=object)
        stresslet_obj = _StressletWrapperNaiveOrBiharmonic(dim=self.dim,
            mu_sym=self.mu, nu_sym=0.5, base_kernel=self.base_kernel)

        # For stokeslet, there's no direction vector involved
        # passing a list of ones instead to remove its usage.
        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += dir_vec_sym[i] * \
                        stresslet_obj._get_int_g((comp, i, j),
                        density_vec_sym[j], [1]*self.dim,
                        qbx_forced_limit, deriv_dirs=[])

        return sym_expr


class _StressletWrapperNaiveOrBiharmonic(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        StressletWrapperBase):
    def apply_pressure(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):
        sym_expr = super().apply_pressure(density_vec_sym, dir_vec_sym,
            qbx_forced_limit, extra_deriv_dirs=extra_deriv_dirs)
        res, = rewrite_using_base_kernel([sym_expr], base_kernel=self.base_kernel)
        return res

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


class StokesletWrapperNaive(_StokesletWrapperNaiveOrBiharmonic,
                            ElasticityWrapperBase):
    def __init__(self, dim, mu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=0.5, base_kernel=None)
        ElasticityWrapperBase.__init__(self, dim=dim, mu_sym=mu_sym, nu_sym=0.5)


class StressletWrapperNaive(_StressletWrapperNaiveOrBiharmonic,
                            ElasticityDoubleLayerWrapperBase):

    def __init__(self, dim, mu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=0.5, base_kernel=None)
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim, mu_sym=mu_sym,
                                                  nu_sym=0.5)


class StokesletWrapperBiharmonic(_StokesletWrapperNaiveOrBiharmonic,
                            ElasticityWrapperBase):
    def __init__(self, dim, mu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=0.5,
                         base_kernel=BiharmonicKernel(dim))
        ElasticityWrapperBase.__init__(self, dim=dim, mu_sym=mu_sym, nu_sym=0.5)


class StressletWrapperBiharmonic(_StressletWrapperNaiveOrBiharmonic,
                            ElasticityDoubleLayerWrapperBase):
    def __init__(self, dim, mu_sym):
        super().__init__(dim=dim, mu_sym=mu_sym, nu_sym=0.5,
                         base_kernel=BiharmonicKernel(dim))
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim, mu_sym=mu_sym,
                                                  nu_sym=0.5)

# }}}


# {{{ Stokeslet/Stresslet using Laplace (Tornberg)

class StokesletWrapperTornberg(StokesletWrapperBase):
    """A Stresslet wrapper using Tornberg and Greengard's method which
    uses Laplace derivatives.

    [1] Tornberg, A. K., & Greengard, L. (2008). A fast multipole method for the
        three-dimensional Stokes equations.
        Journal of Computational Physics, 227(3), 1613-1619.
    """

    def __init__(self, dim=None, mu_sym=_MU_SYM_DEFAULT, nu_sym=0.5):
        self.dim = dim
        if nu_sym != 0.5:
            raise ValueError("nu != 0.5 is not supported")
        self.kernel = LaplaceKernel(dim=self.dim)
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, qbx_forced_limit, extra_deriv_dirs=()):
        stresslet = StressletWrapperTornberg(self.dim, self.mu, self.nu)
        return stresslet.apply_single_and_double_layer(density_vec_sym,
            [0]*self.dim, [0]*self.dim, qbx_forced_limit, 1, 0,
            extra_deriv_dirs)


class StressletWrapperTornberg(StressletWrapperBase):
    """A Stresslet wrapper using Tornberg and Greengard's method which
    uses Laplace derivatives.

    [1] Tornberg, A. K., & Greengard, L. (2008). A fast multipole method for the
        three-dimensional Stokes equations.
        Journal of Computational Physics, 227(3), 1613-1619.
    """
    def __init__(self, dim, mu_sym=_MU_SYM_DEFAULT, nu_sym=0.5):
        self.dim = dim
        if nu_sym != 0.5:
            raise ValueError("nu != 0.5 is not supported")
        self.kernel = LaplaceKernel(dim=self.dim)
        self.mu = mu_sym
        self.nu = nu_sym

    def apply(self, density_vec_sym, dir_vec_sym, qbx_forced_limit,
            extra_deriv_dirs=()):
        return self.apply_single_and_double_layer([0]*self.dim,
            density_vec_sym, dir_vec_sym, qbx_forced_limit, 0, 1, extra_deriv_dirs)

    def apply_single_and_double_layer(self, stokeslet_density_vec_sym,
            stresslet_density_vec_sym, dir_vec_sym,
            qbx_forced_limit, stokeslet_weight, stresslet_weight,
            extra_deriv_dirs=()):

        sym_expr = np.zeros((self.dim,), dtype=object)

        source = [sym.NodeCoordinateComponent(d) for d in range(self.dim)]
        common_source_kernels = [AxisSourceDerivative(k, self.kernel) for
                k in range(self.dim)]
        common_source_kernels.append(self.kernel)

        # The paper in [1] ignores the scaling we use Stokeslet/Stresslet
        # and gives formulae for the kernel expression only
        # stokeslet_weight = StokesletKernel.global_scaling_const /
        #    LaplaceKernel.global_scaling_const
        # stresslet_weight = StressletKernel.global_scaling_const /
        #    LaplaceKernel.global_scaling_const
        stresslet_weight *= 3.0
        stokeslet_weight *= -0.5*self.mu**(-1)

        for i in range(self.dim):
            for j in range(self.dim):
                densities = [(stresslet_weight/6.0)*(
                    stresslet_density_vec_sym[k] * dir_vec_sym[j]
                    + stresslet_density_vec_sym[j] * dir_vec_sym[k])
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
                    target_kernel = self.kernel
                    for deriv_dir in extra_deriv_dirs:
                        target_kernel = AxisTargetDerivative(
                                deriv_dir, target_kernel)

                    sym_expr[i] += sym.IntG(target_kernel=target_kernel,
                        source_kernels=common_source_kernels,
                        densities=densities,
                        qbx_forced_limit=qbx_forced_limit)

            common_density0 = sum(source[k] * stresslet_density_vec_sym[k] for
                    k in range(self.dim))
            common_density1 = sum(source[k] * dir_vec_sym[k] for
                    k in range(self.dim))
            common_density2 = sum(source[k] * stokeslet_density_vec_sym[k] for
                    k in range(self.dim))
            densities = [(stresslet_weight/6.0)*(common_density0 * dir_vec_sym[k]
                    + common_density1 * stresslet_density_vec_sym[k]) for
                    k in range(self.dim)]
            densities.append(stokeslet_weight * common_density2)

            target_kernel = AxisTargetDerivative(i, self.kernel)
            for deriv_dir in extra_deriv_dirs:
                target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)
            sym_expr[i] += sym.IntG(target_kernel=target_kernel,
                source_kernels=tuple(common_source_kernels),
                densities=tuple(densities),
                qbx_forced_limit=qbx_forced_limit)

        return sym_expr

# }}}


# {{{ StokesletWrapper dispatch method

def StokesletWrapper(dim, mu_sym=_MU_SYM_DEFAULT, method=None):  # noqa: N806
    if method is None:
        import warnings
        warnings.warn("Method argument not given. Falling back to 'naive'. "
                "Method argument will be required in the future.")
        method = "naive"
    if method == "naive":
        return StokesletWrapperNaive(dim=dim, mu_sym=mu_sym)
    elif method == "biharmonic":
        return StokesletWrapperBiharmonic(dim=dim, mu_sym=mu_sym)
    elif method == "laplace":
        return StokesletWrapperTornberg(dim=dim, mu_sym=mu_sym)
    else:
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")


def StressletWrapper(dim, mu_sym=_MU_SYM_DEFAULT, method=None):  # noqa: N806
    if method is None:
        import warnings
        warnings.warn("Method argument not given. Falling back to 'naive'. "
                "Method argument will be required in the future.")
        method = "naive"
    if method == "naive":
        return StressletWrapperNaive(dim=dim, mu_sym=mu_sym)
    elif method == "biharmonic":
        return StressletWrapperBiharmonic(dim=dim, mu_sym=mu_sym)
    elif method == "laplace":
        return StressletWrapperTornberg(dim=dim, mu_sym=mu_sym)
    else:
        raise ValueError(f"invalid method: {method}."
                "Needs to be one of naive, laplace, biharmonic")

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

    def __init__(self, ambient_dim, side, stokeslet, stresslet, mu_sym):
        """
        :arg ambient_dim: dimension of the ambient space.
        :arg side: :math:`+1` for exterior or :math:`-1` for interior.
        """
        if side not in [+1, -1]:
            raise ValueError(f"invalid evaluation side: {side}")

        self.ambient_dim = ambient_dim
        self.side = side

        if mu_sym is not None:
            import warnings
            warnings.warn("Explicitly giving mu_sym is deprecated. "
                "Use stokeslet and stresslet arguments.")
        else:
            mu_sym = _MU_SYM_DEFAULT

        if stresslet is None:
            stresslet = StressletWrapper(dim=self.ambient_dim,
                mu_sym=mu_sym)

        if stokeslet is None:
            stokeslet = StokesletWrapper(dim=self.ambient_dim,
                mu_sym=mu_sym)

        self.stokeslet = stokeslet
        self.stresslet = stresslet

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

    @abstractmethod
    def operator(self, sigma):
        """
        :returns: the integral operator that should be solved to obtain the
            density *sigma*.
        """
        raise NotImplementedError

    @abstractmethod
    def velocity(self, sigma, *, normal, qbx_forced_limit=None):
        """
        :returns: a representation of the velocity field in the Stokes flow.
        """
        raise NotImplementedError

    @abstractmethod
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

    def __init__(self, *, omega, alpha=1.0, eta=1.0,
                 stokeslet=None, stresslet=None, mu_sym=None):
        r"""
        :arg omega: farfield behaviour of the velocity field, as defined
            by :math:`A` in [HsiaoKress1985]_ Equation 2.3.
        :arg alpha: real parameter :math:`\alpha > 0`.
        :arg eta: real parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning.
        """
        super().__init__(ambient_dim=2, side=+1, stokeslet=stokeslet,
                stresslet=stresslet, mu_sym=mu_sym)

        # NOTE: in [hsiao-kress], there is an analysis on a circle, which
        # recommends values in
        #   1/2 <= alpha <= 2 and max(1/alpha, 1) <= eta <= min(2, 2/alpha)
        # so we choose alpha = eta = 1, which seems to be in line with some
        # of the presented numerical results too.

        self.omega = omega
        self.alpha = alpha
        self.eta = eta

    def _farfield(self, qbx_forced_limit):
        source_dofdesc = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)
        length = sym.integral(self.ambient_dim, self.dim, 1, dofdesc=source_dofdesc)
        result = self.stresslet.apply_single_and_double_layer(
                -self.omega / length, [0]*self.ambient_dim, [0]*self.ambient_dim,
                qbx_forced_limit=qbx_forced_limit, stokeslet_weight=1,
                stresslet_weight=0)
        return result

    def _operator(self, sigma, normal, qbx_forced_limit):
        # NOTE: we set a dofdesc here to force the evaluation of this integral
        # on the source instead of the target when using automatic tagging
        # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
        dd = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)
        int_sigma = sym.integral(self.ambient_dim, self.dim, sigma, dofdesc=dd)

        meanless_sigma = sym.cse(sigma - sym.mean(self.ambient_dim,
            self.dim, sigma, dofdesc=dd))

        result = self.eta * self.alpha / (2.0 * np.pi) * int_sigma
        result += self.stresslet.apply_single_and_double_layer(meanless_sigma,
                sigma, normal, qbx_forced_limit=qbx_forced_limit,
                stokeslet_weight=-self.eta, stresslet_weight=1)

        return result

    def prepare_rhs(self, b):
        return b + self._farfield(qbx_forced_limit=+1)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        # NOTE: H. K. 1985 Equation 2.18
        return -0.5 * self.side * sigma - self._operator(
            sigma, normal, qbx_forced_limit)

    def velocity(self, sigma, *, normal, qbx_forced_limit=2):
        # NOTE: H. K. 1985 Equation 2.16
        return -self._farfield(qbx_forced_limit) \
                - self._operator(sigma, normal, qbx_forced_limit)

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

    def __init__(self, *, eta=None, stokeslet=None, stresslet=None, mu_sym=None):
        r"""
        :arg eta: a parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning of the operator.
        """

        super().__init__(ambient_dim=3, side=+1, stokeslet=stokeslet,
                stresslet=stresslet, mu_sym=mu_sym)

        # NOTE: eta is chosen here based on H. 1986 Figure 1, which is
        # based on solving on the unit sphere
        if eta is None:
            eta = 0.75

        self.eta = eta
        self.laplace_kernel = LaplaceKernel(3)

    def _operator(self, sigma, normal, qbx_forced_limit):
        result = self.stresslet.apply_single_and_double_layer(sigma,
                sigma, normal, qbx_forced_limit=qbx_forced_limit,
                stokeslet_weight=self.eta, stresslet_weight=1,
                extra_deriv_dirs=())
        return result

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        # NOTE: H. 1986 Equation 17
        return -0.5 * self.side * sigma - self._operator(sigma,
            normal, qbx_forced_limit)

    def velocity(self, sigma, *, normal, qbx_forced_limit=2):
        # NOTE: H. 1986 Equation 16
        return -self._operator(sigma, normal, qbx_forced_limit)

    def pressure(self, sigma, *, normal, qbx_forced_limit=2):
        # FIXME: not given in H. 1986, but should be easy to derive using the
        # equivalent single-/double-layer pressure kernels
        raise NotImplementedError

# }}}
