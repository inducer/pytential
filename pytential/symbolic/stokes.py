from __future__ import annotations


__copyright__ = """
Copyright (C) 2017 Natalie Beams"
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
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import override

from pytools import MovedFunctionDeprecationWrapper, obj_array
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    BiharmonicKernel,
    Kernel,
    LaplaceKernel,
    TargetPointMultiplier,
)
from sumpy.symbolic import SpatialConstant

from pytential import sym
from pytential.symbolic.elasticity import (
    ElasticityDoubleLayerWrapperBase,
    ElasticityWrapperBase,
    Method,
    VectorExpression,
    _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
    _ElasticityWrapperNaiveOrBiharmonic,
)
from pytential.symbolic.pde.system_utils import rewrite_using_base_kernel


if TYPE_CHECKING:
    from pymbolic.typing import ArithmeticExpression

    from pytential.symbolic.primitives import QBXForcedLimit, Side

__doc__ = """
.. autofunction:: make_stokeslet_wrapper
.. autofunction:: make_stresslet_wrapper

.. autoclass:: StokesletWrapperBase
.. autoclass:: StressletWrapperBase

.. autoclass:: StokesletWrapperNaive
.. autoclass:: StressletWrapperNaive

.. autoclass:: StokesletWrapperBiharmonic
.. autoclass:: StressletWrapperBiharmonic

.. autoclass:: StokesletWrapperTornberg
.. autoclass:: StressletWrapperTornberg

.. autoclass:: StokesOperator
.. autoclass:: HsiaoKressExteriorStokesOperator
.. autoclass:: HebekerExteriorStokesOperator
"""


# {{{ StokesletWrapper/StressletWrapper base classes

class StokesletWrapperBase(ElasticityWrapperBase, ABC):
    """Wrapper class for the :class:`~sumpy.kernel.StokesletKernel` kernel.

    In addition to the methods in
    :class:`~pytential.symbolic.elasticity.ElasticityWrapperBase`, this class
    also provides :meth:`apply_stress`, which applies symmetric viscous stress tensor
    in the requested direction, and :meth:`apply_pressure`.

    .. automethod:: apply
    .. automethod:: apply_pressure
    .. automethod:: apply_derivative
    .. automethod:: apply_stress
    """

    dim: int

    def __init__(self, dim: int, mu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5)

    def apply_pressure(self,
                       density_vec_sym: VectorExpression,
                       qbx_forced_limit: QBXForcedLimit,
                       extra_deriv_dirs: tuple[int, ...] = ()) -> ArithmeticExpression:
        """Symbolic expression for pressure field associated with the Stokeslet."""
        # NOTE: the pressure representation does not differ depending on the
        # representation and is implemented in base class only
        lknl = LaplaceKernel(dim=self.dim)

        sym_expr = 0
        for i in range(self.dim):
            deriv_dirs = (*extra_deriv_dirs, i)
            knl = lknl
            for deriv_dir in deriv_dirs:
                knl = AxisTargetDerivative(deriv_dir, knl)
            sym_expr += sym.int_g_vec(knl, density_vec_sym[i],
                                      qbx_forced_limit=qbx_forced_limit)
        return sym_expr

    @abstractmethod
    def apply_stress(self,
                     density_vec_sym: VectorExpression,
                     dir_vec_sym: VectorExpression,
                     qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
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


class StressletWrapperBase(ElasticityDoubleLayerWrapperBase, ABC):
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

    dim: int

    def __init__(self, dim: int, mu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5)

    def apply_pressure(self,
                       density_vec_sym: VectorExpression,
                       dir_vec_sym: VectorExpression,
                       qbx_forced_limit: QBXForcedLimit,
                       extra_deriv_dirs: tuple[int, ...] = ()) -> ArithmeticExpression:
        """Symbolic expression for pressure field associated with the Stresslet.
        """
        # NOTE: the pressure representation does not differ depending on the
        # representation and is implemented in base class only

        import itertools
        lknl = LaplaceKernel(dim=self.dim)

        factor = (2. * self.mu)

        sym_expr = 0

        for i, j in itertools.product(range(self.dim), range(self.dim)):
            deriv_dirs = (*extra_deriv_dirs, i, j)
            knl = lknl
            for deriv_dir in deriv_dirs:
                knl = AxisTargetDerivative(deriv_dir, knl)
            sym_expr += factor * sym.int_g_vec(knl,
                                             density_vec_sym[i] * dir_vec_sym[j],
                                             qbx_forced_limit=qbx_forced_limit)

        return sym_expr

    @abstractmethod
    def apply_stress(self,
                     density_vec_sym: VectorExpression,
                     normal_vec_sym: VectorExpression,
                     dir_vec_sym: VectorExpression,
                     qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
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
    def __init__(self, dim: int, mu: float | SpatialConstant,
                 base_kernel: Kernel | None) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5, base_kernel=base_kernel)
        StokesletWrapperBase.__init__(self, dim=dim, mu=mu)

    @override
    def apply_pressure(self,
                       density_vec_sym: VectorExpression,
                       qbx_forced_limit: QBXForcedLimit,
                       extra_deriv_dirs: tuple[int, ...] = ()) -> ArithmeticExpression:
        sym_expr = super().apply_pressure(density_vec_sym, qbx_forced_limit,
                                          extra_deriv_dirs=extra_deriv_dirs)
        res, = rewrite_using_base_kernel([sym_expr], base_kernel=self.base_kernel)
        return res

    @override
    def apply_stress(self,
                     density_vec_sym: VectorExpression,
                     dir_vec_sym: VectorExpression,
                     qbx_forced_limit: QBXForcedLimit) -> VectorExpression:

        sym_expr = np.zeros((self.dim,), dtype=object)
        stresslet_obj = _StressletWrapperNaiveOrBiharmonic(
            dim=self.dim,
            mu=self.mu, nu=0.5,
            base_kernel=self.base_kernel)

        # For stokeslet, there's no direction vector involved
        # passing a list of ones instead to remove its usage.
        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += dir_vec_sym[i] * stresslet_obj._get_int_g(
                        (comp, i, j),
                        density_vec_sym[j],
                        obj_array.new_1d([1] * self.dim),
                        qbx_forced_limit,
                        deriv_dirs=())

        return sym_expr


class _StressletWrapperNaiveOrBiharmonic(
        _ElasticityDoubleLayerWrapperNaiveOrBiharmonic,
        StressletWrapperBase):

    def __init__(self, dim: int, mu: float | SpatialConstant,
                 base_kernel: Kernel | None) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5, base_kernel=base_kernel)
        StressletWrapperBase.__init__(self, dim=dim, mu=mu)

    @override
    def apply_pressure(self,
                       density_vec_sym: VectorExpression,
                       dir_vec_sym: VectorExpression,
                       qbx_forced_limit: QBXForcedLimit,
                       extra_deriv_dirs: tuple[int, ...] = ()) -> ArithmeticExpression:
        sym_expr = super().apply_pressure(
            density_vec_sym,
            dir_vec_sym,
            qbx_forced_limit,
            extra_deriv_dirs=extra_deriv_dirs)

        res, = rewrite_using_base_kernel([sym_expr], base_kernel=self.base_kernel)
        return res

    @override
    def apply_stress(self,
                     density_vec_sym: VectorExpression,
                     normal_vec_sym: VectorExpression,
                     dir_vec_sym: VectorExpression,
                     qbx_forced_limit: QBXForcedLimit) -> VectorExpression:

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
    r"""Stokeslet wrapper based on the full Stokeslet kernel.

    This representation uses the Stokeslet kernel denoted by
    :attr:`~pytential.symbolic.elasticity.Method.Naive`.
    """

    def __init__(self, dim: int, mu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, base_kernel=None)
        ElasticityWrapperBase.__init__(self, dim=dim, mu=mu, nu=0.5)


class StressletWrapperNaive(_StressletWrapperNaiveOrBiharmonic,
                            ElasticityDoubleLayerWrapperBase):
    r"""Stresslet wrapper based on the full Stresslet kernel.

    This representation uses the Stresslet kernel denoted by
    :attr:`~pytential.symbolic.elasticity.Method.Naive`.
    """

    def __init__(self, dim: int, mu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, base_kernel=None)
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim, mu=mu, nu=0.5)


class StokesletWrapperBiharmonic(_StokesletWrapperNaiveOrBiharmonic,
                                 ElasticityWrapperBase):
    r"""Stokeslet wrapper based on the biharmonic kernel.

    This representation uses the Stokeslet kernel denoted by
    :attr:`~pytential.symbolic.elasticity.Method.Biharmonic`.
    """

    def __init__(self, dim: int, mu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, base_kernel=BiharmonicKernel(dim))
        ElasticityWrapperBase.__init__(self, dim=dim, mu=mu, nu=0.5)


class StressletWrapperBiharmonic(_StressletWrapperNaiveOrBiharmonic,
                                 ElasticityDoubleLayerWrapperBase):
    r"""Stresslet wrapper based on the biharmonic kernel.

    This representation uses the Stresslet kernel denoted by
    :attr:`~pytential.symbolic.elasticity.Method.Biharmonic`.
    """

    def __init__(self, dim: int, mu: float | SpatialConstant) -> None:
        super().__init__(dim=dim, mu=mu, base_kernel=BiharmonicKernel(dim))
        ElasticityDoubleLayerWrapperBase.__init__(self, dim=dim, mu=mu, nu=0.5)

# }}}


# {{{ Stokeslet/Stresslet using Laplace (Tornberg)

@dataclass
class StokesletWrapperTornberg(StokesletWrapperBase):
    """A Stokeslet wrapper based on [Tornberg2008]_.

    This representation uses the Laplace kernel denoted by
    :attr:`~pytential.symbolic.elasticity.Method.Laplace`.

    .. [Tornberg2008] A.-K. Tornberg, L. Greengard,
        *A Fast Multipole Method for the Three-Dimensional Stokes Equations*,
        Journal of Computational Physics, Vol. 227, pp. 1613--1619, 2008,
        `DOI <https://doi.org/10.1016/j.jcp.2007.06.029>`__.
    """

    dim: int
    mu: float | SpatialConstant
    nu: float | SpatialConstant

    def __post_init__(self):
        if self.nu != 0.5:
            raise ValueError("nu != 0.5 is not supported")

    @override
    def apply(self,
              density_vec_sym: VectorExpression,
              qbx_forced_limit: QBXForcedLimit,
              extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:
        stresslet = StressletWrapperTornberg(self.dim, self.mu, self.nu)

        return stresslet.apply_single_and_double_layer(
            density_vec_sym,
            obj_array.new_1d([0]*self.dim),
            obj_array.new_1d([0]*self.dim),
            qbx_forced_limit=qbx_forced_limit,
            stokeslet_weight=1,
            stresslet_weight=0,
            extra_deriv_dirs=extra_deriv_dirs)

    @override
    def apply_stress(self,
                     density_vec_sym: VectorExpression,
                     dir_vec_sym: VectorExpression,
                     qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
        raise NotImplementedError


@dataclass
class StressletWrapperTornberg(StressletWrapperBase):
    """A Stresslet wrapper based on [Tornberg2008]_.

    This representation uses the Laplace kernel denoted by
    :attr:`~pytential.symbolic.elasticity.Method.Laplace`.
    """

    dim: int
    mu: float | SpatialConstant
    nu: float | SpatialConstant

    def __post_init__(self) -> None:
        if self.nu != 0.5:
            raise ValueError("nu != 0.5 is not supported")

    @cached_property
    def laplace_kernel(self) -> Kernel:
        return LaplaceKernel(dim=self.dim)

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
            qbx_forced_limit=qbx_forced_limit,
            stokeslet_weight=0,
            stresslet_weight=1,
            extra_deriv_dirs=extra_deriv_dirs)

    @override
    def apply_stress(self,
                     density_vec_sym: VectorExpression,
                     normal_vec_sym: VectorExpression,
                     dir_vec_sym: VectorExpression,
                     qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
        raise NotImplementedError

    def _create_int_g(self,
                      target_kernel: Kernel,
                      source_kernels: tuple[Kernel, ...],
                      densities: VectorExpression,
                      qbx_forced_limit: QBXForcedLimit) -> sym.IntG | Literal[0]:
        new_source_kernels: list[Kernel] = []
        new_densities: list[ArithmeticExpression] = []
        for source_kernel, density in zip(source_kernels, densities, strict=True):
            if density != 0.0:
                new_source_kernels.append(source_kernel)
                new_densities.append(density)

        if not new_densities:
            return 0

        return sym.IntG(
            target_kernel=target_kernel,
            source_kernels=tuple(new_source_kernels),
            densities=tuple(new_densities),
            source=sym.DEFAULT_DOFDESC,
            target=sym.DEFAULT_DOFDESC,
            qbx_forced_limit=qbx_forced_limit)

    def apply_single_and_double_layer(
            self,
            stokeslet_density_vec_sym: VectorExpression,
            stresslet_density_vec_sym: VectorExpression,
            dir_vec_sym: VectorExpression,
            qbx_forced_limit: QBXForcedLimit,
            stokeslet_weight: ArithmeticExpression,
            stresslet_weight: ArithmeticExpression,
            extra_deriv_dirs: tuple[int, ...] = ()) -> VectorExpression:
        sym_expr = np.zeros((self.dim,), dtype=object)

        source = sym.nodes(self.dim).as_vector()

        # The paper in [1] ignores the scaling we use Stokeslet/Stresslet
        # and gives formulae for the kernel expression only
        # stokeslet_weight = StokesletKernel.global_scaling_const /
        #    LaplaceKernel.global_scaling_const
        # stresslet_weight = StressletKernel.global_scaling_const /
        #    LaplaceKernel.global_scaling_const
        stresslet_weight *= 3.0
        stokeslet_weight *= -0.5*self.mu**(-1)

        common_source_kernels: tuple[Kernel, ...] = tuple(
                [AxisSourceDerivative(k, self.laplace_kernel) for k in range(self.dim)]
                + [self.laplace_kernel])

        for i in range(self.dim):
            for j in range(self.dim):
                densities = obj_array.new_1d([
                    stresslet_weight / 6.0 * (
                        stresslet_density_vec_sym[k] * dir_vec_sym[j]
                        + stresslet_density_vec_sym[j] * dir_vec_sym[k])
                    for k in range(self.dim)
                ] + [stokeslet_weight * stokeslet_density_vec_sym[j]])

                target_kernel = TargetPointMultiplier(j,
                        AxisTargetDerivative(i, self.laplace_kernel))
                for deriv_dir in extra_deriv_dirs:
                    target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

                sym_expr[i] -= self._create_int_g(
                    target_kernel=target_kernel,
                    source_kernels=tuple(common_source_kernels),
                    densities=densities,
                    qbx_forced_limit=qbx_forced_limit)

                if i == j:
                    target_kernel = self.laplace_kernel
                    for deriv_dir in extra_deriv_dirs:
                        target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

                    sym_expr[i] += self._create_int_g(
                        target_kernel=target_kernel,
                        source_kernels=common_source_kernels,
                        densities=densities,
                        qbx_forced_limit=qbx_forced_limit)

            common_density0 = sum(
                source[k] * stresslet_density_vec_sym[k] for k in range(self.dim))
            common_density1 = sum(
                source[k] * dir_vec_sym[k] for k in range(self.dim))
            common_density2 = sum(
                source[k] * stokeslet_density_vec_sym[k] for k in range(self.dim))

            densities = obj_array.new_1d([
                stresslet_weight / 6.0 * (
                    common_density0 * dir_vec_sym[k]
                    + common_density1 * stresslet_density_vec_sym[k])
                for k in range(self.dim)
            ] + [stokeslet_weight * common_density2])

            target_kernel = AxisTargetDerivative(i, self.laplace_kernel)
            for deriv_dir in extra_deriv_dirs:
                target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

            sym_expr[i] += self._create_int_g(
                target_kernel=target_kernel,
                source_kernels=tuple(common_source_kernels),
                densities=densities,
                qbx_forced_limit=qbx_forced_limit)

        return sym_expr

# }}}


# {{{ StokesletWrapper dispatch method

def make_stokeslet_wrapper(
        dim: int,
        mu: float | str | SpatialConstant = "mu",
        method: Method | None = None
        ) -> StokesletWrapperBase:
    """Creates an appropriate :class:`StokesletWrapperBase` object.

    :param dim: ambient dimension of the representation.
    :param nu: expression or value for the viscosity.
    :param method: method to use - defaults to the
        :attr:`~pytential.symbolic.elasticity.Method.Naive` method.
    """
    if method is None:
        import warnings
        warnings.warn(
            "'method' argument not given. Falling back to 'naive'. "
            "This argument will be required in the future.", stacklevel=2)

        method = Method.Naive

    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if method == Method.Naive:
        return StokesletWrapperNaive(dim=dim, mu=mu)
    elif method == Method.Biharmonic:
        return StokesletWrapperBiharmonic(dim=dim, mu=mu)
    elif method == Method.Laplace:
        return StokesletWrapperTornberg(dim=dim, mu=mu, nu=0.5)
    else:
        raise ValueError(f"invalid 'method': {method}")


def make_stresslet_wrapper(
        dim: int,
        mu: float | str | SpatialConstant = "mu",
        method: Method | None = None
        ) -> StressletWrapperBase:
    """Creates an appropriate :class:`StressletWrapperBase` object.

    :param dim: ambient dimension of the representation.
    :param nu: expression or value for the viscosity.
    :param method: method to use - defaults to the
        :attr:`~pytential.symbolic.elasticity.Method.Naive` method.
    """

    if method is None:
        import warnings
        warnings.warn(
            "'method' argument not given. Falling back to 'naive'. "
            "This argument will be required in the future.", stacklevel=2)

        method = Method.Naive

    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if method == Method.Naive:
        return StressletWrapperNaive(dim=dim, mu=mu)
    elif method == Method.Biharmonic:
        return StressletWrapperBiharmonic(dim=dim, mu=mu)
    elif method == Method.Laplace:
        return StressletWrapperTornberg(dim=dim, mu=mu, nu=0.5)
    else:
        raise ValueError(f"invalid 'method': {method}")


StokesletWrapper = MovedFunctionDeprecationWrapper(make_stokeslet_wrapper)
StressletWrapper = MovedFunctionDeprecationWrapper(make_stresslet_wrapper)

# }}}


# {{{ base Stokes operator

class StokesOperator(ABC):
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

    ambient_dim: int
    side: Side
    stokeslet: StokesletWrapperBase
    stresslet: StressletWrapperBase

    def __init__(self,
                 ambient_dim: int,
                 side: Side,
                 stokeslet: StokesletWrapperBase | None = None,
                 stresslet: StressletWrapperBase | None = None,
                 mu: float | str | SpatialConstant | None = None) -> None:
        """
        :arg ambient_dim: dimension of the ambient space.
        :arg side: :math:`+1` for exterior or :math:`-1` for interior.
        """
        if side not in [+1, -1]:
            raise ValueError(f"invalid evaluation side: {side}")

        if mu is not None:
            import warnings
            warnings.warn(
                "Explicitly giving 'mu' is deprecated. Pass in the 'stokeslet' "
                "and 'stresslet' arguments explicitly instead.",
                DeprecationWarning, stacklevel=2)

            if isinstance(mu, str):
                mu = SpatialConstant(mu)

        # attempt to get mu from the inputs
        if mu is None and stokeslet is not None:
            mu = stokeslet.mu

        if mu is None and stresslet is not None:
            mu = stresslet.mu

        # if not found, use the default
        if mu is None:
            mu = SpatialConstant("mu")

        if stresslet is None:
            stresslet = make_stresslet_wrapper(dim=ambient_dim, mu=mu)

        if stokeslet is None:
            stokeslet = make_stokeslet_wrapper(dim=ambient_dim, mu=mu)

        assert mu == stokeslet.mu
        assert mu == stresslet.mu

        self.ambient_dim = ambient_dim
        self.side = side
        self.stokeslet = stokeslet
        self.stresslet = stresslet

    @property
    def dim(self):
        return self.ambient_dim - 1

    def get_density_var(self, name: str = "sigma") -> VectorExpression:
        """
        :returns: a symbolic vector corresponding to the density.
        """
        return sym.make_sym_vector(name, self.ambient_dim)

    def prepare_rhs(self, b: VectorExpression) -> VectorExpression:
        """
        :returns: a (potentially) modified right-hand side *b* that matches
            requirements of the representation.
        """
        return b

    @abstractmethod
    def operator(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = "avg") -> VectorExpression:
        """
        :returns: the integral operator that should be solved to obtain the
            density *sigma*.
        """
        raise NotImplementedError

    @abstractmethod
    def velocity(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = None,
                 ) -> VectorExpression:
        """
        :returns: a representation of the velocity field in the Stokes flow.
        """
        raise NotImplementedError

    @abstractmethod
    def pressure(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = None,
                 ) -> ArithmeticExpression:
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

    omega: VectorExpression
    alpha: float
    eta: float

    def __init__(self, *,
                 omega: VectorExpression,
                 alpha: float = 1.0,
                 eta: float = 1.0,
                 stokeslet: StokesletWrapperBase | None = None,
                 stresslet: StressletWrapperBase | None = None,
                 mu: float | str | SpatialConstant | None = None) -> None:
        r"""
        :arg omega: farfield behaviour of the velocity field, as defined
            by :math:`A` in [HsiaoKress1985]_ Equation 2.3.
        :arg alpha: real parameter :math:`\alpha > 0`.
        :arg eta: real parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning.
        """
        super().__init__(
            ambient_dim=2, side=+1,
            stokeslet=stokeslet, stresslet=stresslet, mu=mu)

        # NOTE: in [hsiao-kress], there is an analysis on a circle, which
        # recommends values in
        #   1/2 <= alpha <= 2 and max(1/alpha, 1) <= eta <= min(2, 2/alpha)
        # so we choose alpha = eta = 1, which seems to be in line with some
        # of the presented numerical results too.

        self.omega = omega
        self.alpha = alpha
        self.eta = eta

    def _farfield(self, qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
        source_dofdesc = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)

        length = sym.integral(self.ambient_dim, self.dim, 1, dofdesc=source_dofdesc)
        result = self.stresslet.apply_single_and_double_layer(
            -self.omega / length,
            obj_array.new_1d([0]*self.ambient_dim),
            obj_array.new_1d([0]*self.ambient_dim),
            qbx_forced_limit=qbx_forced_limit,
            stokeslet_weight=1,
            stresslet_weight=0)

        return result

    def _operator(self,
                  sigma: VectorExpression,
                  normal: VectorExpression,
                  qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
        # NOTE: we set a dofdesc here to force the evaluation of this integral
        # on the source instead of the target when using automatic tagging
        # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
        dd = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)
        int_sigma = sym.integral(self.ambient_dim, self.dim, sigma, dofdesc=dd)

        meanless_sigma = sym.cse(
            sigma - sym.mean(self.ambient_dim, self.dim, sigma, dofdesc=dd)
        )

        result = self.eta * self.alpha / (2.0 * np.pi) * int_sigma
        result += self.stresslet.apply_single_and_double_layer(
            meanless_sigma,
            sigma,
            normal,
            qbx_forced_limit=qbx_forced_limit,
            stokeslet_weight=-self.eta,
            stresslet_weight=1)

        return result

    @override
    def prepare_rhs(self, b: VectorExpression) -> VectorExpression:
        return b + self._farfield(qbx_forced_limit=+1)

    @override
    def operator(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = "avg") -> VectorExpression:
        # NOTE: H. K. 1985 Equation 2.18
        lp = self._operator(sigma, normal, qbx_forced_limit)
        return -0.5 * self.side * sigma - lp

    @override
    def velocity(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = 2) -> VectorExpression:
        # NOTE: H. K. 1985 Equation 2.16
        lp = self._operator(sigma, normal, qbx_forced_limit)
        return -self._farfield(qbx_forced_limit) - lp

    @override
    def pressure(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = 2) -> ArithmeticExpression:
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
    laplace_kernel: LaplaceKernel

    def __init__(self, *,
                 eta: float | None = None,
                 stokeslet: StokesletWrapperBase | None = None,
                 stresslet: StressletWrapperBase | None = None,
                 mu: float | str | SpatialConstant | None = None) -> None:
        r"""
        :arg eta: a parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning of the operator.
        """

        super().__init__(
            ambient_dim=3, side=+1,
            stokeslet=stokeslet, stresslet=stresslet, mu=mu)

        # NOTE: eta is chosen here based on H. 1986 Figure 1, which is
        # based on solving on the unit sphere
        if eta is None:
            eta = 0.75

        self.eta = eta
        self.laplace_kernel = LaplaceKernel(3)

    def _operator(self,
                  sigma: VectorExpression,
                  normal: VectorExpression,
                  qbx_forced_limit: QBXForcedLimit) -> VectorExpression:
        result = self.stresslet.apply_single_and_double_layer(
            sigma,
            sigma,
            normal,
            qbx_forced_limit=qbx_forced_limit,
            stokeslet_weight=self.eta,
            stresslet_weight=1,
            extra_deriv_dirs=())

        return result

    @override
    def operator(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = "avg") -> VectorExpression:
        # NOTE: H. 1986 Equation 17
        op = self._operator(sigma, normal, qbx_forced_limit)
        return -0.5 * self.side * sigma - op

    @override
    def velocity(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = 2) -> VectorExpression:
        # NOTE: H. 1986 Equation 16
        return -self._operator(sigma, normal, qbx_forced_limit)

    @override
    def pressure(self,
                 sigma: VectorExpression, *,
                 normal: VectorExpression,
                 qbx_forced_limit: QBXForcedLimit | None = 2) -> ArithmeticExpression:
        # FIXME: not given in H. 1986, but should be easy to derive using the
        # equivalent single-/double-layer pressure kernels
        raise NotImplementedError

# }}}
