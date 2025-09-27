from __future__ import annotations


__copyright__ = "Copyright (C) 2010-2013 Andreas Kloeckner"

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
.. autoclass:: PotentialMapper
.. autoclass:: VectorExpression

.. autoclass:: L2WeightedPDEOperator

.. autoclass:: DirichletOperator
    :show-inheritance:
.. autoclass:: NeumannOperator
    :show-inheritance:
.. autoclass:: BiharmonicClampedPlateOperator
    :show-inheritance:
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from typing_extensions import override

from pymbolic.typing import ArithmeticExpression
from sumpy.kernel import DirectionalSourceDerivative, Kernel

from pytential import sym


if TYPE_CHECKING:
    from collections.abc import Callable

    from pytools.obj_array import ObjectArray1D

    from pytential.symbolic.dof_desc import DOFDescriptorLike
    from pytential.symbolic.primitives import Operand, QBXForcedLimit, Side


PotentialMapper: TypeAlias = "Callable[[ArithmeticExpression], ArithmeticExpression]"
VectorExpression: TypeAlias = "ObjectArray1D[ArithmeticExpression]"


# {{{ L^2 weighting

class L2WeightedPDEOperator(ABC):
    r""":math:`L^2`-weighting for scalar Integral Equations.

    :math:`L^2`-weighting is performed to help with the solution of integral
    equations that yield locally singular densities. This is done by matching
    the :math:`\ell^2` norm used by the iterative method (e.g. GMRES) with the
    (approximate) :math:`L^2` norm. Clearly, singular densities might not have
    a finite :math:`\ell^2` norm, hampering convergence of the iterative
    method. See [Bremer2012]_ for a detailed description of the construction.

    .. [Bremer2012] J. Bremer,
        *On the Nyström Discretization of Integral Equations on Planar Curves
        With Corners*,
        Applied and Computational Harmonic Analysis, Vol. 32, pp. 45--64, 2012,
        `doi:10.1016/j.acha.2011.03.002 <https://doi.org/10.1016/j.acha.2011.03.002>`__.

    .. autoattribute:: kernel
    .. autoattribute:: use_l2_weighting
    .. autoproperty:: dim

    .. automethod:: get_weight
    .. automethod:: get_sqrt_weight

    .. automethod:: get_density_var
    .. automethod:: prepare_rhs

    .. automethod:: representation
    .. automethod:: operator

    .. automethod:: __init__
    """

    kernel: Kernel
    """The kernel used in the integral operator."""
    use_l2_weighting: bool
    """If *True*, :math:`L^2`-weighting is performed. This can be turned off for
    testing purposes, but should be enabled for other applications.
    """

    def __init__(self, kernel: Kernel, *, use_l2_weighting: bool = True) -> None:
        self.kernel = kernel
        self.use_l2_weighting = use_l2_weighting

        if not use_l2_weighting:
            from warnings import warn
            warn(f"Using L2 weighting in {type(self).__name__} is highly recommended.",
                 stacklevel=3)

    @property
    def dim(self) -> int:
        return self.kernel.dim

    def get_weight(self,
                   dofdesc: DOFDescriptorLike = None) -> ArithmeticExpression:
        """
        :returns: a symbolic expression for the weights (quadrature weights
            and area elements) on *dofdesc* if :attr:`use_l2_weighting` is
            *True* and ``1`` otherwise.
        """
        if self.use_l2_weighting:
            dofdesc = sym.as_dofdesc(dofdesc)
            return sym.cse(
                    sym.area_element(self.dim, dofdesc=dofdesc)
                    * sym.QWeight(dofdesc=dofdesc))
        else:
            return 1

    def get_sqrt_weight(self,
                        dofdesc: DOFDescriptorLike = None) -> ArithmeticExpression:
        """
        :returns: the square root of :meth:`get_weight`.
        """
        if self.use_l2_weighting:
            return sym.sqrt_jac_q_weight(self.dim, dofdesc=dofdesc)
        else:
            return 1

    def prepare_rhs(self, b: ArithmeticExpression) -> ArithmeticExpression:
        """Modify the right-hand side (e.g. boundary conditions) to match the
        operator constructed in :meth:`operator`.
        """
        return self.get_sqrt_weight() * b

    def get_density_var(self, name: str) -> sym.var:
        """
        :param name: a string name for the density.
        :returns: a symbolic variable or array (problem dependent)
            corresponding to the density with the given *name*.
        """
        return sym.var(name)

    @abstractmethod
    def representation(self,
                       u: ArithmeticExpression,
                       *,
                       map_potentials: PotentialMapper | None = None,
                       qbx_forced_limit: QBXForcedLimit = None,
                       source: DOFDescriptorLike = None,
                       target: DOFDescriptorLike = None,
                       **kwargs: Operand) -> ArithmeticExpression:
        """
        :param u: symbolic variable for the density.
        :param map_potentials: a callable that can be used to apply
            additional transformations on all the layer potentials in the
            representation, e.g. to take a target derivative.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.

        :returns: a representation for the unknowns based on an integral
            equation with density *u*. If *qbx_forced_limit* denotes an
            on-surface evaluation, the corresponding jump relations are not
            added to the representation.
        """

    @abstractmethod
    def operator(self,
                 u: ArithmeticExpression,
                 *,
                 dofdesc: DOFDescriptorLike = None,
                 **kwargs: Operand) -> ArithmeticExpression:
        """
        :param u: symbolic variable for the density.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.

        :returns: a boundary integral operator with corresponding jump
            relations that can be used to solve for the density *u*.
        """
        raise NotImplementedError

# }}}


# {{{ dirichlet

class DirichletOperator(L2WeightedPDEOperator):
    """Integral operator and field representation for solving Dirichlet boundary
    value problems with scalar kernels (e.g. :class:`~sumpy.kernel.LaplaceKernel`,
    :class:`~sumpy.kernel.HelmholtzKernel`, :class:`~sumpy.kernel.YukawaKernel`)

    .. note ::

        When testing this as a potential matcher, note that it can only
        access potentials that come from charge distributions having *no* net
        charge. (This is true at least in 2D)

    .. [ColtonKress2012] D. Colton, R. Kress,
        *Inverse Acoustic and Electromagnetic Scattering Theory*,
        Springer Science & Business Media, 2012.
        `doi:10.1007/978-3-030-30351-8 <https://doi.org/10.1007/978-3-030-30351-8>`__.

    .. autoattribute:: loc_sign
    .. autoattribute:: kernel_arguments
    .. autoattribute:: alpha

    .. automethod:: is_unique_only_up_to_constant
    .. automethod:: representation
    .. automethod:: operator
    .. automethod:: __init__
    """

    loc_sign: Side
    """Side of the operator evaluation."""
    kernel_arguments: dict[str, Operand]
    """Additional arguments that will be passed to the layer potential on evaluation."""
    alpha: int | float | complex
    """A complex coefficient with positive imaginary part for the combined-field
    integral representation (CFIE) for the Helmholtz equation (e.g. see
    [ColtonKress2012]_). For other kernels, it does does not offer any benefits.
    """

    def __init__(
            self,
            kernel: Kernel,
            loc_sign: Side, *,
            alpha: int | float | complex | None = None,
            use_l2_weighting: bool = False,
            kernel_arguments: dict[str, Operand] | None = None) -> None:
        assert loc_sign in [-1, 1]
        assert isinstance(kernel, Kernel)

        if kernel_arguments is None:
            kernel_arguments = {}

        if alpha is None:
            # Use a combined-field/Brakhage-Werner representation
            # (alpha != 0) to avoid spurious resonances (mainly for
            # the exterior problem)
            # See:
            # - Brakhage and Werner.
            #    Über das Dirichletsche Außenraumproblem für die
            #    Helmholtzsche Schwingungsgleichung.
            #    https://doi.org/10.1007/BF01220037
            # - Colton and Kress, Chapter 3
            from sumpy.kernel import HelmholtzKernel
            if isinstance(kernel, HelmholtzKernel):
                alpha = 1j
            else:
                alpha = 0

        super().__init__(kernel, use_l2_weighting=use_l2_weighting)

        self.kernel_arguments = kernel_arguments
        self.loc_sign = loc_sign
        self.alpha = alpha

    def is_unique_only_up_to_constant(self) -> bool:
        # No ones matrix needed in Helmholtz case, cf. Hackbusch Lemma 8.5.3.
        from sumpy.kernel import LaplaceKernel
        return isinstance(self.kernel, LaplaceKernel) and self.loc_sign > 0

    @override
    def representation(self,
                       u: ArithmeticExpression,
                       *,
                       map_potentials: PotentialMapper | None = None,
                       qbx_forced_limit: QBXForcedLimit = None,
                       source: DOFDescriptorLike = None,
                       target: DOFDescriptorLike = None,
                       **kwargs: Operand) -> ArithmeticExpression:
        sqrt_w = self.get_sqrt_weight(source)
        inv_sqrt_w_u = sym.cse(u/sqrt_w)

        if map_potentials is None:
            def default_map_potentials(x: ArithmeticExpression) -> ArithmeticExpression:
                return x

            map_potentials = default_map_potentials

        kernel_arguments = self.kernel_arguments
        if kwargs:
            kernel_arguments = {**kernel_arguments, **kwargs}

        S = sym.S(self.kernel, inv_sqrt_w_u,
                  qbx_forced_limit=qbx_forced_limit,
                  kernel_arguments=kernel_arguments,
                  source=source, target=target)
        D = sym.D(self.kernel, inv_sqrt_w_u,
                  qbx_forced_limit=qbx_forced_limit,
                  kernel_arguments=kernel_arguments,
                  source=source, target=target)

        return self.alpha * map_potentials(S) - map_potentials(D)

    @override
    def operator(self,
                 u: ArithmeticExpression, *,
                 dofdesc: DOFDescriptorLike = None,
                 **kwargs: Operand) -> ArithmeticExpression:
        kernel_arguments = self.kernel_arguments
        if kwargs:
            kernel_arguments = {**kernel_arguments, **kwargs}

        dofdesc = sym.as_dofdesc(dofdesc)
        sqrt_w = self.get_sqrt_weight(dofdesc)
        inv_sqrt_w_u = sym.cse(u/sqrt_w)

        if self.is_unique_only_up_to_constant():
            # The exterior Dirichlet operator in this representation
            # has a nullspace. The mean of the density must be matched
            # to the desired solution separately. As is, this operator
            # returns a mean that is not well-specified.
            #
            # See Hackbusch, https://books.google.com/books?id=Ssnf7SZB0ZMC
            # Theorem 8.2.18b

            ones_contribution = (
                    sym.Ones(dofdesc)
                    * sym.mean(self.dim, self.dim - 1, inv_sqrt_w_u, dofdesc=dofdesc))
        else:
            ones_contribution = 0

        def S(density: ArithmeticExpression) -> ArithmeticExpression:
            return sym.S(self.kernel, density,
                    qbx_forced_limit=+1,
                    kernel_arguments=self.kernel_arguments,
                    source=dofdesc, target=dofdesc)

        def D(density: ArithmeticExpression) -> ArithmeticExpression:
            return sym.D(self.kernel, density,
                    qbx_forced_limit="avg",
                    kernel_arguments=self.kernel_arguments,
                    source=dofdesc, target=dofdesc)

        return (
                -0.5 * self.loc_sign * u + sqrt_w * (
                    self.alpha * S(inv_sqrt_w_u)
                    - D(inv_sqrt_w_u)
                    + ones_contribution
                    )
                )

# }}}


# {{{ neumann

class NeumannOperator(L2WeightedPDEOperator):
    """Integral operator and field representation for solving Neumann boundary
    value problems with scalar kernels (e.g. :class:`~sumpy.kernel.LaplaceKernel`,
    :class:`~sumpy.kernel.HelmholtzKernel`, :class:`~sumpy.kernel.YukawaKernel`)

    .. note ::

        When testing this as a potential matcher, note that it can only
        access potentials that come from charge distributions having *no* net
        charge. (This is true at least in 2D)

    .. autoattribute:: loc_sign
    .. autoattribute:: kernel_arguments
    .. autoattribute:: alpha
    .. autoattribute:: use_improved_operator

    .. automethod:: is_unique_only_up_to_constant
    .. automethod:: representation
    .. automethod:: operator
    .. automethod:: __init__
    """

    loc_sign: Side
    """Side of the operator evaluation."""
    kernel_arguments: dict[str, Operand]
    """Additional arguments that will be passed to the layer potential on evaluation."""
    alpha: int | float | complex
    """A complex coefficient with positive imaginary part for the combined-field
    integral representation (CFIE) for the Helmholtz equation (e.g. see
    [ColtonKress2012]_). For other kernels, it does does not offer any benefits.
    """
    use_improved_operator: bool
    """If *True* use the least singular operator available. Only used when
    :attr:`alpha` is not :math:`0`.
    """

    def __init__(self,
            kernel: Kernel,
            loc_sign: Side, *,
            alpha: int | float | complex | None = None,
            use_improved_operator: bool = True,
            use_l2_weighting: bool = False,
            kernel_arguments: dict[str, Any] | None = None):
        assert loc_sign in [-1, 1]
        assert isinstance(kernel, Kernel)

        if kernel_arguments is None:
            kernel_arguments = {}

        if alpha is None:
            # Brakhage-Werner trick for Helmholtz (see DirichletOperator)
            from sumpy.kernel import HelmholtzKernel
            if isinstance(kernel, HelmholtzKernel):
                alpha = 1j
            else:
                alpha = 0

        super().__init__(kernel, use_l2_weighting=use_l2_weighting)

        self.kernel_arguments = kernel_arguments
        self.loc_sign = loc_sign
        self.use_improved_operator = use_improved_operator
        self.alpha = alpha

    def is_unique_only_up_to_constant(self) -> bool:
        from sumpy.kernel import LaplaceKernel
        return isinstance(self.kernel, LaplaceKernel) and self.loc_sign < 0

    @override
    def representation(self,
                       u: ArithmeticExpression,
                       *,
                       map_potentials: PotentialMapper | None = None,
                       qbx_forced_limit: QBXForcedLimit = None,
                       source: DOFDescriptorLike = None,
                       target: DOFDescriptorLike = None,
                       **kwargs: Operand) -> ArithmeticExpression:
        if map_potentials is None:
            def default_map_potentials(x: ArithmeticExpression) -> ArithmeticExpression:
                return x

            map_potentials = default_map_potentials

        kernel_arguments = self.kernel_arguments
        if kwargs:
            kernel_arguments = {**kernel_arguments, **kwargs}

        from sumpy.kernel import LaplaceKernel
        laplace = LaplaceKernel(self.dim)

        sqrt_w = self.get_sqrt_weight(source)
        inv_sqrt_w_u = sym.cse(u/sqrt_w)
        laplace_s_inv_sqrt_w_u = sym.cse(
                sym.S(laplace, inv_sqrt_w_u,
                      qbx_forced_limit=+1,
                      source=source, target=target)
                )

        S = sym.S(self.kernel, inv_sqrt_w_u,
                  qbx_forced_limit=qbx_forced_limit,
                  kernel_arguments=kernel_arguments,
                  source=source, target=target)
        D = sym.D(self.kernel, laplace_s_inv_sqrt_w_u,
                  qbx_forced_limit=qbx_forced_limit,
                  kernel_arguments=kernel_arguments,
                  source=source, target=target)

        return map_potentials(S) - self.alpha * map_potentials(D)

    @override
    def operator(self,
                 u: ArithmeticExpression,
                 *,
                 dofdesc: DOFDescriptorLike = None,
                 **kwargs: Operand) -> ArithmeticExpression:
        kernel_arguments = self.kernel_arguments
        if kwargs:
            kernel_arguments = {**kernel_arguments, **kwargs}

        from sumpy.kernel import HelmholtzKernel, LaplaceKernel
        laplace = LaplaceKernel(self.dim)

        dofdesc = sym.as_dofdesc(dofdesc)
        sqrt_w = self.get_sqrt_weight(dofdesc)
        inv_sqrt_w_u = sym.cse(u/sqrt_w)
        laplace_s_inv_sqrt_w_u = sym.cse(
                sym.S(laplace, inv_sqrt_w_u,
                      qbx_forced_limit=+1,
                      source=dofdesc, target=dofdesc)
                )

        # NOTE: the improved operator here is based on right-precondioning
        # by a single layer and then using some Calderon identities to simplify
        # the result. The integral equation we start with for Neumann is
        #       I + S' + alpha D' = g
        # where D' is hypersingular

        if self.use_improved_operator:
            def Sp(density: ArithmeticExpression) -> ArithmeticExpression:
                return sym.Sp(laplace, density,
                              qbx_forced_limit="avg",
                              source=dofdesc, target=dofdesc)

            # NOTE: using the Calderon identity
            #   D' S = -u/4 + S'^2
            Dp0S0u = -0.25 * u + Sp(Sp(inv_sqrt_w_u))

            if isinstance(self.kernel, HelmholtzKernel):
                Dp0 = sym.Dp(
                    self.kernel, laplace_s_inv_sqrt_w_u,
                    qbx_forced_limit=+1,
                    kernel_arguments=kernel_arguments,
                    source=dofdesc, target=dofdesc)
                Dp1 = sym.Dp(
                    laplace, laplace_s_inv_sqrt_w_u,
                    qbx_forced_limit=+1,
                    kernel_arguments=kernel_arguments,
                    source=dofdesc, target=dofdesc)

                DpS0u = Dp0 - Dp1 + Dp0S0u
            elif isinstance(self.kernel, LaplaceKernel):
                DpS0u = Dp0S0u
            else:
                raise ValueError(f"no improved operator for '{self.kernel}' known")
        else:
            DpS0u = sym.Dp(self.kernel, laplace_s_inv_sqrt_w_u,
                           kernel_arguments=kernel_arguments,
                           qbx_forced_limit=+1,
                           source=dofdesc, target=dofdesc)

        if self.is_unique_only_up_to_constant():
            # The interior Neumann operator in this representation
            # has a nullspace. The mean of the density must be matched
            # to the desired solution separately. As is, this operator
            # returns a mean that is not well-specified.

            ones_contribution = (
                    sym.Ones(dofdesc)
                    * sym.mean(self.dim, self.dim - 1, inv_sqrt_w_u, dofdesc=dofdesc))
        else:
            ones_contribution = 0

        Sp0 = sym.Sp(self.kernel, inv_sqrt_w_u,
                    qbx_forced_limit="avg",
                    kernel_arguments=kernel_arguments,
                    source=dofdesc, target=dofdesc)

        return (
            -0.5 * self.loc_sign * u
            + sqrt_w * (Sp0 - self.alpha * DpS0u + ones_contribution)
        )


class BiharmonicClampedPlateOperator:
    r"""Integral operator and field representation for solving clamped plate biharmonic
    equation.

    .. math::

        \begin{cases}
            \Delta^2 u = 0,   & \quad \text{ on } D, \\
            u = g_1,          & \quad \text{ on } \partial D, \\
            \mathbf{n} \cdot \nabla u = g_2, & \quad \text{ on } \partial D.
        \end{cases}

    This operator assumes that the boundary data :math:`(g_1, g_2)` are
    represented as column vectors and vertically stacked. For details on the
    formulation see Problem C in [Farkas1990]_.

    .. note::

        This operator supports only interior problems (with :attr:`loc_sign`
        set to :math:`-1`).

    .. [Farkas1990] Farkas, Peter, *Mathematical foundations for fast
        algorithms for the biharmonic equation*,
        Technical Report 765, Department of Computer Science,
        Yale University, 1990,
        `PDF <https://cpsc.yale.edu/sites/default/files/files/tr765.pdf>`__.

    .. autoattribute:: kernel
    .. autoattribute:: loc_sign

    .. automethod:: representation
    .. automethod:: operator
    .. automethod:: __init__
    """

    kernel: Kernel
    """The kernel used in the integral operator."""
    loc_sign: Side
    """Side of the operator evaluation."""

    def __init__(self, kernel: Kernel, loc_sign: Side) -> None:
        if loc_sign != -1:
            raise ValueError(
                "only interior problems (loc_sign == -1) are supported")

        if kernel.dim != 2:
            raise ValueError(
                f"unsupported dimension: {kernel.dim} (only 2d problems "
                "are supported)")

        self.kernel = kernel
        self.loc_sign = loc_sign

    @property
    def dim(self) -> int:
        return self.kernel.dim

    def prepare_rhs(self, b: VectorExpression) -> VectorExpression:
        return b

    def get_density_var(self, name: str) -> VectorExpression:
        """
        :returns: a symbolic array corresponding to the density with the given *name*.
        """
        return sym.make_sym_vector(name, 2)

    def representation(self,
                       sigma: VectorExpression,
                       *,
                       map_potentials: PotentialMapper | None = None,
                       qbx_forced_limit: QBXForcedLimit = None,
                       source: DOFDescriptorLike = None,
                       target: DOFDescriptorLike = None,
                       **kwargs: Operand) -> ArithmeticExpression:
        """
        :param sigma: symbolic variable for the density.
        :param map_potentials: a callable that can be used to apply
            additional transformations on all the layer potentials in the
            representation, e.g. to take a target derivative.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.
        """
        assert sigma.shape == (self.dim,)

        if map_potentials is None:
            def default_map_potentials(x: ArithmeticExpression) -> ArithmeticExpression:
                return x

            map_potentials = default_map_potentials

        def dn(knl: Kernel) -> Kernel:
            return DirectionalSourceDerivative(knl, "normal_dir")

        def dt(knl: Kernel) -> Kernel:
            return DirectionalSourceDerivative(knl, "tangent_dir")

        from pytools.obj_array import from_numpy
        normal_dir = from_numpy(sym.normal(self.dim).as_vector(),
                                ArithmeticExpression)
        tangent_dir = from_numpy(np.array([-normal_dir[1], normal_dir[0]]),
                                 ArithmeticExpression)

        Snnn = sym.S(dn(dn(dn(self.kernel))), sigma[0],
                     qbx_forced_limit=qbx_forced_limit,
                     kernel_arguments={**kwargs, "normal_dir": normal_dir},
                     source=source, target=target)
        Sntt = sym.S(dn(dt(dt(self.kernel))), sigma[0],
                     qbx_forced_limit=qbx_forced_limit,
                     kernel_arguments={
                        **kwargs,
                        "normal_dir": normal_dir,
                        "tangent_dir": tangent_dir,
                     },
                     source=source, target=target)

        Snn = sym.S(dn(dn(self.kernel)), sigma[1],
                    qbx_forced_limit=qbx_forced_limit,
                    kernel_arguments={**kwargs, "normal_dir": normal_dir},
                    source=source, target=target)
        Stt = sym.S(dt(dt(self.kernel)), sigma[1],
                    qbx_forced_limit=qbx_forced_limit,
                    kernel_arguments={**kwargs, "tangent_dir": tangent_dir},
                    source=source, target=target)

        return (
                map_potentials(Snnn) + 3 * map_potentials(Sntt)
                - map_potentials(Snn) + map_potentials(Stt)
        )

    def operator(self,
                 sigma: VectorExpression,
                 *,
                 dofdesc: DOFDescriptorLike = None,
                 **kwargs: Operand) -> VectorExpression:
        """
        :param u: symbolic variable for the density.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.

        :returns: the second kind integral operator for the clamped plate
            problem from [Farkas1990]_.
        """
        R = self.representation(
            sigma,
            map_potentials=None,
            qbx_forced_limit="avg",
            source=dofdesc, target=dofdesc,
            **kwargs)
        dR_dn = sym.normal_derivative(self.dim, R, dofdesc=dofdesc)
        kappa = sym.mean_curvature(self.dim, dofdesc=dofdesc)

        int_eq1 = sigma[0] / 2 + R
        int_eq2 = sigma[1] / 2 - kappa * sigma[0] + dR_dn

        from pytools.obj_array import from_numpy
        return from_numpy(np.array([int_eq1, int_eq2]), ArithmeticExpression)

# }}}


# vim: foldmethod=marker
