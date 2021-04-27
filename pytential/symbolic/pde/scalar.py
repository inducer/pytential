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
.. autoclass:: L2WeightedPDEOperator
.. autoclass:: DirichletOperator
.. autoclass:: NeumannOperator
.. autoclass:: BiharmonicClampedPlateOperator
"""

from numbers import Number
from typing import Any, Dict, Optional

import numpy as np

from pytential import sym

from sumpy.kernel import Kernel
from sumpy.kernel import DirectionalSourceDerivative


# {{{ L^2 weighting

class L2WeightedPDEOperator:
    """
    .. attribute:: kernel
    .. attribute:: use_l2_weighting

    .. automethod:: get_weight
    .. automethod:: get_sqrt_weight

    .. automethod:: get_density_var
    .. automethod:: prepare_rhs

    .. automethod:: representation
    .. automethod:: operator

    .. automethod:: __init__
    """

    def __init__(self, kernel: Kernel, use_l2_weighting: bool):
        self.kernel = kernel
        self.use_l2_weighting = use_l2_weighting

        if not use_l2_weighting:
            from warnings import warn
            warn("should use L2 weighting in {}".format(type(self).__name__),
                    stacklevel=3)

    @property
    def dim(self):
        return self.kernel.dim

    def get_weight(self, dofdesc=None):
        """
        :returns: a symbolic expression for the weights (quadrature weights
            and area elements) on *dofdesc* if :attr:`use_l2_weighting` is
            *True* and ``1`` otherwise.
        """
        if self.use_l2_weighting:
            return sym.cse(
                    sym.area_element(self.dim, dofdesc=dofdesc)
                    * sym.QWeight(dofdesc=dofdesc))
        else:
            return 1

    def get_sqrt_weight(self, dofdesc=None):
        """
        :returns: the square root of :meth:`get_weight`.
        """
        if self.use_l2_weighting:
            return sym.sqrt_jac_q_weight(self.dim, dofdesc=dofdesc)
        else:
            return 1

    def prepare_rhs(self, b):
        """Modify the right-hand side (e.g. boundary conditions) to match the
        operator constructed in :meth:`operator`.
        """
        return self.get_sqrt_weight() * b

    def get_density_var(self, name: str):
        """
        :param name: a string name for the density.
        :returns: a symbolic variable or array (problem dependent)
            corresponding to the density with the given *name*.
        """
        return sym.var(name)

    def representation(self, u, qbx_forced_limit=None, **kwargs):
        """
        :returns: a representation for the unknowns based on an integral
            equation with density *u*. If *qbx_forced_limit* denotes an
            on-surface evaluation, the corresponding jump relations are not
            added to the representation.
        """
        raise NotImplementedError

    def operator(self, u, **kwargs):
        """
        :returns: a boundary integral operator with corresponding jump
            relations that can be used to solve for the density *u*.
        """
        raise NotImplementedError

# }}}


# {{{ dirichlet

class DirichletOperator(L2WeightedPDEOperator):
    """IE operator and field representation for solving Dirichlet boundary
    value problems with scalar kernels (e.g. :class:`~sumpy.kernel.LaplaceKernel`,
    :class:`~sumpy.kernel.HelmholtzKernel`, :class:`~sumpy.kernel.YukawaKernel`)

    .. note ::

        When testing this as a potential matcher, note that it can only
        access potentials that come from charge distributions having *no* net
        charge. (This is true at least in 2D)

    Inherits from :class:`L2WeightedPDEOperator`.

    .. automethod:: is_unique_only_up_to_constant
    .. automethod:: representation
    .. automethod:: operator
    .. automethod:: __init__
    """

    def __init__(self, kernel: Kernel, loc_sign: int, *,
            alpha: Optional[Number] = None,
            use_l2_weighting: bool = False,
            kernel_arguments: Optional[Dict[str, Any]] = None):
        """
        :param loc_sign: :math:`+1` for exterior or :math:`-1` for interior
            problems.
        :param alpha: the coefficient for the combined-field representation
            Set to 0 for Laplace.
        """
        assert loc_sign in [-1, 1]
        assert isinstance(kernel, Kernel)

        if kernel_arguments is None:
            kernel_arguments = {}

        if alpha is None:
            from sumpy.kernel import LaplaceKernel
            if isinstance(kernel, LaplaceKernel):
                alpha = 0
            else:
                alpha = 1j

        super().__init__(kernel, use_l2_weighting)

        self.kernel_arguments = kernel_arguments
        self.loc_sign = loc_sign
        self.alpha = alpha

    def is_unique_only_up_to_constant(self):
        # No ones matrix needed in Helmholtz case, cf. Hackbusch Lemma 8.5.3.
        from sumpy.kernel import LaplaceKernel
        return isinstance(self.kernel, LaplaceKernel) and self.loc_sign > 0

    def representation(self, u,
            map_potentials=None, qbx_forced_limit=None, **kwargs):
        """
        :param u: symbolic variable for the density.
        :param map_potentials: a callable that can be used to apply
            additional transformations on all the layer potentials in the
            representation, e.g. to take a target derivative.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.
        """
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = sym.cse(u/sqrt_w)

        if map_potentials is None:
            def map_potentials(x):  # pylint:disable=function-redefined
                return x

        kwargs["kernel_arguments"] = self.kernel_arguments
        kwargs["qbx_forced_limit"] = qbx_forced_limit

        return (
                self.alpha * map_potentials(
                    sym.S(self.kernel, inv_sqrt_w_u, **kwargs)
                    )
                - map_potentials(
                    sym.D(self.kernel, inv_sqrt_w_u, **kwargs)
                    )
                )

    def operator(self, u, **kwargs):
        """
        :param u: symbolic variable for the density.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.
        """
        sqrt_w = self.get_sqrt_weight()
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
                    sym.Ones() * sym.mean(self.dim, self.dim - 1, inv_sqrt_w_u))
        else:
            ones_contribution = 0

        def S(density):  # noqa
            return sym.S(self.kernel, density,
                    kernel_arguments=self.kernel_arguments,
                    qbx_forced_limit=+1, **kwargs)

        def D(density):  # noqa
            return sym.D(self.kernel, density,
                    kernel_arguments=self.kernel_arguments,
                    qbx_forced_limit="avg", **kwargs)

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
    """IE operator and field representation for solving Neumann boundary
    value problems with scalar kernels (e.g. :class:`~sumpy.kernel.LaplaceKernel`,
    :class:`~sumpy.kernel.HelmholtzKernel`, :class:`~sumpy.kernel.YukawaKernel`)

    .. note ::

        When testing this as a potential matcher, note that it can only
        access potentials that come from charge distributions having *no* net
        charge. (This is true at least in 2D)

    Inherits from :class:`L2WeightedPDEOperator`.

    .. automethod:: is_unique_only_up_to_constant
    .. automethod:: representation
    .. automethod:: operator
    .. automethod:: __init__
    """

    def __init__(self, kernel: Kernel, loc_sign: int, *,
            alpha: Optional[Number] = None,
            use_improved_operator: bool = True,
            use_l2_weighting: bool = False,
            kernel_arguments: Optional[Dict[str, Any]] = None):
        """
        :param loc_sign: :math:`+1` for exterior or :math:`-1` for interior
            problems.
        :param alpha: the coefficient for the combined-field representation
            Set to 0 for Laplace.
        :param use_improved_operator: if *True* use the least singular
            operator available.
        """

        assert loc_sign in [-1, 1]
        assert isinstance(kernel, Kernel)

        if kernel_arguments is None:
            kernel_arguments = {}

        from sumpy.kernel import LaplaceKernel
        if alpha is None:
            if isinstance(kernel, LaplaceKernel):
                alpha = 0
            else:
                alpha = 1j

        super().__init__(kernel, use_l2_weighting)

        self.kernel_arguments = kernel_arguments
        self.loc_sign = loc_sign
        self.laplace_kernel = LaplaceKernel(kernel.dim)

        self.alpha = alpha
        self.use_improved_operator = use_improved_operator

    def is_unique_only_up_to_constant(self):
        from sumpy.kernel import LaplaceKernel
        return isinstance(self.kernel, LaplaceKernel) and self.loc_sign < 0

    def representation(self, u,
            map_potentials=None, qbx_forced_limit=None, **kwargs):
        """
        :param u: symbolic variable for the density.
        :param map_potentials: a callable that can be used to apply
            additional transformations on all the layer potentials in the
            representation, e.g. to take a target derivative.
        """
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = sym.cse(u/sqrt_w)
        laplace_s_inv_sqrt_w_u = sym.cse(sym.S(self.laplace_kernel, inv_sqrt_w_u))

        if map_potentials is None:
            def map_potentials(x):  # pylint:disable=function-redefined
                return x

        kwargs["qbx_forced_limit"] = qbx_forced_limit
        kwargs["kernel_arguments"] = self.kernel_arguments

        return (
                map_potentials(
                    sym.S(self.kernel, inv_sqrt_w_u, **kwargs)
                    )
                - self.alpha * map_potentials(
                    sym.D(self.kernel, laplace_s_inv_sqrt_w_u, **kwargs)
                    )
                )

    def operator(self, u, **kwargs):
        """
        :param u: symbolic variable for the density.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.
        """
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = sym.cse(u/sqrt_w)
        laplace_s_inv_sqrt_w_u = sym.cse(
                sym.S(self.laplace_kernel, inv_sqrt_w_u, qbx_forced_limit=+1)
                )

        kwargs["kernel_arguments"] = self.kernel_arguments

        if self.use_improved_operator:
            def Sp(density):
                return sym.Sp(self.laplace_kernel,
                        density,
                        qbx_forced_limit="avg")

            Dp0S0u = -0.25 * u + Sp(Sp(inv_sqrt_w_u))

            from sumpy.kernel import HelmholtzKernel, LaplaceKernel
            if isinstance(self.kernel, HelmholtzKernel):
                DpS0u = (  # noqa
                        sym.Dp(
                            self.kernel - self.laplace_kernel,
                            laplace_s_inv_sqrt_w_u,
                            qbx_forced_limit=+1, **kwargs)
                        + Dp0S0u)
            elif isinstance(self.kernel, LaplaceKernel):
                DpS0u = Dp0S0u
            else:
                raise ValueError(f"no improved operator for '{self.kernel}' known")
        else:
            DpS0u = sym.Dp(self.kernel, laplace_s_inv_sqrt_w_u, **kwargs)

        if self.is_unique_only_up_to_constant():
            # The interior Neumann operator in this representation
            # has a nullspace. The mean of the density must be matched
            # to the desired solution separately. As is, this operator
            # returns a mean that is not well-specified.

            ones_contribution = (
                    sym.Ones() * sym.mean(self.dim, self.dim - 1, inv_sqrt_w_u))
        else:
            ones_contribution = 0

        kwargs["qbx_forced_limit"] = "avg"
        return (
                -0.5 * self.loc_sign * u
                + sqrt_w * (
                    sym.Sp(self.kernel, inv_sqrt_w_u, **kwargs)
                    - self.alpha * DpS0u
                    + ones_contribution
                    )
                )


class BiharmonicClampedPlateOperator:
    r"""IE operator and field representation for solving clamped plate Biharmonic
    equation where,

    .. math::
      \begin{cases}
      \Delta^2 u = 0,   & \quad \text{ on } D, \\
      u = g_1,          & \quad \text{ on } \partial D, \\
      \mathbf{n} \cdot \nabla u = g_2, & \quad \text{ on } \partial D.
      \end{cases}

    This operator assumes that the boundary data :math:`(g_1, g_2)` are
    represented as column vectors and vertically stacked. For details on the
    formulation see Problem C in [Farkas1990]_.

    .. note :: This operator supports only interior problem.

    .. [Farkas1990] Farkas, Peter, *Mathematical foundations for fast
        algorithms for the biharmonic equation*,
        Technical Report 765, Department of Computer Science,
        Yale University, 1990,
        `PDF <https://cpsc.yale.edu/sites/default/files/files/tr765.pdf>`__.

    .. automethod:: representation
    .. automethod:: operator
    .. automethod:: __init__
    """

    def __init__(self, kernel: Kernel, loc_sign: int):
        """
        :param loc_sign: :math:`+1` for exterior or :math:`-1` for interior
            problems.
        """

        if loc_sign != -1:
            raise ValueError("only interior problems (loc_sign == -1) are supported")

        if kernel.dim != 2:
            raise ValueError("unsupported dimension: {kernel.ambient_dim}")

        self.kernel = kernel
        self.loc_sign = loc_sign

    @property
    def dim(self):
        return self.kernel.dim

    def prepare_rhs(self, b):
        return b

    def get_density_var(self, name: str):
        """
        :returns: a symbolic array corresponding to the density
            with the given *name*.
        """
        return sym.make_sym_vector(name, 2)

    def representation(self, sigma,
            map_potentials=None, qbx_forced_limit=None, **kwargs):
        """
        :param sigma: symbolic variable for the density.
        :param map_potentials: a callable that can be used to apply
            additional transformations on all the layer potentials in the
            representation, e.g. to take a target derivative.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.
        """

        if map_potentials is None:
            def map_potentials(x):  # pylint:disable=function-redefined
                return x

        def dv(knl):
            return DirectionalSourceDerivative(knl, "normal_dir")

        def dt(knl):
            return DirectionalSourceDerivative(knl, "tangent_dir")

        normal_dir = sym.normal(self.dim).as_vector()
        tangent_dir = np.array([-normal_dir[1], normal_dir[0]])
        kwargs["qbx_forced_limit"] = qbx_forced_limit

        k1 = (
                map_potentials(
                    sym.S(dv(dv(dv(self.kernel))), sigma[0],
                        kernel_arguments={"normal_dir": normal_dir},
                        **kwargs)
                    )
                + 3 * map_potentials(
                    sym.S(dv(dt(dt(self.kernel))), sigma[0],
                        kernel_arguments={
                            "normal_dir": normal_dir, "tangent_dir": tangent_dir
                            },
                        **kwargs)
                    )
                )

        k2 = (
                -map_potentials(
                    sym.S(dv(dv(self.kernel)), sigma[1],
                        kernel_arguments={"normal_dir": normal_dir},
                        **kwargs)
                    )
                + map_potentials(
                    sym.S(dt(dt(self.kernel)), sigma[1],
                        kernel_arguments={"tangent_dir": tangent_dir},
                        **kwargs)
                    )
                )

        return k1 + k2

    def operator(self, sigma, **kwargs):
        """
        :param u: symbolic variable for the density.
        :param kwargs: additional keyword arguments passed on to the layer
            potential constructor.

        :returns: the second kind integral operator for the clamped plate
            problem from [Farkas1990]_.
        """
        rep = self.representation(sigma, qbx_forced_limit="avg", **kwargs)
        drep_dn = sym.normal_derivative(self.dim, rep)

        int_eq1 = sigma[0] / 2 + rep
        int_eq2 = -sym.mean_curvature(self.dim) * sigma[0] + sigma[1] / 2 + drep_dn

        return np.array([int_eq1, int_eq2])

# }}}


# vim: foldmethod=marker
