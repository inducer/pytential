from __future__ import division, absolute_import, print_function

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


from pytential import sym
from pytential.symbolic.primitives import (
        cse,
        sqrt_jac_q_weight, QWeight, area_element)
from sumpy.kernel import DirectionalSourceDerivative
import numpy as np  # noqa


# {{{ L^2 weighting

class L2WeightedPDEOperator(object):
    def __init__(self, kernel, use_l2_weighting):
        self.kernel = kernel
        self.use_l2_weighting = use_l2_weighting

        if not use_l2_weighting:
            from warnings import warn
            warn("should use L2 weighting in %s" % type(self).__name__,
                    stacklevel=3)

    def get_weight(self, dofdesc=None):
        if self.use_l2_weighting:
            return cse(area_element(self.kernel.dim, dofdesc=dofdesc)
                    * QWeight(dofdesc=dofdesc))
        else:
            return 1

    def get_sqrt_weight(self, dofdesc=None):
        if self.use_l2_weighting:
            return sqrt_jac_q_weight(self.kernel.dim, dofdesc=dofdesc)
        else:
            return 1

    def prepare_rhs(self, b):
        return self.get_sqrt_weight()*b

    def get_density_var(self, name):
        """
        Returns a symbolic variable/array corresponding to the density with the
        given name.
        """
        return sym.var(name)

# }}}


# {{{ dirichlet

class DirichletOperator(L2WeightedPDEOperator):
    """IE operator and field representation for solving Dirichlet boundary
    value problems with scalar kernels (e.g.
    :class:`sumpy.kernel.LaplaceKernel`,
    :class:`sumpy.kernel.HelmholtzKernel`, :class:`sumpy.kernel.YukawaKernel`)

    .. note ::

        When testing this as a potential matcher, note that it can only
        access potentials that come from charge distributions having *no* net
        charge. (This is true at least in 2D.)

    .. automethod:: is_unique_only_up_to_constant
    .. automethod:: representation
    .. automethod:: operator
    """

    def __init__(self, kernel, loc_sign, alpha=None, use_l2_weighting=False,
            kernel_arguments=None):
        """
        :arg loc_sign: +1 for exterior, -1 for interior
        :arg alpha: the coefficient for the combined-field representation
            Set to 0 for Laplace.
        """
        assert loc_sign in [-1, 1]

        from sumpy.kernel import Kernel, LaplaceKernel
        assert isinstance(kernel, Kernel)

        if kernel_arguments is None:
            kernel_arguments = {}
        self.kernel_arguments = kernel_arguments

        self.loc_sign = loc_sign

        L2WeightedPDEOperator.__init__(self, kernel, use_l2_weighting)

        if alpha is None:
            if isinstance(self.kernel, LaplaceKernel):
                alpha = 0
            else:
                alpha = 1j

        self.alpha = alpha

    def is_unique_only_up_to_constant(self):
        # No ones matrix needed in Helmholtz case, cf. Hackbusch Lemma 8.5.3.
        from sumpy.kernel import LaplaceKernel
        return isinstance(self.kernel, LaplaceKernel) and self.loc_sign > 0

    def representation(self,
            u, map_potentials=None, qbx_forced_limit=None):
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        if map_potentials is None:
            def map_potentials(x):  # pylint:disable=function-redefined
                return x

        def S(density):  # noqa
            return sym.S(self.kernel, density,
                    kernel_arguments=self.kernel_arguments,
                    qbx_forced_limit=qbx_forced_limit)

        def D(density):  # noqa
            return sym.D(self.kernel, density,
                    kernel_arguments=self.kernel_arguments,
                    qbx_forced_limit=qbx_forced_limit)

        return (
                self.alpha*map_potentials(S(inv_sqrt_w_u))
                - map_potentials(D(inv_sqrt_w_u)))

    def operator(self, u):
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        if self.is_unique_only_up_to_constant():
            # The exterior Dirichlet operator in this representation
            # has a nullspace. The mean of the density must be matched
            # to the desired solution separately. As is, this operator
            # returns a mean that is not well-specified.
            #
            # See Hackbusch, http://books.google.com/books?id=Ssnf7SZB0ZMC
            # Theorem 8.2.18b

            amb_dim = self.kernel.dim
            ones_contribution = (
                    sym.Ones() * sym.mean(amb_dim, amb_dim-1, inv_sqrt_w_u))
        else:
            ones_contribution = 0

        return (-self.loc_sign*0.5*u
                + sqrt_w*(
                    self.alpha*sym.S(self.kernel, inv_sqrt_w_u,
                        qbx_forced_limit=+1, kernel_arguments=self.kernel_arguments)
                    - sym.D(self.kernel, inv_sqrt_w_u,
                        qbx_forced_limit="avg",
                        kernel_arguments=self.kernel_arguments)
                    + ones_contribution))

# }}}


# {{{ neumann

class NeumannOperator(L2WeightedPDEOperator):
    """IE operator and field representation for solving Dirichlet boundary
    value problems with scalar kernels (e.g.
    :class:`sumpy.kernel.LaplaceKernel`,
    :class:`sumpy.kernel.HelmholtzKernel`, :class:`sumpy.kernel.YukawaKernel`)

    .. note ::

        When testing this as a potential matcher, note that it can only
        access potentials that come from charge distributions having *no* net
        charge. (This is true at least in 2D.)

    .. automethod:: is_unique_only_up_to_constant
    .. automethod:: representation
    .. automethod:: operator
    """

    def __init__(self, kernel, loc_sign, alpha=None,
            use_improved_operator=True,
            laplace_kernel=0, use_l2_weighting=False,
            kernel_arguments=None):
        """
        :arg loc_sign: +1 for exterior, -1 for interior
        :arg alpha: the coefficient for the combined-field representation
            Set to 0 for Laplace.
        :arg use_improved_operator: Whether to use the least singular
            operator available
        """

        assert loc_sign in [-1, 1]

        from sumpy.kernel import Kernel, LaplaceKernel
        assert isinstance(kernel, Kernel)

        if kernel_arguments is None:
            kernel_arguments = {}
        self.kernel_arguments = kernel_arguments

        self.loc_sign = loc_sign
        self.laplace_kernel = LaplaceKernel(kernel.dim)

        L2WeightedPDEOperator.__init__(self, kernel, use_l2_weighting)

        if alpha is None:
            if isinstance(self.kernel, LaplaceKernel):
                alpha = 0
            else:
                alpha = 1j

        self.alpha = alpha
        self.use_improved_operator = use_improved_operator

    def is_unique_only_up_to_constant(self):
        from sumpy.kernel import LaplaceKernel
        return isinstance(self.kernel, LaplaceKernel) and self.loc_sign < 0

    def representation(self, u, map_potentials=None, qbx_forced_limit=None,
            **kwargs):
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        if map_potentials is None:
            def map_potentials(x):  # pylint:disable=function-redefined
                return x

        kwargs["qbx_forced_limit"] = qbx_forced_limit
        kwargs["kernel_arguments"] = self.kernel_arguments

        return (
                map_potentials(
                    sym.S(self.kernel, inv_sqrt_w_u, **kwargs))
                - self.alpha
                * map_potentials(
                    sym.D(self.kernel,
                        sym.S(self.laplace_kernel, inv_sqrt_w_u),
                        **kwargs)))

    def operator(self, u):
        from sumpy.kernel import HelmholtzKernel, LaplaceKernel

        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        knl = self.kernel
        lknl = self.laplace_kernel

        knl_kwargs = {}
        knl_kwargs["kernel_arguments"] = self.kernel_arguments

        DpS0u = sym.Dp(knl,  # noqa
                cse(sym.S(lknl, inv_sqrt_w_u)),
                **knl_kwargs)

        if self.use_improved_operator:
            Dp0S0u = -0.25*u + sym.Sp(  # noqa
                    lknl,  # noqa
                    sym.Sp(lknl, inv_sqrt_w_u, qbx_forced_limit="avg"),
                    qbx_forced_limit="avg")

            if isinstance(self.kernel, HelmholtzKernel):
                DpS0u = (  # noqa
                        sym.Dp(knl - lknl,  # noqa
                            cse(sym.S(lknl, inv_sqrt_w_u, qbx_forced_limit=+1)),
                            qbx_forced_limit=+1, **knl_kwargs)
                        + Dp0S0u)
            elif isinstance(knl, LaplaceKernel):
                DpS0u = Dp0S0u  # noqa
            else:
                raise ValueError("no improved operator for %s known"
                        % self.kernel)

        if self.is_unique_only_up_to_constant():
            # The interior Neumann operator in this representation
            # has a nullspace. The mean of the density must be matched
            # to the desired solution separately. As is, this operator
            # returns a mean that is not well-specified.

            amb_dim = self.kernel.dim
            ones_contribution = (
                    sym.Ones() * sym.mean(amb_dim, amb_dim-1, inv_sqrt_w_u))
        else:
            ones_contribution = 0

        return (-self.loc_sign*0.5*u
                + sqrt_w*(
                    sym.Sp(knl, inv_sqrt_w_u, qbx_forced_limit="avg", **knl_kwargs)
                    - self.alpha*DpS0u
                    + ones_contribution
                    ))


class BiharmonicClampedPlateOperator:
    r"""IE operator and field representation for solving clamped plate Biharmonic
    equation where,

    .. math::
      \begin{align*}
      \Delta^2 u &= 0 \text{ on } D \\
      u &= g_1 \text{ on } \delta D \\
      \frac{\partial u}{\partial \nu} &= g_2 \text{ on } \delta D.
      \end{align*}

    This operator assumes that the boundary data :math:`g_1, g_2` are
    represented as column vectors and vertically stacked.

    .. note :: This operator supports only interior problem.

    Ref: Farkas, Peter. Mathematical foundations for fast algorithms for the
    biharmonic equation. Technical Report 765, Department of Computer Science,
    Yale University, 1990.

    .. automethod:: representation
    .. automethod:: operator
    """

    def __init__(self, knl, loc_sign):
        self.knl = knl
        self.loc_sign = loc_sign
        if loc_sign != -1:
            raise ValueError("loc_sign!=-1 is not supported.")

    def prepare_rhs(self, b):
        return b

    def get_density_var(self, name):
        """
        Returns a symbolic variable of length 2 corresponding to the density with
        the given name.
        """
        return sym.make_sym_vector(name, 2)

    def representation(self,
            sigma, map_potentials=None, qbx_forced_limit=None):

        if map_potentials is None:
            def map_potentials(x):  # pylint:disable=function-redefined
                return x

        def dv(knl):
            return DirectionalSourceDerivative(knl, "normal_dir")

        def dt(knl):
            return DirectionalSourceDerivative(knl, "tangent_dir")

        normal_dir = sym.normal(2).as_vector()
        tangent_dir = np.array([-normal_dir[1], normal_dir[0]])

        k1 = map_potentials(sym.S(dv(dv(dv(self.knl))), sigma[0],
                    kernel_arguments={"normal_dir": normal_dir},
                    qbx_forced_limit=qbx_forced_limit)) + \
             3*map_potentials(sym.S(dv(dt(dt(self.knl))), sigma[0],
                    kernel_arguments={"normal_dir": normal_dir,
                                      "tangent_dir": tangent_dir},
                    qbx_forced_limit=qbx_forced_limit))

        k2 = -map_potentials(sym.S(dv(dv(self.knl)), sigma[1],
                    kernel_arguments={"normal_dir": normal_dir},
                    qbx_forced_limit=qbx_forced_limit)) + \
             map_potentials(sym.S(dt(dt(self.knl)), sigma[1],
                    kernel_arguments={"tangent_dir": tangent_dir},
                    qbx_forced_limit=qbx_forced_limit))

        return k1 + k2

    def operator(self, sigma):
        """
        Returns the two second kind integral equations.
        """
        rep = self.representation(sigma, qbx_forced_limit='avg')
        rep_diff = sym.normal_derivative(2, rep)
        int_eq1 = sigma[0]/2 + rep
        int_eq2 = -sym.mean_curvature(2)*sigma[0] + sigma[1]/2 + rep_diff
        return np.array([int_eq1, int_eq2])

# }}}


# vim: foldmethod=marker
