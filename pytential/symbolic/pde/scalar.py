from __future__ import division

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



from pytential.symbolic.primitives import (
        cse,
        S, D, Sp, Dp,
        Ones, Mean,
        SqrtQuadratureWeights, QuadratureWeights)



class ScalarPDEOperator(object):
    def __init__(self, use_l2_weighting):
        self.use_l2_weighting = use_l2_weighting

        if not use_l2_weighting:
            from warnings import warn
            warn("should use L2 weighting in %s" % type(self).__name__,
                    stacklevel=3)

    def get_weight(self):
        if self.use_l2_weighting:
            return QuadratureWeights()
        else:
            return 1

    def get_sqrt_weight(self):
        if self.use_l2_weighting:
            return SqrtQuadratureWeights()
        else:
            return 1

    def prepare_rhs(self, b):
        return self.get_sqrt_weight()*b




class DirichletOperator(ScalarPDEOperator):
    """When testing this as a potential matcher, note that it can only
    access potentials that come from charge distributions having *no* net
    charge. (This is true at least in 2D.)
    """

    def __init__(self, kernel, loc_sign, alpha=None, use_l2_weighting=False):
        """
        :arg loc_sign: +1 for exterior, -1 for interior
        :arg alpha: the coefficient for the combined-field representation
            Set to 0 for Laplace.
        """
        ScalarPDEOperator.__init__(self, use_l2_weighting)

        from sumpy.kernel import normalize_kernel, LaplaceKernel
        self.kernel = normalize_kernel(kernel)
        self.loc_sign = loc_sign

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

    def representation(self, u):
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        return (self.alpha*S(self.kernel, inv_sqrt_w_u)
                - D(self.kernel, inv_sqrt_w_u))

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

            ones_contribution = Ones() * Mean(inv_sqrt_w_u)
        else:
            ones_contribution = 0

        return (-self.loc_sign*0.5*u
                + sqrt_w*(
                    self.alpha*S(self.kernel, inv_sqrt_w_u)
                    - D(self.kernel, inv_sqrt_w_u)
                    + ones_contribution))

class NeumannOperator(ScalarPDEOperator):
    def __init__(self, kernel, loc_sign, alpha=None, use_improved_operator=True,
            laplace_kernel=0, use_l2_weighting=False):
        """
        :arg loc_sign: +1 for exterior, -1 for interior
        :arg alpha: the coefficient for the combined-field representation
            Set to 0 for Laplace.
        :arg use_improved_operator: Whether to use the least singular operator available
        """
        ScalarPDEOperator.__init__(self, use_l2_weighting)

        from sumpy.kernel import normalize_kernel, LaplaceKernel

        self.kernel = normalize_kernel(kernel)
        self.loc_sign = loc_sign
        self.laplace_kernel = normalize_kernel(laplace_kernel)

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

    def representation(self, u):
        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        return (S(self.kernel, inv_sqrt_w_u)
                - self.alpha*D(self.kernel, S(self.laplace_kernel, inv_sqrt_w_u)))

    def operator(self, u):
        """
        :arg alpha: the coefficient for the combined-field representation
        :arg use_improved_operator: Whether to use the least singular operator available
        """
        from sumpy.kernel import HelmholtzKernel, LaplaceKernel

        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        DpS0u = Dp(self.kernel, cse(S(self.laplace_kernel, inv_sqrt_w_u)))

        if self.use_improved_operator:
            Dp0S0u = -0.25*u + Sp(self.laplace_kernel, Sp(self.laplace_kernel, inv_sqrt_w_u))

            if isinstance(self.kernel, HelmholtzKernel):

                DpS0u = (Dp(self.kernel - self.laplace_kernel,
                    cse(S(self.laplace_kernel, inv_sqrt_w_u))) + Dp0S0u)
            elif isinstance(self.kernel, LaplaceKernel):
                DpS0u = Dp0S0u
            else:
                raise ValueError("no improved operator for %s known"
                        % self.kernel)

        if self.is_unique_only_up_to_constant():
            # The interior Neumann operator in this representation
            # has a nullspace. The mean of the density must be matched
            # to the desired solution separately. As is, this operator
            # returns a mean that is not well-specified.

            ones_contribution = Ones() * Mean(inv_sqrt_w_u)
        else:
            ones_contribution = 0

        return (-self.loc_sign*0.5*u
                + sqrt_w*(
                    Sp(self.kernel, inv_sqrt_w_u)
                    - self.alpha*DpS0u
                    + ones_contribution
                    ))

# vim: foldmethod=marker
