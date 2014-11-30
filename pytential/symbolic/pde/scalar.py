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

2D Dielectric
^^^^^^^^^^^^^

.. autoclass:: Dielectric2DBoundaryOperatorBase
.. autoclass:: TEDielectric2DBoundaryOperator
.. autoclass:: TMDielectric2DBoundaryOperator
.. autoclass:: TEMDielectric2DBoundaryOperator
"""


from pytential import sym
from pytential.symbolic.primitives import (
        cse,
        S, D, Sp, Dp,
        Ones, mean,
        sqrt_jac_q_weight, QWeight, area_element)
import numpy as np
from collections import namedtuple
from six.moves import range


# {{{ L^2 weighting

class L2WeightedPDEOperator(object):
    def __init__(self, use_l2_weighting):
        self.use_l2_weighting = use_l2_weighting

        if not use_l2_weighting:
            from warnings import warn
            warn("should use L2 weighting in %s" % type(self).__name__,
                    stacklevel=3)

    def get_weight(self):
        if self.use_l2_weighting:
            return cse(area_element()*QWeight())
        else:
            return 1

    def get_sqrt_weight(self):
        if self.use_l2_weighting:
            return sqrt_jac_q_weight()
        else:
            return 1

    def prepare_rhs(self, b):
        return self.get_sqrt_weight()*b

# }}}


# {{{ dirichlet

class DirichletOperator(L2WeightedPDEOperator):
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
        L2WeightedPDEOperator.__init__(self, use_l2_weighting)

        assert loc_sign in [-1, 1]

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

            ones_contribution = Ones() * mean(inv_sqrt_w_u)
        else:
            ones_contribution = 0

        return (-self.loc_sign*0.5*u
                + sqrt_w*(
                    self.alpha*S(self.kernel, inv_sqrt_w_u)
                    - D(self.kernel, inv_sqrt_w_u)
                    + ones_contribution))

# }}}


# {{{ neumann

class NeumannOperator(L2WeightedPDEOperator):
    def __init__(self, kernel, loc_sign, alpha=None,
            use_improved_operator=True,
            laplace_kernel=0, use_l2_weighting=False):
        """
        :arg loc_sign: +1 for exterior, -1 for interior
        :arg alpha: the coefficient for the combined-field representation
            Set to 0 for Laplace.
        :arg use_improved_operator: Whether to use the least singular
            operator available
        """
        L2WeightedPDEOperator.__init__(self, use_l2_weighting)

        assert loc_sign in [-1, 1]

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
                - self.alpha
                * D(self.kernel, S(self.laplace_kernel, inv_sqrt_w_u)))

    def operator(self, u):
        from sumpy.kernel import HelmholtzKernel, LaplaceKernel

        sqrt_w = self.get_sqrt_weight()
        inv_sqrt_w_u = cse(u/sqrt_w)

        DpS0u = Dp(self.kernel, cse(S(self.laplace_kernel, inv_sqrt_w_u)))

        if self.use_improved_operator:
            Dp0S0u = -0.25*u + Sp(self.laplace_kernel,
                    Sp(self.laplace_kernel, inv_sqrt_w_u))

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

            ones_contribution = Ones() * mean(inv_sqrt_w_u)
        else:
            ones_contribution = 0

        return (-self.loc_sign*0.5*u
                + sqrt_w*(
                    Sp(self.kernel, inv_sqrt_w_u)
                    - self.alpha*DpS0u
                    + ones_contribution
                    ))

# }}}


# {{{ dielectric

class Dielectric2DBoundaryOperatorBase(L2WeightedPDEOperator):
    r"""
    Solves the following system of BVPs on :math:`\mathbb{R}^2`, in which
    a disjoint family of domains :math:`\Omega_i` is embedded:

    .. math::

        \triangle E + (k_i^2-\beta^2) E = 0\quad \text{on $\Omega_i$}\\
        \triangle H + (k_i^2-\beta^2) H = 0\quad \text{on $\Omega_i$}\\
        [H] = 0 \text{ on $\partial \Omega_i$},\quad
        [E] = 0 \text{ on $\partial \Omega_i$}\\
        \left[ \frac{k_0}{k^2-\beta^2} \partial_{\hat n}H\right] = 0
        \quad\text{on $\partial \Omega_i$},\quad\\
        \left[ \frac{k_0}{k^2-\beta^2} \partial_{\hat n}E\right] = 0
        \quad\text{on $\partial \Omega_i$}

    :math:`E` and :math:`H` are assumed to be of the form

    .. math::

        E(x,y,z,t)=E(x,y)e^{i(\beta z-\omega t)
        H(x,y,z,t)=H(x,y)e^{i(\beta z-\omega t)

    where :math:`[\cdot]` denotes the jump across an interface, and :math:`k`
    (without an index) denotes the value of :math:`k` on either side of the
    interface, for the purpose of computing the jump. :math:`\hat n` denotes
    the unit normal of the interface.

    .. automethod:: make_unknown
    .. automethod:: representation_outer
    .. automethod:: representation_inner
    .. automethod:: operator
    """

    pot_kind_S = 0
    pot_kind_D = 1
    pot_kinds = [pot_kind_S, pot_kind_D]
    potential_ops = {
            pot_kind_S: sym.S,
            pot_kind_D: sym.D,
            }

    field_kind_e = 0
    field_kind_e = 0
    field_kind_h = 1
    field_kinds = [field_kind_e, field_kind_h]

    side_in = 0
    side_out = 1
    sides = [side_in, side_out]
    side_to_sign = {
            side_in: -1,
            side_out: 1,
            }

    dir_none = 0
    dir_normal = 1
    dir_tangential = 2

    BCTermDescriptor = namedtuple("BCDescriptor",
            "i_domain direction field_kind coeff_inner coeff_outer".split())

    def __init__(self, domains,
            k_vacuum_name, k_outer_name, k_inner_names, beta_name,
            use_l2_weighting=True):
        """
        :attr domains: a tuple of source geometry identifiers
        :attr k_outer_name: a variable name for the Helmholtz parameter *k*
            to be used on the exterior domain.
        :attr k_inner_names: a tuple of variable names of the Helmholtz
            parameter *k*, to be used inside each part of the source geometry.
            ``len(k_values)`` must equal ``len(domains)``.
        :attr beta: The wave number in the :math:`z` direction.
        """

        super(Dielectric2DBoundaryOperatorBase, self).__init__(
                use_l2_weighting=use_l2_weighting)

        self.domains = domains

        fk_e = self.field_kind_e
        fk_h = self.field_kind_h

        side_in = self.side_in
        side_out = self.side_out

        dir_none = self.dir_none
        dir_normal = self.dir_normal
        dir_tangential = self.dir_tangential

        beta = sym.var(beta_name)
        k_vacuum = sym.var(k_vacuum_name)
        k_outer = sym.var(k_outer_name)
        k_inner = [sym.var(k_inner_name) for k_inner_name in k_inner_names]

        from sumpy.kernel import HelmholtzKernel
        self.kernel_outer = HelmholtzKernel(2, helmholtz_k_name="K0")
        self.kernel_inner = [
                ]
        kernel_K1 = HelmholtzKernel(2, helmholtz_k_name="K1")

        # {{{ build bc list

        # list of tuples, where each tuple consists of BCTermDescriptor instances

        all_bcs = []
        for i_domain in range(len(self.domains)):
            all_bcs += [
                    (  # [E] = 0
                        self.BCTermDescriptor(
                            i_domain=i_domain,
                            direction=dir_none,
                            field_kind=fk_e,
                            coeff_outer=1,
                            coeff_inner=-1),
                        ),
                    (  # [H] = 0
                        self.BCTermDescriptor(
                            i_domain=i_domain,
                            direction=dir_none,
                            field_kind=fk_h,
                            coeff_outer=1,
                            coeff_inner=-1),
                        ),
                    (
                        self.BCTermDescriptor(
                            i_domain=i_domain,
                            direction=dir_tangential,
                            field_kind=fk_e,
                            coeff_outer=beta/(k_outer**2-beta**2),
                            coeff_inner=-beta/(k_inner[i_domain]**2-beta**2)),
                        self.BCTermDescriptor(
                            i_domain=i_domain,
                            direction=dir_normal,
                            field_kind=fk_h,
                            coeff_outer=-k_vacuum/(k_outer**2-beta**2),
                            coeff_inner=k_vacuum/(k_inner[i_domain]**2-beta**2)),
                        ),
                    (
                        self.BCTermDescriptor(
                            i_domain=i_domain,
                            direction=dir_tangential,
                            field_kind=fk_h,
                            coeff_outer=beta/(k_outer**2-beta**2),
                            coeff_inner=-beta/(k_inner[i_domain]**2-beta**2)),
                        self.BCTermDescriptor(
                            i_domain=i_domain,
                            direction=dir_normal,
                            field_kind=fk_e,
                            coeff_outer=(k_outer**2/k_vacuum)/(k_outer**2-beta**2),
                            coeff_inner=(
                                -(k_inner[i_domain]**2/k_vacuum)
                                / (k_inner[i_domain]**2-beta**2)))
                        ),
                    ]

        self.bcs = []
        for bc in all_bcs:
            any_significant_e = any(
                    term.field_kind == fk_e
                    and term.direction in [dir_normal, dir_none]
                    for term in bc)
            any_significant_h = any(
                    term.field_kind == fk_h
                    and term.direction in [dir_normal, dir_none]
                    for term in bc)
            is_necessary = (
                    (self.e_enabled and any_significant_e)
                    or
                    (self.h_enabled and any_significant_h))

            if is_necessary:
                self.bcs.append(bc)

        assert (len(all_bcs)
                * (int(self.e_enabled) + int(self.h_enabled)) // 2
                == len(self.bcs))

        # }}}

        def find_normal_derivative_bc_coeff(field_kind, i_domain, side):
            result = 0
            for bc in self.bcs:
                for term in bc:
                    if term.field_kind != field_kind:
                        continue
                    if term.i_domain != i_domain:
                        continue
                    if term.direction != dir_normal:
                        continue

                    if side == side_in:
                        result += term.coeff_inner
                    elif side == side_out:
                        result += term.coeff_outer
                    else:
                        raise ValueError("invalid side")

            return result

        self.density_coeffs = np.zeros(
                (len(self.pot_kinds), len(self.field_kinds),
                    len(domains), len(self.sides)),
                dtype=np.object)
        for field_kind in self.field_kinds:
            for i_domain in range(len(self.domains)):
                self.density_coeffs[
                        self.pot_kind_S, field_kind, i_domain, side_in] = 1
                self.density_coeffs[
                        self.pot_kind_S, field_kind, i_domain, side_out] = 1

                # These need to satisfy
                #
                # [dens_coeff_D * bc_coeff * dn D]
                # = dens_coeff_D_out * bc_coeff_out * (dn D)
                #   + dens_coeff_D_in * bc_coeff_in * dn D
                # = 0
                #
                # (because dn D is hypersingular, which we'd like to cancel out)

                dens_coeff_D_in = find_normal_derivative_bc_coeff(
                        field_kind, i_domain, side_out)
                dens_coeff_D_out = - find_normal_derivative_bc_coeff(
                        field_kind, i_domain, side_in)

                self.density_coeffs[
                        self.pot_kind_D, field_kind, i_domain, side_in] \
                                = dens_coeff_D_in
                self.density_coeffs[
                        self.pot_kind_D, field_kind, i_domain, side_in] \
                                = dens_coeff_D_out

    def make_unknown(self, name):
        num_densities = (
                2
                * (int(self.e_enabled) + int(self.h_enabled))
                * len(self.domains))

        assert num_densities == len(self.bcs)

        return sym.make_sym_vector(name, num_densities)

    def _structured_unknown(self, unknown):
        """
        :returns: an array of unknowns, with the following index axes:
            ``[pot_kind, field_kind, i_domain]``, where
            ``pot_kind`` is 0 for the single-layer part and 1 for the double-layer
            part,
            ``field_kind`` is 0 for the E-field and 1 for the H-field part,
            ``i_domain`` is the number of the enclosed domain, starting from 0.
        """
        result = np.zeros((2, 2, len(self.domains)), dtype=np.object)

        sqrt_w = self.get_sqrt_weight()

        i_unknown = 0
        for pot_kind in self.pot_kinds:
            for field_kind in self.field_kinds:
                for i_domain in range(len(self.domains)):

                    is_present = (
                            (field_kind == 0 and self.e_enabled)
                            or
                            (field_kind == 1 and self.h_enabled))

                    if is_present:
                        dens = unknown[i_unknown]
                        i_unknown += 1
                    else:
                        dens = 0

                    result[pot_kind, field_kind, i_domain] = \
                            sym.cse(dens/sqrt_w,
                            "dens_{pot}_{field}_{dom}".format(
                                pot={0: "S", 1: "D"}[pot_kind],
                                field={
                                    self.field_kind_e: "E", self.field_kind_h: "H"}
                                [field_kind],
                                dom=i_domain))

        assert i_unknown == len(unknown)
        return result

    def representation_outer(self, unknown):
        """
        :return: an object array of one axis, with
            axes corresponding to :attr:`field_kinds`
            in the exterior domain.
        """
        unk = self._structured_unknown(unknown)

        return np.array([
            sum(
                self.density_coeffs[pot_kind, field_kind, i_domain, self.side_out]
                * self.potential_ops[pot_kind](self.kernel_outer,
                    unk[pot_kind, field_kind, i_domain])
                for pot_kind in self.pot_kinds
                for i_domain in range(len(self.domains)))
            for field_kind in self.field_kinds
            ], dtype=np.object)

    def representation_inner(self, unknown):
        """
        :return: an object array of two axes ``[field_kind, i_domain]``
        """

        unk = self._structured_unknown(unknown)
        return np.array(
            [
                [
                    sum(
                        self.density_coeffs[
                            pot_kind, field_kind, i_domain, self.side_in]
                        * self.potential_ops[pot_kind](self.kernels[i_domain],
                            unk[pot_kind, field_kind, i_domain])
                        for pot_kind in self.pot_kinds
                        )
                    for i_domain in range(len(self.domains))
                ]
                for field_kind in self.field_kinds
            ], dtype=np.object)

    def operator(self, unknown):
        unk = self._structured_unknown(unknown)

        result = []
        for bc in self.bcs:
            op = 0
            hypersingular_op = 0

            for term in bc:
                for pot_kind in self.pot_kinds:
                    potential_op = self.potential_ops[pot_kind]

                    for side in self.sides:
                        density = unk[pot_kind, term.field_kind, term.i_domain]
                        side_sign = self.side_to_sign[side]

                        if side == self.side_in:
                            kernel = self.kernels[term.i_domain]
                            bc_coeff = term.coeff_inner
                        elif side == self.side_in:
                            kernel = self.kernel_outer
                            bc_coeff = term.coeff_outer
                        else:
                            raise ValueError("invalid value of 'side'")

                        potential_op = potential_op(kernel, density)

                        if term.direction == self.dir_none:
                            if potential_op is sym.S:
                                pass
                            elif potential_op is sym.D:
                                potential_op += (side_sign*0.5) * density
                            else:
                                assert False
                        elif term.direction == self.dir_normal:
                            orig_potential_op = potential_op

                            potential_op = sym.normal_derivative(
                                    potential_op,
                                    self.domains[term.i_domain])

                            if orig_potential_op is sym.S:
                                # S'
                                potential_op += (-side_sign*0.5) * density
                            elif orig_potential_op is sym.D:
                                pass
                            else:
                                assert False
                        elif term.direction == self.dir_tangential:
                            # FIXME
                            raise NotImplementedError("tangential derivative")
                        else:
                            raise ValueError("invalid direction")

                        contrib = (
                                bc_coeff
                                * self.density_coeffs[
                                    pot_kind, term.field_kind, term.i_domain,
                                    side]
                                * potential_op)

                        if (pot_kind == self.pot_kind_D
                                and self.direction == self.dir_normal):
                            hypersingular_op += contrib
                        else:
                            op += contrib

            print(hypersingular_op)
            # FIXME: Check that hypersingular_op disappears
            result.append(op)

        return np.array(result, dtype=np.object)


class TEDielectric2DBoundaryOperator(Dielectric2DBoundaryOperatorBase):
    e_enabled = False
    h_enabled = True


class TMDielectric2DBoundaryOperator(Dielectric2DBoundaryOperatorBase):
    e_enabled = True
    h_enabled = False


class TEMDielectric2DBoundaryOperator(Dielectric2DBoundaryOperatorBase):
    e_enabled = True
    h_enabled = True

# }}}

# vim: foldmethod=marker
