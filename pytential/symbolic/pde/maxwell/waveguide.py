from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2013-2016 Andreas Kloeckner"

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
Second-Kind Waveguide
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SecondKindInfZMuellerOperator

2D Dielectric (old-style)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DielectricSRep2DBoundaryOperator
.. autoclass:: DielectricSDRep2DBoundaryOperator
"""

import numpy as np
from collections import namedtuple
from six.moves import range

from pytential import sym
from pytools import memoize_method
from pytential.symbolic.pde.scalar import L2WeightedPDEOperator


# {{{ second-kind infinite-z Mueller operator

class SecondKindInfZMuellerOperator(L2WeightedPDEOperator):
    """
    Second-kind IE representation by
    `Lai and Jiang <https://arxiv.org/abs/1512.01117>`_.
    """

    def __init__(self, domain_n_exprs, ne,
            interfaces, use_l2_weighting=None):
        """
        :attr interfaces: a tuple of tuples
            ``(outer_domain, inner_domain, interface_id)``,
            where *outer_domain* and *inner_domain* are indices into
            *domain_k_names*,
            and *interface_id* is a symbolic name for the discretization of the
            interface. 'outer' designates the side of the interface to which
            the normal points.
        :attr domain_n_exprs: a tuple of variable names of the Helmholtz
            parameter *k*, to be used inside each part of the source geometry.
            May also be a tuple of strings, which will be transformed into
            variable references of the corresponding names.
        :attr beta: A symbolic expression for the wave number in the :math:`z`
            direction. May be a string, which will be interpreted as a variable
            name.
        """

        self.interfaces = interfaces

        ne = sym.var(ne)
        self.ne = sym.cse(ne, "ne")

        self.domain_n_exprs = [
                sym.var(n_expr)
                for idom, n_expr in enumerate(domain_n_exprs)]
        del domain_n_exprs

        import pymbolic.primitives as p

        def upper_half_square_root(x):
            return p.If(
                    p.Comparison(
                        (x**0.5).a.imag,
                        "<", 0),
                    1j*(-x)**0.5,
                    x**0.5)

        self.domain_K_exprs = [
                sym.cse(
                    upper_half_square_root(n_expr**2-ne**2),
                    "K%d" % i)
                for i, n_expr in enumerate(self.domain_n_exprs)]

        from sumpy.kernel import HelmholtzKernel
        self.kernel = HelmholtzKernel(2, allow_evanescent=True)

    def make_unknown(self, name):
        return sym.make_sym_vector(name, len(self.unknown_index))

    @property
    @memoize_method
    def unknown_index(self):
        result = {}
        for i in range(len(self.interfaces)):
            for current in ["j", "m"]:
                for cur_dir in ["z", "t"]:
                    result[(current + cur_dir, i)] = len(result)

        return result

    def tangent(self, where):
        return sym.cse(
                (sym.pseudoscalar(2, 1, where)
                / sym.area_element(2, 1, where)),
                "tangent")

    def S(self, dom_idx, density, qbx_forced_limit=+1):  # noqa
        return sym.S(
                self.kernel, density,
                k=self.domain_K_exprs[dom_idx],
                qbx_forced_limit=qbx_forced_limit)

    def D(self, dom_idx, density, qbx_forced_limit="avg"):  # noqa
        return sym.D(
                self.kernel, density,
                k=self.domain_K_exprs[dom_idx],
                qbx_forced_limit=qbx_forced_limit)

    def T(self, where, dom_idx, density):  # noqa
        return sym.int_g_dsource(
                2,
                self.tangent(where),
                self.kernel,
                density,
                k=self.domain_K_exprs[dom_idx],
                # ????
                qbx_forced_limit=1).xproject(0)

    def representation(self, domain_idx, unknown):
        """"Return a tuple of vectors [Ex, Ey, Ez] and [Hx, Hy, Hz]
        representing the solution to Maxwell's equation on domain
        *domain_idx*.
        """
        result = np.zeros(4, dtype=object)

        unk_idx = self.unknown_index
        k_v = self.k_vacuum

        for i in range(len(self.interfaces)):
            dom0i, dom1i, where = self.interfaces[i]

            if domain_idx == dom0i:
                domi = dom0i
            elif domain_idx == dom1i:
                domi = dom1i
            else:
                # Interface does not border the requested domain
                continue

            beta = self.ne*k_v
            k = sym.cse(k_v * self.domain_n_exprs[domi],
                    "k%d" % domi)

            jt = unknown[unk_idx["jz", i]]
            jz = unknown[unk_idx["jt", i]]
            mt = unknown[unk_idx["mz", i]]
            mz = unknown[unk_idx["mt", i]]

            def diff_axis(iaxis, operand):
                v = np.array(2, dtype=np.float64)
                v[iaxis] = 1
                d = sym.Derivative()
                return d.resolve(
                        (sym.MultiVector(v).scalar_product(d.dnabla(2)))
                        * d(operand))

            from functools import partial
            dx = partial(diff_axis, 1)
            dy = partial(diff_axis, 1)

            tangent = self.tangent(where)
            tau1 = tangent[0]
            tau2 = tangent[1]

            S = self.S  # noqa
            D = self.D  # noqa
            T = self.T  # noqa

            # Ex
            result[0] += (
                    -1/(1j*k_v) * dx(T(where, domi, jt))
                    + beta/k_v * dx(S(domi, jz))
                    + k**2/(1j*k_v) * S(domi, jt * tau1)
                    - dy(S(domi, mz))
                    + 1j*beta * S(domi, mt*tau2)
                    )

            # Ey
            result[1] += (
                    - 1/(1j*k_v) * dy(T(where, domi, jt))
                    + beta/k_v * dy(S(domi, jz))
                    + k**2/(1j*k_v) * S(domi, jt * tau2)
                    + dx(S(domi, mz))
                    - 1j*beta*S(domi, mt * tau1)
                    )

            # Ez
            result[2] += (
                    - beta/k_v * T(where, domi, jt)
                    + (k**2 - beta**2)/(1j*k_v) * S(domi, jz)
                    + D(domi, mt)
                    )

            # Hx
            result[3] += (
                    1/(1j*k_v) * dx(T(where, domi, mt))
                    - beta/k_v * dx(S(domi, mz))
                    - k**2/(1j*k_v) * S(domi, mt*tau1)
                    - k**2/k_v**2 * dy(S(domi, jz))
                    + 1j * beta * k**2/k_v**2 * S(domi, jt*tau2)
                    )

            # Hy
            result[4] += (
                    1/(1j*k_v) * dy(T(where, domi, mt))
                    - beta/k_v * dy(S(domi, mz))
                    - k**2/(1j*k_v) * S(domi, mt * tau2)
                    + k**2/k_v**2 * dx(S(domi, jz))
                    - 1j*beta * k**2/k_v**2 * S(domi, jt*tau1)
                    )

            # Hz
            result[5] += (
                    beta/k_v * T(where, domi, mt)
                    - (k**2 - beta**2)/(1j*k_v) * S(domi, mz)
                    + k**2/k_v**2 * D(domi, jt)
                    )

        return result

    def operator(self, unknown):
        result = np.zeros(4*len(self.interfaces), dtype=object)

        unk_idx = self.unknown_index

        for i in range(len(self.interfaces)):
            idx_jt = unk_idx["jz", i]
            idx_jz = unk_idx["jt", i]
            idx_mt = unk_idx["mz", i]
            idx_mz = unk_idx["mt", i]

            phi1 = unknown[idx_jt]
            phi2 = unknown[idx_jz]
            phi3 = unknown[idx_mt]
            phi4 = unknown[idx_mz]

            ne = self.ne

            dom0i, dom1i, where = self.interfaces[i]

            tangent = self.tangent(where)
            normal = sym.cse(
                    sym.normal(2, 1, where),
                    "normal")

            S = self.S  # noqa
            D = self.D  # noqa
            T = self.T  # noqa

            def Tt(where, dom, density):  # noqa
                return sym.tangential_derivative(
                        2, T(where, dom, density)).xproject(0)

            def Sn(dom, density):  # noqa
                return sym.normal_derivative(
                        2,
                        S(dom, density,
                            qbx_forced_limit="avg"))

            def St(dom, density):  # noqa
                return sym.tangential_derivative(2, S(dom, density)).xproject(0)

            n0 = self.domain_n_exprs[dom0i]
            n1 = self.domain_n_exprs[dom1i]

            a11 = sym.cse(n0**2 * D(dom0i, phi1) - n1**2 * D(dom1i, phi1), "a11")
            a22 = sym.cse(-n0**2 * Sn(dom0i, phi2) + n1**2 * Sn(dom1i, phi2), "a22")
            a33 = sym.cse(D(dom0i, phi3)-D(dom1i, phi3), "a33")
            a44 = sym.cse(-Sn(dom0i, phi4) + Sn(dom1i, phi4), "a44")

            a21 = sym.cse(-1j * ne * (
                    n0**2 * tangent.scalar_product(
                        S(dom0i, normal * phi1))
                    - n1**2 * tangent.scalar_product(
                        S(dom1i, normal * phi1))), "a21")

            a43 = sym.cse(-1j * ne * (
                    tangent.scalar_product(
                        S(dom0i, normal * phi3))
                    - tangent.scalar_product(
                        S(dom1i, normal * phi3))), "a43")

            a13 = +1*sym.cse(
                    ne*(T(where, dom0i, phi3) - T(where, dom1i, phi3)), "a13")
            a31 = -1*sym.cse(
                    ne*(T(where, dom0i, phi1) - T(where, dom1i, phi1)), "a31")

            a24 = +1*sym.cse(ne*(St(dom0i, phi4) - St(dom1i, phi4)), "a24")
            a42 = -1*sym.cse(ne*(St(dom0i, phi2) - St(dom1i, phi2)), "a42")

            a14 = sym.cse(1j*(
                    (n0**2 - ne**2) * S(dom0i, phi4)
                    - (n1**2 - ne**2) * S(dom1i, phi4)
                    ), "a14")
            a32 = -sym.cse(1j*(
                    (n0**2 - ne**2) * S(dom0i, phi2)
                    - (n1**2 - ne**2) * S(dom1i, phi2)
                    ), "a32")

            def a23_expr(phi):
                return (
                        1j * (Tt(where, dom0i, phi) - Tt(where, dom1i, phi))
                        - 1j * (
                            n0**2 * tangent.scalar_product(
                                S(dom0i, tangent * phi))
                            - n1**2 * tangent.scalar_product(
                                S(dom1i, tangent * phi))))

            a23 = +1*sym.cse(a23_expr(phi3), "a23")
            a41 = -1*sym.cse(a23_expr(phi1), "a41")

            d1 = (n0**2 + n1**2)/2 * phi1
            d2 = (n0**2 + n1**2)/2 * phi2
            d3 = phi3
            d4 = phi4

            result[idx_jt] += d1 + a11 + 000 + a13 + a14
            result[idx_jz] += d2 + a21 + a22 + a23 + a24
            result[idx_mt] += d3 + a31 + a32 + a33 + 0
            result[idx_mz] += d4 + a41 + a42 + a43 + a44

            # TODO: L2 weighting
            # TODO: Add representation contributions to other boundaries
            # abutting the domain
            return result

# }}}


# {{{ old-style waveguide

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
            "i_interface direction field_kind coeff_inner coeff_outer".split())

    # {{{ constructor

    def __init__(self, mode, k_vacuum, domain_k_exprs, beta,
            interfaces, use_l2_weighting=None):
        """
        :attr mode: one of 'te', 'tm', 'tem'
        :attr k_vacuum: A symbolic expression for the wave number in vacuum.
            May be a string, which will be interpreted as a variable name.
        :attr interfaces: a tuple of tuples
            ``(outer_domain, inner_domain, interface_id)``,
            where *outer_domain* and *inner_domain* are indices into
            *domain_k_names*,
            and *interface_id* is a symbolic name for the discretization of the
            interface. 'outer' designates the side of the interface to which
            the normal points.
        :attr domain_k_exprs: a tuple of variable names of the Helmholtz
            parameter *k*, to be used inside each part of the source geometry.
            May also be a tuple of strings, which will be transformed into
            variable references of the corresponding names.
        :attr beta: A symbolic expression for the wave number in the :math:`z`
            direction. May be a string, which will be interpreted as a variable
            name.
        """

        if use_l2_weighting is None:
            use_l2_weighting = False

        super(Dielectric2DBoundaryOperatorBase, self).__init__(
                use_l2_weighting=use_l2_weighting)

        if mode == "te":
            self.ez_enabled = False
            self.hz_enabled = True
        elif mode == "tm":
            self.ez_enabled = True
            self.hz_enabled = False
        elif mode == "tem":
            self.ez_enabled = True
            self.hz_enabled = True
        else:
            raise ValueError("invalid mode '%s'" % mode)

        self.interfaces = interfaces

        fk_e = self.field_kind_e
        fk_h = self.field_kind_h

        dir_none = self.dir_none
        dir_normal = self.dir_normal
        dir_tangential = self.dir_tangential

        if isinstance(beta, str):
            beta = sym.var(beta)
        beta = sym.cse(beta, "beta")

        if isinstance(k_vacuum, str):
            k_vacuum = sym.var(k_vacuum)
        k_vacuum = sym.cse(k_vacuum, "k_vac")

        self.domain_k_exprs = [
                sym.var(k_expr)
                if isinstance(k_expr, str)
                else sym.cse(k_expr, "k%d" % idom)
                for idom, k_expr in enumerate(domain_k_exprs)]
        del domain_k_exprs

        # Note the case of k/K!
        # "K" is the 2D Helmholtz parameter.
        # "k" is the 3D Helmholtz parameter.

        self.domain_K_exprs = [
                sym.cse((k_expr**2-beta**2)**0.5, "K%d" % i)
                for i, k_expr in enumerate(self.domain_k_exprs)]

        from sumpy.kernel import HelmholtzKernel
        self.kernel = HelmholtzKernel(2, allow_evanescent=True)

        # {{{ build bc list

        # list of tuples, where each tuple consists of BCTermDescriptor instances

        all_bcs = []
        for i_interface, (outer_domain, inner_domain, _) in (
                enumerate(self.interfaces)):
            k_outer = self.domain_k_exprs[outer_domain]
            k_inner = self.domain_k_exprs[inner_domain]

            all_bcs += [
                    (  # [E] = 0
                        self.BCTermDescriptor(
                            i_interface=i_interface,
                            direction=dir_none,
                            field_kind=fk_e,
                            coeff_outer=1,
                            coeff_inner=-1),
                        ),
                    (  # [H] = 0
                        self.BCTermDescriptor(
                            i_interface=i_interface,
                            direction=dir_none,
                            field_kind=fk_h,
                            coeff_outer=1,
                            coeff_inner=-1),
                        ),
                    (
                        self.BCTermDescriptor(
                            i_interface=i_interface,
                            direction=dir_tangential,
                            field_kind=fk_e,
                            coeff_outer=beta/(k_outer**2-beta**2),
                            coeff_inner=-beta/(k_inner**2-beta**2)),
                        self.BCTermDescriptor(
                            i_interface=i_interface,
                            direction=dir_normal,
                            field_kind=fk_h,
                            coeff_outer=sym.cse(-k_vacuum/(k_outer**2-beta**2)),
                            coeff_inner=sym.cse(k_vacuum/(k_inner**2-beta**2))),
                        ),
                    (
                        self.BCTermDescriptor(
                            i_interface=i_interface,
                            direction=dir_tangential,
                            field_kind=fk_h,
                            coeff_outer=beta/(k_outer**2-beta**2),
                            coeff_inner=-beta/(k_inner**2-beta**2)),
                        self.BCTermDescriptor(
                            i_interface=i_interface,
                            direction=dir_normal,
                            field_kind=fk_e,
                            coeff_outer=sym.cse(
                                (k_outer**2/k_vacuum)/(k_outer**2-beta**2)),
                            coeff_inner=sym.cse(
                                -(k_inner**2/k_vacuum)
                                / (k_inner**2-beta**2)))
                        ),
                    ]

            del k_outer
            del k_inner

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
                    (self.ez_enabled and any_significant_e)
                    or (self.hz_enabled and any_significant_h))

            # Only keep tangential modes for TEM. Otherwise,
            # no jump in H already implies jump condition on
            # tangential derivative.
            is_tem = self.ez_enabled and self.hz_enabled
            terms = tuple(
                    term
                    for term in bc
                    if term.direction != dir_tangential
                    or is_tem)

            if is_necessary:
                self.bcs.append(terms)

        assert (len(all_bcs)
                * (int(self.ez_enabled) + int(self.hz_enabled)) // 2
                == len(self.bcs))

        # }}}

    # }}}

    def is_field_present(self, field_kind):
        return (
                (field_kind == self.field_kind_e and self.ez_enabled)
                or (field_kind == self.field_kind_h and self.hz_enabled))

    def make_unknown(self, name):
        num_densities = (
                2
                * (int(self.ez_enabled) + int(self.hz_enabled))
                * len(self.interfaces))

        assert num_densities == len(self.bcs)

        return sym.make_sym_vector(name, num_densities)

    def bc_term_to_operator_contrib(self, term, side, raw_potential_op,
            density, discrete):
        potential_op = raw_potential_op

        side_sign = self.side_to_sign[side]

        domain_outer, domain_inner, interface_id = \
                self.interfaces[term.i_interface]
        if side == self.side_in:
            K_expr = self.domain_K_exprs[domain_inner]  # noqa
            bc_coeff = term.coeff_inner
        elif side == self.side_out:
            K_expr = self.domain_K_exprs[domain_outer]  # noqa
            bc_coeff = term.coeff_outer
        else:
            raise ValueError("invalid value of 'side'")

        potential_op = potential_op(
                self.kernel, density, source=interface_id,
                k=K_expr)

        if term.direction == self.dir_none:
            if raw_potential_op is sym.S:
                jump_term = 0
            elif raw_potential_op is sym.D:
                jump_term = (side_sign*0.5) * discrete
            else:
                assert False, raw_potential_op
        elif term.direction == self.dir_normal:
            potential_op = sym.normal_derivative(
                    potential_op, interface_id)

            if raw_potential_op is sym.S:
                # S'
                jump_term = (-side_sign*0.5) * discrete
            elif raw_potential_op is sym.D:
                jump_term = 0
            else:
                assert False, raw_potential_op

        elif term.direction == self.dir_tangential:
            potential_op = sym.tangential_derivative(
                    raw_potential_op(
                        self.kernel, density, source=interface_id,
                        k=K_expr, qbx_forced_limit=side_sign),
                    interface_id).a.as_scalar()

            # Some of these may have jumps, but QBX does the dirty
            # work here by directly computing the limit.
            jump_term = 0

        else:
            raise ValueError("invalid direction")

        potential_op = (
                jump_term
                + self.get_sqrt_weight(interface_id)*potential_op)

        del jump_term

        contrib = bc_coeff * potential_op

        if (raw_potential_op is sym.D
                and term.direction == self.dir_normal):
            # FIXME The hypersingular part should perhaps be
            # treated specially to avoid cancellation.
            pass

        return contrib


# {{{ single-layer representation

class DielectricSRep2DBoundaryOperator(Dielectric2DBoundaryOperatorBase):
    def _structured_unknown(self, unknown, with_l2_weights):
        """
        :arg with_l2_weights: If True, return the 'bare' unknowns
            that do not have the :math:`L^2` weights divided out.
            Note: Those unknowns should *not* be interpreted as
            point values of a density.
        :returns: an array of unknowns, with the following index axes:
            ``[side, field_kind, i_interface]``, where
            ``side`` is o for the outside part and i for the interior part,
            ``field_kind`` is 0 for the E-field and 1 for the H-field part,
            ``i_interface`` is the number of the enclosed domain, starting from 0.
        """
        result = np.zeros((2, 2, len(self.interfaces)), dtype=np.object)

        i_unknown = 0
        for side in self.sides:
            for field_kind in self.field_kinds:
                for i_interface in range(len(self.interfaces)):

                    if self.is_field_present(field_kind):
                        dens = unknown[i_unknown]
                        i_unknown += 1
                    else:
                        dens = 0

                    _, _, interface_id = self.interfaces[i_interface]

                    if not with_l2_weights:
                        dens = sym.cse(
                                dens/self.get_sqrt_weight(interface_id),
                                "dens_{side}_{field}_{dom}".format(
                                    side={
                                        self.side_out: "o",
                                        self.side_in: "i"}
                                    [side],
                                    field={
                                        self.field_kind_e: "E",
                                        self.field_kind_h: "H"
                                        }
                                    [field_kind],
                                    dom=i_interface))

                    result[side, field_kind, i_interface] = dens

        assert i_unknown == len(unknown)
        return result

    def representation(self, unknown, i_domain):
        """
        :return: a symbolic expression for the representation of the PDE solution
            in domain number *i_domain*.
        """
        unk = self._structured_unknown(unknown, with_l2_weights=False)

        result = []

        for field_kind in self.field_kinds:
            if not self.is_field_present(field_kind):
                continue

            field_result = 0
            for i_interface, (i_domain_outer, i_domain_inner, interface_id) in (
                    enumerate(self.interfaces)):
                if i_domain_outer == i_domain:
                    side = self.side_out
                elif i_domain_inner == i_domain:
                    side = self.side_in
                else:
                    continue

                my_unk = unk[side, field_kind, i_interface]
                if my_unk:
                    field_result += sym.S(
                            self.kernel,
                            my_unk,
                            source=interface_id,
                            k=self.domain_K_exprs[i_domain])

            result.append(field_result)

        from pytools.obj_array import make_obj_array
        return make_obj_array(result)

    def operator(self, unknown):
        density_unk = self._structured_unknown(unknown, with_l2_weights=False)
        discrete_unk = self._structured_unknown(unknown, with_l2_weights=True)

        result = []
        for bc in self.bcs:
            op = 0

            for side in self.sides:
                for term in bc:
                    unk_index = (side, term.field_kind, term.i_interface)
                    density = density_unk[unk_index]
                    discrete = discrete_unk[unk_index]

                    op += self.bc_term_to_operator_contrib(
                            term, side, sym.S, density, discrete)

            result.append(op)

        return np.array(result, dtype=np.object)

# }}}


# {{{ single + double layer representation

class DielectricSDRep2DBoundaryOperator(Dielectric2DBoundaryOperatorBase):
    pot_kind_S = 0  # noqa: N815
    pot_kind_D = 1  # noqa: N815
    pot_kinds = [pot_kind_S, pot_kind_D]
    potential_ops = {
            pot_kind_S: sym.S,
            pot_kind_D: sym.D,
            }

    def __init__(self, mode, k_vacuum, domain_k_exprs, beta,
            interfaces, use_l2_weighting=None):

        super(DielectricSDRep2DBoundaryOperator, self).__init__(
                mode, k_vacuum, domain_k_exprs, beta,
                interfaces, use_l2_weighting=use_l2_weighting)

        side_in = self.side_in
        side_out = self.side_out

        def find_normal_derivative_bc_coeff(field_kind, i_interface, side):
            result = 0
            for bc in self.bcs:
                for term in bc:
                    if term.field_kind != field_kind:
                        continue
                    if term.i_interface != i_interface:
                        continue
                    if term.direction != self.dir_normal:
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
                    len(self.interfaces), len(self.sides)),
                dtype=np.object)
        for field_kind in self.field_kinds:
            for i_interface in range(len(self.interfaces)):
                self.density_coeffs[
                        self.pot_kind_S, field_kind, i_interface, side_in] = 1
                self.density_coeffs[
                        self.pot_kind_S, field_kind, i_interface, side_out] = 1

                # These need to satisfy
                #
                # [dens_coeff_D * bc_coeff * dn D]
                # = dens_coeff_D_out * bc_coeff_out * (dn D)
                #   + dens_coeff_D_in * bc_coeff_in * dn D
                # = 0
                #
                # (because dn D is hypersingular, which we'd like to cancel out)
                #
                # NB: bc_coeff_{in,out} already contain the signs to realize
                # the subtraction for the jump. (So the "+" above is as it
                # should be.)

                dens_coeff_D_in = find_normal_derivative_bc_coeff(  # noqa
                        field_kind, i_interface, side_out)
                dens_coeff_D_out = - find_normal_derivative_bc_coeff(  # noqa
                        field_kind, i_interface, side_in)

                self.density_coeffs[
                        self.pot_kind_D, field_kind, i_interface, side_in] \
                                = dens_coeff_D_in
                self.density_coeffs[
                        self.pot_kind_D, field_kind, i_interface, side_out] \
                                = dens_coeff_D_out

    def _structured_unknown(self, unknown, with_l2_weights):
        """
        :arg with_l2_weights: If True, return the 'bare' unknowns
            that do not have the :math:`L^2` weights divided out.
            Note: Those unknowns should *not* be interpreted as
            point values of a density.
        :returns: an array of unknowns, with the following index axes:
            ``[pot_kind, field_kind, i_interface]``, where
            ``pot_kind`` is 0 for the single-layer part and 1 for the double-layer
            part,
            ``field_kind`` is 0 for the E-field and 1 for the H-field part,
            ``i_interface`` is the number of the enclosed domain, starting from 0.
        """
        result = np.zeros((2, 2, len(self.interfaces)), dtype=np.object)

        i_unknown = 0
        for pot_kind in self.pot_kinds:
            for field_kind in self.field_kinds:
                for i_interface in range(len(self.interfaces)):

                    if self.is_field_present(field_kind):
                        dens = unknown[i_unknown]
                        i_unknown += 1
                    else:
                        dens = 0

                    _, _, interface_id = self.interfaces[i_interface]

                    if not with_l2_weights:
                        dens = sym.cse(
                                dens/self.get_sqrt_weight(interface_id),
                                "dens_{pot}_{field}_{intf}".format(
                                    pot={0: "S", 1: "D"}[pot_kind],
                                    field={
                                        self.field_kind_e: "E",
                                        self.field_kind_h: "H"
                                        }
                                    [field_kind],
                                    intf=i_interface))

                    result[pot_kind, field_kind, i_interface] = dens

        assert i_unknown == len(unknown)
        return result

    def representation(self, unknown, i_domain):
        """
        :return: a symbolic expression for the representation of the PDE solution
            in domain number *i_domain*.
        """
        unk = self._structured_unknown(unknown, with_l2_weights=False)

        result = []

        for field_kind in self.field_kinds:
            if not self.is_field_present(field_kind):
                continue

            field_result = 0
            for pot_kind in self.pot_kinds:
                for i_interface, (i_domain_outer, i_domain_inner, interface_id) in (
                        enumerate(self.interfaces)):
                    if i_domain_outer == i_domain:
                        side = self.side_out
                    elif i_domain_inner == i_domain:
                        side = self.side_in
                    else:
                        continue

                    my_unk = unk[pot_kind, field_kind, i_interface]
                    if my_unk:
                        field_result += (
                                self.density_coeffs[
                                    pot_kind, field_kind, i_interface, side]
                                * self.potential_ops[pot_kind](
                                    self.kernel,
                                    my_unk,
                                    source=interface_id,
                                    k=self.domain_K_exprs[i_domain]
                                    ))

            result.append(field_result)

        from pytools.obj_array import make_obj_array
        return make_obj_array(result)

    def operator(self, unknown):
        density_unk = self._structured_unknown(unknown, with_l2_weights=False)
        discrete_unk = self._structured_unknown(unknown, with_l2_weights=True)

        result = []
        for bc in self.bcs:
            op = 0

            for pot_kind in self.pot_kinds:
                for term in bc:

                    for side in self.sides:
                        raw_potential_op = \
                                self.potential_ops[pot_kind]

                        unk_index = (pot_kind, term.field_kind, term.i_interface)
                        density = density_unk[unk_index]
                        discrete = discrete_unk[unk_index]

                        op += (
                                self.density_coeffs[
                                    pot_kind, term.field_kind, term.i_interface,
                                    side]
                                * self.bc_term_to_operator_contrib(
                                    term, side, raw_potential_op, density, discrete)
                                )

            result.append(op)

        return np.array(result, dtype=np.object)

# }}}

# }}}

# vim: foldmethod=marker
