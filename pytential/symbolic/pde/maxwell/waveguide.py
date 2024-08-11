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
"""

import numpy as np

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
                sym.cse(upper_half_square_root(n_expr**2-ne**2), f"K{i}")
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
                    result[current + cur_dir, i] = len(result)

        return result

    def tangent(self, where):
        return sym.cse(
                (sym.pseudoscalar(2, 1, where)
                / sym.area_element(2, 1, where)),
                "tangent")

    def S(self, dom_idx, density, qbx_forced_limit=+1):
        return sym.S(
                self.kernel, density,
                k=self.domain_K_exprs[dom_idx],
                qbx_forced_limit=qbx_forced_limit)

    def D(self, dom_idx, density, qbx_forced_limit="avg"):
        return sym.D(
                self.kernel, density,
                k=self.domain_K_exprs[dom_idx],
                qbx_forced_limit=qbx_forced_limit)

    def T(self, where, dom_idx, density):
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
            k = sym.cse(k_v * self.domain_n_exprs[domi], f"k{domi}")

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

            S = self.S
            D = self.D
            T = self.T

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

            S = self.S
            D = self.D
            T = self.T

            def Tt(where, dom, density):
                return sym.tangential_derivative(
                        2, self.T(where, dom, density)).xproject(0)

            def Sn(dom, density):
                return sym.normal_derivative(
                        2, self.S(dom, density, qbx_forced_limit="avg"))

            def St(dom, density):
                return sym.tangential_derivative(
                    2, self.S(dom, density)).xproject(0)

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

            def a23_expr(phi, tangent, where=where, dom0i=dom0i, dom1i=dom1i):
                return (
                        1j * (Tt(where, dom0i, phi) - Tt(where, dom1i, phi))
                        - 1j * (
                            self.domain_n_exprs[dom0i]**2
                            * tangent.scalar_product(self.S(dom0i, tangent * phi))
                            - self.domain_n_exprs[dom1i]**2
                            * tangent.scalar_product(self.S(dom1i, tangent * phi))
                            )
                        )

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

# vim: foldmethod=marker
