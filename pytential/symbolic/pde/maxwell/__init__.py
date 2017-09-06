from __future__ import division, absolute_import

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

import numpy as np  # noqa
from pytential import sym
from collections import namedtuple
from functools import partial

tangential_to_xyz = sym.tangential_to_xyz
xyz_to_tangential = sym.xyz_to_tangential
cse = sym.cse


# {{{ Charge-Current MFIE

class PECAugmentedMFIEOperator:
    """Magnetic Field Integral Equation operator,
    under the assumption of no surface charges.

    see :file:`contrib/notes/mfie.tm`
    """

    def __init__(self, k=sym.var("k")):
        from sumpy.kernel import HelmholtzKernel
        self.kernel = HelmholtzKernel(3)
        self.k = k

    def j_operator(self, loc, Jt):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")
        return xyz_to_tangential(
                (loc*0.5)*Jxyz - sym.n_cross(
                    sym.curl(sym.S(self.kernel, Jxyz, k=self.k,
                        qbx_forced_limit="avg"))))

    def j_rhs(self, Hinc_xyz):
        return xyz_to_tangential(sym.n_cross(Hinc_xyz))

    def rho_operator(self, loc, rho):
        return (loc*0.5)*rho+sym.Sp(self.kernel, rho, k=self.k)

    def rho_rhs(self, Jt, Einc_xyz):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")

        return (sym.n_dot(Einc_xyz)
                - 1j*self.k*sym.n_dot(sym.S(self.kernel, Jxyz, k=self.k)))

    def scattered_boundary_field(self, Jt, rho, loc):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")

        A = sym.S(self.kernel, Jxyz, k=self.k)
        grad_phi = sym.grad(3, sym.S(self.kernel, rho, k=self.k))

        # use - n x n x v = v_tangential

        E_scat = - 1j*self.k*A - grad_phi + 0.5*loc*rho
        H_scat = sym.curl(sym.S(self.kernel, Jxyz, k=self.k)) + (loc*0.5)*Jxyz

        return sym.join_fields(E_scat, H_scat)

    def scattered_volume_field(self, Jt, rho):
        Jxyz = sym.cse(sym.tangential_to_xyz(Jt), "Jxyz")

        A = sym.S(self.kernel, Jxyz, k=self.k, qbx_forced_limit=None)
        grad_phi = sym.grad(3, sym.S(self.kernel, rho, k=self.k))

        E_scat = - 1j*self.k*A - grad_phi
        H_scat = sym.curl(sym.S(self.kernel, Jxyz, k=self.k))

        return sym.join_fields(E_scat, H_scat)

# }}}


# {{{ Charge-Current Mueller MFIE

class MuellerAugmentedMFIEOperator(object):
    """
    ... warning:: currently untested
    """

    def __init__(self, omega, mus, epss):
        from sumpy.kernel import HelmholtzKernel
        self.kernel = HelmholtzKernel(3)
        self.omega = omega
        self.mus = mus
        self.epss = epss
        self.ks = [
                sym.cse(omega*(eps*mu)**0.5, "k%d" % i)
                for i, (eps, mu) in enumerate(zip(epss, mus))]

    def make_unknown(self, name):
        return sym.make_sym_vector(name, 6)

    unk_structure = namedtuple("MuellerUnknowns", ["jt", "rho_e", "mt", "rho_m"])

    def split_unknown(self, unk):
        return self.unk_structure(
            jt=unk[:2],
            rho_e=unk[2],
            mt=unk[3:5],
            rho_m=unk[5])

    def operator(self, unk):
        u = self.split_unknown(unk)

        Jxyz = cse(tangential_to_xyz(u.jt), "Jxyz")
        Mxyz = cse(tangential_to_xyz(u.mt), "Mxyz")

        omega = self.omega
        mu0, mu1 = self.mus
        eps0, eps1 = self.epss
        k0, k1 = self.ks

        S = partial(sym.S, self.kernel, qbx_forced_limit="avg")

        def curl_S(dens):
            return sym.curl(sym.S(self.kernel, dens, qbx_forced_limit="avg"))

        grad = partial(sym.grad, 3)

        E0 = sym.cse(1j*omega*mu0*eps0*S(Jxyz, k=k0) +
            mu0*curl_S(Mxyz, k=k0) - grad(S(u.rho_e, k=k0)), "E0")
        H0 = sym.cse(-1j*omega*mu0*eps0*S(Mxyz, k=k0) +
            eps0*curl_S(Jxyz, k=k0) + grad(S(u.rho_m, k=k0)), "H0")
        E1 = sym.cse(1j*omega*mu1*eps1*S(Jxyz, k=k1) +
            mu1*curl_S(Mxyz, k=k1) - grad(S(u.rho_e, k=k1)), "E1")
        H1 = sym.cse(-1j*omega*mu1*eps1*S(Mxyz, k=k1) +
            eps1*curl_S(Jxyz, k=k1) + grad(S(u.rho_m, k=k1)), "H1")

        F1 = (xyz_to_tangential(sym.n_cross(H1-H0) + 0.5*(eps0+eps1)*Jxyz))
        F2 = (sym.n_dot(eps1*E1-eps0*E0) + 0.5*(eps1+eps0)*u.rho_e)
        F3 = (xyz_to_tangential(sym.n_cross(E1-E0) + 0.5*(mu0+mu1)*Mxyz))

        # sign flip included
        F4 = -sym.n_dot(mu1*H1-mu0*H0) + 0.5*(mu1+mu0)*u.rho_m

        return sym.join_fields(F1, F2, F3, F4)

    def rhs(self, Einc_xyz, Hinc_xyz):
        mu1 = self.mus[1]
        eps1 = self.epss[1]

        return sym.join_fields(
            xyz_to_tangential(sym.n_cross(Hinc_xyz)),
            sym.n_dot(eps1*Einc_xyz),
            xyz_to_tangential(sym.n_cross(Einc_xyz)),
            sym.n_dot(-mu1*Hinc_xyz))

    def representation(self, i, sol):
        u = self.split_unknown(sol)
        Jxyz = sym.cse(sym.tangential_to_xyz(u.jt), "Jxyz")
        Mxyz = sym.cse(sym.tangential_to_xyz(u.mt), "Mxyz")

        # omega = self.omega
        mu = self.mus[i]
        eps = self.epss[i]
        k = self.ks[i]

        S = partial(sym.S, self.kernel, qbx_forced_limit=None, k=k)

        def curl_S(dens):
            return sym.curl(sym.S(self.kernel, dens, qbx_forced_limit=None, k=k))

        grad = partial(sym.grad, 3)

        E0 = 1j*k*eps*S(Jxyz) + mu*curl_S(Mxyz) - grad(S(u.rho_e))
        H0 = -1j*k*mu*S(Mxyz) + eps*curl_S(Jxyz) + grad(S(u.rho_m))

        return sym.join_fields(E0, H0)

# }}}


# vim: foldmethod=marker
