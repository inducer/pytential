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

from pytools.obj_array import join_fields


# {{{ Charge-Current MFIE

class PECAugmentedMFIEOperator:
    """Magnetic Field Integral Equation operator,
    under the assumption of no surface charges.

    see notes/mfie.tm
    """

    def __init__(self, k=sym.var("k")):
        from sumpy.kernel import HelmholtzKernel
        self.kernel = HelmholtzKernel(3)
        self.k = k

    def j_operator(self, loc, Jt):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")
        return xyz_to_tangential(
                (loc*0.5)*Jxyz - sym.nxcurl_S(self.kernel, 0, Jxyz, k=self.k))

    def j_rhs(self, Hinc_xyz):
        return xyz_to_tangential(sym.n_cross(Hinc_xyz))

    def rho_operator(self, loc, rho):
        return (loc*0.5)*rho+sym.Sp(self.kernel, rho, k=self.k)

    def rho_rhs(self, Jt, Einc_xyz):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")

        return (sym.n_dot(Einc_xyz)
                + 1j*self.k*sym.n_dot(sym.S(self.kernel, Jxyz, k=self.k)))

    def scattered_boundary_field(self, Jt, rho, loc):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")

        A = sym.S(self.kernel, Jxyz, k=self.k)
        grad_phi = sym.grad(3, sym.S(self.kernel, rho, k=self.k))

        # use - n x n x v = v_tangential

        E_scat = 1j*self.k*A - grad_phi + 0.5*loc*rho
        H_scat = sym.curl_S(self.kernel, Jxyz, k=self.k) + (loc*0.5)*Jxyz

        return join_fields(E_scat, H_scat)

    def scattered_volume_field(self, Jt, rho):
        Jxyz = sym.cse(sym.tangential_to_xyz(Jt), "Jxyz")

        A = sym.S(self.kernel, Jxyz, k=self.k, qbx_forced_limit=None)
        grad_phi = sym.grad(3, sym.S(self.kernel, rho, 3, k=self.k))

        E_scat = 1j*self.k*A - grad_phi
        H_scat = sym.curl_S(self.kernel, Jxyz, k=self.k)

        from pytools.obj_array import join_fields
        return E_scat #join_fields(E_scat, H_scat)

# }}}


# {{{ Charge-Current Mueller MFIE

class MuellerAugmentedMFIEOperator:
    """Magnetic Field Integral Equation operator,
    under the assumption of no surface charges.
    """

    def __init__(self,omg,mu0,eps0,mu1,eps1):
        from sumpy.kernel import HelmholtzKernel
        self.kernel = HelmholtzKernel(3)
        self.omg = omg
        self.mu0 = mu0
        self.eps0 = eps0
        self.mu1=mu1
        self.eps1=eps1
        self.k0 = omg*sqrt(eps0*mu0)
        self.k1 = omg*sqrt(eps1*mu1)

    def make_unknown(self, name):
        return sym.make_sym_vector(name, 6)

    unk_structure = namedtuple(["jt", "rho_e", "mt", "rho_m"]
    def split_unknown(self, unk):
        return self.unk_structure(
            jt=unk[:2],
            rho_e=unk[2],
            mt=unk[3:5],
            rho_m=unk[5])

    def augmul_operator(self, unk):
        u = self.split_unknown(unk)
        
        Jxyz = cse(tangential_to_xyz(u.jt), "Jxyz")
        Mxyz = cse(tangential_to_xyz(u.mt), "Mxyz")
        E0 = cse(1j*self.omg*self.mu0*self.eps0*S(self.k0,0,Jxyz) +
            self.mu0*gradScross(self.k0,Mxyz)-grad_S(self.k0,u.rho_e,3),"E0")
        H0 = cse(-1j*self.omg*self.mu0*self.eps0*S(self.k0,0,Mxyz) +
            self.eps0*gradScross(self.k0,Jxyz)+grad_S(self.k0,u.rho_m,3),"H0")
        E1 = cse(1j*self.omg*self.mu1*self.eps1*S(self.k1,0,Jxyz) +
            self.mu1*gradScross(self.k1,Mxyz)-grad_S(self.k1,u.rho_e,3),"E1")
        H1 = cse(-1j*self.omg*self.mu1*self.eps1*S(self.k1,0,Mxyz) +
            self.eps1*gradScross(self.k1,Jxyz)+grad_S(self.k1,u.rho_m,3),"H1")
        F1 = cse(xyz_to_tangential(n_cross(H1-H0) 
            + 0.5*(self.eps0+self.eps1)*Jxyz),"F1")
        F2 = cse(n_dot(self.eps1*E1-self.eps0*E0) 
            + 0.5*(self.eps1+self.eps0)*u.rho_e,"F2")    
        F3 = cse(xyz_to_tangential(n_cross(E1-E0) 
            + 0.5*(self.mu0+self.mu1)*Mxyz),"F3")
        # sign flip included    
        F4 = cse(-n_dot(self.mu1*H1-self.mu0*H0) 
            + 0.5*(self.mu1+self.mu0)*u.rho_m,"F4")    

        return sym.join_fields([F1,F2,F3,F4])

    def augmul_rhs(self, Einc_xyz,Hinc_xyz):
        return sym.join_fields([xyz_to_tangential(n_cross(Hinc_xyz)),
            n_dot(self.eps1*Einc_xyz),xyz_to_tangential(n_cross(Einc_xyz)),
            n_dot(-self.mu1*Hinc_xyz)]

    def scattered_volume_field(self, sol):
        u = self.split_unknown(sol)
        Jxyz = sym.cse(sym.tangential_to_xyz(u.jt), "Jxyz")
        Mxyz = sym.cse(sym.tangential_to_xyz(u.mt), "Mxyz")

        E0 = cse(1j*self.omg*self.mu0*self.eps0*S(self.k0,0,Jxyz) +
            self.mu0*gradScross(self.k0,Mxyz)-grad_S(self.k0,u.rho_e,3),"E0")
        H0 = cse(-1j*self.omg*self.mu0*self.eps0*S(self.k0,0,Mxyz) +
            self.eps0*gradScross(self.k0,Jxyz)+grad_S(self.k0,u.rho_m,3),"H0")

        from pytools.obj_array import join_fields
        return join_fields(E0, H0)

# }}}


# {{{ generalized Debye

def _debye_S0_surf_div(nxnxE):
    # Note integration by parts: S0 div_\Gamma (X) -> - \int \grad g0 \cdot

    nxnxE_t = cse(xyz_to_tangential(nxnxE), "nxnxE_t")

    return - sum([
            IntGdSource(0, nxnxE_t[i],
                ds_direction=make_tangent(i, 3))
            for i in range(3-1)])


class DebyeOperatorBase(object):
    """
    See the `wiki page <http://wiki.tiker.net/HellsKitchen/GeneralizedDebyeRepresentation>`_
    for detailed documentation.

    j, m are the (non-physical) Debye currents
    r, q are (non-physical) Debye charge densities (from the Debye paper)

    r_tilde, defined by: r = surf_lap S_0^2 r_tilde
    q_tilde, defined by: q = surf_lap S_0^2 q_tilde

    :class:`InvertingDebyeOperator` below implements the version based
    on r and q.

    :class:`NonInvertingDebyeOperator` below implements the version based
    on r_tilde and q_tilde.

    A, Q are the Debye vector potentials
    phi, psi are the Debye scalar potentials
    """

    def __init__(self, loc_sign, k, invertible, genus=0,
            a_cycle_names=None,
            b_cycle_names=None, b_spanning_surface_names=None,
            harmonic_vector_field_names=None, h_on_spanning_surface_names=None):
        self.loc_sign = loc_sign

        self.k = k
        self.invertible = invertible

        self.genus = genus

        # {{{ symbolic names for multiply connected case

        if harmonic_vector_field_names is None:
            harmonic_vector_field_names = \
                    ["hvf%d" % i for i in range(2*genus)]

        self.harmonic_vector_field_names = harmonic_vector_field_names
        self.harmonic_vector_field_symbols = [
                make_vector_field(name, 3)
                for name in harmonic_vector_field_names]

        if a_cycle_names is None:
            a_cycle_names = ["acyc%d" % i for i in range(genus)]
        self.a_cycle_names = a_cycle_names

        if b_cycle_names is None:
            b_cycle_names = ["bcyc%d" % i for i in range(genus)]
        self.b_cycle_names = b_cycle_names

        if b_spanning_surface_names is None:
            b_spanning_surface_names = ["span_surf_%d" % i for i in range(genus)]
        self.b_spanning_surface_names = b_spanning_surface_names

        if h_on_spanning_surface_names is None:
            h_on_spanning_surface_names = ["h_%s" % name
                    for name in b_spanning_surface_names]
        self.h_on_spanning_surface_names = h_on_spanning_surface_names

        self.h_on_spanning_surface_symbols = [
                make_vector_field(name, 3)
                for name in h_on_spanning_surface_names]

        # }}}

    def m(self, j):
        return n_cross(j)

    def boundary_field_base(self, r, q, j):
        k = self.k

        surf_grad_phi = surf_grad_S(self.k, r)

        dpsi_dn = Sp(self.k, q)
        dpsi_dn -= self.loc_sign*1/2*q

        m = cse(self.m(j), "m")

        A = cse(S(k, j), "A")
        Q = cse(S(k, m), "Q")

        nxnxE = (
                n_cross(
                    n_cross(1j * k * A)

                    - nxcurl_S(k, self.loc_sign, m))

                # sign: starts out as -grad phi
                # nxnx = - tangential component
                + surf_grad_phi)

        ndotH = (
                np.dot(make_normal(3),
                    # no limit terms: normal part of curl A
                    curl_S_volume(k, j)

                    + 1j*k*Q)

                - dpsi_dn)

        return nxnxE, ndotH

    def volume_field_base(self, r, q, j):
        k = self.k

        grad_phi = grad_S(k, r, 3)
        grad_psi = grad_S(k, q, 3)

        m = cse(self.m(j), "m")

        A = cse(S(k, j), "A")
        Q = cse(S(k, m), "Q")

        E = 1j*k*A - grad_phi - curl_S_volume(k, m)
        H = curl_S_volume(k, j) + 1j*k*Q - grad_psi

        from pytools.obj_array import join_fields
        return join_fields(E, H)

    def integral_equation(self, *args, **kwargs):
        nxnxE, ndotH = self.boundary_field(*args)
        nxnxE = cse(nxnxE, "nxnxE")

        fix = kwargs.pop("fix", 0)
        if kwargs:
            raise TypeError("invalid keyword argument(s)")

        from pytools.obj_array import make_obj_array
        eh_op = make_obj_array([
            2*_debye_S0_surf_div(nxnxE),
            -ndotH,
            ]) + fix

        k = self.k
        j = cse(self.j(*args), "j")
        m = cse(self.m(j), "m")
        A = cse(S(k, j), "A")

        E_minus_grad_phi = 1j*k*A - curl_S_volume(k, m)

        from hellskitchen.fmm import DifferenceKernel
        from pytools.obj_array import join_fields
        return join_fields(
                eh_op,
                # FIXME: These are inefficient. They compute a full volume field,
                # but only actually use the line part of it.

                [
                    # Grad phi integrated on a loop does not contribute.
                    LineIntegral(E_minus_grad_phi, a_cycle_name)
                    for a_cycle_name in self.a_cycle_names
                    ],
                [
                    LineIntegral(
                        # (E(k) - E(0))/(1jk)
                        # Grad phi integrated on a loop does not contribute.
                        (1j*k*A - curl_S_volume(DifferenceKernel(k), m))
                            /(1j*k),
                        b_cycle_name)
                    for b_cycle_name in self.b_cycle_names
                    ]
                )

    def prepare_rhs(self, e_inc, h_inc):
        normal = make_normal(3)
        nxnxE = cse(- project_to_tangential(e_inc), "nxnxE")

        e_rhs = 2*cse(_debye_S0_surf_div(nxnxE))
        h_rhs = cse(-np.dot(normal, h_inc))

        result = [e_rhs, h_rhs]

        result.extend([
            -LineIntegral(h_inc, a_cycle_name)
            for a_cycle_name in self.a_cycle_names
            ] + [
                Integral(
                    np.dot(make_normal(3, ssurf_name), h_on_surf),
                    ssurf_name)

                for ssurf_name, h_on_surf in zip(
                    self.b_spanning_surface_names,
                    self.h_on_spanning_surface_symbols)]
                )

        from pytools.obj_array import make_obj_array
        return make_obj_array(result)


    def harmonic_vector_field_current(self, hvf_coefficients):
        return sum(hvf_coeff_i * hvf_i
                for hvf_coeff_i, hvf_i in
                zip(hvf_coefficients, self.harmonic_vector_field_symbols))

    def cluster_points(self):
        return (-1/4, +1/2)





class InvertingDebyeOperatorBase(DebyeOperatorBase):
    def r_and_q(self, r, q):
        return r, q

    def boundary_field(self, r, q, hvf_coefficients):
        j = cse(self.j(r, q, hvf_coefficients), "j")
        return self.boundary_field_base(r, q, j)

    def volume_field(self, r, q, hvf_coefficients):
        j = cse(self.j(r, q, hvf_coefficients), "j")
        return self.volume_field_base(r, q, j)

    def integral_equation(self, r_tilde, q_tilde, hvf_coefficients):
        fix = 0

        if self.invertible:
            s_ones = cse(S(0,Ones()), "s_ones")

            def inv_rank_one_coeff(u):
                return cse(Mean(u))

            r_coeff = inv_rank_one_coeff(r_tilde)
            q_coeff = inv_rank_one_coeff(q_tilde)

            from pytools.obj_array import join_fields
            factors = self.cluster_points()

            fix = join_fields(
                    factors[0]*s_ones*r_coeff,
                    factors[1]*Ones()*q_coeff,
                    )

        return DebyeOperatorBase.integral_equation(
                self, r_tilde, q_tilde, hvf_coefficients, fix=fix)





class InvertingDebyeOperator(InvertingDebyeOperatorBase):
    "Debye operator based on r and q."

    def j(self, r, q, hvf_coefficients):
        # We would like to solve
        #
        #   surf_lap alpha = f
        #
        # Let alpha = S^2 sigma. Then solve
        #
        #   surf_lap S^2 sigma = f
        #
        # instead. (We need the inner S^2 because that's what we can represent.)

        sigma_solve = Variable("sigma")
        surf_lap_op = surface_laplacian_S_squared(
                sigma_solve, invertibility_scale=1)
        sigma_r = cse(IterativeInverse(surf_lap_op, r, "sigma"))
        sigma_q = cse(IterativeInverse(surf_lap_op, q, "sigma"))

        surf_grad_alpha_r = cse(surf_grad_S(0, S(0, sigma_r)), "surf_grad_alpha_r")
        surf_grad_alpha_q = cse(surf_grad_S(0, S(0, sigma_q)), "surf_grad_alpha_q")

        return (
                1j * self.k * (
                    surf_grad_alpha_r - n_cross(surf_grad_alpha_q))
                + self.harmonic_vector_field_current(hvf_coefficients))


class InvertingSLapSDebyeOperator(InvertingDebyeOperatorBase):
    "Debye operator based on r and q."

    def j(self, r, q, hvf_coefficients):
        # We would like to solve
        #
        #   surf_lap alpha = f
        #
        # Let alpha = S sigma. Then solve
        #
        #   S surf_lap S sigma = S f
        #
        # instead. (We need the inner S and the outer S because that's
        # what we can represent--in this version.)

        sigma_solve = Variable("sigma")
        surf_lap_op = S_surface_laplacian_S(sigma_solve, 3, invertibility_scale=1)
        sigma_r = IterativeInverse(surf_lap_op, S(0, r), "sigma")
        sigma_q = IterativeInverse(surf_lap_op, S(0, q), "sigma")

        surf_grad_alpha_r = cse(surf_grad_S(0, sigma_r), "surf_grad_alpha_r")
        surf_grad_alpha_q = cse(surf_grad_S(0, sigma_q), "surf_grad_alpha_q")

        return (
                1j * self.k * (
                    surf_grad_alpha_r - n_cross(surf_grad_alpha_q))
                + self.harmonic_vector_field_current(hvf_coefficients))


class NonInvertingDebyeOperator(DebyeOperatorBase):
    "Debye operator based on r_tilde and q_tilde."

    def r_and_q(self, r_tilde, q_tilde):
        r = cse(surface_laplacian_S_squared(r_tilde), "r")
        q = cse(surface_laplacian_S_squared(q_tilde), "q")
        return r, q

    def j(self, r_tilde, q_tilde, hvf_coefficients):
        s_r_tilde = cse(S(0, r_tilde), "s_r_tilde")
        s_q_tilde = cse(S(0, q_tilde), "s_q_tilde")
        assert len(hvf_coefficients) == len(self.harmonic_vector_field_symbols)
        surf_grad_s2_r_tilde = cse(surf_grad_S(0, s_r_tilde), "surf_grad_s2_r_tilde")
        surf_grad_s2_q_tilde = cse(surf_grad_S(0, s_q_tilde), "surf_grad_s2_q_tilde")

        return (1j * self.k * (
                surf_grad_s2_r_tilde - n_cross(surf_grad_s2_q_tilde))
                + self.harmonic_vector_field_current(hvf_coefficients))

    def boundary_field(self, r_tilde, q_tilde, hvf_coefficients):
        r, q = self.r_and_q(r_tilde, q_tilde)
        j = cse(self.j(r_tilde, q_tilde, hvf_coefficients), "j")
        return self.boundary_field_base(r, q, j)

    def volume_field(self, r_tilde, q_tilde, hvf_coefficients):
        r, q = self.r_and_q(r_tilde, q_tilde)
        j = cse(self.j(r_tilde, q_tilde, hvf_coefficients), "j")
        return self.volume_field_base(r, q, j)

    def integral_equation(self, r_tilde, q_tilde, hvf_coefficients):
        fix = 0

        if self.invertible:
            s_ones = cse(S(0, Ones()), "s_ones")

            def inv_rank_one_coeff(u):
                return cse(Mean(cse(S(0, cse(S(0, u))))))

            r_coeff = inv_rank_one_coeff(r_tilde)
            q_coeff = inv_rank_one_coeff(q_tilde)

            from pytools.obj_array import join_fields
            factors = self.cluster_points()

            fix = join_fields(
                    factors[0]*s_ones*(r_coeff),
                    factors[1]*Ones()*(q_coeff),
                    )

        return DebyeOperatorBase.integral_equation(
                self, r_tilde, q_tilde, hvf_coefficients, fix=fix)

# }}}

# vim: foldmethod=marker
