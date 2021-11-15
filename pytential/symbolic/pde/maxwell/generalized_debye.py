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


# {{{ generalized Debye

def _debye_S0_surf_div(nxnxE):
    # Note integration by parts: S0 div_\Gamma (X) -> - \int \grad g0 \cdot

    nxnxE_t = cse(xyz_to_tangential(nxnxE), "nxnxE_t")

    return - sum([
            IntGdSource(0, nxnxE_t[i],
                ds_direction=make_tangent(i, 3))
            for i in range(3-1)])


class DebyeOperatorBase:
    """
    See the `wiki page <https://wiki.tiker.net/HellsKitchen/GeneralizedDebyeRepresentation>`_
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
                    [f"hvf{i}" for i in range(2*genus)]

        self.harmonic_vector_field_names = harmonic_vector_field_names
        self.harmonic_vector_field_symbols = [
                make_vector_field(name, 3)
                for name in harmonic_vector_field_names]

        if a_cycle_names is None:
            a_cycle_names = [f"acyc{i}" for i in range(genus)]
        self.a_cycle_names = a_cycle_names

        if b_cycle_names is None:
            b_cycle_names = [f"bcyc{i}" for i in range(genus)]
        self.b_cycle_names = b_cycle_names

        if b_spanning_surface_names is None:
            b_spanning_surface_names = [f"span_surf_{i}" for i in range(genus)]
        self.b_spanning_surface_names = b_spanning_surface_names

        if h_on_spanning_surface_names is None:
            h_on_spanning_surface_names = [f"h_{name}"
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

        from pytools.obj_array import flat_obj_array
        return flat_obj_array(E, H)

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
        from pytools.obj_array import flat_obj_array
        return flat_obj_array(
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

            from pytools.obj_array import flat_obj_array
            factors = self.cluster_points()

            fix = flat_obj_array(
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

            from pytools.obj_array import flat_obj_array
            factors = self.cluster_points()

            fix = flat_obj_array(
                    factors[0]*s_ones*(r_coeff),
                    factors[1]*Ones()*(q_coeff),
                    )

        return DebyeOperatorBase.integral_equation(
                self, r_tilde, q_tilde, hvf_coefficients, fix=fix)

# }}}

