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

__doc__ = """

.. autofunction:: get_sym_maxwell_point_source
.. autofunction:: get_sym_maxwell_plane_wave
.. autoclass:: PECChargeCurrentMFIEOperator
"""


# {{{ point source

def get_sym_maxwell_point_source(kernel, jxyz, k):
    r"""Return a symbolic expression that, when bound to a
    :class:`pytential.source.PointPotentialSource` will yield
    a field satisfying Maxwell's equations.

    Uses the sign convention :math:`\exp(-1 \omega t)` for the time dependency.

    This will return an object of six entries, the first three of which
    represent the electric, and the second three of which represent the
    magnetic field. This satisfies the time-domain Maxwell's equations
    as verified by :func:`sumpy.point_calculus.frequency_domain_maxwell`.
    """
    # This ensures div A = 0, which is simply a consequence of div curl S=0.
    # This means we use the Coulomb gauge to generate this field.

    A = sym.curl(sym.S(kernel, jxyz, k=k, qbx_forced_limit=None))

    # https://en.wikipedia.org/w/index.php?title=Maxwell%27s_equations&oldid=798940325#Alternative_formulations
    # (Vector calculus/Potentials/Any Gauge)
    # assumed time dependence exp(-1j*omega*t)
    return sym.join_fields(
        1j*k*A,
        sym.curl(A))

# }}}


# {{{ plane wave

def get_sym_maxwell_plane_wave(amplitude_vec, v, omega, epsilon=1, mu=1, where=None):
    r"""Return a symbolic expression that, when bound to a
    :class:`pytential.source.PointPotentialSource` will yield
    a field satisfying Maxwell's equations.

    :arg amplitude_vec: should be orthogonal to *v*. If it is not,
        it will be orthogonalized.
    :arg v: a three-vector representing the phase velocity of the wave
        (may be an object array of variables or a vector of concrete numbers)
        While *v* may mathematically be complex-valued, this function
        is for now only tested for real values.
    :arg omega: Accepts the "Helmholtz k" to be compatible with other parts
        of this module.

    Uses the sign convention :math:`\exp(-1 \omega t)` for the time dependency.

    This will return an object of six entries, the first three of which
    represent the electric, and the second three of which represent the
    magnetic field. This satisfies the time-domain Maxwell's equations
    as verified by :func:`sumpy.point_calculus.frequency_domain_maxwell`.
    """

    # See section 7.1 of Jackson, third ed. for derivation.

    # NOTE: for complex, need to ensure real(n).dot(imag(n)) = 0  (7.15)

    x = sym.nodes(3, where).as_vector()

    v_mag_squared = sym.cse(np.dot(v, v), "v_mag_squared")
    n = v/sym.sqrt(v_mag_squared)

    amplitude_vec = amplitude_vec - np.dot(amplitude_vec, n)*n

    c_inv = np.sqrt(mu*epsilon)

    e = amplitude_vec * sym.exp(1j*np.dot(n*omega, x))

    return sym.join_fields(e, c_inv * sym.cross(n, e))

# }}}


# {{{ Charge-Current MFIE

class PECChargeCurrentMFIEOperator:
    r"""Magnetic Field Integral Equation operator with PEC boundary
    conditions, under the assumption of no surface charges.

    See :file:`contrib/notes/mfie.tm` in the repository for a derivation.

    The arguments *loc* below decide between the exterior and the
    interior MFIE. The "exterior" (loc=1) MFIE enforces a zero field
    on the interior of the integration surface, whereas the "interior" MFIE
    (loc=-1) enforces a zero field on the exterior.

    Uses the sign convention :math:`\exp(-1 \omega t)` for the time dependency.

    for the sinusoidal time dependency.

    .. automethod:: j_operator
    .. automethod:: j_rhs
    .. automethod:: rho_operator
    .. automethod:: rho_rhs
    .. automethod:: scattered_volume_field
    """

    def __init__(self, k=sym.var("k")):
        from sumpy.kernel import HelmholtzKernel
        self.kernel = HelmholtzKernel(3)
        self.k = k

    def j_operator(self, loc, Jt):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")
        return xyz_to_tangential(
                loc*0.5*Jxyz - sym.n_cross(
                    sym.curl(sym.S(self.kernel, Jxyz, k=self.k,
                        qbx_forced_limit="avg"))))

    def j_rhs(self, Hinc_xyz):
        return xyz_to_tangential(sym.n_cross(Hinc_xyz))

    def rho_operator(self, loc, rho):
        return loc*0.5*rho+sym.Sp(self.kernel, rho, k=self.k)

    def rho_rhs(self, Jt, Einc_xyz):
        Jxyz = cse(tangential_to_xyz(Jt), "Jxyz")

        return (sym.n_dot(Einc_xyz)
                + 1j*self.k*sym.n_dot(sym.S(
                    self.kernel, Jxyz, k=self.k,
                    # continuous--qbx_forced_limit doesn't really matter
                    qbx_forced_limit="avg")))

    def scattered_volume_field(self, Jt, rho, qbx_forced_limit=None):
        """
        This will return an object array of six entries, the first three of which
        represent the electric, and the second three of which represent the
        magnetic field. This satisfies the time-domain Maxwell's equations
        as verified by :func:`sumpy.point_calculus.frequency_domain_maxwell`.
        """
        Jxyz = sym.cse(sym.tangential_to_xyz(Jt), "Jxyz")

        A = sym.S(self.kernel, Jxyz, k=self.k, qbx_forced_limit=qbx_forced_limit)
        phi = sym.S(self.kernel, rho, k=self.k, qbx_forced_limit=qbx_forced_limit)

        E_scat = 1j*self.k*A - sym.grad(3, phi)
        H_scat = sym.curl(A)

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

        E0 = sym.cse(1j*omega*mu0*eps0*S(Jxyz, k=k0)
            + mu0*curl_S(Mxyz, k=k0) - grad(S(u.rho_e, k=k0)), "E0")
        H0 = sym.cse(-1j*omega*mu0*eps0*S(Mxyz, k=k0)
            + eps0*curl_S(Jxyz, k=k0) + grad(S(u.rho_m, k=k0)), "H0")
        E1 = sym.cse(1j*omega*mu1*eps1*S(Jxyz, k=k1)
            + mu1*curl_S(Mxyz, k=k1) - grad(S(u.rho_e, k=k1)), "E1")
        H1 = sym.cse(-1j*omega*mu1*eps1*S(Mxyz, k=k1)
            + eps1*curl_S(Jxyz, k=k1) + grad(S(u.rho_m, k=k1)), "H1")

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
