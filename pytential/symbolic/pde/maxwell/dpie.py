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
.. autofunction:: get_sym_maxwell_point_source_potentials
.. autofunction:: get_sym_maxwell_plane_wave
.. autoclass:: DPIEOperator
"""


# {{{ point source

def get_sym_maxwell_point_source(kernel, jxyz, k):
    """Return a symbolic expression that, when bound to a
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

# {{{ point source for vector potential

def get_sym_maxwell_point_source_potentials(kernel, jxyz, k):
    """Return a symbolic expression that, when bound to a
    :class:`pytential.source.PointPotentialSource` will yield
    a potential fields satisfying Maxwell's equations.

    Uses the sign convention :math:`\exp(-1 \omega t)` for the time dependency.

    This will return an object of four entries, the first being the
    scalar potential and the last three being the components of the
    vector potential.
    """
    field = get_sym_maxwell_point_source(kernel, jxyz, k)
    return sym.join_fields(
        0*1j,               # scalar potential
        field[:3]/(1j*k)    # vector potential
        )

# }}}


# {{{ plane wave

def get_sym_maxwell_plane_wave(amplitude_vec, v, omega, epsilon=1, mu=1, where=None):
    """Return a symbolic expression that, when bound to a
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



# {{{ Decoupled Potential Integral Equation Operator
class DPIEOperator:
    """
    Decoupled Potential Integral Equation operator with PEC boundary
    conditions, defaults as scaled DPIE.

    See https://arxiv.org/abs/1404.0749 for derivation.

    Uses E(x,t) = Re{E(x) exp(-i omega t)} and H(x,t) = Re{H(x) exp(-i omega t)}
    and solves for the E(x), H(x) fields using vector and scalar potentials via
    the Lorenz Gauage. The DPIE formulates the problem purely in terms of the 
    vector and scalar potentials, A and phi, and then backs out E(x) and H(x) 
    via relationships to the vector and scalar potentials.
    """

    def __init__(self, geometry_list, k=sym.var("k")):
        from sumpy.kernel import HelmholtzKernel

        # specify the frequency variable that will be tuned
        self.k          = k
        self.stype      = type(self.k)

        # specify the 3-D Helmholtz kernel 
        self.kernel     = HelmholtzKernel(3)

        # specify a list of strings representing geometry objects
        self.geometry_list   = geometry_list
        self.nobjs           = len(geometry_list)

        # create the characteristic functions that give a value of
        # 1 when we are on some surface/valume and a value of 0 otherwise
        self.char_funcs = sym.make_sym_vector("chi",len(self.geometry_list))
        for idx in range(0,len(geometry_list)):
            self.char_funcs[idx] = sym.D(self.kernel,1,k=self.k,source=self.geometry_list[idx])

    def numVectorPotentialDensities(self):
        return 4*len(self.geometry_list)

    def numScalarPotentialDensities(self):
        return 2*len(self.geometry_list)

    def D(self, density_vec, target):
        """
        Double layer potential operator across multiple disjoint objects
        """

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # init output symbolic quantity with zeros
        output = np.zeros((ndim,), dtype=type(density))

        # compute individual double layer potential evaluations at the given
        # density across all the disjoint objects
        for i in range(0,nobj):
            output = output + sym.D(self.kernel, density_vec[:,i],
                k=self.k,qbx_forced_limit="avg",
                source=self.geometry_list[i],target=target)

        # return the output summation
        return output

    def S(self, density_vec, target):
        """
        Double layer potential operator across multiple disjoint objects
        """

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # init output symbolic quantity with zeros
        output = np.zeros((ndim,), dtype=self.stype)

        # compute individual double layer potential evaluations at the given
        # density across all the disjoint objects
        for i in range(0,nobj):
            output = output + sym.S(self.kernel, density_vec[:,i],
                k=self.k, qbx_forced_limit="avg",
                source=self.geometry_list[i], target=target)

        # return the output summation
        return output


    def Dp(self, density_vec, target):
        """
        D' layer potential operator across multiple disjoint objects
        """

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # init output symbolic quantity with zeros
        output = np.zeros((ndim,), dtype=self.stype)

        # compute individual double layer potential evaluations at the given
        # density across all the disjoint objects
        for i in range(0,nobj):
            output = output + sym.Dp(self.kernel, density_vec[:,i],
                k=self.k,qbx_forced_limit="avg",
                source=self.geometry_list[i],target=target)

        # return the output summation
        return output

    def Sp(self, density_vec, target):
        """
        S' layer potential operator across multiple disjoint objects
        """

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # init output symbolic quantity with zeros
        output = np.zeros((ndim,), dtype=self.stype)

        # compute individual double layer potential evaluations at the given
        # density across all the disjoint objects
        for i in range(0,nobj):
            output = output + sym.Sp(self.kernel, density_vec[:,i],
                k=self.k, qbx_forced_limit="avg",
                source=self.geometry_list[i], target=target)

        # return the output summation
        return output


    def phi_operator0(self, phi_densities):
        """
        Integral Equation operator for obtaining scalar potential, `phi`
        """

        # extract the densities needed to solve the system of equations
        sigma       = phi_densities[0]
        V_array     = phi_densities[1:]

        # produce integral equation system
        return sym.join_fields(
                        0.5*sigma + sym.D(self.kernel,sigma,k=self.k,qbx_forced_limit="avg")
                         - 1j*self.k*sym.S(self.kernel,sigma,k=self.k,qbx_forced_limit="avg")
                         + np.dot(V_array,self.char_funcs),
                         sym.integral(ambient_dim=3,dim=2,operand=
                            sym.Dp(self.kernel,sigma,k=self.k,qbx_forced_limit="avg")/self.k
                            + 1j*sigma/2.0 - 1j*sym.Sp(self.kernel,sigma,k=self.k,qbx_forced_limit="avg")
                            )
                         )

    def phi_operator(self, phi_densities):
        """
        Integral Equation operator for obtaining scalar potential, `phi`
        """

        # extract the densities needed to solve the system of equations
        sigma   = phi_densities[:self.nobjs]
        sigma_m = sigma.reshape((1,self.nobjs))
        V       = phi_densities[self.nobjs:]

        # init output matvec vector for the phi density IE
        output = np.zeros((2*self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get nth disjoint object 
            obj_n = self.geometry_list[n]

            # setup IE for evaluation over the nth disjoint object's surface
            output[n] = 0.5*sigma[n] + self.D(sigma_m,obj_n) - 1j*self.k*self.S(sigma_m,obj_n) - V[n]

            # setup equation that integrates some integral operators over the nth surface
            output[self.nobjs + n] = sym.integral(ambient_dim=3,dim=2,
                operand=(self.Dp(sigma_m,target=None)/self.k+ 1j*sigma/2.0 - 1j*self.Sp(sigma_m,target=None)))

        # return the resulting system of IE
        return output


    def phi_rhs(self, phi_inc, gradphi_inc):
        """
        The Right-Hand-Side for the Integral Equation for `phi`
        """

        # get the Q_{j} terms inside RHS expression
        Q = np.zeros((self.nobjs,), dtype=self.stype)
        for i in range(0,self.nobjs):
            Q[i] = -sym.integral(3,2,sym.n_dot(gradphi_inc),where=self.geometry_list[i])

        # return the resulting field
        return sym.join_fields(-phi_inc, Q/self.k)

    def A_operator0(self, A_densities):
        """
        Integral Equation operator for obtaining vector potential, `A`
        """

        # extract the densities needed to solve the system of equations
        rho     = A_densities[0]
        a       = sym.tangential_to_xyz(A_densities[1:3])
        v_array = A_densities[3:]

        # define the normal vector in symbolic form
        n = sym.normal(len(a), None).as_vector()
        r = sym.n_cross(a)

        # define system of integral equations for A
        return sym.join_fields(
            0.5*a + sym.n_cross(sym.S(self.kernel,a,k=self.k,qbx_forced_limit="avg"))
                    -self.k * sym.n_cross(sym.S(self.kernel,n*rho,k=self.k,qbx_forced_limit="avg"))
                    + 1j*(
                        self.k*sym.n_cross(sym.cross(sym.S(self.kernel,n,k=self.k,qbx_forced_limit="avg"),a))
                        + sym.n_cross(sym.grad(3,sym.S(self.kernel,rho,k=self.k,qbx_forced_limit="avg")))
                        ),
            0.5*rho + sym.D(self.kernel,rho,k=self.k,qbx_forced_limit="avg")
                    + 1j*(
                        sym.d_dx(3,sym.S(self.kernel,r[0],k=self.k,qbx_forced_limit="avg"))
                        + sym.d_dy(3,sym.S(self.kernel,r[1],k=self.k,qbx_forced_limit="avg"))
                        + sym.d_dz(3,sym.S(self.kernel,r[2],k=self.k,qbx_forced_limit="avg"))
                        - self.k*sym.S(self.kernel,rho,k=self.k,qbx_forced_limit="avg")
                        ) 
                    + np.dot(v_array,self.char_funcs),
            sym.integral(ambient_dim=3,dim=2,operand=sym.n_dot(sym.curl(sym.S(self.kernel,a,k=self.k,qbx_forced_limit="avg")))
                                                     - self.k * sym.n_dot(sym.S(self.kernel,n*rho,k=self.k,qbx_forced_limit="avg"))
                                                     + 1j*(
                                                             self.k*sym.n_dot(sym.S(self.kernel,sym.n_cross(a),k=self.k,qbx_forced_limit="avg"))
                                                             - rho/2.0 + sym.Sp(self.kernel,rho,k=self.k,qbx_forced_limit="avg")
                                                     )
                         )
            )

    def A_operator(self, A_densities):
        """
        Integral Equation operator for obtaining vector potential, `A`
        """

        # extract the densities needed to solve the system of equations
        v       = A_densities[:self.nobjs]
        rho     = A_densities[self.nobjs:(2*self.nobjs)]
        rho_m   = rho.resize((1,self.nobjs))
        a_loc   = A_densities[2*self.nobjs:]
        a       = np.zeros((3,self.nobjs),dtype=self.stype)
        for n in range(0,self.nobjs):
            a[:,n] = sym.tangential_to_xyz(a_loc[2*n:2*(n+1)],where=self.geometry_list[n])

        # define the normal vector in symbolic form
        n = sym.normal(len(a), None).as_vector()
        r = sym.n_cross(a)

        # init output matvec vector for the phi density IE
        output = np.zeros((5*self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

    def A_rhs(self, A_inc, divA_inc):
        """
        The Right-Hand-Side for the Integral Equation for `A`
        """

        # get the q_array
        q = np.zeros((self.nobjs,), dtype=self.stype)
        for i in range(0,self.nobjs):
            q[i] = -sym.integral(3,2,sym.n_dot(A_inc),where=self.geometry_list[i])

        # define RHS for `A` integral equation system
        return sym.join_fields( -sym.n_cross(A_inc), -divA_inc/self.k, q)


    def scalar_potential_rep(self, phi_densities, qbx_forced_limit=None):
        """
        This method is a representation of the scalar potential, phi,
        based on the density `sigma`.
        """

        # extract the densities needed to solve the system of equations
        sigma       = phi_densities[0]

        # evaluate scalar potential representation
        return sym.D(self.kernel,sigma,k=self.k,qbx_forced_limit=qbx_forced_limit)\
               - 1j*self.k*sym.S(self.kernel,sigma,k=self.k,qbx_forced_limit=qbx_forced_limit)

    def vector_potential_rep(self, A_densities, qbx_forced_limit=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities needed to solve the system of equations
        rho     = A_densities[0]
        a       = sym.tangential_to_xyz(A_densities[1:3])

        # define the normal vector in symbolic form
        n = sym.normal(len(a), None).as_vector()

        # define the vector potential representation
        return sym.curl(sym.S(self.kernel,a,k=self.k,qbx_forced_limit=qbx_forced_limit)) \
               - self.k*sym.S(self.kernel,rho*n,k=self.k,qbx_forced_limit=qbx_forced_limit)\
               + 1j*(
                       self.k*sym.S(self.kernel,sym.n_cross(a),k=self.k,qbx_forced_limit=qbx_forced_limit)
                       + sym.grad(3,sym.S(self.kernel,rho,k=self.k,qbx_forced_limit=qbx_forced_limit))
               )


    def scattered_volume_field(self, phi_densities, A_densities, qbx_forced_limit=None):
        """
        This will return an object of six entries, the first three of which
        represent the electric, and the second three of which represent the
        magnetic field. 

        <NOT TRUE YET>
        This satisfies the time-domain Maxwell's equations
        as verified by :func:`sumpy.point_calculus.frequency_domain_maxwell`.
        """

        # obtain expressions for scalar and vector potentials
        A   = self.vector_potential_rep(A_densities,qbx_forced_limit=qbx_forced_limit)
        phi = self.scalar_potential_rep(phi_densities,qbx_forced_limit=qbx_forced_limit)

        # evaluate the potential form for the electric and magnetic fields
        E_scat = 1j*self.k*A - sym.grad(3, phi)
        H_scat = sym.curl(A)

        # join the fields into a vector
        return sym.join_fields(E_scat, H_scat)

# }}}
