from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018 Christian Howard"

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

# import useful tools/libs
import numpy        as np  # noqa
from pytential      import bind, sym
from collections    import namedtuple
from functools      import partial

# define a few functions based on existing functions
tangential_to_xyz = sym.tangential_to_xyz
xyz_to_tangential = sym.xyz_to_tangential
cse = sym.cse

__doc__ = """
.. autoclass:: DPIEOperator
.. autoclass:: DPIEOperatorEvanescent
"""



# {{{ Decoupled Potential Integral Equation Operator - based on Arxiv paper
class DPIEOperator:
    r"""
    Decoupled Potential Integral Equation operator with PEC boundary
    conditions, defaults as scaled DPIE.

    See https://arxiv.org/abs/1404.0749 for derivation.

    Uses :math:`E(x,t) = Re \lbrace E(x) \exp(-i \omega t) \rbrace` and 
    :math:`H(x,t) = Re \lbrace H(x) \exp(-i \omega t) \rbrace` and solves for 
    the :math:`E(x)`, :math:`H(x)` fields using vector and scalar potentials via
    the Lorenz Gauage. The DPIE formulates the problem purely in terms of the 
    vector and scalar potentials, :math:`\boldsymbol{A}` and :math:`\phi`, 
    and then backs out :math:`E(x)` and :math:`H(x)` via relationships to 
    the vector and scalar potentials.
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

    def num_distinct_objects(self):
        return self.nobjs

    def num_vector_potential_densities(self):
        return 4*len(self.geometry_list)

    def num_vector_potential_densities2(self):
        return 5*len(self.geometry_list)

    def num_scalar_potential_densities(self):
        return 2*len(self.geometry_list)

    def get_vector_domain_list(self):
        """
        Method to return domain list that will be used within the scipy_op method to
        solve the system of discretized integral equations. What is returned should just
        be a list with values that are strings or None.
        """

        # initialize domain list
        domain_list = [None]*self.num_vector_potential_densities()

        # get strings for the actual densities
        for n in range(0,self.nobjs):

            # grab nth location identifier
            location = self.geometry_list[n] + "t"

            # assign domain for nth vector density
            domain_list[2*n] = location
            domain_list[2*n+1] = location

            # assign domain for nth scalar density
            domain_list[2*self.nobjs + n] = location

        # return the domain list
        return domain_list

    def get_scalar_domain_list(self):
        """
        Method to return domain list that will be used within the scipy_op method to
        solve the system of discretized integral equations. What is returned should just
        be a list with values that are strings or None.
        """

        # initialize domain list
        domain_list = [None]*self.num_scalar_potential_densities()

        # get strings for the actual densities
        for n in range(0,self.nobjs):

            # grab nth location identifier
            location = self.geometry_list[n] + "t"

            # assign domain for nth scalar density
            domain_list[n] = location

        # return the domain list
        return domain_list

    def get_subproblem_domain_list(self):
        """
        Method to return domain list that will be used within the scipy_op method to
        solve the system of discretized integral equations. What is returned should just
        be a list with values that are strings or None.
        """

        # initialize domain list
        domain_list = [None]*self.nobjs

        # get strings for the actual densities
        for n in range(0,self.nobjs):

            # grab nth location identifier
            location = self.geometry_list[n] + "t"

            # assign domain for nth scalar density
            domain_list[n] = location

        # return the domain list
        return domain_list


    def _layerpot_op(self,layerpot_op,density_vec, target=None, qfl="avg", k=None, kernel=None, use_laplace=False):
        """
        Generic layer potential operator method that works across all objects within the DPIE model
        """
        if kernel is None:
            kernel = self.kernel

        if k is None:
            k = self.k

        kargs = dict()
        if not use_laplace:
            kargs['k'] = k

        # define a convenient integral operator that functions across the multiple objects
        def int_op(idx):
                return layerpot_op(kernel, density_vec[:,idx],qbx_forced_limit=qfl,source=self.geometry_list[idx],target=target, **kargs)

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # init output symbolic quantity with zeros
        output = np.zeros((ndim,), dtype=self.stype)

        # compute individual double layer potential evaluations at the given
        # density across all the disjoint objects
        for i in range(0,nobj):
            output = output + int_op(i)

        # return the output summation
        if ndim == 1:
            return output[0]
        else:
            return output

    def D(self, density_vec, target=None, qfl="avg", k=None, kernel=None, use_laplace=False):
        """
        Double layer potential operator across multiple disjoint objects
        """
        return self._layerpot_op(layerpot_op=sym.D, density_vec=density_vec, target=target, qfl=qfl, k=k, kernel=kernel, use_laplace=use_laplace)

    def S(self, density_vec, target=None, qfl="avg", k=None, kernel=None, use_laplace=False):
        """
        Single layer potential operator across multiple disjoint objects
        """
        return self._layerpot_op(layerpot_op=sym.S, density_vec=density_vec, target=target, qfl=qfl, k=k, kernel=kernel, use_laplace=use_laplace)


    def Dp(self, density_vec, target=None, qfl="avg", k=None, kernel=None, use_laplace=False):
        """
        D' layer potential operator across multiple disjoint objects
        """
        return self._layerpot_op(layerpot_op=sym.Dp, density_vec=density_vec, target=target, qfl=qfl, k=k, kernel=kernel, use_laplace=use_laplace)

    def Sp(self, density_vec, target=None, qfl="avg", k=None, kernel=None, use_laplace=False):
        """
        S' layer potential operator across multiple disjoint objects
        """
        return self._layerpot_op(layerpot_op=sym.Sp, density_vec=density_vec, target=target, qfl=qfl, k=k, kernel=kernel, use_laplace=use_laplace)

    def n_cross(self, density_vec):
        r"""
        This method is such that an cross(n,a) can operate across vectors
        a and n that are local to a set of disjoint source surfaces. Essentially,
        imagine that :math:`\bar{a} = [a_1, a_2, \cdots, a_m]`, where :math:`a_k` represents a vector density
        defined on the :math:`k^{th}` disjoint object. Also imagine that :math:`bar{n} = [n_1, n_2, \cdots, n_m]`,
        where :math:`n_k` represents a normal that exists on the :math:`k^{th}` disjoint object. The goal, then,
        is to have an operator that does element-wise cross products.. ie:

        .. math::
            \bar{n} \times \bar{a}) = [ \left(n_1 \times a_1\right), ..., \left(n_m \times a_m \right)]
        """

        # specify the sources to be evaluated at
        sources = self.geometry_list

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # assert that the ndim value is 3
        assert ndim == 3

        # init output symbolic quantity with zeros
        output = np.zeros(density_vec.shape, dtype=self.stype)

        # loop through the density and sources to construct the appropriate
        # element-wise cross product operation
        for k in range(0,nobj):
            output[:,k] = sym.n_cross(density_vec[:,k],where=sources[k])

        # return result from element-wise cross product
        return output

    def n_times(self, density_vec):
        r"""
        This method is such that an :math:`\boldsymbol{n} \rho`, for some normal :math:`\boldsymbol{n}` and
        some scalar :math:`\rho` can be done across normals and scalars that exist on multiple surfaces. Essentially,
        imagine that :math:`\bar{\rho} = [\rho_1, \cdots, \rho_m]`, where :math:`\rho_k` represents a scalar density
        defined on the :math:`k^{th}` disjoint object. Also imagine that :math:`bar{n} = [\boldsymbol{n}_1, \cdots, \boldsymbol{n}_m]`,
        where :math:`n_k` represents a normal that exists on the :math:`k^{th}` disjoint object. The goal, then,
        is to have an operator that does element-wise products.. ie:

        .. math::
            \bar{n}\bar{\rho} = [ \left(\boldsymbol{n}_1 \rho_1\right), ..., \left(\boldsymbol{n}_m \rho_m \right)]
        """

        # specify the sources to be evaluated at
        sources = self.geometry_list

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # assert that the ndim value is 1
        assert ndim == 1

        # init output symbolic quantity with zeros
        output = np.zeros((3,nobj), dtype=self.stype)

        # loop through the density and sources to construct the appropriate
        # element-wise cross product operation
        for k in range(0,nobj):
            output[:,k] = sym.normal(3,where=sources[k]).as_vector() * density_vec[0,k]

        # return result from element-wise cross product
        return output

    def _extract_phi_densities(self,phi_densities):
        return (phi_densities[:self.nobjs],phi_densities[:self.nobjs].reshape((1,self.nobjs)),phi_densities[self.nobjs:])

    def _extract_tau_densities(self,tau_densities):
        return (tau_densities,tau_densities.reshape((1,self.nobjs)))

    def _extract_a_densities(self,A_densities):
        a0 = A_densities[:(2*self.nobjs)]
        a = np.zeros((3,self.nobjs),dtype=self.stype)
        rho0 = A_densities[(2*self.nobjs):(3*self.nobjs)]
        rho = rho0.reshape((1,self.nobjs))
        v = A_densities[(3*self.nobjs):]
        for n in range(0,self.nobjs):
            a[:,n] = cse(sym.tangential_to_xyz(a0[2*n:2*(n+1)],where=self.geometry_list[n]),"axyz_{0}".format(n))
        return (a0, a, rho0, rho, v)

    def _L(self, a, rho, where):

        # define some useful common sub expressions
        Sa = cse(self.S(a,where),"Sa_"+where)
        Srho = cse(self.S(rho,where),"Srho_"+where)
        Sn_times_rho = cse(self.S(self.n_times(rho),where),"Sn_times_rho_"+where)
        Sn_cross_a = cse(self.S(self.n_cross(a),where),"Sn_cross_a_"+where)
        Drho = cse(self.D(rho,where),"Drho_"+where)

        return sym.join_fields(
            sym.n_cross(sym.curl(self.S(a,where)) - self.k * Sn_times_rho,where=where),
            Drho)

    def _R(self, a, rho, where):
        # define some useful common sub expressions
        Sa = cse(self.S(a,where),"Sa_"+where)
        Srho = cse(self.S(rho,where),"Srho_"+where)
        Sn_times_rho = cse(self.S(self.n_times(rho),where),"Sn_times_rho_"+where)
        Sn_cross_a = cse(self.S(self.n_cross(a),where),"Sn_cross_a_"+where)
        Drho = cse(self.D(rho,where),"Drho_"+where)

        return sym.join_fields(
            sym.n_cross( self.k * Sn_cross_a + sym.grad(ambient_dim=3,operand=self.S(rho,where)),where=where),
            sym.div(self.S(self.n_cross(a),where)) - self.k * Srho
            )

    def _scaledDPIEs_integral(self, sigma, sigma_n, where):
        qfl="avg"

        return sym.integral(
            ambient_dim=3,
            dim=2,
            operand=(self.Dp(sigma,target=where,qfl=qfl)/self.k + 1j*0.5*sigma_n - 1j*self.Sp(sigma,target=where,qfl=qfl)),
            where=where)

    def _scaledDPIEv_integral(self, **kwargs):
        qfl="avg"

        # grab densities and domain to integrate over
        a = kwargs['a']
        rho = kwargs['rho']
        rho_n = kwargs['rho_n']
        where = kwargs['where']

        # define some useful common sub expressions
        Sa = cse(self.S(a,where),"Sa_"+where)
        Srho = cse(self.S(rho,where),"Srho_"+where)
        Sn_times_rho = cse(self.S(self.n_times(rho),where),"Sn_times_rho_"+where)
        Sn_cross_a = cse(self.S(self.n_cross(a),where),"Sn_cross_a_"+where)
        Drho = cse(self.D(rho,where),"Drho_"+where)

        return sym.integral(
            ambient_dim=3,
            dim=2,
            operand=(
                sym.n_dot( sym.curl(self.S(a,where)),where=where) - self.k*sym.n_dot(Sn_times_rho,where=where) \
                + 1j*(self.k*sym.n_dot(Sn_cross_a,where=where) - 0.5*rho_n + self.Sp(rho,target=where,qfl=qfl))
            ),
            where=where)


    def phi_operator(self, phi_densities):
        """
        Integral Equation operator for obtaining scalar potential, `phi`
        """

        # extract the densities needed to solve the system of equations
        (sigma0,sigma,V) = self._extract_phi_densities(phi_densities)

        # init output matvec vector for the phi density IE
        output = np.zeros((2*self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get nth disjoint object 
            obj_n = self.geometry_list[n]

            # setup IE for evaluation over the nth disjoint object's surface
            output[n] = 0.5*sigma0[n] + self.D(sigma,obj_n) - 1j*self.k*self.S(sigma,obj_n) - V[n]

            # setup equation that integrates some integral operators over the nth surface
            output[self.nobjs + n] = self._scaledDPIEs_integral(sigma,sigma[0,n],where=obj_n)

        # return the resulting system of IE
        return output


    def phi_rhs(self, phi_inc, gradphi_inc):
        """
        The Right-Hand-Side for the Integral Equation for `phi`
        """

        # get the scalar f expression for each object
        f = np.zeros((self.nobjs,), dtype=self.stype)
        for i in range(0,self.nobjs):
            f[i] = -phi_inc[i]

        # get the Q_{j} terms inside RHS expression
        Q = np.zeros((self.nobjs,), dtype=self.stype)
        for i in range(0,self.nobjs):
            Q[i] = -sym.integral(3,2,sym.n_dot(gradphi_inc,where=self.geometry_list[i]),where=self.geometry_list[i])

        # return the resulting field
        return sym.join_fields(f, Q/self.k)


    def a_operator(self, A_densities):
        """
        Integral Equation operator for obtaining vector potential, `A`
        """

        # extract the densities needed to solve the system of equations
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)

        # init output matvec vector for the phi density IE
        output = np.zeros((4*self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get the nth target geometry to have IE solved across
            obj_n = self.geometry_list[n]

            # Compute two IE Operators on a and rho densities
            L = self._L(a, rho, obj_n)
            R = self._R(a, rho, obj_n)

            # generate the set of equations for the vector densities, a, coupled
            # across the various geometries involved
            output[2*n:2*(n+1)] = xyz_to_tangential(0.5*a[:,n] + L[:3] + 1j*R[:3], where=obj_n)

            # generate the set of equations for the scalar densities, rho, coupled
            # across the various geometries involved
            output[(2*self.nobjs + n)] = 0.5*rho[0,n] + L[-1] + 1j*R[-1] - v[n]

            # add the equation that integrates everything out into some constant
            output[3*self.nobjs + n] = self._scaledDPIEv_integral(a=a, rho=rho, rho_n=rho[0,n], where=obj_n)

        # return output equations
        return output

    def a_rhs(self, A_inc, divA_inc):
        """
        The Right-Hand-Side for the Integral Equation for `A`
        """

        # get the q , h , and vec(f) associated with each object
        q = np.zeros((self.nobjs,), dtype=self.stype)
        h = np.zeros((self.nobjs,), dtype=self.stype)
        f = np.zeros((2*self.nobjs,), dtype=self.stype)
        for i in range(0,self.nobjs):
            obj_n = self.geometry_list[i]
            q[i] = -sym.integral(3,2,sym.n_dot(A_inc[3*i:3*(i+1)],where=obj_n),where=obj_n)
            h[i] = -divA_inc[i]
            f[2*i:2*(i+1)] = xyz_to_tangential(-sym.n_cross(A_inc[3*i:3*(i+1)],where=obj_n),where=obj_n)

        # define RHS for `A` integral equation system
        return sym.join_fields( f, h/self.k, q )

    def subproblem_operator(self, tau_densities, alpha = 1j):
        """
        Integral Equation operator for obtaining sub problem solution
        """

        # extract the densities from the sub problem solution
        (tau0, tau) = self._extract_tau_densities(tau_densities)

        # init output matvec vector for the phi density IE
        output = np.zeros((self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get nth disjoint object 
            obj_n = self.geometry_list[n]

            # setup IE for evaluation over the nth disjoint object's surface
            output[n] = 0.5*tau0[n] + self.D(tau,obj_n) - alpha*self.S(tau,obj_n)

        # return the resulting system of IE
        return output

    def subproblem_rhs(self, A_densities):
        """
        Integral Equation RHS for obtaining sub problem solution
        """
        # extract the densities needed to solve the system of equations
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)

        # init output matvec vector for the phi density IE
        output = np.zeros((self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get nth disjoint object 
            obj_n = self.geometry_list[n]

            # setup IE for evaluation over the nth disjoint object's surface
            output[n] = sym.div(self.S(a,target=obj_n,qfl="avg"))

        # return the resulting system of IE
        return output

    def subproblem_rhs_func(self, function):
        """
        Integral Equation RHS for obtaining sub problem solution
        """

        # init output matvec vector for the phi density IE
        output = np.zeros((self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get nth disjoint object 
            obj_n = self.geometry_list[n]

            # setup IE for evaluation over the nth disjoint object's surface
            output[n] = function(where=obj_n)

        # return the resulting system of IE
        return output


    def scalar_potential_rep(self, phi_densities, target=None, qfl=None):
        """
        This method is a representation of the scalar potential, phi,
        based on the density `sigma`.
        """

        # extract the densities needed to solve the system of equations
        (sigma0,sigma,V) = self._extract_phi_densities(phi_densities)

        # evaluate scalar potential representation
        return self.D(sigma,target,qfl=qfl) - (1j*self.k)*self.S(sigma,target,qfl=qfl)

    def scalar_potential_constants(self, phi_densities):
        """
        This method is a representation of the scalar potential, phi,
        based on the density `sigma`.
        """

        # extract the densities needed to solve the system of equations
        (sigma0,sigma,V) = self._extract_phi_densities(phi_densities)

        # evaluate scalar potential representation
        return V

    def grad_scalar_potential_rep(self, phi_densities, target=None, qfl=None):
        """
        This method is a representation of the scalar potential, phi,
        based on the density `sigma`.
        """

        # extract the densities needed to solve the system of equations
        (sigma0,sigma,V) = self._extract_phi_densities(phi_densities)

        # evaluate scalar potential representation
        return sym.grad(3,self.D(sigma,target,qfl=qfl)) - (1j*self.k)*sym.grad(3,self.S(sigma,target,qfl=qfl))

    def vector_potential_rep(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities from the main IE solution
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)

        # define the vector potential representation
        return (sym.curl(self.S(a,target,qfl=qfl)) - self.k*self.S(self.n_times(rho),target,qfl=qfl)) \
        + 1j*(self.k*self.S(self.n_cross(a),target,qfl=qfl) + sym.grad(3,self.S(rho,target,qfl=qfl)))

    def div_vector_potential_rep(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities from the main IE solution
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)

        # define the vector potential representation
        return self.k*( self.D(rho,target,qfl=qfl) \
            + 1j*(sym.div(self.S(self.n_cross(a),target,qfl=qfl)) - self.k * self.S(rho,target,qfl=qfl)))

    def subproblem_rep(self, tau_densities, target=None, alpha = 1j, qfl=None):
        """
        This method is a representation of the scalar potential, phi,
        based on the density `sigma`.
        """

        # extract the densities needed to solve the system of equations
        (tau0, tau) = self._extract_tau_densities(tau_densities)

        # evaluate scalar potential representation
        return self.D(tau,target,qfl=qfl) - alpha*self.S(tau,target,qfl=qfl)

    def scattered_volume_field(self, phi_densities, A_densities, tau_densities, target=None, alpha=1j,qfl=None):
        """
        This will return an object of six entries, the first three of which
        represent the electric, and the second three of which represent the
        magnetic field. 

        This satisfies the time-domain Maxwell's equations
        as verified by :func:`sumpy.point_calculus.frequency_domain_maxwell`.
        """

        # extract the densities needed
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)
        (sigma0,sigma, V) = self._extract_phi_densities(phi_densities)
        (tau0, tau) = self._extract_tau_densities(tau_densities)

        # obtain expressions for scalar and vector potentials
        A   = self.vector_potential_rep(A_densities, target=target)
        phi = self.scalar_potential_rep(phi_densities, target=target)

        # evaluate the potential form for the electric and magnetic fields
        E_scat = 1j*self.k*A - sym.grad(3, self.D(sigma,target,qfl=qfl)) + 1j*self.k*sym.grad(3, self.S(sigma,target,qfl=qfl))
        H_scat = sym.grad(3,operand=(self.D(tau,target,qfl=qfl) - alpha*self.S(tau,target,qfl=qfl))) \
            + (self.k**2) * self.S(a,target,qfl=qfl) \
            - self.k * sym.curl(self.S(self.n_times(rho),target,qfl=qfl)) \
            + 1j*self.k*sym.curl(self.S(self.n_cross(a),target,qfl=qfl))
                

        # join the fields into a vector
        return sym.join_fields(E_scat, H_scat)

# }}}



# {{{ Decoupled Potential Integral Equation Operator - Based on Journal Paper
class DPIEOperatorEvanescent(DPIEOperator):
    r"""
    Decoupled Potential Integral Equation operator with PEC boundary
    conditions, defaults as scaled DPIE.

    See https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21585 for journal paper.

    Uses :math:`E(x,t) = Re \lbrace E(x) \exp(-i \omega t) \rbrace` and 
    :math:`H(x,t) = Re \lbrace H(x) \exp(-i \omega t) \rbrace` and solves for 
    the :math:`E(x)`, :math:`H(x)` fields using vector and scalar potentials via
    the Lorenz Gauage. The DPIE formulates the problem purely in terms of the 
    vector and scalar potentials, :math:`\boldsymbol{A}` and :math:`\phi`, 
    and then backs out :math:`E(x)` and :math:`H(x)` via relationships to 
    the vector and scalar potentials.
    """

    def __init__(self, geometry_list, k=sym.var("k")):
        from sumpy.kernel import HelmholtzKernel
        from sumpy.kernel import LaplaceKernel

        # specify the frequency variable that will be tuned
        self.k          = k
        self.ik         = 1j*k
        self.stype      = type(self.k)

        # specify the 3-D Helmholtz kernel 
        self.kernel     = HelmholtzKernel(3)
        self.kernel_ik  = HelmholtzKernel(3, allow_evanescent=True)
        self.kernel_laplace = LaplaceKernel(3)

        # specify a list of strings representing geometry objects
        self.geometry_list   = geometry_list
        self.nobjs           = len(geometry_list)

    def _eval_all_objects(self, density_vec, int_op, qfl="avg", k=None, kernel=None):
        """
        This private method is so some input integral operator and input density can be used to
        evaluate the set of locations defined by the geometry list
        """
        output = np.zeros(density_vec.shape, dtype=self.stype)
        (ndim, nobj) = density_vec.shape
        for i in range(0,nobj):
            output[:,i] = int_op(density_vec=density_vec, target=self.geometry_list[i], qfl=qfl, k=k, kernel=kernel)
        return output

    def _L(self, a, rho, where):

        # define some useful common sub expressions
        Sn_times_rho = cse(self.S(self.n_times(rho),where),"Sn_times_rho_"+where)
        Drho = cse(self.D(rho,where),"Drho_"+where)

        return sym.join_fields(
            sym.n_cross(sym.curl(self.S(a,where)) - self.k * Sn_times_rho,where=where),
            Drho)

    def _R(self, a, rho, where):
        # define some useful common sub expressions
        Sa_ik_nest = cse(self._eval_all_objects(a, self.S, k=self.ik, kernel=self.kernel_ik), "Sa_ik_nest")
        Srho_ik_nest = cse(self._eval_all_objects(rho,self.S, k=self.ik, kernel=self.kernel_ik),"Srho_ik_nest")
        Srho = cse(self.S(Srho_ik_nest,where),"Srho_"+where)
        Sn_cross_a = cse(self.S(self.n_cross(Sa_ik_nest),where),"Sn_cross_a_"+where)

        return self.k*sym.join_fields(
            sym.n_cross( self.k * Sn_cross_a + sym.grad(ambient_dim=3,operand=self.S(Srho_ik_nest,where)),where=where),
            sym.div(self.S(self.n_cross(Sa_ik_nest),where)) - self.k * Srho
            )

    def _scaledDPIEs_integral(self, sigma, sigma_n, where):
        qfl="avg"

        return sym.integral(
            ambient_dim=3,
            dim=2,
            operand=( (self.Dp(sigma,target=where,qfl=qfl) - self.Dp(sigma,target=where,qfl=qfl,kernel=self.kernel_laplace,use_laplace=True))/self.k + 1j*0.5*sigma_n - 1j*self.Sp(sigma,target=where,qfl=qfl)),
            where=where)

    def _scaledDPIEv_integral(self, **kwargs):
        qfl="avg"

        # grab densities and domain to integrate over
        a = kwargs['a']
        rho = kwargs['rho']
        rho_n = kwargs['rho_n']
        where = kwargs['where']

        # define some useful common sub expressions
        Sa_ik_nest = cse(self._eval_all_objects(a, self.S, k=self.ik, kernel=self.kernel_ik), "Sa_ik_nest")
        Srho_ik = cse(self.S(rho,where,k=self.ik,kernel=self.kernel_ik),"Srho_ik"+where)
        Srho_ik_nest = cse(self._eval_all_objects(rho,self.S, k=self.ik, kernel=self.kernel_ik),"Srho_ik_nest")
        Sn_cross_a = cse(self.S(self.n_cross(Sa_ik_nest),where),"Sn_cross_a_nest_"+where)
        Sn_times_rho = cse(self.S(self.n_times(rho),where),"Sn_times_rho_"+where)

        return sym.integral(
            ambient_dim=3,
            dim=2,
            operand=(
                -self.k*sym.n_dot(Sn_times_rho,where=where) \
                + 1j*self.k*(self.k*sym.n_dot(Sn_cross_a,where=where) - 0.5*Srho_ik + self.Sp(Srho_ik_nest,target=where,qfl=qfl))
            ),
            where=where)

    def vector_potential_rep(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities from the main IE solution
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)

        # define some useful quantities
        Sa_ik_nest = cse(self._eval_all_objects(a, self.S, k=self.ik, kernel=self.kernel_ik), "Sa_ik_nest")
        Srho_ik_nest = cse(self._eval_all_objects(rho,self.S, k=self.ik, kernel=self.kernel_ik),"Srho_ik_nest")

        # define the vector potential representation
        return (sym.curl(self.S(a,target,qfl=qfl)) - self.k*self.S(self.n_times(rho),target,qfl=qfl)) \
        + 1j*self.k*(self.k*self.S(self.n_cross(Sa_ik_nest),target,qfl=qfl) + sym.grad(3,self.S(Srho_ik_nest,target,qfl=qfl)))

    def div_vector_potential_rep(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities from the main IE solution
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)

        # define some useful quantities
        Sa_ik_nest = cse(self._eval_all_objects(a, self.S, k=self.ik, kernel=self.kernel_ik), "Sa_ik_nest")
        Srho_ik_nest = cse(self._eval_all_objects(rho,self.S, k=self.ik, kernel=self.kernel_ik),"Srho_ik_nest")

        # define the vector potential representation
        return self.k*( self.D(rho,target,qfl=qfl) \
            + 1j*self.k*(sym.div(self.S(self.n_cross(Sa_ik_nest),target,qfl=qfl)) - self.k * self.S(Srho_ik_nest,target,qfl=qfl)))

    def scattered_volume_field(self, phi_densities, A_densities, tau_densities, target=None, alpha=1j,qfl=None):
        """
        This will return an object of six entries, the first three of which
        represent the electric, and the second three of which represent the
        magnetic field. 

        This satisfies the time-domain Maxwell's equations
        as verified by :func:`sumpy.point_calculus.frequency_domain_maxwell`.
        """

        # extract the densities needed
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)
        (sigma0,sigma, V) = self._extract_phi_densities(phi_densities)
        (tau0, tau) = self._extract_tau_densities(tau_densities)

        # obtain expressions for scalar and vector potentials
        Sa_ik_nest = self._eval_all_objects(a, self.S, k=self.ik, kernel=self.kernel_ik)
        A   = self.vector_potential_rep(A_densities, target=target)
        phi = self.scalar_potential_rep(phi_densities, target=target)

        # evaluate the potential form for the electric and magnetic fields
        E_scat = 1j*self.k*A - sym.grad(3, self.D(sigma,target,qfl=qfl)) + 1j*self.k*sym.grad(3, self.S(sigma,target,qfl=qfl))
        H_scat = sym.grad(3,operand=(self.D(tau,target,qfl=qfl) - alpha*self.S(tau,target,qfl=qfl))) \
            + (self.k**2) * self.S(a,target,qfl=qfl) \
            - self.k * sym.curl(self.S(self.n_times(rho),target,qfl=qfl)) \
            + 1j*(self.k**2)*sym.curl(self.S(self.n_cross(Sa_ik_nest),target,qfl=qfl))
                

        # join the fields into a vector
        return sym.join_fields(E_scat, H_scat)

# }}}
