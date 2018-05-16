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
"""



# {{{ Decoupled Potential Integral Equation Operator
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

    def get_vector_domain_list2(self):
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
            domain_list[3*n] = location
            domain_list[3*n+1] = location
            domain_list[3*n+3] = location

            # assign domain for nth scalar density
            domain_list[3*self.nobjs + n] = location

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


    def D(self, density_vec, target=None, qfl="avg"):
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
            output = output + sym.D(self.kernel, density_vec[:,i],
                k=self.k,qbx_forced_limit=qfl,
                source=self.geometry_list[i],target=target)

        # return the output summation
        if ndim == 1:
            return output[0]
        else:
            return output

    def S(self, density_vec, target=None, qfl="avg"):
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
                k=self.k, qbx_forced_limit=qfl,
                source=self.geometry_list[i], target=target)

        # return the output summation
        if ndim == 1:
            return output[0]
        else:
            return output


    def Dp(self, density_vec, target=None, qfl="avg"):
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
                k=self.k,qbx_forced_limit=qfl,
                source=self.geometry_list[i],target=target)

        # return the output summation
        if ndim == 1:
            return output[0]
        else:
            return output

    def Sp(self, density_vec, target=None, qfl="avg"):
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
                k=self.k, qbx_forced_limit=qfl,
                source=self.geometry_list[i], target=target)

        # return the output summation
        if ndim == 1:
            return output[0]
        else:
            return output

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
            a[:,n] = sym.tangential_to_xyz(a0[2*n:2*(n+1)],where=self.geometry_list[n])
        return (a0, a, rho0, rho, v)

    def _extract_a_densities2(self,A_densities):
        a0 = A_densities[:(3*self.nobjs)]
        a = a0.reshape((3,self.nobjs))
        rho0 = A_densities[(3*self.nobjs):(4*self.nobjs)]
        rho = rho0.reshape((1,self.nobjs))
        v = A_densities[(4*self.nobjs):]
        return (a0, a, rho0, rho, v)

    def _L(self, a, rho, where):
        return sym.join_fields(
            sym.n_cross(self.S(a,where) - self.k * self.S(self.n_times(rho),where),where=where),
            self.D(rho,where))

    def _R(self, a, rho, where):
        return sym.join_fields(
            sym.n_cross( self.k * self.S(self.n_cross(a),where) + sym.grad(ambient_dim=3,operand=self.S(rho,where)),where=where),
            sym.div(self.S(self.n_cross(a),where)) - self.k * self.S(rho,where)
            )

    def _scaledDPIEs_integral(self, sigma, sigma_n, where):
        qfl="avg"
        return sym.integral(
            ambient_dim=3,
            dim=2,
            operand=(self.Dp(sigma,target=where,qfl=qfl)/self.k + 1j*0.5*sigma_n - 1j*self.Sp(sigma,target=where,qfl=qfl)),
            where=where)

    def _scaledDPIEv_integral(self, a, rho, rho_n, where):
        qfl="avg"
        return sym.integral(
            ambient_dim=3,
            dim=2,
            operand=(
                sym.n_dot(sym.curl(self.S(a,target=where,qfl=qfl)),where=where) - self.k*sym.n_dot(self.S(self.n_times(rho),target=where,qfl=qfl),where=where) \
                + 1j*(self.k*sym.n_dot(self.S(self.n_cross(a),target=where,qfl=qfl),where=where) - 0.5*rho_n + self.Sp(rho,target=where,qfl=qfl))
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
            output[3*self.nobjs + n] = self._scaledDPIEv_integral(a, rho, rho[0,n], where=obj_n)

        # return output equations
        return output

    def a_operator2(self, A_densities):
        """
        Integral Equation operator for obtaining vector potential, `A`
        """

        # extract the densities needed to solve the system of equations
        (a0, a, rho0, rho, v) = self._extract_a_densities2(A_densities)

        # init output matvec vector for the phi density IE
        output = np.zeros((5*self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get the nth target geometry to have IE solved across
            obj_n = self.geometry_list[n]

            # Compute two IE Operators on a and rho densities
            L = self._L(a, rho, obj_n)
            R = self._R(a, rho, obj_n)

            # generate the set of equations for the vector densities, a, coupled
            # across the various geometries involved
            output[3*n:3*(n+1)] = 0.5*a[:,n] + L[:3] + 1j*R[:3]

            # generate the set of equations for the scalar densities, rho, coupled
            # across the various geometries involved
            output[(3*self.nobjs + n)] = 0.5*rho[0,n] + L[-1] + 1j*R[-1] - v[n]

            # add the equation that integrates everything out into some constant
            output[4*self.nobjs + n] = self._scaledDPIEv_integral(a, rho, rho[0,n], where=obj_n)

        # return output equations
        return output

    def a_operator0(self, A_densities):
        """
        Integral Equation operator for obtaining vector potential, `A`
        """

        # get object this will be working on
        obj = self.geometry_list[0]

        # extract the densities needed to solve the system of equations
        a = sym.tangential_to_xyz(A_densities[:(2*self.nobjs)],where=obj)
        a_n = a.reshape((3,1))
        rho = A_densities[(2*self.nobjs):(3*self.nobjs)][0]
        #rho_n = rho.reshape((1,1))
        v = A_densities[(3*self.nobjs):]

        # init output matvec vector for the phi density IE
        output = np.zeros((4,), dtype=self.stype)

        # Compute useful sub-density expressions
        n_times_rho = (sym.normal(3,where=obj).as_vector() * rho)
        n_cross_a = sym.n_cross(a,where=obj)

        # generate the set of equations for the vector densities, a, coupled
        # across the various geometries involved
        # a_lhs = 0.5*a \
        # + sym.n_cross(self.S(a_n,obj) \
        #     - self.k*self.S(n_times_rho,obj) \
        #     + 1j*(
        #         self.k*self.S(n_cross_a,obj) + sym.grad(3,self.S(rho_n,obj))
        #         ),
        #     where=obj)
        a_lhs = 0.5*a \
        + sym.n_cross(sym.S(self.kernel, a, k=self.k, qbx_forced_limit="avg",source=obj, target=obj) \
            - self.k*sym.S(self.kernel, n_times_rho, k=self.k, qbx_forced_limit="avg",source=obj, target=obj) \
            + 1j*(
                self.k*sym.S(self.kernel, n_cross_a, k=self.k, qbx_forced_limit="avg",source=obj, target=obj) \
                + sym.grad(3,sym.S(self.kernel, rho, k=self.k, qbx_forced_limit="avg",source=obj, target=obj))
                ),
            where=obj)
        output[:2] = xyz_to_tangential(a_lhs, where=obj)

        # generate the set of equations for the scalar densities, rho, coupled
        # across the various geometries involved
        # output[2] = 0.5*rho[0] \
        # + self.D(rho_n,obj) \
        # + 1j*(sym.div(self.S(n_cross_a,obj)) \
        #     - self.k*self.S(rho_n,obj)) \
        # - v[0]

        output[2] = 0.5*rho \
        + sym.D(self.kernel, rho,k=self.k, qbx_forced_limit="avg",source=obj,target=obj) \
        + 1j*(sym.div(sym.S(self.kernel, n_cross_a, k=self.k, qbx_forced_limit="avg",source=obj, target=obj)) \
            - self.k*sym.S(self.kernel, rho, k=self.k, qbx_forced_limit="avg",source=obj, target=obj)) \
        - v[0]

        # add the equation that integrates everything out into some constant
        output[3] = 0

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

    def subproblem_rhs2(self, function):
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
        return sym.curl(self.S(a,target,qfl=qfl)) - self.k*self.S(self.n_times(rho),target,qfl=qfl) \
        + 1j*(self.k*self.S(self.n_cross(a),target,qfl=qfl) + sym.grad(3,self.S(rho,target,qfl=qfl)))

    def div_vector_potential_rep(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities from the main IE solution
        (a0, a, rho0, rho, v) = self._extract_a_densities(A_densities)

        # define the vector potential representation
        return self.k*(self.D(self.n_times(rho),target,qfl=qfl) \
            + 1j*(sym.div(self.S(self.n_cross(a),target,qfl=qfl)) - self.k * self.S(rho,target,qfl=qfl)))

    def vector_potential_rep0(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # get object this will be working on
        obj = self.geometry_list[0]

        # extract the densities needed to solve the system of equations
        a = sym.tangential_to_xyz(A_densities[:(2*self.nobjs)],where=obj)
        a_n = a.reshape((3,1))
        rho = A_densities[(2*self.nobjs):(3*self.nobjs)][0]
        #rho_n = rho.reshape((1,1))
        v = A_densities[(3*self.nobjs):]

        # Compute useful sub-density expressions
        n_times_rho = (sym.normal(3,where=obj).as_vector() * rho)
        n_cross_a = sym.n_cross(a,where=obj)

        # define the vector potential representation
        # return sym.curl(self.S(a_n,target,qfl=qfl)) - self.k*self.S(n_times_rho,target,qfl=qfl) \
        # + 1j*(self.k*self.S(n_cross_a,target,qfl=qfl) + sym.grad(3,self.S(rho_n,target,qfl=qfl)))

        return sym.curl(sym.S(self.kernel, a, k=self.k, qbx_forced_limit=qfl, source=obj, target=target)) \
        - self.k*sym.S(self.kernel, n_times_rho, k=self.k, qbx_forced_limit=qfl, source=obj, target=target) \
        + 1j*(self.k*sym.S(self.kernel, n_cross_a, k=self.k, qbx_forced_limit=qfl, source=obj, target=target) \
            + sym.grad(3,sym.S(self.kernel, rho, k=self.k, qbx_forced_limit=qfl, source=obj, target=target)))

    def div_vector_potential_rep0(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # get object this will be working on
        obj = self.geometry_list[0]

        # extract the densities needed to solve the system of equations
        a = sym.tangential_to_xyz(A_densities[:(2*self.nobjs)],where=obj)
        a_n = a.reshape((3,1))
        rho = A_densities[(2*self.nobjs):(3*self.nobjs)][0]
        #rho_n = rho.reshape((1,1))
        v = A_densities[(3*self.nobjs):]

        # Compute useful sub-density expressions
        n_times_rho = (sym.normal(3,where=obj).as_vector() * rho)
        n_cross_a = sym.n_cross(a,where=obj)

        # define the vector potential representation
        return self.k*(sym.D(self.kernel, n_times_rho, k=self.k, qbx_forced_limit=qfl, source=obj, target=target) \
            + 1j*(sym.div(sym.S(self.kernel, n_cross_a, k=self.k, qbx_forced_limit=qfl, source=obj, target=target)) \
                - self.k * sym.S(self.kernel, rho, k=self.k, qbx_forced_limit=qfl, source=obj, target=target)))

    def vector_potential_rep2(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities from the main IE solution
        (a0, a, rho0, rho, v) = self._extract_a_densities2(A_densities)

        # define the vector potential representation
        return sym.curl(self.S(a,target,qfl=qfl)) - self.k*self.S(self.n_times(rho),target,qfl=qfl) \
        + 1j*(self.k*self.S(self.n_cross(a),target,qfl=qfl) + sym.grad(3,self.S(rho,target,qfl=qfl)))

    def div_vector_potential_rep2(self, A_densities, target=None, qfl=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities from the main IE solution
        (a0, a, rho0, rho, v) = self._extract_a_densities2(A_densities)

        # define the vector potential representation
        return self.k*(self.D(self.n_times(rho),target,qfl=qfl) \
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

        <NOT TRUE YET>
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
