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
from pytential      import sym
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

        # create the characteristic functions that give a value of
        # 1 when we are on some surface/valume and a value of 0 otherwise
        self.char_funcs = sym.make_sym_vector("chi",len(self.geometry_list))
        for idx in range(0,len(geometry_list)):
            self.char_funcs[idx] = sym.D(self.kernel, 1, k=self.k,source=self.geometry_list[idx])

    def num_vector_potential_densities(self):
        return 4*len(self.geometry_list)

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
            location                            = self.geometry_list[n] + "t"

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
            location                            = self.geometry_list[n] + "t"

            # assign domain for nth scalar density
            domain_list[n]                      = location

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

    def n_cross_multi(self, density_vec, sources):
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

    def n_times_multi(self, density_vec, sources):
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

    def n_cross(self, density_vec, where=None):
        r"""
        This method is so, given a single surface identifier, we can compute the cross product of a normal
        on this surface for a set of vectors represented as columns of some matrix
        :math:`\bar{a} = [a_1, a_2, \cdots, a_m]`. The goal, then, is to perform the following, given 
        some normal :math:`\hat{n}`:

        .. math::
            \hat{n} \times \bar{a}) = [ \left(\hat{n} \times a_1\right), ..., \left(\hat{n} \times a_m \right)]
        """

        # get the shape of density_vec
        (ndim, nobj) = density_vec.shape

        # assert that the ndim value is 1
        assert ndim == 3

        # init output symbolic quantity with zeros
        output = np.zeros(density_vec.shape, dtype=self.stype)

        # loop through the density and sources to construct the appropriate
        # element-wise cross product operation
        for k in range(0,nobj):
            output[:,k] = sym.n_cross(density_vec[:,k],where=where)

        # return result from element-wise cross product
        return output


    def phi_operator(self, phi_densities):
        """
        Integral Equation operator for obtaining scalar potential, `phi`
        """

        # extract the densities needed to solve the system of equations
        sigma   = phi_densities[:self.nobjs]
        sigma_m = sigma.reshape((1,self.nobjs))

        # extract the scalar quantities, { V_j }, that remove the nullspace
        V = phi_densities[self.nobjs:]

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
                operand=(self.Dp(sigma_m,target=obj_n)/self.k+ 1j*sigma[n]/2.0 - 1j*self.Sp(sigma_m,target=obj_n)),\
                where=obj_n)

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
        a_loc   = A_densities[:(2*self.nobjs)]
        a       = np.zeros((3,self.nobjs),dtype=self.stype)
        rho     = A_densities[(2*self.nobjs):(3*self.nobjs)]
        rho_m   = rho.reshape((1,self.nobjs))
        v       = A_densities[(3*self.nobjs):]
        for n in range(0,self.nobjs):
            a[:,n] = sym.tangential_to_xyz(a_loc[2*n:2*(n+1)],where=self.geometry_list[n])

        # init output matvec vector for the phi density IE
        output = np.zeros((4*self.nobjs,), dtype=self.stype)

        # produce integral equation system over each disjoint object
        for n in range(0,self.nobjs):

            # get the nth target geometry to have IE solved across
            obj_n = self.geometry_list[n]

            # generate the set of equations for the vector densities, a, coupled
            # across the various geometries involved
            output[2*n:2*(n+1)] = xyz_to_tangential((0.5*a[:,n] + sym.n_cross(self.S(a,obj_n),where=obj_n) \
                                             + -self.k * sym.n_cross(self.S(self.n_times_multi(rho_m,self.geometry_list),obj_n),where=obj_n) \
                                             + 1j*( self.k* sym.n_cross(self.S(self.n_cross_multi(a,self.geometry_list),obj_n),where=obj_n) + \
                                                    sym.n_cross(sym.grad(ambient_dim=3,operand=self.S(rho_m,obj_n)),where=obj_n)
                                                )), where=obj_n)

            # generate the set of equations for the scalar densities, rho, coupled
            # across the various geometries involved
            output[(2*self.nobjs + n)] = 0.5*rho[n] + self.D(rho_m,obj_n) \
                                            + 1j*(  sym.div(self.S(self.n_cross_multi(a,self.geometry_list),obj_n)) \
                                                    -self.k*self.S(rho_m,target=obj_n)
                                                )\
                                            + v[n]

            # add the equation that integrates everything out into some constant
            output[3*self.nobjs + n] = sym.integral(ambient_dim=3,dim=2,\
                operand=(sym.n_dot(sym.curl(self.S(a,target=obj_n)),where=obj_n) - self.k*sym.n_dot(self.S(self.n_times_multi(rho_m,self.geometry_list),target=obj_n),where=obj_n) + \
                    1j*(self.k*sym.n_dot(self.S(self.n_cross_multi(a,self.geometry_list),target=obj_n),where=obj_n) - 0.5*rho[n] + self.Sp(rho_m,target=obj_n))),\
                where=obj_n)

        # print something to help with debugging
        #print(sym.pretty(output))

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
            q[i] = -sym.integral(3,2,sym.n_dot(A_inc[3*i:3*(i+1)],where=self.geometry_list[i]),where=self.geometry_list[i])
            h[i] = -divA_inc[i]/self.k
            f[2*i:2*(i+1)] = xyz_to_tangential(-sym.n_cross(A_inc[3*i:3*(i+1)],where=self.geometry_list[i]),where=self.geometry_list[i])

        # define RHS for `A` integral equation system
        return sym.join_fields( f, h, q)

    def scalar_potential_rep(self, phi_densities, target=None):
        """
        This method is a representation of the scalar potential, phi,
        based on the density `sigma`.
        """

        # extract the densities needed to solve the system of equations
        sigma   = phi_densities[:self.nobjs]
        sigma_m = sigma.reshape((1,self.nobjs))

        # evaluate scalar potential representation
        return self.D(sigma_m,target,qfl=None) - 1j*self.k*self.S(sigma_m,target,qfl=None)

    def vector_potential_rep(self, A_densities, target=None):
        """
        This method is a representation of the vector potential, phi,
        based on the vector density `a` and scalar density `rho`
        """

        # extract the densities needed to solve the system of equations
        a_loc   = A_densities[:(2*self.nobjs)]
        a       = np.zeros((3,self.nobjs),dtype=self.stype)
        rho     = A_densities[(2*self.nobjs):(3*self.nobjs)]
        rho_m   = rho.reshape((1,self.nobjs))
        for n in range(0,self.nobjs):
            a[:,n] = sym.tangential_to_xyz(a_loc[2*n:2*(n+1)],where=self.geometry_list[n])

        # define the vector potential representation
        return sym.curl(self.S(a,target,qfl=None)) - self.k*self.S(self.n_times_multi(rho_m,self.geometry_list),target,qfl=None) \
                + 1j*(self.k*self.S(self.n_cross_multi(a,self.geometry_list),target,qfl=None) + sym.grad(3,self.S(rho_m,target,qfl=None)))

    def scattered_volume_field(self, phi_densities, A_densities, target=None):
        """
        This will return an object of six entries, the first three of which
        represent the electric, and the second three of which represent the
        magnetic field. 

        <NOT TRUE YET>
        This satisfies the time-domain Maxwell's equations
        as verified by :func:`sumpy.point_calculus.frequency_domain_maxwell`.
        """

        # extract the densities needed to solve the system of equations
        sigma   = phi_densities[:self.nobjs]
        sigma_m = sigma.reshape((1,self.nobjs))

        # extract the densities needed to solve the system of equations
        a_loc   = A_densities[:(2*self.nobjs)]
        a       = np.zeros((3,self.nobjs),dtype=self.stype)
        rho     = A_densities[(2*self.nobjs):(3*self.nobjs)]
        rho_m   = rho.reshape((1,self.nobjs))
        for n in range(0,self.nobjs):
            a[:,n] = sym.tangential_to_xyz(a_loc[2*n:2*(n+1)],where=self.geometry_list[n])

        # obtain expressions for scalar and vector potentials
        A   = self.vector_potential_rep(A_densities, target=target)
        phi = self.scalar_potential_rep(phi_densities, target=target)

        # evaluate the potential form for the electric and magnetic fields
        E_scat = 1j*self.k*A - sym.grad(3, self.D(sigma_m,target,qfl=None)) + 1j*self.k*sym.grad(3,self.S(sigma_m,target,qfl=None))
        H_scat = sym.curl(sym.curl(self.S(a,target,qfl=None))) - self.k*sym.curl(self.S(self.n_times_multi(rho_m,self.geometry_list),target,qfl=None)) \
                + 1j*(self.k*sym.curl(self.S(self.n_cross_multi(a,self.geometry_list),target,qfl=None)) )

        # join the fields into a vector
        return sym.join_fields(E_scat, H_scat)

# }}}
