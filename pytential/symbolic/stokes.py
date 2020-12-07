__copyright__ = "Copyright (C) 2017 Natalie Beams"

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

import numpy as np

from pytential import sym
from sumpy.kernel import (StokesletKernel, StressletKernel, LaplaceKernel,
    AxisTargetDerivative, BiharmonicKernel)


class StokesletWrapperMixin:
    """A base class for StokesletWrapper and StressletWrapper

    """
    def get_int_g(self, idx, density, mu_sym, qbx_forced_limit,
            deriv_dirs):

        """
        Returns the Integral of the Stokeslet/Stresslet kernel given by `idx`
        and its derivatives. If `use_biharmonic` is set, Biharmonic Kernel
        and its derivatives will be used instead of Stokeslet/Stresslet
        """

        def create_int_g(knl, deriv_dirs, **kwargs):
            for deriv_dir in deriv_dirs:
                knl = AxisTargetDerivative(deriv_dir, knl)

            res = sym.S(knl, density,
                    qbx_forced_limit=qbx_forced_limit, **kwargs)
            return res

        if not self.use_biharmonic:
            knl = self.kernel_dict[idx]
            return create_int_g(knl, deriv_dirs, mu=mu_sym)

        deriv_relation = self.deriv_relation_dict[idx]
        from pytential.symbolic.primitives import as_dofdesc, DEFAULT_SOURCE
        const = deriv_relation[0]
        const *= sym.integral(self.dim, self.dim-1, density,
                              dofdesc=as_dofdesc(DEFAULT_SOURCE))

        result = const
        for mi, coeff in deriv_relation[1]:
            new_deriv_dirs = list(deriv_dirs)
            for idx, val in enumerate(mi):
                new_deriv_dirs.extend([idx]*val)
            result += create_int_g(self.base_kernel, new_deriv_dirs) * coeff

        return result


class StokesletWrapper(StokesletWrapperMixin):
    """ Wrapper class for the Stokeslet kernel.

    This class is meant to shield the user from the messiness of writing
    out every term in the expansion of the double-indexed Stokeslet kernel
    applied to the density vector.  The object is created
    to do some of the set-up and bookkeeping once, rather than every
    time we want to create a symbolic expression based on the kernel -- say,
    once when we solve for the density, and once when we want a symbolic
    representation for the solution, for example.

    The apply() function returns the integral expressions needed for
    the vector velocity resulting from convolution with the vector density,
    and is meant to work similarly to
    calling S() (which is IntG()).

    Similar functions are available for other useful things related to
    the flow: apply_pressure, apply_derivative (target derivative),
    apply_stress (applies symmetric viscous stress tensor in
    the requested direction).

   .. attribute:: kernel_dict

       The dictionary allows us to exploit symmetry -- that
       StokesletKernel(icomp=0, jcomp=1) is identical to
       StokesletKernel(icomp=1, jcomp=0) -- and avoid creating multiple expansions
       for the same kernel in a different ordering.
    """

    def __init__(self, dim=None, use_biharmonic=True):
        self.use_biharmonic = use_biharmonic
        self.dim = dim
        if not (dim == 3 or dim == 2):
            raise ValueError("unsupported dimension given to StokesletWrapper")

        self.kernel_dict = {}

        self.base_kernel = BiharmonicKernel(dim=dim)

        for i in range(dim):
            for j in range(i, dim):
                self.kernel_dict[(i, j)] = StokesletKernel(dim=dim, icomp=i,
                                                           jcomp=j)

        for i in range(dim):
            for j in range(i):
                self.kernel_dict[(i, j)] = self.kernel_dict[(j, i)]

        if self.use_biharmonic:
            from pytential.symbolic.pde.system_utils import get_deriv_relation
            results = get_deriv_relation(list(self.kernel_dict.values()),
                                         self.base_kernel, tol=1e-10, order=2)
            self.deriv_relation_dict = {}
            for deriv_eq, (idx, knl) in zip(results, self.kernel_dict.items()):
                self.deriv_relation_dict[idx] = deriv_eq

    def apply(self, density_vec_sym, mu_sym, qbx_forced_limit):
        """ Symbolic expressions for integrating Stokeslet kernel

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the dyadic Stokeslet kernel with
        variable *density_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector
        :arg mu_sym: a symbolic variable for the viscosity
        :arg qbx_forced_limit: the qbx_forced_limit argument to be passed on
            to IntG.  +/-1 for exterior/interior one-sided boundary limit,
            +/-2 for exterior/interior off-boundary evaluation, and 'avg'
            for the average of the two one-sided boundary limits.
        """

        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp in range(self.dim):
            for i in range(self.dim):
                sym_expr[comp] += self.get_int_g((comp, i),
                        density_vec_sym[i], mu_sym, qbx_forced_limit, deriv_dirs=[])

        return sym_expr

    def apply_pressure(self, density_vec_sym, mu_sym, qbx_forced_limit):
        """ Symbolic expression for pressure field associated with the Stokeslet"""

        from pytential.symbolic.mappers import DerivativeTaker
        kernel = LaplaceKernel(dim=self.dim)
        sym_expr = 0

        for i in range(self.dim):
            sym_expr += (DerivativeTaker(i).map_int_g(
                         sym.S(kernel, density_vec_sym[i],
                         qbx_forced_limit=qbx_forced_limit)))

        return sym_expr

    def apply_derivative(self, deriv_dir, density_vec_sym,
                             mu_sym, qbx_forced_limit):
        """ Symbolic derivative of velocity from Stokeslet.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* derivative of the
        dyadic Stokeslet kernel (wrt target, not source) with
        variable *density_vec_sym*.

        :arg deriv_dir: which derivative we want: 0, 1, or 2 for x, y, z
        :arg density_vec_sym: a symbolic vector variable for the density vector
        :arg mu_sym: a symbolic variable for the viscosity
        :arg qbx_forced_limit: the qbx_forced_limit argument to be passed
            on to IntG.  +/-1 for exterior/interior one-sided boundary limit,
            +/-2 for exterior/interior off-boundary evaluation, and 'avg'
            for the average of the two one-sided boundary limits.
        """

        sym_expr = self.apply(density_vec_sym, mu_sym, qbx_forced_limit)

        for comp in range(self.dim):
            for i in range(self.dim):
                sym_expr[comp] += self.get_int_g((comp, i),
                        density_vec_sym[i], mu_sym, qbx_forced_limit,
                        deriv_dirs=[deriv_dir])

        return sym_expr

    def apply_stress(self, density_vec_sym, dir_vec_sym,
                        mu_sym, qbx_forced_limit):
        """ Symbolic expression for viscous stress applied to direction

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress:
        -pressure * I + mu * ( grad U + (grad U).T)),
        applied in the direction of *dir_vec_sym*.

        Note that this computation is very similar to computing
        a double-layer potential with the stresslet kernel.
        The difference is that here the direction vector is the
        direction applied to the stress tensor and is applied
        outside of the integration, whereas the stresslet calculation
        uses the normal vectors at every source point.  As such, the
        length of the argument passed in for the stresslet velocity
        calculation (after binding) is the same length as the number
        of source points/nodes; when calling this routine, the number
        of direction vectors should be the same as the number of targets.

        :arg density_vec_sym: a symbolic vector variable for the density vector
        :arg dir_vec_sym: a symbolic vector for the application direction
        :arg mu_sym: a symbolic variable for the viscosity
        :arg qbx_forced_limit: the qbx_forced_limit argument to be passed
            on to IntG.  +/-1 for exterior/interior one-sided boundary limit,
            +/-2 for exterior/interior off-boundary evaluation, and 'avg'
            for the average of the two one-sided boundary limits.
        """

        sym_expr = np.zeros((self.dim,), dtype=object)
        stresslet_obj = StressletWrapper(dim=self.dim,
                                         use_biharmonic=self.use_biharmonic)

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += dir_vec_sym[i] * \
                        stresslet_obj.get_int_g((comp, i, j),
                        density_vec_sym[j],
                        mu_sym, qbx_forced_limit, deriv_dirs=[])

        return sym_expr


class StressletWrapper(StokesletWrapperMixin):
    """ Wrapper class for the Stresslet kernel.

    This class is meant to shield the user from the messiness of writing
    out every term in the expansion of the triple-indexed Stresslet
    kernel applied to both a normal vector and the density vector.
    The object is created to do some of the set-up and bookkeeping once,
    rather than every time we want to create a symbolic expression based
    on the kernel -- say, once when we solve for the density, and once when
    we want a symbolic representation for the solution, for example.

    The apply() function returns the integral expressions needed for convolving
    the kernel with a vector density, and is meant to work similarly to
    calling S() (which is IntG()).

    Similar functions are available for other useful things related to
    the flow: apply_pressure, apply_derivative (target derivative),
    apply_stress (applies symmetric viscous stress tensor in
    the requested direction).

    .. attribute:: kernel_dict

        The dictionary allows us to exploit symmetry -- that
        StressletKernel(icomp=0, jcomp=1, kcomp=2) is identical to
        StressletKernel(icomp=1, jcomp=2, kcomp=0) -- and avoid creating
        multiple expansions for the same kernel in a different ordering.

    """

    def __init__(self, dim=None, use_biharmonic=True):
        self.use_biharmonic = use_biharmonic
        self.dim = dim
        if not (dim == 3 or dim == 2):
            raise ValueError("unsupported dimension given to StokesletWrapper")

        self.kernel_dict = {}

        self.base_kernel = BiharmonicKernel(dim=dim)

        for i in range(dim):
            for j in range(i, dim):
                for k in range(j, dim):
                    self.kernel_dict[(i, j, k)] = StressletKernel(dim=dim, icomp=i,
                                                               jcomp=j, kcomp=k)

        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    if (i, j, k) in self.kernel_dict:
                        continue
                    s = tuple(sorted([i, j, k]))
                    self.kernel_dict[(i, j, k)] = self.kernel_dict[s]

        if self.use_biharmonic:
            from pytential.symbolic.pde.system_utils import get_deriv_relation
            results = get_deriv_relation(list(self.kernel_dict.values()),
                                         self.base_kernel, tol=1e-10, order=3)
            self.deriv_relation_dict = {}
            for deriv_eq, (idx, knl) in zip(results, self.kernel_dict.items()):
                self.deriv_relation_dict[idx] = deriv_eq

    def apply(self, density_vec_sym, dir_vec_sym, mu_sym, qbx_forced_limit):
        """ Symbolic expressions for integrating stresslet kernel

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the dyadic stresslet kernel with
        variable *density_vec_sym* and source direction vectors *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector
        :arg dir_vec_sym: a symbolic vector variable for the direction vector
        :arg mu_sym: a symbolic variable for the viscosity
        :arg qbx_forced_limit: the qbx_forced_limit argument to be passed
            on to IntG.  +/-1 for exterior/interior one-sided boundary limit,
            +/-2 for exterior/interior off-boundary evaluation, and 'avg'
            for the average of the two one-sided boundary limits.
        """

        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += self.get_int_g((comp, i, j),
                        dir_vec_sym[i] * density_vec_sym[j],
                        mu_sym, qbx_forced_limit, deriv_dirs=[])

        return sym_expr

    def apply_pressure(self, density_vec_sym, dir_vec_sym, mu_sym, qbx_forced_limit):
        """ Symbolic expression for pressure field associated with the stresslet"""

        import itertools
        from pytential.symbolic.mappers import DerivativeTaker
        kernel = LaplaceKernel(dim=self.dim)

        factor = (2. * mu_sym)

        sym_expr = 0

        for i, j in itertools.product(range(self.dim), range(self.dim)):
            sym_expr += factor * DerivativeTaker(i).map_int_g(
                                   DerivativeTaker(j).map_int_g(
                                       sym.S(kernel,
                                             density_vec_sym[i] * dir_vec_sym[j],
                                             qbx_forced_limit=qbx_forced_limit)))

        return sym_expr

    def apply_derivative(self, deriv_dir, density_vec_sym, dir_vec_sym,
                             mu_sym, qbx_forced_limit):
        """ Symbolic derivative of velocity from stresslet.

        Returns an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* derivative of the
        dyadic stresslet kernel (wrt target, not source) with
        variable *density_vec_sym* and source direction vectors *dir_vec_sym*.

        :arg deriv_dir: which derivative we want: 0, 1, or 2 for x, y, z
        :arg density_vec_sym: a symbolic vector variable for the density vector
        :arg dir_vec_sym: a symbolic vector variable for the normal direction
        :arg mu_sym: a symbolic variable for the viscosity
        :arg qbx_forced_limit: the qbx_forced_limit argument to be passed
            on to IntG.  +/-1 for exterior/interior one-sided boundary limit,
            +/-2 for exterior/interior off-boundary evaluation, and 'avg'
            for the average of the two one-sided boundary limits.
        """

        sym_expr = np.zeros((self.dim,), dtype=object)

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += self.get_int_g((comp, i, j),
                        dir_vec_sym[i] * density_vec_sym[j],
                        mu_sym, qbx_forced_limit, deriv_dirs=[deriv_dir])

        return sym_expr

    def apply_stress(self, density_vec_sym, normal_vec_sym, dir_vec_sym,
                        mu_sym, qbx_forced_limit):
        """ Symbolic expression for viscous stress applied to direction

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress:
        -pressure * I + mu * ( grad U + (grad U).T)),
        applied in the direction of *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector
        :arg normal_vec_sym: a symbolic vector variable for the normal vectors
            (outward facing normals at source locations)
        :arg dir_vec_sym: a symbolic vector for the application direction
        :arg mu_sym: a symbolic variable for the viscosity
        :arg qbx_forced_limit: the qbx_forced_limit argument to be passed
            on to IntG.  +/-1 for exterior/interior one-sided boundary limit,
            +/-2 for exterior/interior off-boundary evaluation, and 'avg'
            for the average of the two one-sided boundary limits.
        """

        sym_expr = np.empty((self.dim,), dtype=object)

        # Build velocity derivative matrix
        sym_grad_matrix = np.empty((self.dim, self.dim), dtype=object)
        for i in range(self.dim):
            sym_grad_matrix[:, i] = self.apply_derivative(i, density_vec_sym,
                                     normal_vec_sym, mu_sym, qbx_forced_limit)

        for comp in range(self.dim):

            # First, add the pressure term:
            sym_expr[comp] = - dir_vec_sym[comp] * self.apply_pressure(
                                            density_vec_sym, normal_vec_sym,
                                            mu_sym, qbx_forced_limit)

            # Now add the velocity derivative components
            for j in range(self.dim):
                sym_expr[comp] = sym_expr[comp] + (
                                    dir_vec_sym[j] * mu_sym * (
                                        sym_grad_matrix[comp][j]
                                        + sym_grad_matrix[j][comp])
                                        )

        return sym_expr
