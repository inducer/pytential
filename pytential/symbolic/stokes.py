from __future__ import division, absolute_import
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
from sumpy.kernel import StokesletKernel, StressletKernel, LaplaceKernel


class StokesletWrapper(object):
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

    def __init__(self, dim=None):

        self.dim = dim
        if dim == 2:
            self.kernel_dict = {
                        (2, 0): StokesletKernel(dim=2, icomp=0, jcomp=0),
                        (1, 1): StokesletKernel(dim=2, icomp=0, jcomp=1),
                        (0, 2): StokesletKernel(dim=2, icomp=1, jcomp=1)
                               }

        elif dim == 3:
            self.kernel_dict = {
                        (2, 0, 0): StokesletKernel(dim=3, icomp=0, jcomp=0),
                        (1, 1, 0): StokesletKernel(dim=3, icomp=0, jcomp=1),
                        (1, 0, 1): StokesletKernel(dim=3, icomp=0, jcomp=2),
                        (0, 2, 0): StokesletKernel(dim=3, icomp=1, jcomp=1),
                        (0, 1, 1): StokesletKernel(dim=3, icomp=1, jcomp=2),
                        (0, 0, 2): StokesletKernel(dim=3, icomp=2, jcomp=2)
                               }

        else:
            raise ValueError("unsupported dimension given to StokesletWrapper")

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

        sym_expr = np.empty((self.dim,), dtype=object)

        for comp in range(self.dim):

            # Start variable count for kernel with 1 for the requested result
            #  component
            base_count = np.zeros(self.dim, dtype=np.int)
            base_count[comp] += 1

            for i in range(self.dim):
                var_ctr = base_count.copy()
                var_ctr[i] += 1
                ctr_key = tuple(var_ctr)

                if i < 1:
                    sym_expr[comp] = sym.IntG(
                                     self.kernel_dict[ctr_key], density_vec_sym[i],
                                     qbx_forced_limit=qbx_forced_limit, mu=mu_sym)

                else:
                    sym_expr[comp] = sym_expr[comp] + sym.IntG(
                                     self.kernel_dict[ctr_key], density_vec_sym[i],
                                     qbx_forced_limit=qbx_forced_limit, mu=mu_sym)

        return sym_expr

    def apply_pressure(self, density_vec_sym, mu_sym, qbx_forced_limit):
        """ Symbolic expression for pressure field associated with the Stokeslet"""

        from pytential.symbolic.mappers import DerivativeTaker
        kernel = LaplaceKernel(dim=self.dim)

        for i in range(self.dim):

            if i < 1:
                sym_expr = DerivativeTaker(i).map_int_g(
                                sym.S(kernel, density_vec_sym[i],
                                qbx_forced_limit=qbx_forced_limit))
            else:
                sym_expr = sym_expr + (DerivativeTaker(i).map_int_g(
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

        from pytential.symbolic.mappers import DerivativeTaker

        sym_expr = np.empty((self.dim,), dtype=object)

        for comp in range(self.dim):

            # Start variable count for kernel with 1 for the requested result
            #  component
            base_count = np.zeros(self.dim, dtype=np.int)
            base_count[comp] += 1

            for i in range(self.dim):
                var_ctr = base_count.copy()
                var_ctr[i] += 1
                ctr_key = tuple(var_ctr)

                if i < 1:
                    sym_expr[comp] = DerivativeTaker(deriv_dir).map_int_g(
                                         sym.IntG(self.kernel_dict[ctr_key],
                                             density_vec_sym[i],
                                             qbx_forced_limit=qbx_forced_limit,
                                             mu=mu_sym))

                else:
                    sym_expr[comp] = sym_expr[comp] + DerivativeTaker(
                                         deriv_dir).map_int_g(
                                             sym.IntG(self.kernel_dict[ctr_key],
                                             density_vec_sym[i],
                                             qbx_forced_limit=qbx_forced_limit,
                                             mu=mu_sym))

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

        import itertools

        sym_expr = np.empty((self.dim,), dtype=object)
        stresslet_obj = StressletWrapper(dim=self.dim)

        for comp in range(self.dim):

            # Start variable count for kernel with 1 for the requested result
            #   component
            base_count = np.zeros(self.dim, dtype=np.int)
            base_count[comp] += 1

            for i, j in itertools.product(range(self.dim), range(self.dim)):
                var_ctr = base_count.copy()
                var_ctr[i] += 1
                var_ctr[j] += 1
                ctr_key = tuple(var_ctr)

                if i + j < 1:
                    sym_expr[comp] = dir_vec_sym[i] * sym.IntG(
                                     stresslet_obj.kernel_dict[ctr_key],
                                     density_vec_sym[j],
                                     qbx_forced_limit=qbx_forced_limit, mu=mu_sym)

                else:
                    sym_expr[comp] = sym_expr[comp] + dir_vec_sym[i] * sym.IntG(
                                                stresslet_obj.kernel_dict[ctr_key],
                                                density_vec_sym[j],
                                                qbx_forced_limit=qbx_forced_limit,
                                                mu=mu_sym)

        return sym_expr


class StressletWrapper(object):
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

    def __init__(self, dim=None):

        self.dim = dim
        if dim == 2:
            self.kernel_dict = {
                (3, 0): StressletKernel(dim=2, icomp=0, jcomp=0, kcomp=0),
                (2, 1): StressletKernel(dim=2, icomp=0, jcomp=0, kcomp=1),
                (1, 2): StressletKernel(dim=2, icomp=0, jcomp=1, kcomp=1),
                (0, 3): StressletKernel(dim=2, icomp=1, jcomp=1, kcomp=1)
                               }

        elif dim == 3:
            self.kernel_dict = {
                (3, 0, 0): StressletKernel(dim=3, icomp=0, jcomp=0, kcomp=0),
                (2, 1, 0): StressletKernel(dim=3, icomp=0, jcomp=0, kcomp=1),
                (2, 0, 1): StressletKernel(dim=3, icomp=0, jcomp=0, kcomp=2),
                (1, 2, 0): StressletKernel(dim=3, icomp=0, jcomp=1, kcomp=1),
                (1, 1, 1): StressletKernel(dim=3, icomp=0, jcomp=1, kcomp=2),
                (1, 0, 2): StressletKernel(dim=3, icomp=0, jcomp=2, kcomp=2),
                (0, 3, 0): StressletKernel(dim=3, icomp=1, jcomp=1, kcomp=1),
                (0, 2, 1): StressletKernel(dim=3, icomp=1, jcomp=1, kcomp=2),
                (0, 1, 2): StressletKernel(dim=3, icomp=1, jcomp=2, kcomp=2),
                (0, 0, 3): StressletKernel(dim=3, icomp=2, jcomp=2, kcomp=2)
                               }

        else:
            raise ValueError("unsupported dimension given to StressletWrapper")

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

        import itertools

        sym_expr = np.empty((self.dim,), dtype=object)

        for comp in range(self.dim):

            # Start variable count for kernel with 1 for the requested result
            #   component
            base_count = np.zeros(self.dim, dtype=np.int)
            base_count[comp] += 1

            for i, j in itertools.product(range(self.dim), range(self.dim)):
                var_ctr = base_count.copy()
                var_ctr[i] += 1
                var_ctr[j] += 1
                ctr_key = tuple(var_ctr)

                if i + j < 1:
                    sym_expr[comp] = sym.IntG(
                                     self.kernel_dict[ctr_key],
                                     dir_vec_sym[i] * density_vec_sym[j],
                                     qbx_forced_limit=qbx_forced_limit, mu=mu_sym)

                else:
                    sym_expr[comp] = sym_expr[comp] + sym.IntG(
                                                self.kernel_dict[ctr_key],
                                                dir_vec_sym[i] * density_vec_sym[j],
                                                qbx_forced_limit=qbx_forced_limit,
                                                mu=mu_sym)

        return sym_expr

    def apply_pressure(self, density_vec_sym, dir_vec_sym, mu_sym, qbx_forced_limit):
        """ Symbolic expression for pressure field associated with the stresslet"""

        import itertools
        from pytential.symbolic.mappers import DerivativeTaker
        kernel = LaplaceKernel(dim=self.dim)

        factor = (2. * mu_sym)

        for i, j in itertools.product(range(self.dim), range(self.dim)):

            if i + j < 1:
                sym_expr = factor * DerivativeTaker(i).map_int_g(
                             DerivativeTaker(j).map_int_g(
                                 sym.S(kernel, density_vec_sym[i] * dir_vec_sym[j],
                                 qbx_forced_limit=qbx_forced_limit)))
            else:
                sym_expr = sym_expr + (
                               factor * DerivativeTaker(i).map_int_g(
                                   DerivativeTaker(j).map_int_g(
                                       sym.S(kernel,
                                             density_vec_sym[i] * dir_vec_sym[j],
                                             qbx_forced_limit=qbx_forced_limit))))

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

        import itertools
        from pytential.symbolic.mappers import DerivativeTaker

        sym_expr = np.empty((self.dim,), dtype=object)

        for comp in range(self.dim):

            # Start variable count for kernel with 1 for the requested result
            #   component
            base_count = np.zeros(self.dim, dtype=np.int)
            base_count[comp] += 1

            for i, j in itertools.product(range(self.dim), range(self.dim)):
                var_ctr = base_count.copy()
                var_ctr[i] += 1
                var_ctr[j] += 1
                ctr_key = tuple(var_ctr)

                if i + j < 1:
                    sym_expr[comp] = DerivativeTaker(deriv_dir).map_int_g(
                                     sym.IntG(self.kernel_dict[ctr_key],
                                     dir_vec_sym[i] * density_vec_sym[j],
                                     qbx_forced_limit=qbx_forced_limit, mu=mu_sym))

                else:
                    sym_expr[comp] = sym_expr[comp] + DerivativeTaker(
                                        deriv_dir).map_int_g(
                                        sym.IntG(self.kernel_dict[ctr_key],
                                        dir_vec_sym[i] * density_vec_sym[j],
                                        qbx_forced_limit=qbx_forced_limit,
                                        mu=mu_sym))

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
