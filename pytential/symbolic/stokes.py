from __future__ import division, absolute_import
import numpy as np

from pytential import sym
from sumpy.kernel import StokesletKernel, StressletKernel


class StokesletWrapper(object):
    """ Wrapper class for the Stokeslet kernel.  This class is meant to
     shield the user from the messiness of writing out every term
     in the expansion of the double-indiced Stokeslet kernel applied to
     the density vector.  The object is created
     to do some of the set-up and bookkeeping once, rather than every
     time we want to create a symbolic expression based on the kernel -- say,
     once when we solve for the density, and once when we want a symbolic
     representation for the solution, for example.

    The apply() function returns the integral expressions needed for
     the vector velocity resulting from convolution with the vectory density,
     and is meant to work similarly to
     calling S() (which is IntG()).

    .. attribute:: kernel_dict
        The dictionary allows us to exploit symmetry -- that
        StokesletKernel(icomp=0, jcomp=1) is identical to
        StokesletKernel(icomp=1, jcomp=0) -- and avoid creating multiple expansions for
        the same kernel in a different ordering.

    """

    def __init__(self, dim=None):

        self.dim = dim
        if dim == 2:
            self.kernel_dict = {
                        (2, 0) : StokesletKernel(dim=2, icomp=0, jcomp=0),
                        (1, 1) : StokesletKernel(dim=2, icomp=0, jcomp=1),
                        (0, 2) : StokesletKernel(dim=2, icomp=1, jcomp=1)
                               }

        elif dim == 3:
            self.kernel_dict = { 
                        (2, 0, 0) : StokesletKernel(dim=3, icomp=0, jcomp=0),
                        (1, 1, 0) : StokesletKernel(dim=3, icomp=0, jcomp=1),
                        (1, 0, 1) : StokesletKernel(dim=3, icomp=0, jcomp=2),
                        (0, 2, 0) : StokesletKernel(dim=3, icomp=1, jcomp=1),
                        (0, 1, 1) : StokesletKernel(dim=3, icomp=1, jcomp=2),
                        (0, 0, 2) : StokesletKernel(dim=3, icomp=2, jcomp=2)
                               }


        else:
            raise ValueError("unsupported dimension given to StokesletWrapper")


    def apply(self, density_vec_sym, mu_sym, qbx_forced_limit):
        """ Returns an object array of symbolic expressions for the vector
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
            base_count[comp] +=1

            for i in range(self.dim):
                var_ctr = base_count.copy()
                var_ctr[i] += 1
                ctr_key = tuple(var_ctr)

                if i < 0.1:
                    sym_expr[comp] = sym.IntG(
                                     self.kernel_dict[ctr_key], density_vec_sym[i],
                                     qbx_forced_limit=qbx_forced_limit, mu=mu_sym)

                else:
                    sym_expr[comp] = sym_expr[comp] + sym.IntG(
                                     self.kernel_dict[ctr_key], density_vec_sym[i],
                                     qbx_forced_limit=qbx_forced_limit, mu=mu_sym)
            

        return sym_expr


class StressletWrapper(object):
    """ Wrapper class for the Stresslet kernel.  This class is meant to
     shield the user from the messiness of writing out every term
     in the expansion of the triple-indiced Stresslet kernel applied to both
     a normal vector and the density vector.  The object is created
     to do some of the set-up and bookkeeping once, rather than every
     time we want to create a symbolic expression based on the kernel -- say,
     once when we solve for the density, and once when we want a symbolic
     representation for the solution, for example.

    The apply() function returns the integral expressions needed for convolving
     the kernel with a vectory density, and is meant to work similarly to
     calling S() (which is IntG()).

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
                (3, 0) : StressletKernel(dim=2, icomp=0, jcomp=0, kcomp=0),
                (2, 1) : StressletKernel(dim=2, icomp=0, jcomp=0, kcomp=1),
                (1, 2) : StressletKernel(dim=2, icomp=0, jcomp=1, kcomp=1),
                (0, 3) : StressletKernel(dim=2, icomp=1, jcomp=1, kcomp=1)
                               }

        elif dim == 3:
            self.kernel_dict = {
                (3, 0, 0) : StressletKernel(dim=3, icomp=0, jcomp=0, kcomp=0),
                (2, 1, 0) : StressletKernel(dim=3, icomp=0, jcomp=0, kcomp=1),
                (2, 0, 1) : StressletKernel(dim=3, icomp=0, jcomp=0, kcomp=2),
                (1, 2, 0) : StressletKernel(dim=3, icomp=0, jcomp=1, kcomp=1),
                (1, 1, 1) : StressletKernel(dim=3, icomp=0, jcomp=1, kcomp=2),
                (1, 0, 2) : StressletKernel(dim=3, icomp=0, jcomp=2, kcomp=2),
                (0, 3, 0) : StressletKernel(dim=3, icomp=1, jcomp=1, kcomp=1),
                (0, 2, 1) : StressletKernel(dim=3, icomp=1, jcomp=1, kcomp=2),
                (0, 1, 2) : StressletKernel(dim=3, icomp=1, jcomp=2, kcomp=2),
                (0, 0, 3) : StressletKernel(dim=3, icomp=2, jcomp=2, kcomp=2)
                               }


        else:
            raise ValueError("unsupported dimension given to StressletWrapper")             


    def apply(self, density_vec_sym, dir_vec_sym, mu_sym, qbx_forced_limit):
        """ Returns an object array of symbolic expressions for the vector
            resulting from integrating the dyadic Stresslet kernel with
            direction vector variable name *dir_vec_sym* and
            variable *density_vec_sym*.

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
            base_count[comp] +=1

            for i, j in itertools.product(range(self.dim), range(self.dim)):
                var_ctr = base_count.copy()
                var_ctr[i] += 1
                var_ctr[j] += 1
                ctr_key = tuple(var_ctr)

                if i + j < 0.1:
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


