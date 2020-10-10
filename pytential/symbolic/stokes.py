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


# {{{ StokesletWrapper

class StokesletWrapper:
    """Wrapper class for the Stokeslet kernel.

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

# }}}


# {{{ StressletWrapper

class StressletWrapper:
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

# }}}

# {{{ base Stokes operator

class StokesOperator:
    """
    .. attribute:: ambient_dim
    .. attribute:: side

    .. automethod:: __init__
    .. automethod:: get_density_var
    .. automethod:: prepare_rhs
    .. automethod:: representation
    .. automethod:: operator
    """

    def __init__(self, ambient_dim, side):
        """
        :arg ambient_dim: dimension of the ambient space.
        :arg side: :math:`+`` for exterior or :math:`-1` for interior.
        """

        if abs(side) != 1:
            raise ValueError(f"invalid evaluation side: {side}")

        self.ambient_dim = ambient_dim
        self.side = side

        self.stresslet = StressletWrapper(dim=self.ambient_dim)
        self.stokeslet = StokesletWrapper(dim=self.ambient_dim)

    @property
    def dim(self):
        return self.ambient_dim - 1

    def get_density_var(self, name="sigma"):
        """
        :returns: a symbolic vector corresponding to the density.
        """
        return sym.make_sym_vector(name, self.ambient_dim)

    def prepare_rhs(self, b, *, mu):
        """
        :returns: a (potentially) modified right-hand side *b* that matches
            requirements of the representation.
        """
        return b

    def operator(self, sigma):
        """
        :returns: the integral operator that should be solved to obtain the
            density *sigma*.
        """
        raise NotImplementedError

    def velocity(self, sigma, *, normal, mu, qbx_forced_limit=None):
        """
        :returns: a representation of the velocity field in the Stokes flow.
        """
        raise NotImplementedError

    def pressure(self, sigma, *, normal, mu, qbx_forced_limit=None):
        """
        :returns: a representation of the pressure in the Stokes flow.
        """
        raise NotImplementedError

# }}}


# {{{ exterior Stokes flow

class HsiaoKressExteriorStokesOperator(StokesOperator):
    """Representation for 2D Stokes Flow based on [hsiao-kress]_.

    .. [hsiao-kress] Hsiao & Kress, *On an Integral Equation for the
        Two-Dimensional Exterior Stokes Problem*,
        Applied Numerical Mathematics, Vol. 1, 1985.

    .. automethod:: __init__
    """

    def __init__(self, *, omega, alpha=None, eta=None):
        r"""
        :arg omega: farfield behaviour of the velocity field, as defined
            by :math:`A` in [hsiao-kress]_ Equation 2.3.
        :arg alpha: real parameter :math:`\alpha > 0`.
        :arg eta: real parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning.
        """
        super().__init__(ambient_dim=2, side=+1)

        # NOTE: in [hsiao-kress], there is an analysis on a circle, which
        # recommends values in
        #   1/2 <= alpha <= 2 and max(1/alpha, 1) <= eta <= min(2, 2/alpha)
        # so we choose alpha = eta = 1, which seems to be in line with some
        # of the presented numerical results too.

        if alpha is None:
            alpha = 1.0

        if eta is None:
            eta = 1.0

        self.omega = omega
        self.alpha = alpha
        self.eta = eta

    def _farfield(self, mu, qbx_forced_limit):
        length = sym.integral(self.ambient_dim, self.dim, 1)
        return self.stokeslet.apply(
                self.omega / length,
                mu,
                qbx_forced_limit=qbx_forced_limit)

    def _operator(self, sigma, normal, mu, qbx_forced_limit):
        # NOTE: we set a dofdesc here to force the evaluation of this integral
        # on the source instead of the target when using automatic tagging
        # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
        dd = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)

        int_sigma = sym.integral(self.ambient_dim, self.dim, sigma, dofdesc=dd)
        meanless_sigma = sym.cse(sigma - sym.mean(self.ambient_dim, self.dim, sigma))

        op_k = self.stresslet.apply(sigma, normal, mu,
                qbx_forced_limit=qbx_forced_limit)
        op_s = (
                self.alpha / (2.0 * np.pi) * int_sigma
                - self.stokeslet.apply(meanless_sigma, mu,
                    qbx_forced_limit=qbx_forced_limit)
                )

        return op_k + self.eta * op_s

    def prepare_rhs(self, b, *, mu):
        return b + self._farfield(mu, qbx_forced_limit=+1)

    def operator(self, sigma, *, normal, mu):
        # NOTE: H. K. 1985 Equation 2.18
        return -0.5 * self.side * sigma - self._operator(sigma, normal, mu, "avg")

    def velocity(self, sigma, *, normal, mu, qbx_forced_limit=2):
        # NOTE: H. K. 1985 Equation 2.16
        return (
                self._farfield(mu, qbx_forced_limit)
                - self._operator(sigma, normal, mu, qbx_forced_limit)
                )

    def pressure(self, sigma, *, normal, mu, qbx_forced_limit=2):
        # FIXME: H. K. 1985 Equation 2.17
        raise NotImplementedError


class HebekerExteriorStokesOperator(StokesOperator):
    """Representation for 3D Stokes Flow based on [hebeker]_.

    .. [hebeker] Hebeker, *Efficient Boundary Element Methods for
        Three-Dimensional Exterior Viscous Flow*, Numerical Methods for
        Partial Differential Equations, Vol. 2, 1986.

    .. automethod:: __init__
    """

    def __init__(self, *, eta=None):
        r"""
        :arg eta: a parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning of the operator.
        """

        super().__init__(ambient_dim=3, side=+1)

        # NOTE: eta is chosen here based on H. 1986 Figure 1, which is
        # based on solving on the unit sphere
        if eta is None:
            eta = 0.5

        self.eta = eta

    def _operator(self, sigma, normal, mu, qbx_forced_limit):
        op_w = self.stresslet.apply(sigma, normal, mu,
                qbx_forced_limit=qbx_forced_limit)
        op_v = self.stokeslet.apply(sigma, mu,
                qbx_forced_limit=qbx_forced_limit)

        return op_w + self.eta * op_v

    def operator(self, sigma, *, normal, mu):
        # NOTE: H. 1986 Equation 17
        return -0.5 * self.side * sigma - self._operator(sigma, normal, mu, "avg")

    def velocity(self, sigma, *, normal, mu, qbx_forced_limit=2):
        # NOTE: H. 1986 Equation 16
        return -self._operator(sigma, normal, mu, qbx_forced_limit)

    def pressure(self, sigma, *, normal, mu, qbx_forced_limit=2):
        # FIXME: not given in H. 1986, but should be easy to derive using the
        # equivalent single-/double-layer pressure kernels
        raise NotImplementedError

# }}}
