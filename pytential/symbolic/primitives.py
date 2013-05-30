from __future__ import division

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


import numpy as np
from pymbolic.primitives import (  # noqa
        Expression as ExpressionBase, Variable,
        make_sym_vector, cse_scope,
        make_common_subexpression as cse)

__doc__ = """
.. |where-blurb| replace:: A symbolic name for a
    :class:`pytential.discretization.Discretization`
"""


class DEFAULT_SOURCE:
    pass


class DEFAULT_TARGET:
    pass


# {{{ helper functions

def array_to_tuple(ary):
    """This function is typically used to make :class:`numpy.ndarray`
    instances hashable by converting them to tuples.
    """

    if isinstance(ary, np.ndarray):
        return tuple(ary)
    else:
        return ary

# }}}


class Expression(ExpressionBase):
    def stringifier(self):
        from pytential.symbolic.mappers import StringifyMapper
        return StringifyMapper


class Function(Variable):
    def __call__(self, operand, *args, **kwargs):
        # If the call is handed an object array full of operands,
        # return an object array of the operator applied to each of the
        # operands.

        from pytools.obj_array import is_obj_array, with_object_array_or_scalar
        if is_obj_array(operand):
            def make_op(operand_i):
                return self(operand_i, *args, **kwargs)

            return with_object_array_or_scalar(make_op, operand)
        else:
            return Variable.__call__(self, operand)

real = Function("real")
imag = Function("imag")
sqrt = Function("sqrt")


class DiscretizationProperty(Expression):
    """A quantity that depends exclusively on the discretization (and has no
    further arguments.
    """

    def __init__(self, where):
        """
        :arg where: |where-blurb|
        """

        self.where = where

    def __getinitargs__(self):
        return (self.where,)


# {{{ discretization properties

class QWeight(DiscretizationProperty):
    """Bare quadrature weights (without Jacobians)."""

    mapper_method = intern("map_q_weights")


class ParametrizationDerivativeComponent(DiscretizationProperty):
    def __init__(self, ambient_axis, ref_axis, where):
        """
        :arg where: |where-blurb|
        """
        self.ambient_axis = ambient_axis
        self.ref_axis = ref_axis
        DiscretizationProperty.__init__(self, where)

    def __getinitargs__(self):
        return (self.ambient_axis, self.ref_axis, self.where)

    mapper_method = intern("map_parametrization_derivative_component")


class ParametrizationGradient(DiscretizationProperty):
    """Return a *(ambient_dimension, dimension)*-shaped object array
    containing the gradient of the parametrization.
    """

    mapper_method = "map_parametrization_gradient"


class ParametrizationDerivative(DiscretizationProperty):
    """A :class:`pymbolic.geometric_algebra.MultiVector` for the
    parametrization derivative.
    """

    mapper_method = "map_parametrization_derivative"


def jacobian(where):
    return cse(
            sqrt(ParametrizationDerivative(where).attr("norm_squared")()),
            "jacobian", cse_scope.GLOBAL)


def sqrt_jac_q_weight(where=None):
    return cse(sqrt(jacobian(where) * QWeight(where)),
            "sqrt_jac_q_weight", cse_scope.GLOBAL)


def normal(where=None):
    pder = ParametrizationDerivative(where)
    return cse(-pder.attr("I") | pder, "normal", cse_scope.GLOBAL)


def mean_curvature(dph):
    """
    :arg dph: a :class:`DiscretizationPlaceholder`
    """
    raise NotImplementedError()


def gaussian_curvature(dph):
    """
    :arg dph: a :class:`DiscretizationPlaceholder`
    """
    raise NotImplementedError()

# FIXME: make sense of this in the context of MultiVectors
# def xyz_to_local_matrix(dim, where=None):
#     """First two rows are tangents."""
#     result = np.zeros((dim, dim), dtype=np.object)
#
#     for i in range(dim-1):
#         result[i] = make_tangent(i, dim, where)
#     result[-1] = make_normal(dim, where)
#
#     return result

# }}}


class Upsample(Expression):
    def __init__(self, to_where, from_where, operand):
        """
        :arg to_where: |where-blurb|
        :arg from_where: |where-blurb|
        """
        self.to_where = to_where
        self.from_where = from_where
        self.operand = operand

    mapper_method = "map_upsample"


# {{{ operators

# {{{ operator base classes

class OperatorBase(Expression):
    def __init__(self, operand):
        self.operand = operand

    def __getinitargs__(self):
        return (self.operand,)


class LayerPotentialOperatorBase(OperatorBase):
    def __new__(cls, kernel, operand, *args, **kwargs):
        # If the constructor is handed an object array full of operands,
        # return an object array of the operator applied to each of the
        # operands.

        from pytools.obj_array import is_obj_array, with_object_array_or_scalar
        if is_obj_array(operand):

            def make_op(operand_i):
                return cls(kernel, operand_i, *args, **kwargs)

            return with_object_array_or_scalar(make_op, operand)
        else:
            return OperatorBase.__new__(cls)

    def __init__(self, kernel, operand, qbx_forced_limit=None,
            source=None, target=None):
        OperatorBase.__init__(self, operand)

        from sumpy.kernel import normalize_kernel
        self.kernel = normalize_kernel(kernel)

        assert qbx_forced_limit is not DEFAULT_SOURCE
        assert qbx_forced_limit is not DEFAULT_TARGET

        self.qbx_forced_limit = qbx_forced_limit
        self.source = source
        self.target = target

    def __getinitargs__(self):
        return (self.kernel, self.operand, self.qbx_forced_limit,
                self.source, self.target)


class SourceDiffLayerPotentialOperatorBase(LayerPotentialOperatorBase):
    pass

# }}}


class NodeSum(Expression):
    def __init__(self, operand):
        self.operand = operand

    def __getinitargs__(self):
        return (self.operand)

    def get_hash(self):
        return hash((type(self),) + (array_to_tuple(self.operand), self.where))

    mapper_method = "map_node_sum"


def integral(operand, where=None):
    return NodeSum(jacobian(where) * QWeight(where) * operand)


def mean(operand, where=None):
    return integral(operand, where) / integral(Ones(where), where)


#class LineIntegral(IntegralBase):
    #mapper_method = intern("map_line_integral")

class Ones(Expression):
    def __init__(self, where=None):
        self.where = where

    def __getinitargs__(self):
        return (self.where,)

    mapper_method = intern("map_ones")


class IterativeInverse(Expression):
    def __init__(self, expression, rhs, variable_name, extra_vars={},
            where=None):
        self.expression = expression
        self.rhs = rhs
        self.variable_name = variable_name
        self.extra_vars = extra_vars
        self.where = where

    def __getinitargs__(self):
        return (self.expression, self.rhs, self.variable_name,
                self.extra_vars, self.where)

    def get_hash(self):
        return hash((self.__class__,) + (self.expression,
            self.rhs, self.variable_name,
            frozenset(self.extra_vars.iteritems()), self.where))

    mapper_method = intern("map_inverse")


# {{{ building blocks

class IntG(LayerPotentialOperatorBase):
    r"""
    .. math::

        \int_\Gamma g_k(x-y) \sigma(y) dS_y

    where :math:`\sigma` is *operand*.
    """

    def what(self):
        return self.target, "p", ()

    mapper_method = intern("map_int_g")


class IntGdSource(SourceDiffLayerPotentialOperatorBase):
    r"""
    .. math::

        \int_\Gamma d \cdot \nabla_y g_k(x-y) \sigma(y) dS_y

    where :math:`\sigma` is *operand*,
    :math:`d` is *ds_direction*, a vector defaulting to the unit
    surface normal of :math:`\Gamma`.
    """

    def __init__(self, kernel, operand, ds_direction=None,
            qbx_forced_limit=None, source=None, target=None):
        LayerPotentialOperatorBase.__init__(self, kernel, operand,
                qbx_forced_limit, source, target)
        self.ds_direction = ds_direction

    def what(self):
        return self.target, "p", ()

    def __getinitargs__(self):
        return (self.kernel, self.operand, self.ds_direction,
                self.qbx_forced_limit, self.source, self.target, )

    def get_hash(self):
        return hash((self.__class__,) + (
            self.kernel, self.operand, array_to_tuple(self.ds_direction),
            self.qbx_forced_limit, self.source, self.target))

    mapper_method = intern("map_int_g_ds")


class IntGdTarget(LayerPotentialOperatorBase):
    r"""
    .. math::

        \frac \partial {\partial x_i} \int_\Gamma d \cdot \nabla_y g_k(x-y)
        \sigma(y) dS_y

    where :math:`\sigma` is *operand*, and
    :math:`i` is *dt_axis*.
    """
    def __init__(self, kernel, operand, dt_axis, qbx_forced_limit=None,
            source=None, target=None):
        LayerPotentialOperatorBase.__init__(self, kernel, operand,
                qbx_forced_limit, source, target)
        self.dt_axis = dt_axis

    def what(self):
        return self.target, "g", self.dt_axis

    def __getinitargs__(self):
        return (self.kernel, self.operand, self.dt_axis,
                self.qbx_forced_limit, self.source, self.target)

    mapper_method = intern("map_int_g_dt")


class IntGdMixed(SourceDiffLayerPotentialOperatorBase):
    r"""
    .. math::

        \frac \partial {\partial x_i} \int_\Gamma d \cdot \nabla_y g_k(x-y)
        \sigma(y) dS_y

    where :math:`\sigma` is *operand*,
    :math:`d` is *ds_direction*, a vector defaulting to the unit
    surface normal of :math:`\Gamma`, and :math:`i` is *dt_axis*.
    """
    def __init__(self, kernel, operand, dt_axis, ds_direction=None,
            qbx_forced_limit=None, source=None, target=None):
        LayerPotentialOperatorBase.__init__(self, kernel, operand,
                qbx_forced_limit, source, target)
        self.dt_axis = dt_axis
        self.ds_direction = ds_direction

    def what(self):
        return self.target, "g", self.dt_axis

    def __getinitargs__(self):
        return (self.kernel, self.operand, self.dt_axis, self.ds_direction,
                self.qbx_forced_limit, self.source, self.target)

    def get_hash(self):
        return hash((self.__class__,) + (
            self.kernel, self.operand, self.dt_axis,
            array_to_tuple(self.ds_direction),
            self.qbx_forced_limit, self.source, self.target))

    mapper_method = intern("map_int_g_dmix")


class IntGd2Target(LayerPotentialOperatorBase):
    r"""
    .. math::

        \frac \partial {\partial x_i}\frac \partial {\partial x_j}
        \int_\Gamma g_k(x-y) \sigma(y) dS_y

    where :math:`\sigma` is *operand*,
    :math:`i` is *dt_axis_a*, and :math:`j` is *dt_axis_b*.
    """
    def __init__(self, kernel, operand, dt_axis_a, dt_axis_b,
            qbx_forced_limit=None, source=None, target=None):
        LayerPotentialOperatorBase.__init__(self, kernel, operand,
                qbx_forced_limit, source, target)
        self.dt_axis_a = dt_axis_a
        self.dt_axis_b = dt_axis_b

    def what(self):
        return self.target, "h", (self.dt_axis_a, self.dt_axis_b)

    def __getinitargs__(self):
        return (self.kernel, self.operand, self.dt_axis_a, self.dt_axis_b,
                self.qbx_forced_limit, self.source, self.target)

    mapper_method = intern("map_int_g_d2t")

# }}}

# {{{ non-dimension-specific operators, get eliminated upon entry into bind()

S = IntG


def D(kernel, arg, qbx_forced_limit=None, source=None, target=None):
    return IntGdSource(kernel, arg,
            qbx_forced_limit=qbx_forced_limit, source=source, target=target)


class Sp(LayerPotentialOperatorBase):
    mapper_method = "map_single_layer_prime"


class Spp(LayerPotentialOperatorBase):
    mapper_method = "map_single_layer_2prime"


class Dp(LayerPotentialOperatorBase):
    mapper_method = "map_double_layer_prime"

# }}}

# }}}


# {{{ differential operators on layer potentials

def grad_S(kernel, arg, dim):
    from pytools.obj_array import log_shape
    arg_shape = log_shape(arg)
    result = np.zeros(arg_shape+(dim,), dtype=object)
    from pytools import indices_in_shape
    for i in indices_in_shape(arg_shape):
        for j in range(dim):
            result[i+(j,)] = IntGdTarget(kernel, arg[i], j)
    return result


def grad_D(kernel, arg, dim):
    from pytools.obj_array import log_shape
    arg_shape = log_shape(arg)
    result = np.zeros(arg_shape+(dim,), dtype=object)
    from pytools import indices_in_shape
    for i in indices_in_shape(arg_shape):
        for j in range(dim):
            result[i+(j,)] = IntGdMixed(kernel, arg[i], j)
    return result


def tangential_surf_grad_source_S(kernel, arg, dim=3):
    from pytools.obj_array import make_obj_array
    return make_obj_array([
        IntGdSource(kernel, arg,
            ds_direction=make_tangent(i, dim, "src"))
        for i in range(dim-1)])


def surf_grad_S(kernel, arg, dim):
    """
    :arg dim: The dimension of the ambient space.
    """

    return project_to_tangential(cse(grad_S(kernel, arg, dim)))


def div_S_volume(kernel, arg):
    return sum(IntGdTarget(kernel, arg_n, n) for n, arg_n in enumerate(arg))


def curl_S_volume(kernel, arg):
    from pytools import levi_civita
    from pytools.obj_array import make_obj_array

    return make_obj_array([
        sum(
            levi_civita((l, m, n)) * IntGdTarget(kernel, arg[n], m)
            for m in range(3) for n in range(3))
        for l in range(3)])


def curl_curl_S_volume(k, arg):
    # By vector identity, this is grad div S volume + k^2 S_k(arg),
    # since S_k(arg) satisfies a Helmholtz equation.

    from pytools.obj_array import make_obj_array

    def swap_min_first(i, j):
        if i < j:
            return i, j
        else:
            return j, i

    return make_obj_array([
        sum(IntGd2Target(k, arg[m], *swap_min_first(m, n)) for m in range(3))
        for n in range(3)]) + k**2*S(k, arg)


def nxcurl_S(kernel, loc, arg):
    """
    :arg loc: one of three values:
      * +1 on the side of the surface toward which
        the normal points ('exterior' of the surface),
      * 0 on the surface, or evaluated at a volume target
      * -1 on the interior of the surface.
    """
    nxcurl_S = np.cross(normal(3), curl_S_volume(kernel, arg))
    assert loc in [-1, 0, 1], "invalid value for 'loc' (%s)" % loc
    return nxcurl_S + loc*(1/2)*arg


def surface_laplacian_S_squared(u, invertibility_scale=0):
    """
    :arg u: The field to which the surface Laplacian is applied.
    """
    # http://wiki.tiker.net/HellsKitchen/SurfaceLaplacian

    Su = cse(S(0, u), "su_from_surflap")

    return (
            - 2*mean_curvature()*Sp(0, Su)
            - ((Spp(0, Su)+Dp(0, Su))-(-1/4*u+Sp(0, Sp(0, u))))
            - invertibility_scale * mean(S(0, Su))*Ones())


def S_surface_laplacian_S(u, dim, invertibility_scale=0, qbx_fix_scale=0):
    """
    :arg u: The field to which the surface Laplacian is applied.
    """

    # This converges, but appears to work quite poorly compared to the above.

    tgrad_Su = cse(
            project_to_tangential(grad_S(0, u, dim)),
            "tgrad_su_from_surflap")

    return (
            - IntGdSource(0, Ones(), ds_direction=real(tgrad_Su))
            - 1j*IntGdSource(0, Ones(), ds_direction=imag(tgrad_Su))
            - invertibility_scale * S(0, Ones()*mean(S(0, u)))
            - qbx_fix_scale * (
                u
                # D+ - D- = identity (but QBX will compute the
                # 'compact part of the identity' -- call that I*)
                - (
                    D(0, u, qbx_forced_limit=+1)
                    - D(0, u, qbx_forced_limit=-1))
                )
                # The above is I - I*, which means only the high-frequency
                # bits of the identity are left.
            )

# }}}


# {{{ geometric operations

def xyz_to_tangential(xyz_vec, which=None):
    d = len(xyz_vec)
    x2l = xyz_to_local_matrix(d)
    return np.dot(x2l[:-1], xyz_vec)


def tangential_to_xyz(tangential_vec, which=None):
    d = len(tangential_vec) + 1
    x2l = xyz_to_local_matrix(d)
    return np.dot(x2l[:-1].T, tangential_vec)


def project_to_tangential(xyz_vec, which=None):
    return tangential_to_xyz(
            cse(xyz_to_tangential(xyz_vec, which), which))


def n_dot(vec, which=None):
    return np.dot(normal(len(vec), which), vec)


def n_cross(vec, which=None):
    nrm = normal(3, which)

    from pytools import levi_civita
    from pytools.obj_array import make_obj_array
    return make_obj_array([
        sum(
            levi_civita((i, j, k)) * nrm[j] * vec[k]
            for j in range(3) for k in range(3))
        for i in range(3)])


def surf_n_cross(tangential_vec):
    assert len(tangential_vec) == 2
    from pytools.obj_array import make_obj_array
    return make_obj_array([-tangential_vec[1], tangential_vec[0]])

# }}}

# vim: foldmethod=marker
