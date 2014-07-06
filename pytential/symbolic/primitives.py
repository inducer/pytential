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
        Expression as ExpressionBase, Variable as var,
        cse_scope as cse_scope_base,
        make_common_subexpression as cse)
from pymbolic.geometric_algebra import MultiVector, componentwise

__doc__ = """
.. |where-blurb| replace:: A symbolic name for a
    :class:`pytential.discretization.Discretization`
"""


class DEFAULT_SOURCE:
    pass


class DEFAULT_TARGET:
    pass


class cse_scope(cse_scope_base):
    DISCRETIZATION = "pytential_discretization"


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


class VectorVariable(ExpressionBase):
    """:class:`pytential.symbolic.mappers.Dimensionalizer`
    turns this into a :class:`pymbolic.geometric_algebra.MultiVector`
    of scalar variables.
    """

    def __init__(self, name, num_components=None):
        """
        :arg num_components: if None, defaults to the dimension of the
            ambient space
        """
        self.name = name
        self.num_components = num_components

    mapper_method = "map_vector_variable"


class DimensionalizedExpression(Expression):
    """Preserves an already-dimensionalized expression until it hits
    the :class:`pytential.symbolic.mappers.Dimensionalizer`, which
    will unpack it and discard this wrapper.
    """

    def __init__(self, child):
        self.child = child

    mapper_method = "map_dimensionalized_expression"


class Function(var):
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
            return var.__call__(self, operand)

real = Function("real")
imag = Function("imag")
conj = Function("conj")
sqrt = Function("sqrt")
abs = Function("abs")


class DiscretizationProperty(Expression):
    """A quantity that depends exclusively on the discretization (and has no
    further arguments.
    """

    def __init__(self, where=None):
        """
        :arg where: |where-blurb|
        """

        self.where = where

    def __getinitargs__(self):
        return (self.where,)


# {{{ discretization properties

class QWeight(DiscretizationProperty):
    """Bare quadrature weights (without Jacobians)."""

    mapper_method = intern("map_q_weight")


class NodeCoordinateComponent(DiscretizationProperty):
    def __init__(self, ambient_axis, where=None):
        """
        :arg where: |where-blurb|
        """
        self.ambient_axis = ambient_axis
        DiscretizationProperty.__init__(self, where)

    mapper_method = intern("map_node_coordinate_component")


class Nodes(DiscretizationProperty):
    """Node location of the discretization.
    """

    def __init__(self, where=None):
        DiscretizationProperty.__init__(self, where)

    mapper_method = intern("map_nodes")


class NumReferenceDerivative(DiscretizationProperty):
    def __init__(self, ref_axes, operand, where=None):
        """
        :arg ref_axes: a :class:`frozenset` of indices of
            reference coordinates along which derivatives
            will be taken.
        :arg where: |where-blurb|
        """

        if not isinstance(ref_axes, frozenset):
            raise ValueError("ref_axes must be a frozenset")

        self.ref_axes = ref_axes
        self.operand = operand
        DiscretizationProperty.__init__(self, where)

    def __getinitargs__(self):
        return (self.ref_axes, self.operand, self.where)

    mapper_method = intern("map_num_reference_derivative")


class ParametrizationDerivative(DiscretizationProperty):
    """A :class:`pymbolic.geometric_algebra.MultiVector` for the
    parametrization derivative.
    """

    mapper_method = "map_parametrization_derivative"


def area_element(where=None):
    return cse(
            sqrt(ParametrizationDerivative(where).attr("norm_squared")()),
            "area_element", cse_scope.DISCRETIZATION)


def sqrt_jac_q_weight(where=None):
    return cse(sqrt(area_element(where) * QWeight(where)),
            "sqrt_jac_q_weight", cse_scope.DISCRETIZATION)


def normal(where=None):
    """Exterior unit normals."""

    # Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = ParametrizationDerivative(where) / area_element()
    return cse(pder.attr("I") | pder, "normal",
            cse_scope.DISCRETIZATION)


def mean_curvature(where):
    raise NotImplementedError()


# FIXME: make sense of this in the context of GA
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


# {{{ operators

class NodeSum(Expression):
    def __init__(self, operand):
        self.operand = operand

    def __getinitargs__(self):
        return (self.operand,)

    mapper_method = "map_node_sum"


def integral(operand, where=None):
    return NodeSum(area_element(where) * QWeight(where) * operand)


class Ones(Expression):
    def __init__(self, where=None):
        self.where = where

    def __getinitargs__(self):
        return (self.where,)

    mapper_method = intern("map_ones")


def area(where=None):
    return cse(integral(Ones(where), where), "area",
            cse_scope.DISCRETIZATION)


def mean(operand, where=None):
    return integral(operand, where) / area(where)


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


# {{{ potentials

class IntG(Expression):
    r"""
    .. math::

        \int_\Gamma g_k(x-y) \sigma(y) dS_y

    where :math:`\sigma` is *density*.
    """

    def __new__(cls, kernel, density, *args, **kwargs):
        # If the constructor is handed a multivector object, return an
        # object array of the operator applied to each of the
        # coefficients in the multivector.

        if isinstance(density, MultiVector):
            def make_op(operand_i):
                return cls(kernel, operand_i, *args, **kwargs)

            return componentwise(make_op, density)
        else:
            return Expression.__new__(cls)

    def __init__(self, kernel, density,
            qbx_forced_limit=0, source=None, target=None):
        """*target_derivatives* and later arguments should be considered
        keyword-only.

        :arg kernel: a kernel as accepted by
            :func:`sumpy.kernel.normalize_kernel`
        :arg qbx_forced_limit: +1 if the output is required to originate from a
            QBX center on the "+" side of the boundary. -1 for the other side. 0 if
            either side of center (or no center at all) is acceptable.
        """

        if qbx_forced_limit not in [-1, 0, 1]:
                raise ValueError("invalid value (%s) of qbx_forced_limit"
                        % qbx_forced_limit)

        from sumpy.kernel import normalize_kernel
        self.kernel = normalize_kernel(kernel)
        self.density = density
        self.qbx_forced_limit = qbx_forced_limit
        self.source = source
        self.target = target

    def copy(self, kernel=None, density=None, qbx_forced_limit=None,
            source=None, target=None):
        kernel = kernel or self.kernel
        density = density or self.density
        qbx_forced_limit = qbx_forced_limit or self.qbx_forced_limit
        source = source or self.source
        target = target or self.target
        return type(self)(kernel, density, qbx_forced_limit, source, target)

    def __getinitargs__(self):
        return (self.kernel, self.density, self.qbx_forced_limit,
                self.source, self.target)

    mapper_method = intern("map_int_g")


class IntGdSource(IntG):
    r"""
    .. math::

        \int_\Gamma \operatorname{dsource} \overdot \nabla_y
            \overdot g(x-y) \sigma(y) dS_y

    where :math:`\sigma` is *density*, and
    *dsource*, a multivector.
    Note that the first product in the integrand
    is a geometric product.

    .. attribute:: dsource

        A :class:`pymbolic.geometric_algebra.MultiVector`.
    """

    def __init__(self, dsource, kernel, density,
            qbx_forced_limit=0, source=None, target=None):
        IntG.__init__(self, kernel, density,
                qbx_forced_limit, source, target)
        self.dsource = dsource

    def __getinitargs__(self):
        return (self.dsource,) + IntG.__getinitargs__(self)

    mapper_method = intern("map_int_g_ds")

# }}}


# {{{ non-dimension-specific operators
#
# (these get made specific to dimensionality by
# pytential.symbolic.mappers.Dimensionalizer)

# {{{ geometric calculus

class NablaComponent(Expression):
    def __init__(self, ambient_axis, nabla_id):
        self.ambient_axis = ambient_axis
        self.nabla_id = nabla_id

    def __getinitargs__(self):
        return (self.ambient_axis, self.nabla_id)

    mapper_method = "map_nabla_component"


class Nabla(Expression):
    def __init__(self, nabla_id):
        self.nabla_id = nabla_id

    def __getinitargs__(self):
        return (self.nabla_id,)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Nabla subscript must be an integer")

        return NablaComponent(index, self.nabla_id)

    mapper_method = "map_nabla"


class DerivativeSource(Expression):
    def __init__(self, operand, nabla_id=None):
        self.operand = operand
        self.nabla_id = nabla_id

    def __getinitargs__(self):
        return (self.operand, self.nabla_id)

    mapper_method = "map_derivative_source"


class Derivative(object):
    _next_id = [0]

    def __init__(self):
        self.my_id = "id%s" % self._next_id[0]
        self._next_id[0] += 1

    @property
    def nabla(self):
        return Nabla(self.my_id)

    def __call__(self, operand):
        return DerivativeSource(operand, self.my_id)

# }}}

S = IntG


def normal_derivative(operand, where=None):
    d = Derivative()
    return (normal(where) * d.nabla) * d(operand)


def Sp(*args, **kwargs):
    where = kwargs.get("where")
    return normal_derivative(S(*args, **kwargs), where).attr("xproject")(0)


def Spp(*args, **kwargs):
    where = kwargs.get("where")
    return normal_derivative(Sp(*args, **kwargs), where).attr("xproject")(0)


def D(*args, **kwargs):
    return IntGdSource(normal(), *args, **kwargs).attr("xproject")(0)


def Dp(*args, **kwargs):
    where = kwargs.get("where")
    return normal_derivative(D(*args, **kwargs), where).attr("xproject")(0)

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


def pretty(expr):
    # Doesn't quite belong here, but this is exposed to the user as
    # "pytential.sym", so in here it goes.

    from pytential.symbolic.mappers import PrettyStringifyMapper
    stringify_mapper = PrettyStringifyMapper()
    from pymbolic.mapper.stringifier import PREC_NONE
    result = stringify_mapper(expr, PREC_NONE)

    splitter = "="*75 + "\n"

    cse_strs = stringify_mapper.get_cse_strings()
    if cse_strs:
        result = "\n".join(cse_strs)+"\n"+splitter+result

    return result

# vim: foldmethod=marker
