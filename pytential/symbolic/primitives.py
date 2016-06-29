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

import six
from six.moves import range, intern
from warnings import warn

import numpy as np
from pymbolic.primitives import (  # noqa
        Expression as ExpressionBase, Variable as var,
        cse_scope as cse_scope_base,
        make_common_subexpression as cse)
from pymbolic.geometric_algebra import MultiVector, componentwise
from pymbolic.geometric_algebra.primitives import (  # noqa
        Nabla, NablaComponent, DerivativeSource, Derivative)
from pymbolic.primitives import make_sym_vector  # noqa


__doc__ = """
.. |where-blurb| replace:: A symbolic name for a
    :class:`pytential.discretization.Discretization`

.. autoclass:: Variable
.. autoclass:: VectorVariable

Functions
^^^^^^^^^

.. data:: real
.. data:: imag
.. data:: conj
.. data:: sqrt
.. data:: abs

Discretization properties
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: QWeight
.. autoclass:: Nodes
.. autoclass:: ParametrizationDerivative
.. autoclass:: pseudoscalar
.. autoclass:: area_element
.. autoclass:: sqrt_jac_q_weight
.. autoclass:: normal

Elementary numerics
^^^^^^^^^^^^^^^^^^^

.. autoclass:: NumReferenceDerivative
.. autoclass:: NodeSum
.. autofunction:: integral
.. autoclass:: Ones
.. autofunction:: ones_vec
.. autofunction:: area
.. autoclass:: IterativeInverse

Layer potentials
^^^^^^^^^^^^^^^^

.. autoclass:: IntG
.. autoclass:: IntGdSource

Internal helpers
^^^^^^^^^^^^^^^^

.. autoclass:: DimensionalizedExpression
"""


class DEFAULT_SOURCE:  # noqa
    pass


class DEFAULT_TARGET:  # noqa
    pass


class cse_scope(cse_scope_base):  # noqa
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

    def __getinitargs__(self):
        return (self.ambient_axis, self.where)

    mapper_method = intern("map_node_coordinate_component")


class Nodes(DiscretizationProperty):
    """Node location of the discretization.
    """

    def __init__(self, where=None):
        DiscretizationProperty.__init__(self, where)

    def __getinitargs__(self):
        return (self.where,)

    mapper_method = intern("map_nodes")


class NumReferenceDerivative(DiscretizationProperty):
    """An operator that takes a derivative
    of *operand* with respect to the the element
    reference coordinates.
    """

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
    """A :class:`pymbolic.geometric_algebra.MultiVector` representing
    the derivative of the reference-to-global parametrization.
    """

    mapper_method = "map_parametrization_derivative"


def pseudoscalar(where=None):
    return cse(
            ParametrizationDerivative(where).a.project_max_grade(),
            "pseudoscalar", cse_scope.DISCRETIZATION)


def area_element(where=None):
    return cse(
            sqrt(pseudoscalar(where).a.norm_squared()),
            "area_element", cse_scope.DISCRETIZATION)


def sqrt_jac_q_weight(where=None):
    return cse(sqrt(area_element(where) * QWeight(where)),
            "sqrt_jac_q_weight", cse_scope.DISCRETIZATION)


def normal(where=None):
    """Exterior unit normals."""

    # Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = pseudoscalar(where) / area_element(where)
    return cse(pder.a.I | pder, "normal",
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
    """Implements a global sum over all discretization nodes."""

    def __init__(self, operand):
        self.operand = operand

    def __getinitargs__(self):
        return (self.operand,)

    mapper_method = "map_node_sum"


def integral(operand, where=None):
    """A volume integral of *operand*."""

    return NodeSum(area_element(where) * QWeight(where) * operand)


class Ones(Expression):
    """A vector that is constant *one* on the whole
    discretization.
    """

    def __init__(self, where=None):
        self.where = where

    def __getinitargs__(self):
        return (self.where,)

    mapper_method = intern("map_ones")


def ones_vec(dim, where=None):
    from pytools.obj_array import make_obj_array
    return DimensionalizedExpression(
            MultiVector(
                make_obj_array(dim*[Ones(where)])))


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
            frozenset(six.iteritems(self.extra_vars)), self.where))

    mapper_method = intern("map_inverse")


# {{{ potentials

def hashable_kernel_args(kernel_arguments):
    hashable_args = []
    for key, val in sorted(kernel_arguments.items()):
        if isinstance(val, np.ndarray):
            val = tuple(val)
        hashable_args.append((key, val))

    return tuple(hashable_args)


class _NoArgSentinel(object):
    pass


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
            qbx_forced_limit, source=None, target=None,
            kernel_arguments=None,
            **kwargs):
        """*target_derivatives* and later arguments should be considered
        keyword-only.

        :arg kernel: a kernel as accepted by
            :func:`sumpy.kernel.to_kernel_and_args`,
            likely a :class:`sumpy.kernel.Kernel`.
        :arg qbx_forced_limit: +1 if the output is required to originate from a
            QBX center on the "+" side of the boundary. -1 for the other side.
            Evaluation at a target with a value of +/- 1 in *qbx_forced_limit*
            will fail if no QBX center is found.

            +2 may be used to *allow* evaluation QBX center on the "+" side of the
            (but disallow evaluation using a center on the "-" side). Potential
            evaluation at the target still succeeds if no applicable QBX center
            is found. (-2 for the analogous behavior on the "-" side.)

            *None* may be used to avoid expressing a side preference for close
            evaluation.

            ``'avg'`` may be used as a shorthand to evaluate this potential
            as an average of the ``+1`` and the ``-1`` value.

        :arg kernel_arguments: A dictionary mapping named
            :class:`sumpy.kernel.Kernel` arguments
            (see :meth:`sumpy.kernel.Kernel.get_args`
            and :meth:`sumpy.kernel.Kernel.get_source_args`
            to expressions that determine them)

        *kwargs* has the same meaning as *kernel_arguments* can be used as a
        more user-friendly interface.
        """

        if kernel_arguments is None:
            kernel_arguments = {}

        if isinstance(kernel_arguments, tuple):
            kernel_arguments = dict(kernel_arguments)

        from sumpy.kernel import to_kernel_and_args
        kernel, kernel_arguments_2 = to_kernel_and_args(kernel)

        for name, val in kernel_arguments_2.items():
            if name in kernel_arguments:
                raise ValueError("'%s' already set in kernel_arguments"
                        % name)
            kernel_arguments[name] = val

        del kernel_arguments_2

        if qbx_forced_limit not in [-1, +1, -2, +2, "avg", None]:
            raise ValueError("invalid value (%s) of qbx_forced_limit"
                    % qbx_forced_limit)

        kernel_arg_names = set(
                karg.loopy_arg.name
                for karg in (
                    kernel.get_args()
                    +
                    kernel.get_source_args()))

        kernel_arguments = kernel_arguments.copy()
        if kwargs:
            for name, val in kwargs.items():
                if name in kernel_arguments:
                    raise ValueError("'%s' already set in kernel_arguments"
                            % name)

                if name not in kernel_arg_names:
                    raise TypeError("'%s' not recognized as kernel argument"
                            % name)

                kernel_arguments[name] = val

        provided_arg_names = set(kernel_arguments.keys())
        missing_args = kernel_arg_names - provided_arg_names
        if missing_args:
            raise TypeError("kernel argument(s) '%s' not supplied"
                    % ", ".join(missing_args))

        extraneous_args = provided_arg_names - kernel_arg_names
        if missing_args:
            raise TypeError("kernel arguments '%s' not recognized"
                    % ", ".join(extraneous_args))

        self.kernel = kernel
        self.density = density
        self.qbx_forced_limit = qbx_forced_limit
        self.source = source
        self.target = target
        self.kernel_arguments = kernel_arguments

    def copy(self, kernel=None, density=None, qbx_forced_limit=_NoArgSentinel,
            source=None, target=None, kernel_arguments=None):
        kernel = kernel or self.kernel
        density = density or self.density
        if qbx_forced_limit is _NoArgSentinel:
            qbx_forced_limit = self.qbx_forced_limit
        source = source or self.source
        target = target or self.target
        kernel_arguments = kernel_arguments or self.kernel_arguments
        return type(self)(kernel, density, qbx_forced_limit, source, target,
                kernel_arguments)

    def __getinitargs__(self):
        return (self.kernel, self.density, self.qbx_forced_limit,
                self.source, self.target,
                hashable_kernel_args(self.kernel_arguments))

    mapper_method = intern("map_int_g")


class IntGdSource(IntG):
    # FIXME: Unclear if this class is still fully needed
    # now that the kernel_arguments mechanism exists.

    r"""
    .. math::

        \int_\Gamma \operatorname{dsource} \dot \nabla_y
            \dot g(x-y) \sigma(y) dS_y

    where :math:`\sigma` is *density*, and
    *dsource*, a multivector.
    Note that the first product in the integrand
    is a geometric product.

    .. attribute:: dsource

        A :class:`pymbolic.geometric_algebra.MultiVector`.
    """

    def __init__(self, dsource, kernel, density,
            qbx_forced_limit=0, source=None, target=None,
            kernel_arguments=None,
            **kwargs):
        IntG.__init__(self, kernel, density,
                qbx_forced_limit, source, target,
                kernel_arguments=kernel_arguments,
                **kwargs)
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

class _unspecified:
    pass

def S(kernel, density,
        qbx_forced_limit=_unspecified, source=None, target=None,
        kernel_arguments=None, **kwargs):

    if qbx_forced_limit is _unspecified:
        warn("not specifying qbx_forced_limit on call to 'S' is deprecated, "
                "defaulting to +1", DeprecationWarning, stacklevel=2)
        qbx_forced_limit = +1

    return IntG(kernel, density, qbx_forced_limit, source, target,
            kernel_arguments, **kwargs)


def tangential_derivative(operand, where=None):
    pder = pseudoscalar(where) / area_element(where)

    # FIXME: Should be formula (3.25) in Dorst et al.
    d = Derivative()
    return (d.nabla * d(operand)) >> pder


def normal_derivative(operand, where=None):
    d = Derivative()
    return (normal(where).a.scalar_product(d.nabla)) * d(operand)


def Sp(*args, **kwargs):  # noqa
    where = kwargs.get("target")
    if "qbx_forced_limit" not in kwargs:
        warn("not specifying qbx_forced_limit on call to 'Sp' is deprecated, "
                "defaulting to 'avg'", DeprecationWarning, stacklevel=2)
        kwargs["qbx_forced_limit"] = "avg"

    return normal_derivative(S(*args, **kwargs), where)


def Spp(*args, **kwargs):  # noqa
    where = kwargs.get("target")
    return normal_derivative(Sp(*args, **kwargs), where)


def D(*args, **kwargs):  # noqa
    where = kwargs.get("source")
    if "qbx_forced_limit" not in kwargs:
        warn("not specifying qbx_forced_limit on call to 'D' is deprecated, "
                "defaulting to 'avg'", DeprecationWarning, stacklevel=2)
        kwargs["qbx_forced_limit"] = "avg"
    return IntGdSource(normal(where), *args, **kwargs).a.xproject(0)


def Dp(*args, **kwargs):  # noqa
    target = kwargs.get("target")
    if "qbx_forced_limit" not in kwargs:
        warn("not specifying qbx_forced_limit on call to 'Dp' is deprecated, "
                "defaulting to +1", DeprecationWarning, stacklevel=2)
        kwargs["qbx_forced_limit"] = +1
    return normal_derivative(D(*args, **kwargs), target)

# }}}

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
