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
from six.moves import intern
from warnings import warn

import numpy as np
from pymbolic.primitives import (  # noqa: F401, N813
        Expression as ExpressionBase, Variable as var,
        cse_scope as cse_scope_base,
        make_common_subexpression as cse)
from pymbolic.geometric_algebra import MultiVector, componentwise
from pymbolic.geometric_algebra.primitives import (  # noqa: F401
        NablaComponent, DerivativeSource, Derivative as DerivativeBase)
from pymbolic.primitives import make_sym_vector  # noqa: F401
from pytools.obj_array import make_obj_array  # noqa: F401

from functools import partial


__doc__ = """
.. |where-blurb| replace:: A symbolic name for a
    :class:`pytential.discretization.Discretization`

Object types
^^^^^^^^^^^^
Based on the mathematical quantity being represented, the following types of
objects occur as part of a symbolic operator representation:

*   If a quantity is a scalar, it is just a symbolic expression--that is, a nested
    combination of placeholders (see below), arithmetic on them (see
    :mod:`pymbolic.primitives`. These objects are created simply by doing
    arithmetic on placeholders.

*   If the quantity is "just a bunch of scalars" (like, say, rows in a system
    of integral equations), the symbolic representation an object array. Each
    element of the object array contains a symbolic expression.

    :func:`pytools.obj_array.make_obj_array` and
    :func:`pytools.obj_array.join_fields`
    can help create those.

*   If it is a geometric quantity (that makes sense without explicit reference to
    coordinates), it is a :class:`pymbolic.geometric_algebra.MultiVector`.
    This can be converted to an object array by calling :
    :meth:`pymbolic.geometric_algebra.MultiVector.as_vector`.

:mod:`pyopencl.array.Array` instances do not occur on the symbolic of
:mod:`pymbolic` at all.  Those hold per-node degrees of freedom (and only
those), which is not visible as an array axis in symbolic code. (They're
visible only once evaluated.)

Placeholders
^^^^^^^^^^^^

.. autoclass:: Variable
.. autoclass:: make_sym_vector
.. autoclass:: make_sym_mv
.. autoclass:: make_sym_surface_mv

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
.. autofunction:: nodes
.. autofunction:: parametrization_derivative
.. autofunction:: parametrization_derivative_matrix
.. autofunction:: pseudoscalar
.. autofunction:: area_element
.. autofunction:: sqrt_jac_q_weight
.. autofunction:: normal

Elementary numerics
^^^^^^^^^^^^^^^^^^^

.. autoclass:: NumReferenceDerivative
.. autoclass:: NodeSum
.. autofunction:: integral
.. autoclass:: Ones
.. autofunction:: ones_vec
.. autofunction:: area
.. autoclass:: IterativeInverse

Calculus
^^^^^^^^
.. autoclass:: Derivative
.. autofunction:: dd_axis
.. autofunction:: d_dx
.. autofunction:: d_dy
.. autofunction:: d_dz
.. autofunction:: grad_mv
.. autofunction:: grad

Layer potentials
^^^^^^^^^^^^^^^^

.. autoclass:: IntG
.. autofunction:: int_g_dsource
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


def make_sym_mv(name, num_components):
    return MultiVector(make_sym_vector(name, num_components))


def make_sym_surface_mv(name, ambient_dim, dim, where=None):
    par_grad = parametrization_derivative_matrix(ambient_dim, dim, where)

    return sum(
            var("%s%d" % (name, i))
            *
            cse(MultiVector(vec), "tangent%d" % i, cse_scope.DISCRETIZATION)
            for i, vec in enumerate(par_grad.T))


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
            return var.__call__(self, operand, *args, **kwargs)


class EvalMapperFunction(Function):
    pass


class CLMathFunction(Function):
    pass


real = EvalMapperFunction("real")
imag = EvalMapperFunction("imag")
conj = EvalMapperFunction("conj")
abs = EvalMapperFunction("abs")

sqrt = CLMathFunction("sqrt")

sin = CLMathFunction("sin")
cos = CLMathFunction("cos")
tan = CLMathFunction("tan")

asin = CLMathFunction("asin")
acos = CLMathFunction("acos")
atan = CLMathFunction("atan")
atan2 = CLMathFunction("atan2")

sinh = CLMathFunction("sinh")
cosh = CLMathFunction("cosh")
tanh = CLMathFunction("tanh")

asinh = CLMathFunction("asinh")
acosh = CLMathFunction("acosh")
atanh = CLMathFunction("atanh")

exp = CLMathFunction("exp")
log = CLMathFunction("log")


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


def nodes(ambient_dim, where=None):
    """Return a :class:`pymbolic.geometric_algebra.MultiVector` of node
    locations.
    """

    return MultiVector(
            make_obj_array([
                NodeCoordinateComponent(i, where)
                for i in range(ambient_dim)]))


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


def reference_jacobian(field, field_dim, dim, where=None):
    """Return a :class:`np.array` representing the Jacobian of a vector field with
    respect to the reference coordinates.
    """
    jac = np.zeros((field_dim, dim), np.object)

    for i in range(field_dim):
        field_component = field[i]
        for j in range(dim):
            jac[i, j] = NumReferenceDerivative(
                frozenset([j]),
                field_component,
                where)

    return jac


def parametrization_derivative_matrix(ambient_dim, dim, where=None):
    """Return a :class:`np.array` representing the derivative of the
    reference-to-global parametrization.
    """

    return reference_jacobian(
            [NodeCoordinateComponent(i, where) for i in range(ambient_dim)],
            ambient_dim, dim)


def parametrization_derivative(ambient_dim, dim, where=None):
    """Return a :class:`pymbolic.geometric_algebra.MultiVector` representing
    the derivative of the reference-to-global parametrization.
    """

    par_grad = parametrization_derivative_matrix(ambient_dim, dim, where)

    from pytools import product
    return product(MultiVector(vec) for vec in par_grad.T)


def pseudoscalar(ambient_dim, dim=None, where=None):
    """
    Same as the outer product of all parametrization derivative columns.
    """
    if dim is None:
        dim = ambient_dim - 1

    return cse(
            parametrization_derivative(ambient_dim, dim, where)
            .project_max_grade(),
            "pseudoscalar", cse_scope.DISCRETIZATION)


def area_element(ambient_dim, dim=None, where=None):
    return cse(
            sqrt(pseudoscalar(ambient_dim, dim, where).norm_squared()),
            "area_element", cse_scope.DISCRETIZATION)


def sqrt_jac_q_weight(ambient_dim, dim=None, where=None):
    return cse(
            sqrt(
                area_element(ambient_dim, dim, where)
                * QWeight(where)),
            "sqrt_jac_q_weight", cse_scope.DISCRETIZATION)


def normal(ambient_dim, dim=None, where=None):
    """Exterior unit normals."""

    # Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = (
            pseudoscalar(ambient_dim, dim, where)
            / area_element(ambient_dim, dim, where))
    return cse(
            # Dorst Section 3.7.2
            pder << pder.I.inv(),
            cse_scope.DISCRETIZATION)


def mean_curvature(ambient_dim, dim=None, where=None):
    """(Numerical) mean curvature."""

    if dim is None:
        dim = ambient_dim - 1

    if not (dim == 1 and ambient_dim == 2):
        raise NotImplementedError(
                "only know how to calculate curvature for a curve in 2D")

    xp, yp = cse(
            parametrization_derivative_matrix(ambient_dim, dim, where),
            "pd_matrix", cse_scope.DISCRETIZATION)

    xpp, ypp = cse(
            reference_jacobian([xp[0], yp[0]], ambient_dim, dim, where),
            "p2d_matrix", cse_scope.DISCRETIZATION)

    return (xp[0]*ypp[0] - yp[0]*xpp[0]) / (xp[0]**2 + yp[0]**2)**(3/2)

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

    def __new__(cls, operand):
        # If the constructor is handed a multivector object, return an
        # object array of the operator applied to each of the
        # coefficients in the multivector.

        if isinstance(operand, (np.ndarray, MultiVector)):
            def make_op(operand_i):
                return cls(operand_i)

            return componentwise(make_op, operand)
        else:
            return Expression.__new__(cls)

    def __init__(self, operand):
        self.operand = operand

    def __getinitargs__(self):
        return (self.operand,)

    mapper_method = "map_node_sum"


def integral(ambient_dim, dim, operand, where=None):
    """A volume integral of *operand*."""

    return NodeSum(
            area_element(ambient_dim, dim, where)
            * QWeight(where)
            * operand)


class Ones(Expression):
    """A DOF-vector that is constant *one* on the whole
    discretization.
    """

    def __init__(self, where=None):
        self.where = where

    def __getinitargs__(self):
        return (self.where,)

    mapper_method = intern("map_ones")


def ones_vec(dim, where=None):
    from pytools.obj_array import make_obj_array
    return MultiVector(
                make_obj_array(dim*[Ones(where)]))


def area(ambient_dim, dim, where=None):
    return cse(integral(ambient_dim, dim, Ones(where), where), "area",
            cse_scope.DISCRETIZATION)


def mean(ambient_dim, dim, operand, where=None):
    return (
            integral(ambient_dim, dim, operand, where)
            /
            area(ambient_dim, dim, where))


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


class Derivative(DerivativeBase):
    @property
    def nabla(self):
        raise ValueError("Derivative.nabla should not be used"
                "--use Derivative.dnabla instead. (Note the extra 'd')"
                "To explain: 'nabla' was intended to be "
                "dimension-independent, which turned out to be a bad "
                "idea.")

    @staticmethod
    def resolve(expr):
        from pytential.symbolic.mappers import DerivativeBinder
        return DerivativeBinder()(expr)


def dd_axis(axis, ambient_dim, operand):
    """Return the derivative along (XYZ) axis *axis*
    (in *ambient_dim*-dimensional space) of *operand*.
    """
    d = Derivative()

    unit_vector = np.zeros(ambient_dim)
    unit_vector[axis] = 1

    unit_mvector = MultiVector(unit_vector)

    return d.resolve(
            (unit_mvector.scalar_product(d.dnabla(ambient_dim)))
            * d(operand))


d_dx = partial(dd_axis, 0)
d_dy = partial(dd_axis, 1)
d_dz = partial(dd_axis, 2)


def grad_mv(ambient_dim, operand):
    """Return the *ambient_dim*-dimensional gradient of
    *operand* as a :class:`pymbolic.geometric_algebra.MultiVector`.
    """

    d = Derivative()
    return d.resolve(
            d.dnabla(ambient_dim) * d(operand))


def grad(ambient_dim, operand):
    """Return the *ambient_dim*-dimensional gradient of
    *operand* as a :class:`numpy.ndarray`.
    """
    return grad_mv(ambient_dim, operand).as_vector()


def laplace(ambient_dim, operand):
    d = Derivative()
    nabla = d.dnabla(ambient_dim)
    return d.resolve(nabla | d(
        d.resolve((nabla * d(operand))))).as_scalar()


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

        if isinstance(density, (np.ndarray, MultiVector)):
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

        :arg source: The symbolic name of the source discretization. This name
            is bound to a concrete :class:`pytential.source.LayerPotentialSourceBase`
            by :func:`pytential.bind`.

        :arg target: The symbolic name of the set of targets. This name gets
            assigned to a concrete target set by :func:`pytential.bind`.

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


_DIR_VEC_NAME = "dsource_vec"


def _insert_source_derivative_into_kernel(kernel):
    # Inserts the source derivative at the innermost
    # kernel wrapping level.
    from sumpy.kernel import DirectionalSourceDerivative

    if kernel.get_base_kernel() is kernel:
        return DirectionalSourceDerivative(
                kernel, dir_vec_name=_DIR_VEC_NAME)
    else:
        return kernel.replace_inner_kernel(
                _insert_source_derivative_into_kernel(kernel.kernel))


def _get_dir_vec(dsource, ambient_dim):
    from pymbolic.mapper.coefficient import (
            CoefficientCollector as CoefficientCollectorBase)

    class _DSourceCoefficientFinder(CoefficientCollectorBase):
        def map_nabla_component(self, expr):
            return {expr: 1}

        def map_variable(self, expr):
            return {1: expr}

        def map_common_subexpression(self, expr):
            return {1: expr}

        def map_quotient(self, expr):
            return {1: expr}

    coeffs = _DSourceCoefficientFinder()(dsource)

    dir_vec = np.zeros(ambient_dim, np.object)
    for i in range(ambient_dim):
        dir_vec[i] = coeffs.pop(NablaComponent(i, None), 0)

    if coeffs:
        raise RuntimeError("source derivative expression contained constant term")

    return dir_vec


def int_g_dsource(ambient_dim, dsource, kernel, density,
            qbx_forced_limit, source=None, target=None,
            kernel_arguments=None, **kwargs):
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

    if kernel_arguments is None:
        kernel_arguments = {}

    if isinstance(kernel_arguments, tuple):
        kernel_arguments = dict(kernel_arguments)

    kernel = _insert_source_derivative_into_kernel(kernel)

    from pytools.obj_array import make_obj_array
    nabla = MultiVector(make_obj_array(
        [NablaComponent(axis, None)
            for axis in range(ambient_dim)]))

    def add_dir_vec_to_kernel_args(coeff):
        result = kernel_arguments.copy()
        result[_DIR_VEC_NAME] = _get_dir_vec(coeff, ambient_dim)
        return result

    density = cse(density)
    return (dsource*nabla).map(
            lambda coeff: IntG(
                kernel,
                density, qbx_forced_limit, source, target,
                kernel_arguments=add_dir_vec_to_kernel_args(coeff),
                **kwargs))

# }}}


# {{{ non-dimension-specific operators
#
# (these get made specific to dimensionality by
# pytential.symbolic.mappers.Dimensionalizer)

# {{{ geometric calculus


class _unspecified:  # noqa
    pass


def S(kernel, density,  # noqa
        qbx_forced_limit=_unspecified, source=None, target=None,
        kernel_arguments=None, **kwargs):

    if qbx_forced_limit is _unspecified:
        warn("not specifying qbx_forced_limit on call to 'S' is deprecated, "
                "defaulting to +1", DeprecationWarning, stacklevel=2)
        qbx_forced_limit = +1

    return IntG(kernel, density, qbx_forced_limit, source, target,
            kernel_arguments, **kwargs)


def tangential_derivative(ambient_dim, operand, dim=None, where=None):
    pder = (
            pseudoscalar(ambient_dim, dim, where)
            / area_element(ambient_dim, dim, where))

    # FIXME: Should be formula (3.25) in Dorst et al.
    d = Derivative()
    return d.resolve(
            (d.dnabla(ambient_dim) * d(operand)) >> pder)


def normal_derivative(ambient_dim, operand, dim=None, where=None):
    d = Derivative()
    return d.resolve(
            (normal(ambient_dim, dim, where).scalar_product(d.dnabla(ambient_dim)))
            * d(operand))


def Sp(kernel, *args, **kwargs):  # noqa
    where = kwargs.get("target")
    if "qbx_forced_limit" not in kwargs:
        warn("not specifying qbx_forced_limit on call to 'Sp' is deprecated, "
                "defaulting to 'avg'", DeprecationWarning, stacklevel=2)
        kwargs["qbx_forced_limit"] = "avg"

    ambient_dim = kwargs.get("ambient_dim")
    from sumpy.kernel import Kernel
    if ambient_dim is None and isinstance(kernel, Kernel):
        ambient_dim = kernel.dim
    if ambient_dim is None:
        raise ValueError("ambient_dim must be specified, either through "
                "the kernel, or directly")
    dim = kwargs.pop("dim", None)

    return normal_derivative(
            ambient_dim,
            S(kernel, *args, **kwargs),
            dim=dim, where=where)


def Spp(kernel, *args, **kwargs):  # noqa
    ambient_dim = kwargs.get("ambient_dim")
    from sumpy.kernel import Kernel
    if ambient_dim is None and isinstance(kernel, Kernel):
        ambient_dim = kernel.dim
    if ambient_dim is None:
        raise ValueError("ambient_dim must be specified, either through "
                "the kernel, or directly")
    dim = kwargs.pop("dim", None)

    where = kwargs.get("target")
    return normal_derivative(
            ambient_dim,
            Sp(kernel, *args, **kwargs),
            dim=dim, where=where)


def D(kernel, *args, **kwargs):  # noqa
    ambient_dim = kwargs.get("ambient_dim")
    from sumpy.kernel import Kernel
    if ambient_dim is None and isinstance(kernel, Kernel):
        ambient_dim = kernel.dim
    if ambient_dim is None:
        raise ValueError("ambient_dim must be specified, either through "
                "the kernel, or directly")
    dim = kwargs.pop("dim", None)

    where = kwargs.get("source")

    if "qbx_forced_limit" not in kwargs:
        warn("not specifying qbx_forced_limit on call to 'D' is deprecated, "
                "defaulting to 'avg'", DeprecationWarning, stacklevel=2)
        kwargs["qbx_forced_limit"] = "avg"

    return int_g_dsource(
            ambient_dim,
            normal(ambient_dim, dim, where),
            kernel, *args, **kwargs).xproject(0)


def Dp(kernel, *args, **kwargs):  # noqa
    ambient_dim = kwargs.get("ambient_dim")
    from sumpy.kernel import Kernel
    if ambient_dim is None and isinstance(kernel, Kernel):
        ambient_dim = kernel.dim
    if ambient_dim is None:
        raise ValueError("ambient_dim must be specified, either through "
                "the kernel, or directly")
    dim = kwargs.pop("dim", None)
    target = kwargs.get("target")
    if "qbx_forced_limit" not in kwargs:
        warn("not specifying qbx_forced_limit on call to 'Dp' is deprecated, "
                "defaulting to +1", DeprecationWarning, stacklevel=2)
        kwargs["qbx_forced_limit"] = +1
    return normal_derivative(
            ambient_dim,
            D(kernel, *args, **kwargs),
            dim=dim, where=target)

# }}}

# }}}

# }}}


# {{{ geometric operations

def tangential_onb(ambient_dim, dim=None, where=None):
    if dim is None:
        dim = ambient_dim - 1

    pd_mat = cse(parametrization_derivative_matrix(ambient_dim, dim, where),
            "pd_matrix", cse_scope.DISCRETIZATION)

    # {{{ Gram-Schmidt

    orth_pd_mat = np.zeros_like(pd_mat)
    for k in range(pd_mat.shape[1]):
        avec = pd_mat[:, k]
        q = avec
        for j in range(k):
            q = q - np.dot(avec, orth_pd_mat[:, j])*orth_pd_mat[:, j]

        orth_pd_mat[:, k] = cse(q/sqrt(np.sum(q**2)), "orth_pd_vec")

    # }}}

    return orth_pd_mat


def xyz_to_tangential(xyz_vec, where=None):
    ambient_dim = len(xyz_vec)
    tonb = tangential_onb(ambient_dim, where=where)
    return make_obj_array([
        np.dot(tonb[:, i], xyz_vec)
        for i in range(ambient_dim - 1)
        ])


def tangential_to_xyz(tangential_vec, where=None):
    ambient_dim = len(tangential_vec) + 1
    tonb = tangential_onb(ambient_dim, where=where)
    return sum(
        tonb[:, i] * tangential_vec[i]
        for i in range(ambient_dim - 1))


def project_to_tangential(xyz_vec, which=None):
    return tangential_to_xyz(
            cse(xyz_to_tangential(xyz_vec, which), which))

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
