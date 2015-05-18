from __future__ import division
from __future__ import absolute_import
import six
from six.moves import range
from functools import reduce

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
from pymbolic.mapper.stringifier import (
        CSESplittingStringifyMapperMixin,
        PREC_NONE, PREC_PRODUCT)
from pymbolic.mapper import (
        Mapper,
        CSECachingMapperMixin
        )
from pymbolic.mapper.dependency import (
        DependencyMapper as DependencyMapperBase)
from pymbolic.mapper.coefficient import (
        CoefficientCollector as CoefficientCollectorBase)
from pymbolic.geometric_algebra import MultiVector
from pymbolic.geometric_algebra.mapper import (
        CombineMapper as CombineMapperBase,
        IdentityMapper as IdentityMapperBase,
        Collector as CollectorBase,
        DerivativeBinder as DerivativeBinderBase,
        EvaluationMapper as EvaluationMapperBase,

        Dimensionalizer as DimensionalizerBase,

        StringifyMapper as BaseStringifyMapper,

        DerivativeSourceAndNablaComponentCollector
        as DerivativeSourceAndNablaComponentCollectorBase,
        NablaComponentToUnitVector
        as NablaComponentToUnitVectorBase,
        DerivativeSourceFinder
        as DerivativeSourceFinderBase,


        GraphvizMapper as GraphvizMapperBase)
import pytential.symbolic.primitives as prim


class IdentityMapper(IdentityMapperBase):
    def map_dimensionalized_expression(self, expr):
        return type(expr)(self.rec(expr.child))

    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

    def map_num_reference_derivative(self, expr):
        return type(expr)(expr.ref_axes, self.rec(expr.operand),
                expr.where)

    # {{{ childless -- no need to rebuild

    def map_vector_variable(self, expr):
        return expr

    map_ones = map_vector_variable
    map_q_weight = map_vector_variable
    map_node_coordinate_component = map_vector_variable
    map_parametrization_gradient = map_vector_variable
    map_parametrization_derivative = map_vector_variable

    # }}}

    def map_inverse(self, expr):
        return type(expr)(
                # don't recurse into expression--it is a separate world that
                # will be processed once it's executed.

                expr.expression, self.rec(expr.rhs), expr.variable_name,
                dict([
                    (name, self.rec(name_expr))
                    for name, name_expr in six.iteritems(expr.extra_vars)]),
                expr.where)

    def map_int_g(self, expr):
        return expr.copy(
                density=self.rec(expr.density),
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

    def map_int_g_ds(self, expr):
        return expr.copy(
                density=self.rec(expr.density),
                dsource=self.rec(expr.dsource),
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))


class CombineMapper(CombineMapperBase):
    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    map_num_reference_derivative = map_node_sum

    def map_int_g(self, expr):
        result = self.rec(expr.density)
        for arg_expr in expr.kernel_arguments.values():
            result.update(self.rec(arg_expr))
        return result

    def map_int_g_ds(self, expr):
        return self.rec(expr.dsource) | self.map_int_g(expr)

    def map_inverse(self, expr):
        from operator import or_
        return self.rec(expr.rhs) | reduce(or_,
                (self.rec(name_expr)
                for name_expr in six.itervalues(expr.extra_vars)),
                set())


class Collector(CollectorBase, CombineMapper):
    def map_vector_variable(self, expr):
        return set()

    map_ones = map_vector_variable
    map_node_coordinate_component = map_vector_variable
    map_parametrization_derivative = map_vector_variable
    map_q_weight = map_vector_variable


class OperatorCollector(Collector):
    def map_int_g(self, expr):
        return set([expr]) | Collector.map_int_g(self, expr)

    def map_int_g_ds(self, expr):
        return set([expr]) | Collector.map_int_g(self, expr)


class DependencyMapper(DependencyMapperBase, Collector):
    pass


class EvaluationMapper(EvaluationMapperBase):
    """Unlike :mod:`pymbolic.mapper.evaluation.EvaluationMapper`, this class
    does evaluation mostly to get :class:`pymbolic.geometric_algebra.MultiVector`
    instances to to do their thing, and perhaps to automatically kill terms
    that are multiplied by zero. Otherwise it intends to largely preserve
    the structure of the input expression.
    """

    def map_variable(self, expr):
        return expr

    def map_vector_variable(self, expr):
        return expr

    def map_subscript(self, expr):
        return self.rec(expr.aggregate)[self.rec(expr.index)]

    map_q_weight = map_variable
    map_ones = map_variable

    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

    def map_node_coordinate_component(self, expr):
        return expr

    def map_num_reference_derivative(self, expr):
        return type(expr)(expr.ref_axes, self.rec(expr.operand), expr.where)

    def map_int_g(self, expr):
        return type(expr)(
                expr.kernel,
                self.rec(expr.density),
                expr.qbx_forced_limit, expr.source, expr.target,
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

    def map_int_g_ds(self, expr):
        return type(expr)(
                self.rec(expr.dsource),
                expr.kernel,
                self.rec(expr.density),
                expr.qbx_forced_limit, expr.source, expr.target,
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

    def map_common_subexpression(self, expr):
        return prim.cse(
                self.rec(expr.child),
                expr.prefix,
                expr.scope)


# {{{ target/source tagging

class LocationTagger(CSECachingMapperMixin, IdentityMapper):
    """Used internally by :class:`ToTargetTagger`. Tags all src/target taggable
    leaves/operators as going back onto the source.
    """
    def __init__(self, default_where, default_source=prim.DEFAULT_SOURCE):
        self.default_source = default_source
        self.default_where = default_where

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_ones(self, expr):
        if expr.where is None:
            return type(expr)(where=self.default_where)
        else:
            return expr

    map_q_weight = map_ones

    def map_parametrization_derivative_component(self, expr):
        if expr.where is None:
            return type(expr)(
                    expr.ambient_axis, expr.ref_axis, self.default_where)
        else:
            return expr

    map_parametrization_derivative = map_ones
    map_nodes = map_ones

    def map_node_coordinate_component(self, expr):
        if expr.where is None:
            return type(expr)(
                    expr.ambient_axis, self.default_where)
        else:
            return expr

    def map_num_reference_derivative(self, expr):
        if expr.where is None:
            return type(expr)(
                    expr.ref_axes, self.rec(expr.operand), self.default_where)
        else:
            return expr

    def map_int_g(self, expr):
        source = expr.source
        target = expr.target

        if source is None:
            source = self.default_source
        if target is None:
            target = self.default_where

        return type(expr)(
                expr.kernel,
                self.operand_rec(expr.density),
                expr.qbx_forced_limit, source, target,
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

    def map_int_g_ds(self, expr):
        source = expr.source
        target = expr.target

        if source is None:
            source = self.default_source
        if target is None:
            target = self.default_where

        return type(expr)(
                self.operand_rec(expr.dsource),
                expr.kernel,
                self.operand_rec(expr.density),
                expr.qbx_forced_limit, source, target,
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

    def map_inverse(self, expr):
        where = expr.where

        if where is None:
            where = self.default_where

        return type(expr)(
                # don't recurse into expression--it is a separate world that
                # will be processed once it's executed.

                expr.expression, self.rec(expr.rhs), expr.variable_name,
                dict([
                    (name, self.rec(name_expr))
                    for name, name_expr in six.iteritems(expr.extra_vars)]),
                where)

    def operand_rec(self, expr):
        return self.rec(expr)


class ToTargetTagger(LocationTagger):
    """Descends into the expression tree, marking everything up to the first
    layer potential operator as operating on the targets, and everything below
    there as operating on the source. Also performs a consistency check such
    that only 'target' and 'source' items are combined arithmetically.
    """

    def __init__(self, default_source, default_target):
        LocationTagger.__init__(self, default_target)
        self.operand_rec = LocationTagger(default_source)

# }}}


# {{{ dimensionalizer

class _DSourceCoefficientFinder(CoefficientCollectorBase):
    def map_nabla_component(self, expr):
        return {expr: 1}

    def map_variable(self, expr):
        return {1: expr}

    def map_common_subexpression(self, expr):
        return {1: expr}


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
    coeffs = _DSourceCoefficientFinder()(dsource)

    dir_vec = np.zeros(ambient_dim, np.object)
    for i in range(ambient_dim):
        dir_vec[i] = coeffs.pop(prim.NablaComponent(i, None), 0)

    if coeffs:
        raise RuntimeError("source derivative expression contained constant term")

    return dir_vec


class Dimensionalizer(DimensionalizerBase, EvaluationMapper):
    """Once the discretization is known, the dimension count is, too.
    This mapper plugs in dimension-specific quantities for their
    non-dimensional symbolic counterparts.
    """

    def __init__(self, discr_dict):
        self.discr_dict = discr_dict
        super(Dimensionalizer, self).__init__()

    @property
    def ambient_dim(self):
        from pytools import single_valued
        return single_valued(
                discr.ambient_dim
                for discr in six.itervalues(self.discr_dict))

    def map_vector_variable(self, expr):
        from pymbolic import make_sym_vector
        num_components = expr.num_components

        if num_components is None:
            num_components = self.ambient_dim

        return MultiVector(make_sym_vector(expr.name, num_components))

    def map_dimensionalized_expression(self, expr):
        return expr.child

    def map_parametrization_derivative(self, expr):
        discr = self.discr_dict[expr.where]

        from pytential.qbx import LayerPotentialSource
        if isinstance(discr, LayerPotentialSource):
            discr = discr.fine_density_discr

        from meshmode.discretization import Discretization
        if not isinstance(discr, Discretization):
            raise RuntimeError("Cannot compute the parametrization derivative "
                    "of something that is not a discretization (a target perhaps?). "
                    "For example, you will receive this error if you try to "
                    "evaluate S' in the volume.")

        par_grad = np.zeros((discr.ambient_dim, discr.dim), np.object)
        for i in range(discr.ambient_dim):
            for j in range(discr.dim):
                par_grad[i, j] = prim.NumReferenceDerivative(
                        frozenset([j]),
                        prim.NodeCoordinateComponent(i, expr.where),
                        expr.where)

        from pytools import product
        return product(MultiVector(vec) for vec in par_grad.T)

    def map_nodes(self, expr):
        discr = self.discr_dict[expr.where]
        from pytools.obj_array import make_obj_array
        return MultiVector(
                make_obj_array([
                    prim.NodeCoordinateComponent(i, expr.where)
                    for i in range(discr.ambient_dim)]))

    def map_int_g(self, expr):
        from sumpy.kernel import KernelDimensionSetter
        return type(expr)(
                KernelDimensionSetter(self.ambient_dim)(expr.kernel),
                self.rec(expr.density),
                expr.qbx_forced_limit, expr.source, expr.target,
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

    def map_int_g_ds(self, expr):
        dsource = self.rec(expr.dsource)

        ambient_dim = self.ambient_dim

        from sumpy.kernel import KernelDimensionSetter
        kernel = _insert_source_derivative_into_kernel(
                KernelDimensionSetter(ambient_dim)(expr.kernel))

        from pytools.obj_array import make_obj_array
        nabla = MultiVector(make_obj_array(
            [prim.NablaComponent(axis, None)
                for axis in range(ambient_dim)]))

        kernel_arguments = dict(
                (name, self.rec(arg_expr))
                for name, arg_expr in expr.kernel_arguments.items()
                )

        def add_dir_vec_to_kernel_args(coeff):
            result = kernel_arguments.copy()
            result[_DIR_VEC_NAME] = _get_dir_vec(coeff, ambient_dim)
            return result

        rec_operand = prim.cse(self.rec(expr.density))
        return (dsource*nabla).map(
                lambda coeff: prim.IntG(
                    kernel,
                    rec_operand, expr.qbx_forced_limit, expr.source, expr.target,
                    kernel_arguments=add_dir_vec_to_kernel_args(coeff)))

# }}}


# {{{ derivative binder

class DerivativeTaker(Mapper):
    def __init__(self, ambient_axis):
        self.ambient_axis = ambient_axis

    def map_int_g(self, expr):
        from sumpy.kernel import AxisTargetDerivative
        return expr.copy(kernel=AxisTargetDerivative(self.ambient_axis, expr.kernel))


class DerivativeSourceAndNablaComponentCollector(
        Collector,
        DerivativeSourceAndNablaComponentCollectorBase):
    pass


class NablaComponentToUnitVector(
        EvaluationMapper,
        NablaComponentToUnitVectorBase):
    pass


class DerivativeSourceFinder(EvaluationMapper,
        DerivativeSourceFinderBase):
    pass


class DerivativeBinder(DerivativeBinderBase, IdentityMapper):
    derivative_source_and_nabla_component_collector = \
            DerivativeSourceAndNablaComponentCollector
    nabla_component_to_unit_vector = NablaComponentToUnitVector
    derivative_source_finder = DerivativeSourceFinder

    def take_derivative(self, ambient_axis, expr):
        return DerivativeTaker(ambient_axis)(expr)

# }}}


# {{{ QBX preprocessor

class QBXPreprocessor(IdentityMapper):
    def __init__(self, source_name, places):
        self.source_name = source_name
        self.places = places

    def map_int_g(self, expr):
        if expr.source != self.source_name:
            # not ours
            return IdentityMapper.map_int_g(self, expr)

        source = self.places[self.source_name]
        target_discr = self.places[expr.target]

        from pytential.qbx import LayerPotentialSource
        if isinstance(target_discr, LayerPotentialSource):
            target_discr = target_discr.density_discr

        assert expr.qbx_forced_limit is not None
        if expr.qbx_forced_limit != 0 \
                or source.density_discr is not target_discr:
            # Not computing the self-on-surface value, nothing to do.
            return IdentityMapper.map_int_g(self, expr)

        expr = expr.copy(
                kernel=expr.kernel,
                density=self.rec(expr.density),
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

        from sumpy.kernel import DerivativeCounter
        num_derivatives = DerivativeCounter()(expr.kernel)

        if num_derivatives == 0:
            # either side will do
            return expr.copy(qbx_forced_limit=+1)
        else:  # if num_derivatives == 1:
            # Assume it's a PV integral, preserve 'numerical compactness' by using
            # two-sided average.
            return 0.5*(
                    expr.copy(qbx_forced_limit=+1)
                    + expr.copy(qbx_forced_limit=-1))
        # else:
        #     # FIXME
        #     # from sumpy.qbx import find_jump_term
        #     # jump_term = find_jump_term(expr.kernel,
        #     #         _QBXJumpTermSymbolicArgumentProvider(expr.source))
        #     raise NotImplementedError()

    def map_int_g_ds(self, expr):
        raise RuntimeError("user-facing source derivative operators are expected "
                "to have been eliminated by the time QBXPreprocessor is called.")

# }}}


# {{{ stringifier

def stringify_where(where):
    if where is None:
        return "?"
    elif where is prim.DEFAULT_SOURCE:
        return "s"
    elif where is prim.DEFAULT_TARGET:
        return "t"
    else:
        return str(where)


class StringifyMapper(BaseStringifyMapper):

    def map_nodes(self, expr, enclosing_prec):
        return "x"

    def map_vector_variable(self, expr, enclosing_prec):
        return " %s> " % expr.name

    def map_dimensionalized_expression(self, expr, enclosing_prec):
        return self.rec(expr.child, enclosing_prec)

    def map_ones(self, expr, enclosing_prec):
        return "Ones.%s" % stringify_where(expr.where)

    def map_inverse(self, expr, enclosing_prec):
        return "Solve(%s = %s {%s})" % (
                self.rec(expr.expression, PREC_NONE),
                self.rec(expr.rhs, PREC_NONE),
                ", ".join("%s=%s" % (var_name, self.rec(var_expr, PREC_NONE))
                    for var_name, var_expr in six.iteritems(expr.extra_vars)))

        from operator import or_

        return self.rec(expr.rhs) | reduce(or_,
                (self.rec(name_expr)
                for name_expr in six.itervalues(expr.extra_vars)),
                set())

    def map_node_sum(self, expr, enclosing_prec):
        return "NodeSum(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_coordinate_component(self, expr, enclosing_prec):
        return "x%d.%s" % (expr.ambient_axis,
                stringify_where(expr.where))

    def map_num_reference_derivative(self, expr, enclosing_prec):
        result = "d/dr%s.%s %s" % (
                ",".join(str(ax) for ax in expr.ref_axes),
                stringify_where(expr.where),
                self.rec(expr.operand, PREC_PRODUCT),
                )

        if enclosing_prec >= PREC_PRODUCT:
            return "(%s)" % result
        else:
            return result

    def map_parametrization_derivative(self, expr, enclosing_prec):
        return "dx/dr.%s" % (stringify_where(expr.where))

    def map_q_weight(self, expr, enclosing_prec):
        return "w_quad.%s" % stringify_where(expr.where)

    def _stringify_kernel_args(self, kernel_arguments):
        if not kernel_arguments:
            return ""
        else:
            return "{%s}" % ", ".join(
                    "%s: %s" % (name, self.rec(arg_expr, PREC_NONE))
                    for name, arg_expr in kernel_arguments.items())

    def map_int_g(self, expr, enclosing_prec):
        return u"Int[%s->%s]@(%d)%s (%s * %s)" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                self._stringify_kernel_args(
                    expr.kernel_arguments),
                expr.kernel,
                self.rec(expr.density, PREC_PRODUCT))

    def map_int_g_ds(self, expr, enclosing_prec):
        if isinstance(expr.dsource, MultiVector):
            deriv_term = r"(%s*\/)" % self.rec(expr.dsource, PREC_PRODUCT)
        else:
            deriv_term = self.rec(expr.dsource, PREC_PRODUCT)

        result = u"Int[%s->%s]@(%d)%s %s G_%s %s" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                self._stringify_kernel_args(
                    expr.kernel_arguments),
                deriv_term,
                expr.kernel,
                self.rec(expr.density, PREC_NONE))

        if enclosing_prec >= PREC_PRODUCT:
            return "(%s)" % result
        else:
            return result

# }}}


class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin, StringifyMapper):
    pass


# {{{ graphviz

class GraphvizMapper(GraphvizMapperBase):
    def __init__(self):
        super(GraphvizMapper, self).__init__()

    def map_pytential_leaf(self, expr):
        self.lines.append(
                "%s [label=\"%s\", shape=box];" % (
                    self.get_id(expr),
                    str(expr).replace("\\", "\\\\")))

        if self.visit(expr, node_printed=True):
            self.post_visit(expr)

    map_nodes = map_pytential_leaf
    map_vector_variable = map_pytential_leaf

    def map_dimensionalized_expression(self, expr):
        self.lines.append(
                "%s [label=\"%s\",shape=circle];" % (
                    self.get_id(expr), type(expr).__name__))
        if not self.visit(expr, node_printed=True):
            return

        self.rec(expr.child)
        self.post_visit(expr)

    map_ones = map_pytential_leaf

    def map_map_node_sum(self, expr):
        self.lines.append(
                "%s [label=\"%s\",shape=circle];" % (
                    self.get_id(expr), type(expr).__name__))
        if not self.visit(expr, node_printed=True):
            return

        self.rec(expr.operand)
        self.post_visit(expr)

    map_node_coordinate_component = map_pytential_leaf
    map_num_reference_derivative = map_pytential_leaf
    map_parametrization_derivative = map_pytential_leaf

    map_q_weight = map_pytential_leaf

    def map_int_g(self, expr):
        descr = u"Int[%s->%s]@(%d) (%s)" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                expr.kernel,
                )
        self.lines.append(
                "%s [label=\"%s\",shape=box];" % (
                    self.get_id(expr), descr))
        if not self.visit(expr, node_printed=True):
            return

        self.rec(expr.density)
        for arg_expr in expr.kernel_arguments.values():
            self.rec(arg_expr)

        self.post_visit(expr)

    def map_int_g_ds(self, expr):
        descr = u"Int[%s->%s]@(%d) (%s)" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                expr.kernel,
                )
        self.lines.append(
                "%s [label=\"%s\",shape=box];" % (
                    self.get_id(expr), descr))
        if not self.visit(expr, node_printed=True):
            return

        self.rec(expr.density)
        for arg_expr in expr.kernel_arguments.values():
            self.rec(arg_expr)
        self.rec(expr.dsource)
        self.post_visit(expr)

# }}}


# vim: foldmethod=marker
