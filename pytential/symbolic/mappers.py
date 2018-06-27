from __future__ import division, absolute_import, print_function

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
from functools import reduce

from pymbolic.mapper.stringifier import (
        CSESplittingStringifyMapperMixin,
        PREC_NONE, PREC_PRODUCT)
from pymbolic.mapper import (
        Mapper,
        CSECachingMapperMixin
        )
from pymbolic.mapper.dependency import (
        DependencyMapper as DependencyMapperBase)
from pymbolic.geometric_algebra import componentwise
from pymbolic.geometric_algebra.mapper import (
        CombineMapper as CombineMapperBase,
        IdentityMapper as IdentityMapperBase,
        Collector as CollectorBase,
        DerivativeBinder as DerivativeBinderBase,
        EvaluationMapper as EvaluationMapperBase,

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
    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

    map_node_max = map_node_sum

    def map_elementwise_sum(self, expr):
        return type(expr)(self.rec(expr.operand), expr.where)

    map_elementwise_min = map_elementwise_sum
    map_elementwise_max = map_elementwise_sum

    def map_num_reference_derivative(self, expr):
        return type(expr)(expr.ref_axes, self.rec(expr.operand),
                expr.where)

    # {{{ childless -- no need to rebuild

    def map_ones(self, expr):
        return expr

    map_q_weight = map_ones
    map_node_coordinate_component = map_ones
    map_parametrization_gradient = map_ones
    map_parametrization_derivative = map_ones

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


class CombineMapper(CombineMapperBase):
    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    map_node_max = map_node_sum
    map_num_reference_derivative = map_node_sum
    map_elementwise_sum = map_node_sum
    map_elementwise_min = map_node_sum
    map_elementwise_max = map_node_sum

    def map_int_g(self, expr):
        return self.combine(
                [self.rec(expr.density)]
                + [self.rec(arg_expr)
                    for arg_expr in expr.kernel_arguments.values()])

    def map_inverse(self, expr):
        return self.combine([
            self.rec(expr.rhs)] + [
                (self.rec(name_expr)
                for name_expr in six.itervalues(expr.extra_vars))
                ])


class Collector(CollectorBase, CombineMapper):
    def map_ones(self, expr):
        return set()

    map_node_coordinate_component = map_ones
    map_parametrization_derivative = map_ones
    map_q_weight = map_ones


class OperatorCollector(Collector):
    def map_int_g(self, expr):
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

    def map_subscript(self, expr):
        return self.rec(expr.aggregate)[self.rec(expr.index)]

    map_q_weight = map_variable
    map_ones = map_variable

    def map_node_sum(self, expr):
        return componentwise(type(expr), self.rec(expr.operand))

    map_node_max = map_node_sum

    def map_node_coordinate_component(self, expr):
        return expr

    def map_num_reference_derivative(self, expr):
        return componentwise(
                lambda subexpr: type(expr)(
                        expr.ref_axes, self.rec(subexpr), expr.where),
                expr.operand)

    def map_int_g(self, expr):
        return componentwise(
                lambda subexpr: type(expr)(
                        expr.kernel,
                        self.rec(subexpr),
                        expr.qbx_forced_limit, expr.source, expr.target,
                        kernel_arguments=dict(
                                (name, self.rec(arg_expr))
                                for name, arg_expr in expr.kernel_arguments.items()
                                )),
                expr.density)

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

    def map_elementwise_sum(self, expr):
        return type(expr)(
                self.rec(expr.operand),
                expr.where if expr.where is not None else self.default_where)

    map_elementwise_min = map_elementwise_sum
    map_elementwise_max = map_elementwise_sum

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
                    (name, self.operand_rec(arg_expr))
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


# {{{ derivative binder

class DerivativeTaker(Mapper):
    def __init__(self, ambient_axis):
        self.ambient_axis = ambient_axis

    def map_sum(self, expr):
        from pymbolic.primitives import flattened_sum
        return flattened_sum(tuple(self.rec(child) for child in expr.children))

    def map_product(self, expr):
        from pymbolic.primitives import is_constant
        const = []
        nonconst = []
        for subexpr in expr.children:
            if is_constant(subexpr):
                const.append(subexpr)
            else:
                nonconst.append(subexpr)

        if len(nonconst) > 1:
            raise RuntimeError("DerivativeTaker doesn't support products with "
                    "more than one non-constant")

        if not nonconst:
            nonconst = [1]

        from pytools import product
        return product(const) * self.rec(nonconst[0])

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


# {{{ Unregularized preprocessor

class UnregularizedPreprocessor(IdentityMapper):

    def __init__(self, source_name, places):
        self.source_name = source_name
        self.places = places

    def map_int_g(self, expr):
        if expr.qbx_forced_limit in (-1, 1):
            raise ValueError(
                    "Unregularized evaluation does not support one-sided limits")

        expr = expr.copy(
                qbx_forced_limit=None,
                kernel=expr.kernel,
                density=self.rec(expr.density),
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

        return expr

# }}}


# {{{ QBX preprocessor

class QBXPreprocessor(IdentityMapper):
    def __init__(self, source_name, places):
        self.source_name = source_name
        self.places = places

    def map_int_g(self, expr):
        source = self.places[self.source_name]
        target_discr = self.places[expr.target]

        from pytential.source import LayerPotentialSourceBase
        if isinstance(target_discr, LayerPotentialSourceBase):
            target_discr = target_discr.density_discr

        if expr.qbx_forced_limit == 0:
            raise ValueError("qbx_forced_limit == 0 was a bad idea and "
                    "is no longer supported. Use qbx_forced_limit == 'avg' "
                    "to request two-sided averaging explicitly if needed.")

        is_self = source.density_discr is target_discr

        expr = expr.copy(
                kernel=expr.kernel,
                density=self.rec(expr.density),
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))

        if not is_self:
            # non-self evaluation
            if expr.qbx_forced_limit in ['avg', 1, -1]:
                raise ValueError("May not specify +/-1 or \"avg\" for "
                        "qbx_forced_limit for non-self evaluation. "
                        "Specify 'None' for automatic choice or +/-2 "
                        "to force a QBX side in the near-evaluation "
                        "regime.")

            return expr

        if expr.qbx_forced_limit is None:
            raise ValueError("qbx_forced_limit == None is not supported "
                    "for self evaluation--must pick evaluation side")

        if (isinstance(expr.qbx_forced_limit, int)
                and abs(expr.qbx_forced_limit) == 2):
            raise ValueError("May not specify qbx_forced_limit == +/-2 "
                    "for self-evaluation. "
                    "Specify +/-1 or \"avg\" instead.")

        if expr.qbx_forced_limit == "avg":
            return 0.5*(
                    expr.copy(qbx_forced_limit=+1)
                    + expr.copy(qbx_forced_limit=-1))
        else:
            return expr

# }}}


# {{{ stringifier

def stringify_where(where):
    if isinstance(where, prim._QBXSourceStage2):
        return "stage2(%s)" % stringify_where(where.where)

    if where is None:
        return "?"
    elif where is prim.DEFAULT_SOURCE:
        return "s"
    elif where is prim.DEFAULT_TARGET:
        return "t"
    else:
        return str(where)


class StringifyMapper(BaseStringifyMapper):

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

    def map_elementwise_sum(self, expr, enclosing_prec):
        return "ElwiseSum.%s(%s)" % (
                stringify_where(expr.where),
                self.rec(expr.operand, PREC_NONE))

    def map_elementwise_min(self, expr, enclosing_prec):
        return "ElwiseMin.%s(%s)" % (
                stringify_where(expr.where),
                self.rec(expr.operand, PREC_NONE))

    def map_elementwise_max(self, expr, enclosing_prec):
        return "ElwiseMax.%s(%s)" % (
                stringify_where(expr.where),
                self.rec(expr.operand, PREC_NONE))

    def map_node_max(self, expr, enclosing_prec):
        return "NodeMax(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_sum(self, expr, enclosing_prec):
        return "NodeSum(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_coordinate_component(self, expr, enclosing_prec):
        return "x%d.%s" % (expr.ambient_axis,
                stringify_where(expr.where))

    def map_num_reference_derivative(self, expr, enclosing_prec):
        diff_op = " ".join(
                "d/dr%d" % axis
                if mult == 1 else
                "d/dr%d^%d" % (axis, mult)
                for axis, mult in expr.ref_axes)

        result = "%s.%s %s" % (
                diff_op,
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
        return u"Int[%s->%s]@(%s)%s (%s * %s)" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                self._stringify_kernel_args(
                    expr.kernel_arguments),
                expr.kernel,
                self.rec(expr.density, PREC_PRODUCT))

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

# }}}


# vim: foldmethod=marker
