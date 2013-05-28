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
from pymbolic.mapper.stringifier import (
        StringifyMapper as BaseStringifyMapper,
        CSESplittingStringifyMapperMixin,
        PREC_NONE, PREC_PRODUCT)
from pymbolic.mapper.evaluator import EvaluationMapper
from pymbolic.mapper import (
        IdentityMapper as BaseIdentityMapper,
        CombineMapper as BaseCombineMapper,
        CSECachingMapperMixin
        )
from pymbolic.mapper.dependency import (
        DependencyMapper as BaseDependencyMapper)
import pytential.symbolic.primitives as prim




class OperatorReducerMixin:
    def map_int_g(self, *args, **kwargs):
        return self.map_operator(*args, **kwargs)

    map_int_g_ds = map_int_g
    map_int_g_dt = map_int_g
    map_int_g_dmix = map_int_g
    map_int_g_d2t = map_int_g

    map_single_layer_prime = map_int_g
    map_single_layer_2prime = map_int_g
    map_double_layer_prime = map_int_g

    map_quad_kernel_op = map_int_g

class IdentityMapper(BaseIdentityMapper, OperatorReducerMixin):
    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

    # {{{ childless -- no need to rebuild

    def map_ones(self, expr):
        return expr

    map_q_weights = map_ones
    map_parametrization_derivative_component = map_ones
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
                    for name, name_expr in expr.extra_vars.iteritems()]),
                expr.where)

    def map_operator(self, expr, *args, **kwargs):
        def map_arg(arg):
            from hellskitchen.fmm import KernelBase

            if arg is None:
                return None
            elif isinstance(arg, KernelBase):
                return arg
            elif isinstance(arg, str):
                return arg
            elif not isinstance(arg, np.ndarray) and \
                    arg in (prim.DEFAULT_SOURCE, prim.DEFAULT_TARGET):
                return arg
            else:
                return self.rec(arg, *args, **kwargs)

        return type(expr)(
                *[map_arg(arg) for arg in expr.__getinitargs__()])

class Collector(OperatorReducerMixin, BaseCombineMapper):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    def map_ones(self, expr):
        return set()

    map_parametrization_derivative_component = map_ones
    map_parametrization_gradient = map_ones
    map_parametrization_derivative = map_ones
    map_q_weights = map_ones

    map_operator = map_node_sum

    def map_inverse(self, expr):
        from operator import or_
        return self.rec(expr.rhs) | reduce(or_,
                (self.rec(name_expr)
                for name_expr in expr.extra_vars.itervalues()),
                set())

    def map_variable(self, expr):
        return set()





class OperatorCollector(Collector):
    def map_operator(self, expr):
        result = set([expr]) | self.rec(expr.operand)

        from pytential.symbolic.primitives import SourceDiffLayerPotentialOperatorBase
        if (isinstance(expr, SourceDiffLayerPotentialOperatorBase)
                and expr.ds_direction is not None):
            result |= self.rec(expr.ds_direction)

        return result

    map_integral = map_operator
    map_mean = map_operator
    map_line_integral = map_operator







class DependencyMapper(BaseDependencyMapper, OperatorReducerMixin):
    # {{{ childless

    def map_ones(self, expr):
        return set()

    map_parametrization_derivative_component = map_ones
    map_parametrization_gradient = map_ones
    map_parametrization_derivative = map_ones
    map_q_weights = map_ones

    # }}}

    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    def map_inverse(self, expr):
        from operator import or_
        return self.rec(expr.rhs) | reduce(or_,
                (self.rec(name_expr)
                for name_expr in expr.extra_vars.itervalues()),
                set())

    map_operator = map_node_sum




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

    def map_parametrization_derivative_component(self, expr):
        if expr.where is None:
            return ParametrizationDerivativeComponent(
                    expr.ambient_axis, expr.ref_axis, self.default_where)
        else:
            return expr

    map_parametrization_gradient = map_ones
    map_parametrization_derivative = map_ones

    def map_operator(self, op):
        if isinstance(op, prim.LayerPotentialOperatorBase):
            def operand_rec_if_possible(subexpr):
                from pymbolic.primitives import Expression
                if isinstance(subexpr, np.ndarray) or isinstance(subexpr, Expression):
                    return self.operand_rec(subexpr)
                else:
                    return subexpr

            args = op.__getinitargs__()
            k = args[0]
            operand = self.operand_rec(args[1])
            rest = tuple(operand_rec_if_possible(rest_i) for rest_i in args[2:-2])
            source, target = args[-2:]

            if source is None:
                source = self.default_source
            if target is None:
                target = self.default_where

            new_args = (k, operand,) + rest + (source, target)

            return type(op)(*new_args)
        else:
            return IdentityMapper.map_operator(self, op)


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
                    for name, name_expr in expr.extra_vars.iteritems()]),
                where)

    def operand_rec(self, expr):
        return self.rec(expr)




class ToTargetTagger(LocationTagger):
    """Descends into the expression tree, marking everything up to the first layer
    potential operator as operating on the targets, and everything below there
    as operating on the source. Also performs a consistency check such that only
    'target' and 'source' items are combined arithmetically.
    """

    def __init__(self, default_source, default_target):
        LocationTagger.__init__(self, default_target)
        self.operand_rec = LocationTagger(default_source)

# }}}

# {{{ discretization plugger-inner

class DiscretizationPluggerInner(EvaluationMapper):
    def __init__(self, discr_dict):
        self.discr_dict = discr_dict
        EvaluationMapper.__init__(self)

    def map_variable(self, expr):
        return expr

    map_q_weights = map_variable
    map_ones = map_variable

    def map_parametrization_gradient(self, expr):
        discr = self.discr_dict[expr.where]

        result = np.zeros((discr.ambient_dim, discr.dim), np.object)
        for i in xrange(discr.ambient_dim):
            for j in xrange(discr.dim):
                result[i,j] = prim.ParametrizationDerivativeComponent(i, j, expr.where)

        return result

    def map_parametrization_derivative(self, expr):
        par_grad = self.rec(prim.ParametrizationGradient(expr.where))
        from pytools import product
        from pymbolic.geometric_algebra import MultiVector
        return product(MultiVector(vec) for vec in par_grad.T)

    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

# }}}

# {{{ auto-upsampler

class AutoUpsampler(IdentityMapper):
    pass

# }}}

# {{{ stringifier

class StringifyMapper(BaseStringifyMapper):
    def _stringify_where(self, where):
        if where is None:
            return "?"
        elif where is prim.DEFAULT_SOURCE:
            return "s"
        elif where is prim.DEFAULT_TARGET:
            return "t"
        else:
            return str(where)

    def map_ones(self, expr, enclosing_prec):
        return "Ones.%s" % self._stringify_where(expr.where)

    def map_inverse(self, expr, enclosing_prec):
        return "Solve(%s = %s {%s})" % (
                self.rec(expr.expression, PREC_NONE),
                self.rec(expr.rhs, PREC_NONE),
                ", ".join("%s=%s" % (var_name, self.rec(var_expr, PREC_NONE))
                    for var_name, var_expr in expr.extra_vars.iteritems()))

        from operator import or_

        return self.rec(expr.rhs) | reduce(or_,
                (self.rec(name_expr)
                for name_expr in expr.extra_vars.itervalues()),
                set())

    def map_node_sum(self, expr, enclosing_prec):
        return "NodeSum(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_parametrization_derivative_component(self, expr, enclosing_prec):
        return "dx%d/dr%d" % (expr.ambient_axis, expr.ref_axis)

    def map_parametrization_gradient(self, expr, enclosing_prec):
        return "[[dx/dr]]"

    def map_parametrization_derivative(self, expr, enclosing_prec):
        return "dx/dr"

    def map_q_weights(self, expr, enclosing_prec):
        return "w_quad"

    def map_int_g(self, op, enclosing_prec):
        return u"S_%s[%s->%s](%s)" % (
                op.kernel,
                self._stringify_where(op.source),
                self._stringify_where(op.target),
                self.rec(op.operand, PREC_NONE))

    def map_int_g_ds(self, op, enclosing_prec):
        if op.ds_direction is None:
            return "D_%s[%s->%s](%s)" % (
                    op.kernel,
                    self._stringify_where(op.source),
                    self._stringify_where(op.target),
                    self.rec(op.operand, PREC_NONE))
        else:
            result = u"Int[%s->%s] (%s,d_s) G_%s %s" % (
                    self._stringify_where(op.source),
                    self._stringify_where(op.target),
                    self.rec(op.ds_direction, PREC_PRODUCT),
                    op.kernel,
                    self.rec(op.operand, PREC_NONE))

            if enclosing_prec >= PREC_PRODUCT:
                return "(%s)" % result
            else:
                return result

    def map_int_g_dt(self, op, enclosing_prec):
        result = u"d_t%d (Int[%s->%s] G_%s %s)" % (
                op.dt_axis,
                self._stringify_where(op.source),
                self._stringify_where(op.target),
                op.kernel,
                self.rec(op.operand, PREC_NONE))

        if enclosing_prec > PREC_PRODUCT:
            return "(%s)" % result
        else:
            return result

    def map_int_g_dmix(self, op, enclosing_prec):
        if op.ds_direction is None:
            ds_direction = "normal"
        else:
            ds_direction = self.rec(op.ds_direction, PREC_PRODUCT)

        result = u"d_t%d (Int[%s->%s] (%s,d_s) G_%s %s)" % (
                op.dt_axis,
                self._stringify_where(op.source),
                self._stringify_where(op.target),
                ds_direction,
                op.kernel,
                self.rec(op.operand, PREC_PRODUCT))

        if enclosing_prec >= PREC_PRODUCT:
            return "(%s)" % result
        else:
            return result

    def map_int_g_d2t(self, op, enclosing_prec):
        result = u"d_t%d d_t%d (Int[%s->%s] G_%s %s)" % (
                op.dt_axis_a,
                op.dt_axis_b,
                self._stringify_where(op.source),
                self._stringify_where(op.target),
                op.kernel,
                self.rec(op.operand, PREC_PRODUCT))

        if enclosing_prec > PREC_PRODUCT:
            return "(%s)" % result
        else:
            return result

    def map_quad_kernel_op(self, op, enclosing_prec):
        doc = op.interact_func.__doc__
        func_name = doc[:doc.index(" ")]
        if op.parameter_gen is not None:
            return "QInt[%s->%s](%s(%s), %s)" % (
                    self._stringify_where(op.source),
                    self._stringify_where(op.target),
                    func_name,
                    str(op.parameter_gen),
                    self.rec(op.operand, PREC_NONE))
        else:
            return "QInt%s(%s, %s)" % (
                    self._stringify_where(op.source),
                    self._stringify_where(op.target),
                    func_name, self.rec(op.operand, PREC_NONE))

    def map_single_layer_prime(self, op, enclosing_prec):
        return "S'_%s[%s->%s](%s)" % (
                op.kernel,
                self._stringify_where(op.source),
                self._stringify_where(op.target),
                self.rec(op.operand, PREC_NONE))

    def map_double_layer_prime(self, op, enclosing_prec):
        return "D'_%s[%s->%s](%s)" % (
                op.kernel,
                self._stringify_where(op.source),
                self._stringify_where(op.target),
                self.rec(op.operand, PREC_NONE))

    def map_single_layer_2prime(self, op, enclosing_prec):
        return "S''_%s[%s->%s](%s)" % (
                op.kernel,
                self._stringify_where(op.source),
                self._stringify_where(op.target),
                self.rec(op.operand, PREC_NONE))

# }}}

# {{{ pretty printing ---------------------------------------------------------

class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin, StringifyMapper):
    pass

def pretty_print_optemplate(optemplate):
    stringify_mapper = PrettyStringifyMapper()
    from pymbolic.mapper.stringifier import PREC_NONE
    result = stringify_mapper(optemplate, PREC_NONE)

    splitter = "="*75 + "\n"

    cse_strs = stringify_mapper.get_cse_strings()
    if cse_strs:
        result = "\n".join(cse_strs)+"\n"+splitter+result

    return result

# }}}

# vim: foldmethod=marker
