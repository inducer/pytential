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
        IdentityMapper as IdentityMapperBase,
        CombineMapper as CombineMapperBase,
        CSECachingMapperMixin
        )
from pymbolic.mapper.dependency import (
        DependencyMapper as BaseDependencyMapper)
from pymbolic.geometric_algebra import MultiVector
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


class IdentityMapper(IdentityMapperBase, OperatorReducerMixin):
    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

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
                    for name, name_expr in expr.extra_vars.iteritems()]),
                expr.where)

    def map_operator(self, expr, *args, **kwargs):
        def map_arg(arg):
            from sumpy.kernel import Kernel

            if arg is None:
                return None
            elif isinstance(arg, Kernel):
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


class Collector(OperatorReducerMixin, CombineMapperBase):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    map_nabla = map_constant
    map_nabla_component = map_constant

    # from sumpy.kernel
    map_kernel_partial_derivative = map_constant

    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    map_num_reference_derivative = map_node_sum

    map_ones = map_constant

    map_node_coordinate_component = map_ones
    map_parametrization_derivative = map_ones
    map_q_weight = map_ones

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
    def map_int_g(self, expr):
        return set([expr]) | self.rec(expr.density)

    def map_int_g_ds(self, expr):
        result = set([expr]) | self.rec(expr.density)
        result |= self.rec(expr.dsource)
        return result


class DependencyMapper(BaseDependencyMapper, OperatorReducerMixin):
    # {{{ childless

    def map_ones(self, expr):
        return set()

    map_node_coordinate_component = map_ones
    map_parametrization_derivative = map_ones
    map_q_weight = map_ones

    # }}}

    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    map_num_reference_derivative = map_node_sum

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

    map_q_weight = map_ones

    def map_parametrization_derivative_component(self, expr):
        if expr.where is None:
            return type(self)(
                    expr.ambient_axis, expr.ref_axis, self.default_where)
        else:
            return expr

    map_parametrization_derivative = map_ones
    map_points = map_ones

    def map_int_g(self, expr):
        source = expr.source
        target = expr.target

        if source is None:
            source = self.default_source
        if target is None:
            target = self.default_where

        return type(expr)(expr.kernel, self.operand_rec(expr.density),
                expr.qbx_forced_limit, source, target)

    def map_int_g_ds(self, expr):
        source = expr.source
        target = expr.target

        if source is None:
            source = self.default_source
        if target is None:
            target = self.default_where

        return type(expr)(
                self.operand_rec(expr.dsource),
                expr.kernel, self.operand_rec(expr.density),
                expr.qbx_forced_limit, source, target)

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

def _insert_dsource_into_kernel(kernel, dsource):
    from sumpy.kernel import DirectionalSourceDerivative

    if kernel.get_base_kernel() is kernel:
        return DirectionalSourceDerivative(kernel, dsource)
    else:
        return kernel.replace_inner_kernel(
                _insert_dsource_into_kernel(kernel.kernel, dsource))


class Dimensionalizer(EvaluationMapper, OperatorReducerMixin):
    """Once the discretization is known, the dimension count is, too.
    This mapper plugs in dimension-specific quantities for their
    non-dimensional symbolic counterparts.
    """

    def __init__(self, discr_dict):
        self.discr_dict = discr_dict
        EvaluationMapper.__init__(self)

    def map_variable(self, expr):
        return expr

    def map_nabla(self, expr):
        from pytools import single_valued
        ambient_dim = single_valued(
                discr.ambient_dim
                for discr in self.discr_dict.itervalues())

        from pytools.obj_array import make_obj_array
        return MultiVector(make_obj_array(
            [prim.NablaComponent(axis, expr.nabla_id)
                for axis in xrange(ambient_dim)]))

    map_q_weight = map_variable
    map_ones = map_variable

    def map_parametrization_derivative(self, expr):
        discr = self.discr_dict[expr.where]

        par_grad = np.zeros((discr.ambient_dim, discr.dim), np.object)
        for i in xrange(discr.ambient_dim):
            for j in xrange(discr.dim):
                par_grad[i, j] = prim.NumReferenceDerivative(
                        frozenset([j]),
                        prim.NodeCoordinateComponent(i, expr.where),
                        expr.where)

        from pytools import product
        return product(MultiVector(vec) for vec in par_grad.T)

    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

    def map_points(self, expr):
        discr = self.discr_dict[expr.where]
        from pytools.obj_array import make_obj_array
        return MultiVector(
                make_obj_array([
                    prim.NodeCoordinateComponent(i, expr.where)
                    for i in xrange(discr.ambient_dim)]))

    def map_int_g(self, expr):
        return type(expr)(expr.kernel, self.rec(expr.density),
                expr.qbx_forced_limit, expr.source, expr.target)

    def map_int_g_ds(self, expr):
        dsource = self.rec(expr.dsource)

        from pytools import single_valued
        ambient_dim = single_valued(
                discr.ambient_dim
                for discr in self.discr_dict.itervalues())

        from pytools.obj_array import make_obj_array
        from sumpy.kernel import KernelPartialDerivative
        nabla = MultiVector(make_obj_array(
            [KernelPartialDerivative(axis)
                for axis in xrange(ambient_dim)]))

        rec_operand = prim.cse(self.rec(expr.density))
        return (dsource*nabla).map(
                lambda coeff: prim.IntG(
                    _insert_dsource_into_kernel(expr.kernel, coeff),
                    rec_operand, expr.qbx_forced_limit, expr.source, expr.target))

    def map_common_subexpression(self, expr):
        return prim.cse(
                self.rec(expr.child),
                expr.prefix,
                expr.scope)

# }}}


# {{{ QBX on-surface mapper

class QBXOnSurfaceMapper(IdentityMapper):
    def __init__(self, source_name):
        self.source_name = source_name

    def map_int_g(self, expr):
        if expr.source != self.source_name:
            # not ours
            return IdentityMapper.map_int_g(self, expr)

        if expr.qbx_forced_limit is not None:
            # Not computing the on-surface value, nothing to do.
            return IdentityMapper.map_int_g(self, expr)

        from sumpy.kernel import count_derivatives
        num_derivatives = count_derivatives(expr.kernel)

        if num_derivatives == 0:
            # nothing to do
            return IdentityMapper.map_int_g(self, expr)
        elif num_derivatives == 1:
            # Assume it's a PV integral, preserve numerical compactness by using
            # two-sided average.
            return 0.5*(
                    expr.copy(qbx_forced_limit=+1)
                    + expr.copy(qbx_forced_limit=-1))
        else:
            # FIXME
            # from sumpy.layerpot import find_jump_term
            # jump_term = find_jump_term(expr.kernel,
            #         _QBXJumpTermSymbolicArgumentProvider(expr.source))
            raise NotImplementedError()

    def map_int_g_ds(self, expr):
        raise RuntimeError("user-facing source derivative operators are expected "
                "to have been eliminated by the time QBXLimitFinder is called.")

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

    def map_nabla(self, expr, enclosing_prec):
        return r"\/"

    def map_nabla_component(self, expr, enclosing_prec):
        return r"d/dx%d" % expr.ambient_axis

    def map_ones(self, expr, enclosing_prec):
        return "Ones.%s" % stringify_where(expr.where)

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

    def map_node_coordinate_component(self, expr, enclosing_prec):
        return "x%d.%s" % (expr.ambient_axis,
                stringify_where(expr.where))

    def map_num_reference_derivative(self, expr, enclosing_prec):
        return "d/dr%s.%s %s" % (
                ",".join(str(ax) for ax in expr.ref_axes),
                stringify_where(expr.where),
                self.rec(expr.operand, PREC_PRODUCT),
                )

    def map_parametrization_derivative(self, expr, enclosing_prec):
        return "dx/dr.%s" % (stringify_where(expr.where))

    def map_q_weight(self, expr, enclosing_prec):
        return "w_quad.%s" % stringify_where(expr.where)

    def map_int_g(self, op, enclosing_prec):
        return u"S_%s[%s->%s](%s)" % (
                op.kernel,
                stringify_where(op.source),
                stringify_where(op.target),
                self.rec(op.density, PREC_NONE))

    def map_int_g_ds(self, op, enclosing_prec):
        if isinstance(op.dsource, MultiVector):
            deriv_term = r"(%s*\/)" % self.rec(op.dsource, PREC_PRODUCT)
        else:
            deriv_term = self.rec(op.dsource, PREC_PRODUCT)

        result = u"Int[%s->%s] %s G_%s %s" % (
                stringify_where(op.source),
                stringify_where(op.target),
                deriv_term,
                op.kernel,
                self.rec(op.density, PREC_NONE))

        if enclosing_prec >= PREC_PRODUCT:
            return "(%s)" % result
        else:
            return result

# }}}


# {{{ pretty-printing ---------------------------------------------------------

class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin, StringifyMapper):
    pass


def pretty(optemplate):
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
