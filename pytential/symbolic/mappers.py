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
from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper import (
        Mapper,
        IdentityMapper as IdentityMapperBase,
        CombineMapper as CombineMapperBase,
        CSECachingMapperMixin
        )
from pymbolic.mapper.dependency import (
        DependencyMapper as DependencyMapperBase)
from pymbolic.mapper.coefficient import (
        CoefficientCollector as CoefficientCollectorBase)
from pymbolic.geometric_algebra import MultiVector
import pytential.symbolic.primitives as prim
from sumpy.kernel import (
        KernelIdentityMapper as KernelIdentityMapperBase,
        KernelCombineMapper as KernelCombineMapperBase)


# {{{ mappers for sumpy.kernel kernels

class ExpressionKernelIdentityMapper(KernelIdentityMapperBase):
    def __init__(self, expr_map):
        self.expr_map = expr_map

    def map_directional_target_derivative(self, kernel):
        return type(kernel)(
                self.rec(kernel.inner_kernel), kernel.dir_vec_name,
                self.expr_map(kernel.dir_vec_data))

    map_directional_source_derivative = map_directional_target_derivative


class ExpressionKernelCombineMapper(KernelCombineMapperBase):
    def __init__(self, expr_map):
        self.expr_map = expr_map

    def combine(self, sets):
        from pytools import set_sum
        return set_sum(sets)

    def map_laplace_kernel(self, kernel):
        return set()

    def map_helmholtz_kernel(self, kernel):
        return set()

    def map_one_kernel(self, kernel):
        return set()

    def map_directional_target_derivative(self, kernel):
        return self.expr_map(kernel.dir_vec_data) | self.rec(kernel.inner_kernel)

    map_directional_source_derivative = map_directional_target_derivative


class KernelEvalArgumentCollector(KernelCombineMapperBase):
    """Collect a mapping (:class:`dict`) from expression names
    to expressions for their value. These arguments are mostly used
    for direction vectors in kernels.
    """

    def combine(self, dicts):
        result = {}
        for d in dicts:
            result.update(d)
        return result

    def map_laplace_kernel(self, kernel):
        return {}

    def map_helmholtz_kernel(self, kernel):
        return {}

    def map_one_kernel(self, kernel):
        return {}

    def map_directional_target_derivative(self, kernel):
        result = {kernel.dir_vec_name: kernel.dir_vec_data}
        result.update(self.rec(kernel.inner_kernel))
        return result

    map_directional_source_derivative = map_directional_target_derivative

# }}}


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
    map_nabla = map_vector_variable
    map_nabla_component = map_vector_variable

    # }}}

    def map_derivative_source(self, expr):
        return type(expr)(self.rec(expr.operand), expr.nabla_id)

    def map_inverse(self, expr):
        return type(expr)(
                # don't recurse into expression--it is a separate world that
                # will be processed once it's executed.

                expr.expression, self.rec(expr.rhs), expr.variable_name,
                dict([
                    (name, self.rec(name_expr))
                    for name, name_expr in expr.extra_vars.iteritems()]),
                expr.where)

    def map_int_g(self, expr):
        return expr.copy(
                density=self.rec(expr.density),
                kernel=ExpressionKernelIdentityMapper(self.rec)(expr.kernel))

    def map_int_g_ds(self, expr):
        return expr.copy(
                density=self.rec(expr.density),
                dsource=self.rec(expr.dsource),
                kernel=ExpressionKernelIdentityMapper(self.rec)(expr.kernel))


class CombineMapper(CombineMapperBase):
    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    map_num_reference_derivative = map_node_sum

    def map_int_g(self, expr):
        return (
                self.rec(expr.density)
                | ExpressionKernelCombineMapper(self.rec)(expr.kernel))

    def map_int_g_ds(self, expr):
        return (
                self.rec(expr.density)
                | self.rec(expr.dsource)
                | ExpressionKernelCombineMapper(self.rec)(expr.kernel))

    def map_inverse(self, expr):
        from operator import or_
        return self.rec(expr.rhs) | reduce(or_,
                (self.rec(name_expr)
                for name_expr in expr.extra_vars.itervalues()),
                set())


class Collector(CombineMapper):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    map_vector_variable = map_constant
    map_nabla = map_constant
    map_nabla_component = map_constant
    map_ones = map_constant
    map_node_coordinate_component = map_constant
    map_parametrization_derivative = map_constant
    map_q_weight = map_constant
    map_variable = map_constant


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

    map_q_weight = map_variable
    map_ones = map_variable

    map_nabla_component = map_variable
    map_nabla = map_variable

    def map_derivative_source(self, expr):
        return type(expr)(self.rec(expr.operand), expr.nabla_id)

    def map_node_sum(self, expr):
        return type(expr)(self.rec(expr.operand))

    def map_node_coordinate_component(self, expr):
        return expr

    def map_num_reference_derivative(self, expr):
        return type(expr)(expr.ref_axes, self.rec(expr.operand), expr.where)

    def map_int_g(self, expr):
        return type(expr)(
                ExpressionKernelIdentityMapper(self.rec)(expr.kernel),
                self.rec(expr.density),
                expr.qbx_forced_limit, expr.source, expr.target)

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

    def map_int_g(self, expr):
        source = expr.source
        target = expr.target

        if source is None:
            source = self.default_source
        if target is None:
            target = self.default_where

        return type(expr)(
                ExpressionKernelIdentityMapper(self.rec)(expr.kernel),
                self.operand_rec(expr.density),
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
                ExpressionKernelIdentityMapper(self.rec)(expr.kernel),
                self.operand_rec(expr.density),
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

class _DSourceCoefficientFinder(CoefficientCollectorBase):
    def map_nabla_component(self, expr):
        return {expr: 1}

    def map_variable(self, expr):
        return {1: expr}

    def map_common_subexpression(self, expr):
        return {1: expr}


def _insert_dir_vec_into_kernel(kernel, dir_vec):
    from sumpy.kernel import DirectionalSourceDerivative

    if kernel.get_base_kernel() is kernel:
        return DirectionalSourceDerivative(kernel, dir_vec_data=dir_vec)
    else:
        return kernel.replace_inner_kernel(
                _insert_dsource_into_kernel(kernel.kernel, dir_vec))


def _insert_dsource_into_kernel(kernel, dsource, ambient_dim):
    coeffs = _DSourceCoefficientFinder()(dsource)

    dir_vec = np.zeros(ambient_dim, np.object)
    for i in xrange(ambient_dim):
        dir_vec[i] = coeffs.pop(prim.NablaComponent(i, None), 0)

    if coeffs:
        raise RuntimeError("source derivative expression contained constant term")

    return _insert_dir_vec_into_kernel(kernel, dir_vec)


class Dimensionalizer(EvaluationMapper):
    """Once the discretization is known, the dimension count is, too.
    This mapper plugs in dimension-specific quantities for their
    non-dimensional symbolic counterparts.
    """

    def __init__(self, discr_dict):
        self.discr_dict = discr_dict
        EvaluationMapper.__init__(self)

    @property
    def ambient_dim(self):
        from pytools import single_valued
        return single_valued(
                discr.ambient_dim
                for discr in self.discr_dict.itervalues())

    def map_vector_variable(self, expr):
        from pymbolic import make_sym_vector
        num_components = expr.num_components

        if num_components is None:
            num_components = self.ambient_dim

        return MultiVector(make_sym_vector(expr.name, num_components))

    def map_dimensionalized_expression(self, expr):
        return expr.child

    def map_nabla(self, expr):
        from pytools import single_valued
        ambient_dim = single_valued(
                discr.ambient_dim
                for discr in self.discr_dict.itervalues())

        from pytools.obj_array import make_obj_array
        return MultiVector(make_obj_array(
            [prim.NablaComponent(axis, expr.nabla_id)
                for axis in xrange(ambient_dim)]))

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
        for i in xrange(discr.ambient_dim):
            for j in xrange(discr.dim):
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
                    for i in xrange(discr.ambient_dim)]))

    def map_int_g(self, expr):
        from sumpy.kernel import KernelDimensionSetter
        return type(expr)(
                KernelDimensionSetter(self.ambient_dim)(
                    ExpressionKernelIdentityMapper(self.rec)(expr.kernel)),
                self.rec(expr.density),
                expr.qbx_forced_limit, expr.source, expr.target)

    def map_int_g_ds(self, expr):
        dsource = self.rec(expr.dsource)

        ambient_dim = self.ambient_dim

        from sumpy.kernel import KernelDimensionSetter
        kernel = KernelDimensionSetter(ambient_dim)(
                ExpressionKernelIdentityMapper(self.rec)(expr.kernel))

        from pytools.obj_array import make_obj_array
        nabla = MultiVector(make_obj_array(
            [prim.NablaComponent(axis, None)
                for axis in xrange(ambient_dim)]))

        rec_operand = prim.cse(self.rec(expr.density))
        return (dsource*nabla).map(
                lambda coeff: prim.IntG(
                    _insert_dsource_into_kernel(kernel, coeff, ambient_dim),
                    rec_operand, expr.qbx_forced_limit, expr.source, expr.target))

# }}}


# {{{ derivative binder

class DerivativeSourceAndNablaComponentCollector(Collector):
    def map_nabla(self, expr):
        raise RuntimeError("DerivativeOccurrenceMapper must be invoked after "
                "Dimensionalizer--Nabla found, not allowed")

    def map_nabla_component(self, expr):
        return set([expr])

    def map_derivative_source(self, expr):
        return set([expr])


class NablaComponentToUnitVector(EvaluationMapper):
    def __init__(self, nabla_id, ambient_axis):
        self.nabla_id = nabla_id
        self.ambient_axis = ambient_axis

    def map_nabla_component(self, expr):
        if expr.nabla_id == self.nabla_id:
            if expr.ambient_axis == self.ambient_axis:
                return 1
            else:
                return 0
        else:
            return EvaluationMapper.map_nabla_component(self, expr)


class DerivativeTaker(Mapper):
    def __init__(self, ambient_axis):
        self.ambient_axis = ambient_axis

    def map_int_g(self, expr):
        from sumpy.kernel import AxisTargetDerivative
        return expr.copy(kernel=AxisTargetDerivative(self.ambient_axis, expr.kernel))


class DerivativeSourceFinder(EvaluationMapper):
    """Recurses down until it finds the :class:`pytential.sym.DerivativeSource`
    with the right *nabla_id*, then calls *taker* on the
    source's argument.
    """

    def __init__(self, nabla_id, taker):
        self.nabla_id = nabla_id
        self.taker = taker

    def map_derivative_source(self, expr):
        if expr.nabla_id == self.nabla_id:
            return self.taker(expr.operand)
        else:
            return EvaluationMapper.map_derivative_source(self, expr)


class DerivativeBinder(IdentityMapper):
    def __init__(self):
        self.derivative_collector = DerivativeSourceAndNablaComponentCollector()

    def map_product(self, expr):
        # {{{ gather NablaComponents and DerivativeSources

        rec_children = []
        d_source_nabla_ids_per_child = []

        # id to set((child index, axis), ...)
        nabla_finder = {}

        for child_idx, child in enumerate(expr.children):
            rec_expr = self.rec(child)
            rec_children.append(rec_expr)

            nabla_component_ids = set()
            derivative_source_ids = set()

            nablas = []
            for d_or_n in self.derivative_collector(rec_expr):
                if isinstance(d_or_n, prim.NablaComponent):
                    nabla_component_ids.add(d_or_n.nabla_id)
                    nablas.append(d_or_n)
                elif isinstance(d_or_n, prim.DerivativeSource):
                    derivative_source_ids.add(d_or_n.nabla_id)
                else:
                    raise RuntimeError("unexpected result from "
                            "DerivativeSourceAndNablaComponentCollector")

            d_source_nabla_ids_per_child.append(
                    derivative_source_ids - nabla_component_ids)

            for ncomp in nablas:
                nabla_finder.setdefault(
                        ncomp.nabla_id, set()).add((child_idx, ncomp.ambient_axis))

        # }}}

        # a list of lists, the outer level presenting a sum, the inner a product
        result = [rec_children]

        for child_idx, (d_source_nabla_ids, child) in enumerate(
                zip(d_source_nabla_ids_per_child, rec_children)):
            if not d_source_nabla_ids:
                continue

            if len(d_source_nabla_ids) > 1:
                raise NotImplementedError("more than one DerivativeSource per "
                        "child in a product")

            nabla_id, = d_source_nabla_ids
            nablas = nabla_finder[nabla_id]
            n_axes = max(axis for _, axis in nablas) + 1

            new_result = []
            for prod_term_list in result:
                for axis in xrange(n_axes):
                    new_ptl = prod_term_list[:]
                    dsfinder = DerivativeSourceFinder(nabla_id,
                            DerivativeTaker(axis))

                    new_ptl[child_idx] = dsfinder(new_ptl[child_idx])
                    for nabla_child_index, _ in nablas:
                        new_ptl[nabla_child_index] = \
                                NablaComponentToUnitVector(nabla_id, axis)(
                                        new_ptl[nabla_child_index])

                    new_result.append(new_ptl)

            result = new_result

        from pymbolic.primitives import flattened_sum
        return flattened_sum(
                type(expr)(tuple(prod_term_list)) for prod_term_list in result)

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
                kernel=ExpressionKernelIdentityMapper(self.rec)(expr.kernel),
                density=self.rec(expr.density))

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

    def map_nabla(self, expr, enclosing_prec):
        return r"\/[%s]" % expr.nabla_id

    def map_nabla_component(self, expr, enclosing_prec):
        return r"d/dx%d[%s]" % (expr.ambient_axis, expr.nabla_id)

    def map_derivative_source(self, expr, enclosing_prec):
        return r"D[%s](%s)" % (expr.nabla_id, self.rec(expr.operand, PREC_NONE))

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

    def map_int_g(self, expr, enclosing_prec):
        return u"Int[%s->%s]@(%d) (%s * %s)" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                expr.kernel,
                self.rec(expr.density, PREC_PRODUCT))

    def map_int_g_ds(self, expr, enclosing_prec):
        if isinstance(expr.dsource, MultiVector):
            deriv_term = r"(%s*\/)" % self.rec(expr.dsource, PREC_PRODUCT)
        else:
            deriv_term = self.rec(expr.dsource, PREC_PRODUCT)

        result = u"Int[%s->%s]@(%d) %s G_%s %s" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
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

# vim: foldmethod=marker
