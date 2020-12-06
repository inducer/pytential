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
    map_node_min = map_node_sum

    def map_elementwise_sum(self, expr):
        return type(expr)(self.rec(expr.operand), expr.dofdesc)

    map_elementwise_min = map_elementwise_sum
    map_elementwise_max = map_elementwise_sum

    def map_num_reference_derivative(self, expr):
        return type(expr)(expr.ref_axes, self.rec(expr.operand),
                expr.dofdesc)

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
                {
                    name: self.rec(name_expr)
                    for name, name_expr in expr.extra_vars.items()},
                expr.dofdesc)

    def map_int_g(self, expr):
        return expr.copy(
                densities=self.rec(expr.densities),
                kernel_arguments={
                    name: self.rec(arg_expr)
                    for name, arg_expr in expr.kernel_arguments.items()
                    })

    def map_interpolation(self, expr):
        return type(expr)(expr.from_dd, expr.to_dd, self.rec(expr.operand))


class CombineMapper(CombineMapperBase):
    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    map_node_max = map_node_sum
    map_node_min = map_node_sum
    map_num_reference_derivative = map_node_sum
    map_elementwise_sum = map_node_sum
    map_elementwise_min = map_node_sum
    map_elementwise_max = map_node_sum
    map_interpolation = map_node_sum

    def map_int_g(self, expr):
        return self.combine(
                [self.rec(density) for density in expr.densities]
                + [self.rec(arg_expr)
                    for arg_expr in expr.kernel_arguments.values()])

    def map_inverse(self, expr):
        return self.combine([
            self.rec(expr.rhs)] + [
                (self.rec(name_expr)
                for name_expr in expr.extra_vars.values())
                ])


class Collector(CollectorBase, CombineMapper):
    def map_ones(self, expr):
        return set()

    map_node_coordinate_component = map_ones
    map_parametrization_derivative = map_ones
    map_q_weight = map_ones


class OperatorCollector(Collector):
    def map_int_g(self, expr):
        return {expr} | Collector.map_int_g(self, expr)


class DependencyMapper(DependencyMapperBase, Collector):
    pass


class EvaluationMapper(EvaluationMapperBase):
    """Unlike :mod:`pymbolic.mapper.evaluation.EvaluationMapper`, this class
    does evaluation mostly to get :class:`pymbolic.geometric_algebra.MultiVector`
    instances to do their thing, and perhaps to automatically kill terms
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
    map_node_min = map_node_sum

    def map_node_coordinate_component(self, expr):
        return expr

    def map_num_reference_derivative(self, expr):
        return componentwise(
                lambda subexpr: type(expr)(
                        expr.ref_axes, self.rec(subexpr), expr.dofdesc),
                expr.operand)

    def map_int_g(self, expr):
        return expr.copy(
                    densities=[self.rec(density) for density in expr.densities],
                    kernel_arguments={
                        name: self.rec(arg_expr)
                        for name, arg_expr in expr.kernel_arguments.items()
                    })

    def map_common_subexpression(self, expr):
        return prim.cse(
                self.rec(expr.child),
                expr.prefix,
                expr.scope)


# {{{ dofdesc tagging

class LocationTagger(CSECachingMapperMixin, IdentityMapper):
    """Used internally by :class:`ToTargetTagger`."""

    def __init__(self, default_where, default_source=prim.DEFAULT_SOURCE):
        self.default_source = default_source
        self.default_where = default_where

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def _default_dofdesc(self, dofdesc):
        if dofdesc.geometry is None:
            if dofdesc.discr_stage is None \
                    and dofdesc.granularity == prim.GRANULARITY_NODE:
                dofdesc = dofdesc.copy(geometry=self.default_where)
            else:
                dofdesc = dofdesc.copy(geometry=self.default_source)

        return dofdesc

    def map_ones(self, expr):
        return type(expr)(dofdesc=self._default_dofdesc(expr.dofdesc))

    map_q_weight = map_ones

    def map_parametrization_derivative_component(self, expr):
        return type(expr)(
                expr.ambient_axis,
                expr.ref_axis,
                self._default_dofdesc(expr.dofdesc))

    def map_node_coordinate_component(self, expr):
        return type(expr)(
                expr.ambient_axis,
                self._default_dofdesc(expr.dofdesc))

    def map_num_reference_derivative(self, expr):
        return type(expr)(
                expr.ref_axes,
                self.rec(expr.operand),
                self._default_dofdesc(expr.dofdesc))

    def map_elementwise_sum(self, expr):
        return type(expr)(
                self.rec(expr.operand),
                self._default_dofdesc(expr.dofdesc))

    map_elementwise_min = map_elementwise_sum
    map_elementwise_max = map_elementwise_sum

    def map_int_g(self, expr):
        source = expr.source
        if source.geometry is None:
            source = source.copy(geometry=self.default_source)

        target = expr.target
        if target.geometry is None:
            target = target.copy(geometry=self.default_where)

        return type(expr)(
                expr.target_kernel,
                expr.source_kernels,
                self.operand_rec(expr.densities),
                expr.qbx_forced_limit, source, target,
                kernel_arguments={
                    name: self.operand_rec(arg_expr)
                    for name, arg_expr in expr.kernel_arguments.items()
                    })

    def map_inverse(self, expr):
        dofdesc = expr.dofdesc
        if dofdesc.geometry is None:
            dofdesc = dofdesc.copy(geometry=self.default_where)

        return type(expr)(
                # don't recurse into expression--it is a separate world that
                # will be processed once it's executed.

                expr.expression, self.rec(expr.rhs), expr.variable_name,
                {
                    name: self.rec(name_expr)
                    for name, name_expr in expr.extra_vars.items()},
                dofdesc)

    def map_interpolation(self, expr):
        from_dd = expr.from_dd
        if from_dd.geometry is None:
            from_dd = from_dd.copy(geometry=self.default_source)

        to_dd = expr.to_dd
        if to_dd.geometry is None:
            to_dd = to_dd.copy(geometry=self.default_source)

        return type(expr)(from_dd, to_dd, self.operand_rec(expr.operand))

    def operand_rec(self, expr):
        return self.rec(expr)


class ToTargetTagger(LocationTagger):
    """Descends into the expression tree, marking expressions based on two
    heuristics:

    * everything up to the first layer potential operator is marked as
    operating on the targets, and everything below there as operating on the
    source.
    * if an expression has a :class:`~pytential.symbolic.primitives.DOFDescriptor`
    that requires a :class:`~pytential.source.LayerPotentialSourceBase` to be
    used (e.g. by being defined on
    :class:`~pytential.symbolic.primitives.QBX_SOURCE_QUAD_STAGE2`), then
    it is marked as operating on a source.
    """

    def __init__(self, default_source, default_target):
        LocationTagger.__init__(self, default_target,
                                default_source=default_source)
        self.operand_rec = LocationTagger(default_source,
                                          default_source=default_source)


class DiscretizationStageTagger(IdentityMapper):
    """Descends into an expression tree and changes the
    :attr:`~pytential.symbolic.primitives.DOFDescriptor.discr_stage` to
    :attr:`discr_stage`.

    .. attribute:: discr_stage

        The new discretization for the DOFs in the expression. For valid
        values, see
        :attr:`~pytential.symbolic.primitives.DOFDescriptor.discr_stage`.
    """

    def __init__(self, discr_stage):
        if not (discr_stage == prim.QBX_SOURCE_STAGE1
                or discr_stage == prim.QBX_SOURCE_STAGE2
                or discr_stage == prim.QBX_SOURCE_QUAD_STAGE2):
            raise ValueError(f'unknown discr stage tag: "{discr_stage}"')

        self.discr_stage = discr_stage

    def map_node_coordinate_component(self, expr):
        dofdesc = expr.dofdesc
        if dofdesc.discr_stage == self.discr_stage:
            return expr

        return type(expr)(
                expr.ambient_axis,
                dofdesc.copy(discr_stage=self.discr_stage))

    def map_num_reference_derivative(self, expr):
        dofdesc = expr.dofdesc
        if dofdesc.discr_stage == self.discr_stage:
            return expr

        return type(expr)(
                expr.ref_axes,
                self.rec(expr.operand),
                dofdesc.copy(discr_stage=self.discr_stage))

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
        return expr.copy(target_kernel=AxisTargetDerivative(self.ambient_axis,
            expr.target_kernel))


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

    def __init__(self, geometry, places):
        self.geometry = geometry
        self.places = places

    def map_int_g(self, expr):
        if expr.qbx_forced_limit in (-1, 1):
            raise ValueError(
                    "Unregularized evaluation does not support one-sided limits")

        expr = expr.copy(
                qbx_forced_limit=None,
                densities=self.rec(expr.densities),
                kernel_arguments={
                    name: self.rec(arg_expr)
                    for name, arg_expr in expr.kernel_arguments.items()
                    })

        return expr

# }}}


# {{{ interpolation preprocessor

class InterpolationPreprocessor(IdentityMapper):
    """Handle expressions that require upsampling or downsampling by inserting
    a :class:`~pytential.symbolic.primitives.Interpolation`. This is used to

    * do differentiation on
    :class:`~pytential.symbolic.primitives.QBX_SOURCE_QUAD_STAGE2`.
    by performing it on :attr:`from_discr_stage` and upsampling.
    * upsample layer potential sources to
    :attr:`~pytential.symbolic.primitives.QBX_SOURCE_QUAD_STAGE2`,
    """

    def __init__(self, places, from_discr_stage=None):
        """
        .. attribute:: from_discr_stage

            Sets the stage on which to compute the data before interpolation.
            For valid values, see
            :attr:`~pytential.symbolic.primitives.DOFDescriptor.discr_stage`.
        """
        self.places = places
        self.from_discr_stage = (prim.QBX_SOURCE_STAGE2
                if from_discr_stage is None else from_discr_stage)
        self.tagger = DiscretizationStageTagger(self.from_discr_stage)

    def map_num_reference_derivative(self, expr):
        to_dd = expr.dofdesc
        if to_dd.discr_stage != prim.QBX_SOURCE_QUAD_STAGE2:
            return expr

        from pytential.qbx import QBXLayerPotentialSource
        lpot_source = self.places.get_geometry(to_dd.geometry)
        if not isinstance(lpot_source, QBXLayerPotentialSource):
            return expr

        from_dd = to_dd.copy(discr_stage=self.from_discr_stage)
        return prim.interp(from_dd, to_dd, self.rec(self.tagger(expr)))

    def map_int_g(self, expr):
        if expr.target.discr_stage is None:
            expr = expr.copy(target=expr.target.to_stage1())

        if expr.source.discr_stage is not None:
            return expr

        from pytential.qbx import QBXLayerPotentialSource
        lpot_source = self.places.get_geometry(expr.source.geometry)
        if not isinstance(lpot_source, QBXLayerPotentialSource):
            return expr

        from_dd = expr.source.to_stage1()
        to_dd = from_dd.to_quad_stage2()
        densities = [prim.interp(from_dd, to_dd, self.rec(density)) for
            density in expr.densities]

        from_dd = from_dd.copy(discr_stage=self.from_discr_stage)
        kernel_arguments = {
                name: prim.interp(from_dd, to_dd,
                    self.rec(self.tagger(arg_expr)))
                for name, arg_expr in expr.kernel_arguments.items()}

        return expr.copy(
                densities=densities,
                kernel_arguments=kernel_arguments,
                source=to_dd)

# }}}


# {{{ QBX preprocessor

class QBXPreprocessor(IdentityMapper):
    def __init__(self, geometry, places):
        self.geometry = geometry
        self.places = places

    def map_int_g(self, expr):
        if expr.source.geometry != self.geometry:
            return expr

        source_discr = self.places.get_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_discretization(
                expr.target.geometry, expr.target.discr_stage)

        if expr.qbx_forced_limit == 0:
            raise ValueError("qbx_forced_limit == 0 was a bad idea and "
                    "is no longer supported. Use qbx_forced_limit == 'avg' "
                    "to request two-sided averaging explicitly if needed.")

        is_self = source_discr is target_discr

        expr = expr.copy(
                densities=self.rec(expr.densities),
                kernel_arguments={
                    name: self.rec(arg_expr)
                    for name, arg_expr in expr.kernel_arguments.items()
                    })

        if not is_self:
            # non-self evaluation
            if expr.qbx_forced_limit in ["avg", 1, -1]:
                raise ValueError("May not specify +/-1 or 'avg' for "
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
                    "for self-evaluation. Specify +/-1 or 'avg' instead.")

        if expr.qbx_forced_limit == "avg":
            return 0.5*(
                    expr.copy(qbx_forced_limit=+1)
                    + expr.copy(qbx_forced_limit=-1))
        else:
            return expr

# }}}


# {{{ stringifier

def stringify_where(where):
    return str(prim.as_dofdesc(where))


class StringifyMapper(BaseStringifyMapper):

    def map_ones(self, expr, enclosing_prec):
        return "Ones[%s]" % stringify_where(expr.dofdesc)

    def map_inverse(self, expr, enclosing_prec):
        return "Solve(%s = %s {%s})" % (
                self.rec(expr.expression, PREC_NONE),
                self.rec(expr.rhs, PREC_NONE),
                ", ".join("{}={}".format(var_name, self.rec(var_expr, PREC_NONE))
                    for var_name, var_expr in expr.extra_vars.items()))

        from operator import or_

        return self.rec(expr.rhs) | reduce(or_,
                (self.rec(name_expr)
                for name_expr in expr.extra_vars.values()),
                set())

    def map_elementwise_sum(self, expr, enclosing_prec):
        return "ElwiseSum[{}]({})".format(
                stringify_where(expr.dofdesc),
                self.rec(expr.operand, PREC_NONE))

    def map_elementwise_min(self, expr, enclosing_prec):
        return "ElwiseMin[{}]({})".format(
                stringify_where(expr.dofdesc),
                self.rec(expr.operand, PREC_NONE))

    def map_elementwise_max(self, expr, enclosing_prec):
        return "ElwiseMax[{}]({})".format(
                stringify_where(expr.dofdesc),
                self.rec(expr.operand, PREC_NONE))

    def map_node_max(self, expr, enclosing_prec):
        return "NodeMax(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_min(self, expr, enclosing_prec):
        return "NodeMin(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_sum(self, expr, enclosing_prec):
        return "NodeSum(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_coordinate_component(self, expr, enclosing_prec):
        return "x%d[%s]" % (expr.ambient_axis,
                stringify_where(expr.dofdesc))

    def map_num_reference_derivative(self, expr, enclosing_prec):
        diff_op = " ".join(
                "d/dr%d" % axis
                if mult == 1 else
                "d/dr%d^%d" % (axis, mult)
                for axis, mult in expr.ref_axes)

        result = "{}[{}] {}".format(
                diff_op,
                stringify_where(expr.dofdesc),
                self.rec(expr.operand, PREC_PRODUCT),
                )

        if enclosing_prec >= PREC_PRODUCT:
            return "(%s)" % result
        else:
            return result

    def map_parametrization_derivative(self, expr, enclosing_prec):
        return "dx/dr[%s]" % (stringify_where(expr.dofdesc))

    def map_q_weight(self, expr, enclosing_prec):
        return "w_quad[%s]" % stringify_where(expr.dofdesc)

    def _stringify_kernel_args(self, kernel_arguments):
        if not kernel_arguments:
            return ""
        else:
            return "{%s}" % ", ".join(
                    "{}: {}".format(name, self.rec(arg_expr, PREC_NONE))
                    for name, arg_expr in kernel_arguments.items())

    def map_int_g(self, expr, enclosing_prec):
        source_kernels_strs = [
            "{} * {}".format(self.rec(density, PREC_PRODUCT), source_kernel)
            for source_kernel, density in zip(expr.source_kernels, expr.densities)
        ]
        source_kernels_str = " + ".join(source_kernels_strs)
        target_kernel_str = str(expr.target_kernel)
        base_kernel_str = str(expr.target_kernel.get_base_kernel())
        kernel_str = target_kernel_str.replace(base_kernel_str,
            f"({source_kernels_str})")

        return "Int[{}->{}]@({}){} {}".format(
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                self._stringify_kernel_args(
                    expr.kernel_arguments),
                kernel_str)

    def map_interpolation(self, expr, enclosing_prec):
        return "Interp[{}->{}]({})".format(
                stringify_where(expr.from_dd),
                stringify_where(expr.to_dd),
                self.rec(expr.operand, PREC_PRODUCT))

# }}}


class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin, StringifyMapper):
    pass


# {{{ graphviz

class GraphvizMapper(GraphvizMapperBase):
    def __init__(self):
        super().__init__()

    def map_pytential_leaf(self, expr):
        self.lines.append(
                '{} [label="{}", shape=box];'.format(
                    self.get_id(expr),
                    str(expr).replace("\\", "\\\\")))

        if self.visit(expr, node_printed=True):
            self.post_visit(expr)

    map_ones = map_pytential_leaf

    def map_map_node_sum(self, expr):
        self.lines.append(
                '{} [label="{}",shape=circle];'.format(
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
        descr = "Int[%s->%s]@(%d) (%s)" % (
                stringify_where(expr.source),
                stringify_where(expr.target),
                expr.qbx_forced_limit,
                expr.target_kernel,
                )
        self.lines.append(
                '{} [label="{}",shape=box];'.format(self.get_id(expr), descr))
        if not self.visit(expr, node_printed=True):
            return

        [self.rec(density) for density in expr.densities]
        for arg_expr in expr.kernel_arguments.values():
            self.rec(arg_expr)

        self.post_visit(expr)

# }}}


# vim: foldmethod=marker
