from __future__ import annotations


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

from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, cast

from typing_extensions import Self, override

import pymbolic.primitives as p
from pymbolic import ArithmeticExpression, ExpressionNode
from pymbolic.geometric_algebra import componentwise
from pymbolic.geometric_algebra.mapper import (
    Collector as CollectorBase,
    CombineMapper as CombineMapperBase,
    DerivativeBinder as DerivativeBinderBase,
    DerivativeSourceAndNablaComponentCollector as DerivativeSourceAndNablaComponentCollectorBase,  # noqa: E501
    DerivativeSourceFinder as DerivativeSourceFinderBase,
    EvaluationRewriter as EvaluationRewriterBase,
    GraphvizMapper as GraphvizMapperBase,
    IdentityMapper as IdentityMapperBase,
    NablaComponentToUnitVector as NablaComponentToUnitVectorBase,
    StringifyMapper as BaseStringifyMapper,
)
from pymbolic.mapper import (
    CachedMapper,
    CollectedT,
    CSECachingMapperMixin,
    Mapper,
    ResultT,
)
from pymbolic.mapper.dependency import (
    Dependency,
    DependencyMapper as DependencyMapperBase,
)
from pymbolic.mapper.flattener import FlattenMapper as FlattenMapperBase
from pymbolic.mapper.stringifier import (
    PREC_NONE,
    PREC_PRODUCT,
    CSESplittingStringifyMapperMixin,
)
from pymbolic.typing import Expression

import pytential.symbolic.primitives as pp


if TYPE_CHECKING:
    from sumpy.symbolic import SpatialConstant

    from pytential.collection import GeometryCollection
    from pytential.symbolic.dof_desc import (
        DiscretizationStage,
        DOFDescriptor,
        DOFDescriptorLike,
        GeometryId,
    )


def rec_int_g_arguments(mapper: IdentityMapper | EvaluationRewriter, expr: pp.IntG):
    densities = [mapper.rec_arith(d) for d in expr.densities]
    kernel_arguments = {
            name: componentwise(mapper.rec_arith, arg)
            for name, arg in expr.kernel_arguments.items()
            }

    changed = not (
            all(d is orig for d, orig in zip(densities, expr.densities, strict=True))
            and all(
                arg is orig for arg, orig in zip(
                    kernel_arguments.values(),
                    expr.kernel_arguments.values(), strict=True))
            )

    return densities, kernel_arguments, changed


# {{{ IdentityMapper

class IdentityMapper(IdentityMapperBase[[]]):
    def _map_nodal_red(self,
                expr: pp.NodeSum | pp.NodeMax | pp.NodeMin
            ) -> ArithmeticExpression:
        operand = self.rec_arith(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(operand)

    map_node_sum: Callable[[Self, pp.NodeSum], ArithmeticExpression] = _map_nodal_red
    map_node_max: Callable[[Self, pp.NodeMax], ArithmeticExpression] = _map_nodal_red
    map_node_min: Callable[[Self, pp.NodeMin], ArithmeticExpression] = _map_nodal_red

    def _map_elwise_red(self,
                expr: pp.ElementwiseSum | pp.ElementwiseMin | pp.ElementwiseMax
            ) -> ArithmeticExpression:
        operand = self.rec_arith(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(operand, expr.dofdesc)

    map_elementwise_sum: \
         Callable[[Self, pp.ElementwiseSum], ArithmeticExpression] = _map_elwise_red
    map_elementwise_min: \
         Callable[[Self, pp.ElementwiseMin], ArithmeticExpression] = _map_elwise_red
    map_elementwise_max: \
         Callable[[Self, pp.ElementwiseMax], ArithmeticExpression] = _map_elwise_red

    def map_num_reference_derivative(self,
                expr: pp.NumReferenceDerivative
            ) -> ArithmeticExpression:
        operand = self.rec_arith(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(expr.ref_axes, operand, expr.dofdesc)

    # {{{ childless -- no need to rebuild

    def _map_childless(self,
                expr: pp.SpatialConstant
                    | pp.Ones
                    | pp.QWeight
                    | pp.NodeCoordinateComponent
                    | pp.IsShapeClass
                    | pp.ErrorExpression,
           ) -> ArithmeticExpression:
        return expr

    map_spatial_constant: \
        Callable[[Self, pp.SpatialConstant], ArithmeticExpression] = _map_childless
    map_ones: \
        Callable[[Self, pp.Ones], ArithmeticExpression] = _map_childless
    map_q_weight: \
        Callable[[Self, pp.QWeight], ArithmeticExpression] = _map_childless
    map_node_coordinate_component: \
        Callable[[Self, pp.NodeCoordinateComponent],
        ArithmeticExpression] = _map_childless
    map_is_shape_class: \
        Callable[[Self, pp.IsShapeClass], ArithmeticExpression] = _map_childless
    map_error_expression: \
        Callable[[Self, pp.ErrorExpression], ArithmeticExpression] = _map_childless

    # }}}

    def map_inverse(self, expr: pp.IterativeInverse):
        return type(expr)(
                # don't recurse into expression--it is a separate world that
                # will be processed once it's evaluated.

                expr.expression, self.rec_arith(expr.rhs), expr.variable_name,
                {
                    name: self.rec(name_expr)
                    for name, name_expr in expr.extra_vars.items()},
                expr.dofdesc)

    def map_int_g(self, expr: pp.IntG) -> ArithmeticExpression:
        densities, kernel_arguments, changed = rec_int_g_arguments(self, expr)
        if not changed:
            return expr

        return replace(expr, densities=densities, kernel_arguments=kernel_arguments)

    def map_interpolation(self, expr: pp.Interpolation):
        operand = self.rec_arith(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(expr.from_dd, expr.to_dd, operand)

    def map_interleave(self, expr: pp.Interleave):
        operand_1 = self.rec_arith(expr.operand_1)
        operand_2 = self.rec_arith(expr.operand_2)
        if (operand_1 is expr.operand_1) and (operand_2 is expr.operand_2):
            return expr

        return type(expr)(expr.from_dd, operand_1, operand_2)


class CachedIdentityMapper(CachedMapper[Expression, []], IdentityMapper):
    def __call__(self, expr):
        try:
            return CachedMapper.__call__(self, expr)
        except TypeError:
            # Fallback to no cached behaviour for unhashable types
            # like list, numpy.array
            return IdentityMapper.__call__(self, expr)

    rec = __call__

# }}}


# {{{ CombineMapper

class CombineMapper(CombineMapperBase[ResultT, []]):
    def _map_with_operand(self, expr:
                pp.NodeSum
                | pp.NodeMin
                | pp.NodeMax
                | pp.NumReferenceDerivative
                | pp.ElementwiseSum
                | pp.ElementwiseMin
                | pp.ElementwiseMax
                | pp.Interpolation
            ):
        return self.rec(expr.operand)

    map_node_sum: Callable[[Self, pp.NodeSum], ResultT] = _map_with_operand
    map_node_max: Callable[[Self, pp.NodeMax], ResultT] = _map_with_operand
    map_node_min: Callable[[Self, pp.NodeMax], ResultT] = _map_with_operand
    map_num_reference_derivative: \
        Callable[[Self, pp.NumReferenceDerivative], ResultT] = _map_with_operand
    map_elementwise_sum: \
        Callable[[Self, pp.ElementwiseSum], ResultT] = _map_with_operand
    map_elementwise_min: \
        Callable[[Self, pp.ElementwiseMin], ResultT] = _map_with_operand
    map_elementwise_max: \
        Callable[[Self, pp.ElementwiseMax], ResultT] = _map_with_operand
    map_interpolation: Callable[[Self, pp.Interpolation], ResultT] = _map_with_operand

    def map_interleave(self, expr: pp.Interleave):
        return self.combine([self.rec(expr.operand_1), self.rec(expr.operand_2)])

    def map_spatial_constant(self, expr: pp.SpatialConstant, /) -> ResultT:
        raise NotImplementedError()

    def map_int_g(self, expr: pp.IntG) -> ResultT:
        from pytential.symbolic.primitives import hashable_kernel_args
        return self.combine(
                [self.rec(density) for density in expr.densities]
                + [self.rec(arg_expr)
                    for _, arg_expr in hashable_kernel_args(expr.kernel_arguments)])

    def map_inverse(self, expr: pp.IterativeInverse) -> ResultT:
        return self.combine([
            self.rec(expr.rhs),
            *(self.rec(name_expr) for name_expr in expr.extra_vars.values())
            ])

# }}}


# {{{ Collector

class Collector(CollectorBase[CollectedT, []], CombineMapper[Set[CollectedT]]):
    def _map_leaf(self,
                expr: pp.Ones
                    | pp.ErrorExpression
                    | pp.IsShapeClass
                    | pp.NodeCoordinateComponent
                    | pp.QWeight
                    | pp.SpatialConstant
            ) -> Set[CollectedT]:
        return set()

    map_ones: \
        Callable[[Self, pp.Ones], Set[CollectedT]] = _map_leaf
    map_is_shape_class: \
        Callable[[Self, pp.IsShapeClass], Set[CollectedT]] = _map_leaf
    map_error_expression: \
        Callable[[Self, pp.ErrorExpression], Set[CollectedT]] = _map_leaf
    map_node_coordinate_component: \
        Callable[[Self, pp.NodeCoordinateComponent], Set[CollectedT]] = _map_leaf
    map_q_weight: \
        Callable[[Self, pp.QWeight], Set[CollectedT]] = _map_leaf
    map_spatial_constant: \
        Callable[[Self, pp.SpatialConstant], Set[CollectedT]] = _map_leaf


class OperatorCollector(Collector[pp.IntG]):
    @override
    def map_int_g(self, expr: pp.IntG):
        return {expr} | Collector[pp.IntG].map_int_g(self, expr)


class DependencyMapper(DependencyMapperBase[[]], Collector[Dependency]):
    pass

# }}}


# {{{ EvaluationRewriter

class EvaluationRewriter(EvaluationRewriterBase):
    """Unlike :mod:`pymbolic.mapper.evaluation.EvaluationMapper`, this class
    does evaluation mostly to get :class:`pymbolic.geometric_algebra.MultiVector`
    instances to do their thing, and perhaps to automatically kill terms
    that are multiplied by zero. Otherwise it intends to largely preserve
    the structure of the input expression.
    """

    def rec_arith(self,
                expr: ArithmeticExpression,
            ) -> ArithmeticExpression:
        res = self.rec(expr)
        assert p.is_arithmetic_expression(res)
        return res

    @override
    def map_variable(self, expr):
        return expr

    @override
    def map_subscript(self, expr: p.Subscript):
        aggregate = self.rec(expr.aggregate)
        index = self.rec(expr.index)
        if aggregate is expr.aggregate and index is expr.index:
            return expr

        return cast("ExpressionNode", aggregate)[index]

    map_q_weight = map_variable
    map_ones = map_variable

    def map_node_sum(self, expr):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(operand)

    map_node_max = map_node_sum
    map_node_min = map_node_sum

    def map_node_coordinate_component(self, expr: pp.NodeCoordinateComponent):
        return expr

    def map_num_reference_derivative(self, expr: pp.NumReferenceDerivative):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(expr.ref_axes, operand, expr.dofdesc)

    def map_int_g(self, expr: pp.IntG):
        densities, kernel_arguments, changed = rec_int_g_arguments(self, expr)
        if not changed:
            return expr

        return replace(expr, densities=densities, kernel_arguments=kernel_arguments)

    @override
    def map_common_subexpression(self, expr: p.CommonSubexpression):
        child = self.rec(expr.child)
        if child is expr.child:
            return expr

        return pp.cse(
                child,
                expr.prefix,
                expr.scope)

# }}}


# {{{ FlattenMapper

class FlattenMapper(FlattenMapperBase, IdentityMapper):
    pass


def flatten(expr: ArithmeticExpression):
    return FlattenMapper().rec_arith(expr)

# }}}


# {{{ LocationTagger

class LocationTagger(CSECachingMapperMixin[Expression, []],
                     IdentityMapper):
    """Used internally by :class:`ToTargetTagger`."""

    def __init__(self,
                default_target: DOFDescriptorLike,
                default_source: DOFDescriptorLike
            ):
        self.default_source: DOFDescriptor = pp.as_dofdesc(default_source)
        self.default_target: DOFDescriptor = pp.as_dofdesc(default_target)

    @override
    def map_common_subexpression_uncached(self, expr: p.CommonSubexpression):
        return IdentityMapper.map_common_subexpression(self, expr)

    def _default_dofdesc(self, dofdesc: DOFDescriptor):
        if dofdesc.geometry is None:
            # NOTE: this is a heuristic to determine how to tag things:
            #   * if no `discr_stage` is given, it's probably a target, since
            #   only `QBXLayerPotentialSource` has stages.
            #   * if some stage is present, assume it's a source
            if (dofdesc.discr_stage is None
                    and dofdesc.granularity == pp.GRANULARITY_NODE):
                dofdesc = dofdesc.copy(geometry=self.default_target)
            else:
                dofdesc = dofdesc.copy(geometry=self.default_source)
        elif dofdesc.geometry is pp.DEFAULT_SOURCE:
            dofdesc = dofdesc.copy(geometry=self.default_source)
        elif dofdesc.geometry is pp.DEFAULT_TARGET:
            dofdesc = dofdesc.copy(geometry=self.default_target)

        return dofdesc

    @override
    def map_ones(self, expr: pp.Ones | pp.QWeight):
        return type(expr)(dofdesc=self._default_dofdesc(expr.dofdesc))

    map_q_weight = map_ones

    @override
    def map_node_coordinate_component(self, expr: pp.NodeCoordinateComponent):
        return type(expr)(
                expr.ambient_axis,
                self._default_dofdesc(expr.dofdesc))

    @override
    def map_num_reference_derivative(self, expr: pp.NumReferenceDerivative):
        return type(expr)(
                expr.ref_axes,
                self.rec_arith(expr.operand),
                self._default_dofdesc(expr.dofdesc))

    @override
    def map_elementwise_sum(self,
                expr: pp.ElementwiseSum | pp.ElementwiseMin | pp.ElementwiseMax
            ):
        return type(expr)(
                self.rec_arith(expr.operand),
                self._default_dofdesc(expr.dofdesc))

    map_elementwise_min = map_elementwise_sum
    map_elementwise_max = map_elementwise_sum

    @override
    def map_int_g(self, expr: pp.IntG):
        source = expr.source
        if source.geometry is None:
            source = source.copy(geometry=self.default_source)

        target = expr.target
        if target.geometry is None:
            target = target.copy(geometry=self.default_target)

        return type(expr)(
                expr.target_kernel,
                expr.source_kernels,
                tuple(self.rec_arith(d) for d in expr.densities),
                expr.qbx_forced_limit, source, target,
                kernel_arguments={
                    name: componentwise(self.rec_arith, arg_expr)
                    for name, arg_expr in expr.kernel_arguments.items()
                    })

    @override
    def map_inverse(self, expr: pp.IterativeInverse):
        # NOTE: this doesn't use `_default_dofdesc` because it should be always
        # evaluated at the targets (ignores `discr_stage`)
        dofdesc = expr.dofdesc
        if dofdesc.geometry is None:
            dofdesc = dofdesc.copy(geometry=self.default_target)

        return type(expr)(
                # don't recurse into expression--it is a separate world that
                # will be processed once it's executed.

                expr.expression, self.rec_arith(expr.rhs), expr.variable_name,
                {
                    name: self.rec(name_expr)
                    for name, name_expr in expr.extra_vars.items()},
                dofdesc)

    @override
    def map_interpolation(self, expr: pp.Interpolation):
        from_dd = expr.from_dd
        if from_dd.geometry is None:
            from_dd = from_dd.copy(geometry=self.default_source)

        to_dd = expr.to_dd
        if to_dd.geometry is None:
            to_dd = to_dd.copy(geometry=self.default_source)

        return type(expr)(from_dd, to_dd, self.rec_arith(expr.operand))

    @override
    def map_interleave(self, expr: pp.Interleave):
        from_dd = expr.from_dd
        if from_dd.geometry is None:
            from_dd = from_dd.copy(geometry=self.default_source)

        return type(expr)(
                from_dd,
                self.rec_arith(expr.operand_1),
                self.rec_arith(expr.operand_2))

    @override
    def map_is_shape_class(self, expr: pp.IsShapeClass):
        return type(expr)(expr.shape, self._default_dofdesc(expr.dofdesc))

    @override
    def map_error_expression(self, expr: pp.ErrorExpression):
        return expr

    def operand_rec(self, expr, /):
        return self.rec(expr)


class ToTargetTagger(LocationTagger):
    """Descends into the expression tree, marking expressions based on two
    heuristics:

    * everything up to the first layer potential operator is marked as
    operating on the targets, and everything below there as operating on the
    source.
    * if an expression has a :class:`~pytential.symbolic.dof_desc.DOFDescriptor`
    that requires a :class:`~pytential.source.LayerPotentialSourceBase` to be
    used (e.g. by being defined on
    :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_QUAD_STAGE2`), then
    it is marked as operating on a source.
    """

    def __init__(self, default_source, default_target):
        LocationTagger.__init__(self, default_target,
                                default_source=default_source)
        self.rec = LocationTagger(default_source,
                                          default_source=default_source)

# }}}


# {{{ DiscretizationStageTagger

class DiscretizationStageTagger(IdentityMapper):
    """Descends into an expression tree and changes the
    :attr:`~pytential.symbolic.dof_desc.DOFDescriptor.discr_stage` to
    :attr:`discr_stage`.

    .. attribute:: discr_stage

        The new discretization for the DOFs in the expression. For valid
        values, see
        :attr:`~pytential.symbolic.dof_desc.DOFDescriptor.discr_stage`.
    """

    def __init__(self, discr_stage):
        if not (discr_stage == pp.QBX_SOURCE_STAGE1
                or discr_stage == pp.QBX_SOURCE_STAGE2
                or discr_stage == pp.QBX_SOURCE_QUAD_STAGE2):
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
                self.rec_arith(expr.operand),
                dofdesc.copy(discr_stage=self.discr_stage))

# }}}


# {{{ DerivativeBinder

class _IsSptiallyVaryingMapper(CombineMapper[bool]):
    @override
    def combine(self, values: Iterable[bool]):
        return any(values)

    @override
    def map_constant(self, expr: object):
        return False

    @override
    def map_spatial_constant(self, expr: SpatialConstant):
        return False

    @override
    def map_variable(self, expr: p.Variable):
        return True

    @override
    def map_int_g(self, expr: pp.IntG):
        return True


class _DerivativeTakerUnsupoortedProductError(Exception):
    pass


class DerivativeTaker(Mapper[ArithmeticExpression, []]):
    def __init__(self, ambient_axis: int):
        self.ambient_axis: int = ambient_axis

    @override
    def map_constant(self, expr: object):
        return 0

    @override
    def map_sum(self, expr: p.Sum):
        children = [self.rec(child) for child in expr.children]
        if all(child is orig for child, orig in zip(
                children, expr.children, strict=True)):
            return expr

        from pymbolic.primitives import flattened_sum
        return flattened_sum(children)

    @override
    def map_product(self, expr: p.Product):
        const: list[ArithmeticExpression] = []
        nonconst: list[ArithmeticExpression] = []
        for subexpr in expr.children:
            if _IsSptiallyVaryingMapper()(subexpr):
                nonconst.append(subexpr)
            else:
                const.append(subexpr)

        if len(nonconst) > 1:
            raise _DerivativeTakerUnsupoortedProductError(
                    "DerivativeTaker doesn't support products with "
                    "more than one non-constant. "
                    "The following were recognized as non-constant: "
                    f"{', '.join(str(nc) for nc in nonconst)}. "
                    "If some of these are spatially constant, use sym.SpatialConstant "
                    "when creating them."
                )

        if not nonconst:
            nonconst = [1]

        from pytools import product
        return product(const) * self.rec(nonconst[0])

    def map_int_g(self, expr: pp.IntG):
        from sumpy.kernel import AxisTargetDerivative

        target_kernel = AxisTargetDerivative(self.ambient_axis, expr.target_kernel)
        return replace(expr, target_kernel=target_kernel)


class DerivativeSourceAndNablaComponentCollector(
        Collector,
        DerivativeSourceAndNablaComponentCollectorBase):
    pass


class NablaComponentToUnitVector(
        EvaluationRewriter,
        NablaComponentToUnitVectorBase):
    pass


class DerivativeSourceFinder(
        EvaluationRewriter,
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


# {{{ UnregularizedPreprocessor

@dataclass(frozen=True)
class UnregularizedPreprocessor(IdentityMapper):
    geometry: GeometryId
    places: GeometryCollection

    @override
    def map_int_g(self, expr: pp.IntG):
        if expr.qbx_forced_limit in (-1, 1):
            raise ValueError(
                    "Unregularized evaluation does not support one-sided limits")

        return replace(
            expr,
            qbx_forced_limit=None,
            densities=self.rec(expr.densities),
            kernel_arguments={
                name: componentwise(self.rec_arith, arg_expr)
                for name, arg_expr in expr.kernel_arguments.items()
            }
        )

# }}}


# {{{ InterpolationPreprocessor

class InterpolationPreprocessor(IdentityMapper):
    """Handle expressions that require upsampling or downsampling by inserting
    a :class:`~pytential.symbolic.primitives.Interpolation`. This is used to

    * do differentiation on
      :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_QUAD_STAGE2`.
      by performing it on :attr:`from_discr_stage` and upsampling.
    * upsample layer potential sources to
      :attr:`~pytential.symbolic.dof_desc.QBX_SOURCE_QUAD_STAGE2`, if a
      stage is not already assigned to the source descriptor.

    .. attribute:: from_discr_stage
    .. automethod:: __init__
    """
    places: GeometryCollection
    from_discr_stage: DiscretizationStage

    def __init__(self,
                places: GeometryCollection,
                from_discr_stage: DiscretizationStage | None = None
            ):
        """
        :arg from_discr_stage: sets the stage on which to evaluate the expression
            before interpolation. For valid values, see
            :attr:`~pytential.symbolic.dof_desc.DOFDescriptor.discr_stage`.
        """
        self.places = places
        self.from_discr_stage = (pp.QBX_SOURCE_STAGE2
                if from_discr_stage is None else from_discr_stage)
        self.tagger = DiscretizationStageTagger(self.from_discr_stage)

    @override
    def map_num_reference_derivative(self, expr: pp.NumReferenceDerivative):
        to_dd = expr.dofdesc
        if to_dd.discr_stage != pp.QBX_SOURCE_QUAD_STAGE2:
            return expr

        from pytential.qbx import QBXLayerPotentialSource
        lpot_source = self.places.get_geometry(to_dd.geometry)
        if not isinstance(lpot_source, QBXLayerPotentialSource):
            return expr

        from_dd = to_dd.copy(discr_stage=self.from_discr_stage)
        return pp.interpolate(self.rec_arith(self.tagger(expr)), from_dd, to_dd)

    @override
    def map_int_g(self, expr: pp.IntG):
        if expr.target.discr_stage is None:
            expr = replace(expr, target=expr.target.to_stage1())

        if expr.source.discr_stage is not None:
            return expr

        from pytential.qbx import QBXLayerPotentialSource
        lpot_source = self.places.get_geometry(expr.source.geometry)
        if not isinstance(lpot_source, QBXLayerPotentialSource):
            return expr

        from_dd = expr.source.to_stage1()
        to_dd = from_dd.to_quad_stage2()
        densities = tuple(
            pp.interpolate(self.rec_arith(density), from_dd, to_dd)
            for density in expr.densities)

        from_dd = from_dd.copy(discr_stage=self.from_discr_stage)
        kernel_arguments = {
                name: componentwise(
                    lambda aexpr: pp.interpolate(
                        self.rec_arith(
                            self.tagger.rec_arith(aexpr)), from_dd, to_dd),
                    arg_expr)
                for name, arg_expr in expr.kernel_arguments.items()}

        return replace(
                expr,
                densities=densities,
                kernel_arguments=kernel_arguments,
                source=to_dd)

# }}}


# {{{ QBXPreprocessor

@dataclass(frozen=True)
class QBXPreprocessor(IdentityMapper):
    geometry: GeometryId
    places: GeometryCollection

    @override
    def map_int_g(self, expr: pp.IntG):
        if expr.source.geometry != self.geometry:
            return expr

        if expr.qbx_forced_limit == 0:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError("qbx_forced_limit == 0 was a bad idea and "
                    "is no longer supported. Use qbx_forced_limit == 'avg' "
                    "to request two-sided averaging explicitly if needed.")

        is_self = (
            expr.source.geometry == expr.target.geometry
            and expr.source.discr_stage == expr.target.discr_stage
            )

        expr = replace(
                expr,
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
                    replace(expr, qbx_forced_limit=+1)
                    + replace(expr, qbx_forced_limit=-1))
        else:
            return expr

# }}}


# {{{ StringifyMapper

def stringify_where(where: DOFDescriptorLike):
    return str(pp.as_dofdesc(where))


class StringifyMapper(BaseStringifyMapper):

    def map_ones(self, expr: pp.Ones, enclosing_prec: int):
        return "Ones[%s]" % stringify_where(expr.dofdesc)

    def map_inverse(self, expr: pp.IterativeInverse, enclosing_prec: int):
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

    def map_elementwise_sum(self, expr: pp.ElementwiseSum, enclosing_prec: int):
        return "ElwiseSum[{}]({})".format(
                stringify_where(expr.dofdesc),
                self.rec(expr.operand, PREC_NONE))

    def map_elementwise_min(self, expr: pp.ElementwiseMin, enclosing_prec: int):
        return "ElwiseMin[{}]({})".format(
                stringify_where(expr.dofdesc),
                self.rec(expr.operand, PREC_NONE))

    def map_elementwise_max(self, expr: pp.ElementwiseMax, enclosing_prec: int):
        return "ElwiseMax[{}]({})".format(
                stringify_where(expr.dofdesc),
                self.rec(expr.operand, PREC_NONE))

    def map_node_max(self, expr: pp.NodeMax, enclosing_prec: int):
        return "NodeMax(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_min(self, expr: pp.NodeMin, enclosing_prec: int):
        return "NodeMin(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_sum(self, expr: pp.NodeSum, enclosing_prec: int):
        return "NodeSum(%s)" % self.rec(expr.operand, PREC_NONE)

    def map_node_coordinate_component(self,
                expr: pp.NodeCoordinateComponent,
                enclosing_prec: int):
        return "x%d[%s]" % (expr.ambient_axis,
                stringify_where(expr.dofdesc))

    def map_num_reference_derivative(self,
                expr: pp.NumReferenceDerivative,
                enclosing_prec: int):
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

    def map_q_weight(self, expr: pp.QWeight, enclosing_prec: int):
        return "w_quad[%s]" % stringify_where(expr.dofdesc)

    def _stringify_kernel_args(self, kernel_arguments):
        if not kernel_arguments:
            return ""
        else:
            return "{%s}" % ", ".join(
                    "{}: {}".format(name, self.rec(arg_expr, PREC_NONE))
                    for name, arg_expr in kernel_arguments.items())

    def map_int_g(self, expr: pp.IntG, enclosing_prec: int):
        source_kernels_str = " + ".join([
            "{} * {}".format(self.rec(density, PREC_PRODUCT), source_kernel)
            for source_kernel, density in zip(
                expr.source_kernels, expr.densities, strict=True)
        ])
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

    def map_interpolation(self, expr: pp.Interpolation, enclosing_prec: int):
        return "Interp[{}->{}]({})".format(
                stringify_where(expr.from_dd),
                stringify_where(expr.to_dd),
                self.rec(expr.operand, PREC_PRODUCT))

    def map_interleave(self, expr: pp.Interleave, enclosing_prec: int):
        return "Interleave[{}]({}, {})".format(
                stringify_where(expr.from_dd),
                self.rec(expr.operand_1, PREC_NONE),
                self.rec(expr.operand_2, PREC_NONE),
            )

    def map_is_shape_class(self, expr: pp.IsShapeClass, enclosing_prec: int):
        return "IsShape[{}]({})".format(stringify_where(expr.dofdesc),
                                        expr.shape.__name__)


class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin[[]], StringifyMapper):
    pass

# }}}


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

        self.rec(expr.densities)
        for arg_expr in expr.kernel_arguments.values():
            self.rec(arg_expr)

        self.post_visit(expr)

# }}}


# vim: foldmethod=marker
