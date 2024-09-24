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
        CachedMapper,
        CSECachingMapperMixin,
        )
from pymbolic.mapper.dependency import (
        DependencyMapper as DependencyMapperBase)
from pymbolic.mapper.flattener import FlattenMapper as FlattenMapperBase
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
from pymbolic.typing import ExpressionT
import pytential.symbolic.primitives as prim


def rec_int_g_arguments(mapper, expr):
    densities = mapper.rec(expr.densities)
    kernel_arguments = {
            name: mapper.rec(arg) for name, arg in expr.kernel_arguments.items()
            }

    changed = (
            all(d is orig for d, orig in zip(densities, expr.densities, strict=True))
            and all(
                arg is orig for arg, orig in zip(
                    kernel_arguments.values(),
                    expr.kernel_arguments.values(), strict=True))
            )

    return densities, kernel_arguments, changed


# {{{ IdentityMapper

class IdentityMapper(IdentityMapperBase[[]]):
    def map_node_sum(self, expr):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(operand)

    map_node_max = map_node_sum
    map_node_min = map_node_sum

    def map_elementwise_sum(self, expr):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(operand, expr.dofdesc)

    map_elementwise_min = map_elementwise_sum
    map_elementwise_max = map_elementwise_sum

    def map_num_reference_derivative(self, expr):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(expr.ref_axes, operand, expr.dofdesc)

    # {{{ childless -- no need to rebuild

    def map_ones(self, expr):
        return expr

    map_q_weight = map_ones
    map_node_coordinate_component = map_ones
    map_parametrization_gradient = map_ones
    map_parametrization_derivative = map_ones
    map_is_shape_class = map_ones
    map_error_expression = map_ones

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
        densities, kernel_arguments, changed = rec_int_g_arguments(self, expr)
        if not changed:
            return expr

        return expr.copy(
                densities=densities,
                kernel_arguments=kernel_arguments)

    def map_interpolation(self, expr):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(expr.from_dd, expr.to_dd, operand)


class CachedIdentityMapper(CachedMapper, IdentityMapper):
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
        from pytential.symbolic.primitives import hashable_kernel_args
        return self.combine(
                [self.rec(density) for density in expr.densities]
                + [self.rec(arg_expr)
                    for _, arg_expr in hashable_kernel_args(expr.kernel_arguments)])

    def map_inverse(self, expr):
        return self.combine([
            self.rec(expr.rhs),
            *(self.rec(name_expr) for name_expr in expr.extra_vars.values())
            ])

    def map_is_shape_class(self, expr):
        return set()

    map_error_expression = map_is_shape_class

# }}}


# {{{ Collector

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

# }}}


# {{{ EvaluationMapper

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
        aggregate = self.rec(expr.aggregate)
        index = self.rec(expr.index)
        if aggregate is expr.aggregate and index is expr.index:
            return expr

        return aggregate[index]

    map_q_weight = map_variable
    map_ones = map_variable

    def map_node_sum(self, expr):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(operand)

    map_node_max = map_node_sum
    map_node_min = map_node_sum

    def map_node_coordinate_component(self, expr):
        return expr

    def map_num_reference_derivative(self, expr):
        operand = self.rec(expr.operand)
        if operand is expr.operand:
            return expr

        return type(expr)(expr.ref_axes, operand, expr.dofdesc)

    def map_int_g(self, expr):
        densities, kernel_arguments, changed = rec_int_g_arguments(self, expr)
        if not changed:
            return expr

        return expr.copy(
                densities=densities,
                kernel_arguments=kernel_arguments,
                )

    def map_common_subexpression(self, expr):
        child = self.rec(expr.child)
        if child is expr.child:
            return expr

        return prim.cse(
                child,
                expr.prefix,
                expr.scope)

# }}}


# {{{ FlattenMapper

class FlattenMapper(FlattenMapperBase, IdentityMapper):
    pass


def flatten(expr):
    return FlattenMapper()(expr)

# }}}


# {{{ LocationTagger

class LocationTagger(CSECachingMapperMixin[ExpressionT, []],
                     IdentityMapper):
    """Used internally by :class:`ToTargetTagger`."""

    def __init__(self, default_target, default_source):
        self.default_source = default_source
        self.default_target = default_target

    def map_common_subexpression_uncached(self, expr) -> ExpressionT:
        # Mypy 1.13 complains about this:
        # error: Too few arguments for "map_common_subexpression" of "IdentityMapper"  [call-arg]  # noqa: E501
        # error: Argument 1 to "map_common_subexpression" of "IdentityMapper" has incompatible type "LocationTagger"; expected "IdentityMapper[P]"  [arg-type]  # noqa: E501
        # This seems spurious?
        return IdentityMapper.map_common_subexpression(self, expr)  # type: ignore[arg-type, call-arg]

    def _default_dofdesc(self, dofdesc):
        if dofdesc.geometry is None:
            # NOTE: this is a heuristic to determine how to tag things:
            #   * if no `discr_stage` is given, it's probably a target, since
            #   only `QBXLayerPotentialSource` has stages.
            #   * if some stage is present, assume it's a source
            if (dofdesc.discr_stage is None
                    and dofdesc.granularity == prim.GRANULARITY_NODE):
                dofdesc = dofdesc.copy(geometry=self.default_target)
            else:
                dofdesc = dofdesc.copy(geometry=self.default_source)
        elif dofdesc.geometry is prim.DEFAULT_SOURCE:
            dofdesc = dofdesc.copy(geometry=self.default_source)
        elif dofdesc.geometry is prim.DEFAULT_TARGET:
            dofdesc = dofdesc.copy(geometry=self.default_target)

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
            target = target.copy(geometry=self.default_target)

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
        # NOTE: this doesn't use `_default_dofdesc` because it should be always
        # evaluated at the targets (ignores `discr_stage`)
        dofdesc = expr.dofdesc
        if dofdesc.geometry is None:
            dofdesc = dofdesc.copy(geometry=self.default_target)

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

    def map_is_shape_class(self, expr):
        return type(expr)(expr.shape, self._default_dofdesc(expr.dofdesc))

    def map_error_expression(self, expr):
        return expr

    def operand_rec(self, expr):
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
        self.operand_rec = LocationTagger(default_source,
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


# {{{ DerivativeBinder

class _IsSptiallyVaryingMapper(CombineMapper):
    def combine(self, values):
        import operator
        from functools import reduce
        return reduce(operator.or_, values, False)

    def map_constant(self, expr):
        return False

    def map_spatial_constant(self, expr):
        return False

    def map_variable(self, expr):
        return True

    def map_int_g(self, expr):
        return True


class _DerivativeTakerUnsupoortedProductError(Exception):
    pass


class DerivativeTaker(Mapper):
    def __init__(self, ambient_axis):
        self.ambient_axis = ambient_axis

    def map_constant(self, expr):
        return 0

    def map_sum(self, expr):
        children = [self.rec(child) for child in expr.children]
        if all(child is orig for child, orig in zip(
                children, expr.children, strict=True)):
            return expr

        from pymbolic.primitives import flattened_sum
        return flattened_sum(children)

    def map_product(self, expr):
        const = []
        nonconst = []
        for subexpr in expr.children:
            if _IsSptiallyVaryingMapper()(subexpr):
                nonconst.append(subexpr)
            else:
                const.append(subexpr)

        if len(nonconst) > 1:
            raise _DerivativeTakerUnsupoortedProductError(
                    "DerivativeTaker doesn't support products with "
                    "more than one non-constant")

        if not nonconst:
            nonconst = [1]

        from pytools import product
        return product(const) * self.rec(nonconst[0])

    def map_int_g(self, expr):
        from sumpy.kernel import AxisTargetDerivative
        return expr.copy(
                target_kernel=AxisTargetDerivative(
                    self.ambient_axis, expr.target_kernel))


class DerivativeSourceAndNablaComponentCollector(
        Collector,
        DerivativeSourceAndNablaComponentCollectorBase):
    pass


class NablaComponentToUnitVector(
        EvaluationMapper,
        NablaComponentToUnitVectorBase):
    pass


class DerivativeSourceFinder(
        EvaluationMapper,
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

    def __init__(self, places, from_discr_stage=None):
        """
        :arg from_discr_stage: sets the stage on which to evaluate the expression
            before interpolation. For valid values, see
            :attr:`~pytential.symbolic.dof_desc.DOFDescriptor.discr_stage`.
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


# {{{ QBXPreprocessor

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


# {{{ StringifyMapper

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

    def map_interpolation(self, expr, enclosing_prec):
        return "Interp[{}->{}]({})".format(
                stringify_where(expr.from_dd),
                stringify_where(expr.to_dd),
                self.rec(expr.operand, PREC_PRODUCT))

    def map_is_shape_class(self, expr, enclosing_prec):
        return "IsShape[{}]({})".format(stringify_where(expr.dofdesc),
                                        expr.shape.__name__)


class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin, StringifyMapper):
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
