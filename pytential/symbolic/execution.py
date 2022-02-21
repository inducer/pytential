__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
"""

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

from typing import Optional

from pymbolic.mapper.evaluator import (
        EvaluationMapper as PymbolicEvaluationMapper)
import numpy as np

from arraycontext import PyOpenCLArrayContext, thaw, freeze
from meshmode.dof_array import DOFArray

from pytools import memoize_in, memoize_method
from pytential.qbx.cost import AbstractQBXCostModel

from pytential import sym

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass :: BoundExpression
"""


# FIXME caches: fix up queues

class EvaluationMapperCSECacheKey:
    """Serves as a unique key for the common subexpression cache in
    :meth:`GeometryCollection._get_cache`.
    """


class EvaluationMapperBoundOpCacheKey:
    """Serves as a unique key for the bound operator cache in
    :meth:`GeometryCollection._get_cache`.
    """


# {{{ evaluation mapper base (shared, between actual eval and cost model)

class EvaluationMapperBase(PymbolicEvaluationMapper):
    def __init__(self, bound_expr, actx: PyOpenCLArrayContext, context=None,
            target_geometry=None,
            target_points=None, target_normals=None, target_tangents=None):
        if context is None:
            context = {}
        PymbolicEvaluationMapper.__init__(self, context)

        self.bound_expr = bound_expr
        self.places = bound_expr.places
        self.array_context = actx

        from arraycontext import PyOpenCLArrayContext
        if not isinstance(actx, PyOpenCLArrayContext):
            raise NotImplementedError("evaluation with non-PyOpenCL array context")

        self.queue = actx.queue

    # {{{ map_XXX

    def _map_minmax(self, func, inherited_func, expr):
        ev_children = [self.rec(ch) for ch in expr.children]
        from functools import reduce
        from meshmode.dof_array import DOFArray
        if any(isinstance(ch, self.array_context.array_types + (DOFArray,))
                for ch in ev_children):
            return reduce(func, ev_children)
        else:
            return inherited_func(expr)

    def map_max(self, expr):
        return self._map_minmax(
                self.array_context.np.maximum,
                super().map_max,
                expr)

    def map_min(self, expr):
        return self._map_minmax(
                self.array_context.np.minimum,
                super().map_min,
                expr)

    def map_node_sum(self, expr):
        actx = self.array_context
        return sum(actx.np.sum(grp_ary) for grp_ary in self.rec(expr.operand))

    def map_node_max(self, expr):
        from functools import reduce
        actx = self.array_context
        return reduce(
            actx.np.maximum,
            (actx.np.max(grp_ary) for grp_ary in self.rec(expr.operand)))

    def map_node_min(self, expr):
        from functools import reduce
        actx = self.array_context
        return reduce(
            actx.np.minimum,
            (actx.np.min(grp_ary) for grp_ary in self.rec(expr.operand)))

    def _map_elementwise_reduction(self, reduction_name, expr):
        import loopy as lp
        from arraycontext import make_loopy_program
        from meshmode.transform_metadata import (
                ConcurrentElementInameTag, ConcurrentDOFInameTag)

        @memoize_in(self.places, "elementwise_node_"+reduction_name)
        def node_knl():
            t_unit = make_loopy_program(
                    """{[iel, idof, jdof]:
                        0<=iel<nelements and
                        0<=idof, jdof<ndofs}""",
                    """
                    result[iel, idof] = %s(jdof, operand[iel, jdof])
                    """ % reduction_name,
                    name="nodewise_reduce")

            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag(),
                })

        @memoize_in(self.places, "elementwise_"+reduction_name)
        def element_knl():
            # FIXME: This computes the reduction value redundantly for each
            # output DOF.
            t_unit = make_loopy_program(
                    """{[iel, jdof]:
                        0<=iel<nelements and
                        0<=jdof<ndofs}
                    """,
                    """
                    result[iel, 0] = %s(jdof, operand[iel, jdof])
                    """ % reduction_name,
                    name="elementwise_reduce")

            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                })

        discr = self.places.get_discretization(
                expr.dofdesc.geometry, expr.dofdesc.discr_stage)
        operand = self.rec(expr.operand)
        assert operand.shape == (len(discr.groups),)

        def _reduce(knl, result):
            for grp in discr.groups:
                self.array_context.call_loopy(knl,
                        operand=operand[grp.index],
                        result=result[grp.index])

            return result

        dtype = operand.entry_dtype
        granularity = expr.dofdesc.granularity
        if granularity is sym.GRANULARITY_NODE:
            return _reduce(node_knl(),
                    discr.empty(self.array_context, dtype=dtype))
        elif granularity is sym.GRANULARITY_ELEMENT:
            result = DOFArray(self.array_context, tuple([
                self.array_context.empty((grp.nelements, 1), dtype=dtype)
                for grp in discr.groups
                ]))
            return _reduce(element_knl(), result)
        else:
            raise ValueError(f"unsupported granularity: {granularity}")

    def map_elementwise_sum(self, expr):
        return self._map_elementwise_reduction("sum", expr)

    def map_elementwise_min(self, expr):
        return self._map_elementwise_reduction("min", expr)

    def map_elementwise_max(self, expr):
        return self._map_elementwise_reduction("max", expr)

    def map_ones(self, expr):
        discr = self.places.get_discretization(
                expr.dofdesc.geometry, expr.dofdesc.discr_stage)
        result = discr.empty(actx=self.array_context, dtype=discr.real_dtype)

        for grp_ary in result:
            grp_ary.fill(1)
        return result

    def map_node_coordinate_component(self, expr):
        discr = self.places.get_discretization(
                expr.dofdesc.geometry, expr.dofdesc.discr_stage)

        from arraycontext import thaw
        x = discr.nodes()[expr.ambient_axis]
        return thaw(x, self.array_context)

    def map_num_reference_derivative(self, expr):
        from pytools import flatten
        ref_axes = flatten([axis] * mult for axis, mult in expr.ref_axes)

        from meshmode.discretization import num_reference_derivative
        discr = self.places.get_discretization(
                expr.dofdesc.geometry, expr.dofdesc.discr_stage)

        return num_reference_derivative(discr, ref_axes, self.rec(expr.operand))

    def map_q_weight(self, expr):
        from arraycontext import thaw
        discr = self.places.get_discretization(
                expr.dofdesc.geometry, expr.dofdesc.discr_stage)
        return thaw(discr.quad_weights(), self.array_context)

    def map_inverse(self, expr):
        bound_op_cache = self.bound_expr.places._get_cache(
                EvaluationMapperBoundOpCacheKey)

        try:
            bound_op = bound_op_cache[expr]
        except KeyError:
            bound_op = bind(
                    expr.expression,
                    self.places.get_geometry(expr.dofdesc.geometry),
                    self.bound_expr.iprec)
            bound_op_cache[expr] = bound_op

        scipy_op = bound_op.scipy_op(expr.variable_name, expr.dofdesc,
                **{var_name: self.rec(var_expr)
                    for var_name, var_expr in expr.extra_vars.items()})

        from pytential.solve import gmres
        rhs = self.rec(expr.rhs)
        result = gmres(scipy_op, rhs)
        return result

    def map_interpolation(self, expr):
        operand = self.rec(expr.operand)

        if isinstance(operand,
                self.array_context.array_types
                + (list, np.ndarray, DOFArray)):
            conn = self.places.get_connection(expr.from_dd, expr.to_dd)
            return conn(operand)
        elif isinstance(operand, (int, float, complex, np.number)):
            return operand
        else:
            raise TypeError(f"cannot interpolate '{type(operand).__name__}'")

    def map_common_subexpression(self, expr):
        if expr.scope == sym.cse_scope.EXPRESSION:
            cache = self.bound_expr._get_cache(EvaluationMapperCSECacheKey)
        elif expr.scope == sym.cse_scope.DISCRETIZATION:
            cache = self.places._get_cache(EvaluationMapperCSECacheKey)
        else:
            return self.rec(expr.child)

        from numbers import Number
        try:
            rec = cache[expr.child]
            if (expr.scope == sym.cse_scope.DISCRETIZATION
                    and not isinstance(rec, Number)):
                rec = thaw(rec, self.array_context)
        except KeyError:
            cached_rec = rec = self.rec(expr.child)
            if (expr.scope == sym.cse_scope.DISCRETIZATION
                    and not isinstance(rec, Number)):
                cached_rec = freeze(cached_rec, self.array_context)

            cache[expr.child] = cached_rec

        return rec

    # }}}

    def map_error_expression(self, expr):
        raise RuntimeError(expr.message)

    def map_is_shape_class(self, expr):
        discr = self.places.get_discretization(
            expr.dofdesc.geometry, expr.dofdesc.discr_stage)

        from pytools import is_single_valued
        if not is_single_valued(type(grp.mesh_el_group) for grp in discr.groups):
            # FIXME Conceivably, one could stick per-group bools into a DOFArray.
            raise NotImplementedError(
                    "non-homogeneous element groups are not supported")

        from meshmode.mesh import _ModepyElementGroup
        meg = discr.groups[0].mesh_el_group
        if isinstance(meg, _ModepyElementGroup):
            return isinstance(meg._modepy_shape, expr.shape)
        else:
            raise TypeError(f"element type not supported: '{type(meg).__name__}'")

    def exec_assign(self, actx: PyOpenCLArrayContext, insn, bound_expr, evaluate):
        return [(name, evaluate(expr))
                for name, expr in zip(insn.names, insn.exprs)]

    def exec_compute_potential_insn(
            self, actx: PyOpenCLArrayContext, insn, bound_expr, evaluate):
        raise NotImplementedError

    def map_call(self, expr):
        from pytential.symbolic.primitives import NumpyMathFunction

        if isinstance(expr.function, NumpyMathFunction):
            args = [self.rec(arg) for arg in expr.parameters]
            from numbers import Number
            if all(isinstance(arg, Number) for arg in args):
                return getattr(np, expr.function.name)(*args)
            else:
                return getattr(self.array_context.np, expr.function.name)(*args)

        else:
            return super().map_call(expr)

# }}}


# {{{ evaluation mapper

class EvaluationMapper(EvaluationMapperBase):

    def __init__(self, bound_expr, actx, context=None,
            timing_data=None):
        EvaluationMapperBase.__init__(self, bound_expr, actx, context)
        self.timing_data = timing_data

    def exec_compute_potential_insn(
            self, actx: PyOpenCLArrayContext, insn, bound_expr, evaluate):
        source = bound_expr.places.get_geometry(insn.source.geometry)

        return_timing_data = self.timing_data is not None

        result, timing_data = (
                source.exec_compute_potential_insn(
                    actx, insn, bound_expr, evaluate, return_timing_data))

        if return_timing_data:
            # The compiler ensures this.
            assert insn not in self.timing_data

            self.timing_data[insn] = timing_data

        return result

# }}}


# {{{ cost model evaluation mapper

class CostModelMapper(EvaluationMapperBase):
    """Mapper for evaluating cost models.

    This executes everything *except* the layer potential operator. Instead of
    executing the operator, the cost model gets run and the cost
    data is collected.

    .. attribute:: kernel_to_calibration_params

        Can either be a :class:`str` "constant_one", which uses the constant 1.0 as
        calibration parameters for all stages of all kernels, or be a :class:`dict`,
        which maps from kernels to the calibration parameters, returned from
        `estimate_kernel_specific_calibration_params`.

    """

    def __init__(self, bound_expr, actx,
                 kernel_to_calibration_params, per_box,
                 context=None,
                 target_geometry=None,
                 target_points=None, target_normals=None, target_tangents=None):
        if context is None:
            context = {}
        EvaluationMapperBase.__init__(
                self, bound_expr, actx, context,
                target_geometry,
                target_points,
                target_normals,
                target_tangents)

        self.kernel_to_calibration_params = kernel_to_calibration_params
        self.modeled_cost = {}
        self.metadata = {}
        self.per_box = per_box

    def exec_compute_potential_insn(
            self, actx: PyOpenCLArrayContext, insn, bound_expr, evaluate):
        source = bound_expr.places.get_geometry(insn.source.geometry)
        knls = frozenset(knl for knl in insn.target_kernels)

        if (isinstance(self.kernel_to_calibration_params, str)
                and self.kernel_to_calibration_params == "constant_one"):
            calibration_params = \
                AbstractQBXCostModel.get_unit_calibration_params()
        else:
            calibration_params = self.kernel_to_calibration_params[knls]

        result, (cost_model_result, metadata) = \
            source.cost_model_compute_potential_insn(
                actx, insn, bound_expr, evaluate, calibration_params,
                self.per_box)

        # The compiler ensures this.
        assert insn not in self.modeled_cost

        self.modeled_cost[insn] = cost_model_result
        self.metadata[insn] = metadata

        return result

    def get_modeled_cost(self):
        return self.modeled_cost, self.metadata

# }}}


# {{{ scipy-like mat-vec op

class MatVecOp:
    """A :class:`scipy.sparse.linalg.LinearOperator` work-alike.
    Exposes a :mod:`pytential` operator as a generic matrix operation,
    i.e., given :math:`x`, compute :math:`Ax`.

    .. attribute:: shape
    .. attribute:: dtype
    .. automethod:: matvec
    """

    def __init__(self,
            bound_expr, actx: PyOpenCLArrayContext,
            arg_name, dtype, total_dofs, discrs, starts_and_ends, extra_args):
        self.bound_expr = bound_expr
        self.array_context = actx
        self.arg_name = arg_name
        self.dtype = dtype
        self.total_dofs = total_dofs
        self.discrs = discrs
        self.starts_and_ends = starts_and_ends
        self.extra_args = extra_args

    @property
    def shape(self):
        return (self.total_dofs, self.total_dofs)

    @property
    def _operator_uses_obj_array(self):
        return len(self.discrs) > 1

    def flatten(self, ary):
        # Return a flat version of *ary*. The returned value is suitable for
        # use with solvers whose API expects a one-dimensional array.
        if not self._operator_uses_obj_array:
            ary = [ary]

        from arraycontext import flatten
        result = self.array_context.empty(self.total_dofs, self.dtype)
        for res_i, (start, end) in zip(ary, self.starts_and_ends):
            result[start:end] = flatten(res_i, self.array_context)

        return result

    def unflatten(self, ary):
        # Convert a flat version of *ary* into a structured version.
        components = []
        for discr, (start, end) in zip(self.discrs, self.starts_and_ends):
            component = ary[start:end]

            from meshmode.discretization import Discretization
            if isinstance(discr, Discretization):
                from arraycontext import unflatten
                template_ary = thaw(discr.nodes()[0], self.array_context)
                component = unflatten(
                        template_ary, component, self.array_context,
                        strict=False)

            components.append(component)

        if self._operator_uses_obj_array:
            from pytools.obj_array import make_obj_array
            return make_obj_array(components)
        else:
            return components[0]

    def matvec(self, x):
        # Three types of inputs are supported:
        # * flat NumPy arrays
        #    => output is a flat NumPy array
        # * flat PyOpenCL arrays
        #    => output is a flat PyOpenCL array
        # * structured arrays (object arrays/DOFArrays)
        #    => output has same structure as input
        if isinstance(x, DOFArray):
            flat, host = False, False
        elif isinstance(x, np.ndarray) and x.dtype.char == "O":
            flat, host = False, False
        elif isinstance(x, self.array_context.array_types):
            flat, host = True, False
            assert x.shape == (self.total_dofs,)
        elif isinstance(x, np.ndarray) and x.dtype.char != "O":
            x = self.array_context.from_numpy(x)
            flat, host = True, True
            assert x.shape == (self.total_dofs,)
        else:
            raise ValueError(f"unsupported input type: {type(x).__name__}")

        args = self.extra_args.copy()
        args[self.arg_name] = self.unflatten(x) if flat else x
        result = self.bound_expr(self.array_context, **args)

        if flat:
            result = self.flatten(result)
        if host:
            result = self.array_context.to_numpy(result)

        return result

# }}}


# {{{ expression prep

def _prepare_domains(nresults, places, domains, default_domain):
    """
    :arg nresults: number of results.
    :arg places: a :class:`~pytential.GeometryCollection`.
    :arg domains: recommended domains.
    :arg default_domain: default value for domains which are not provided.

    :return: a list of domains for each result. If domains is `None`, each
        element in the list is *default_domain*. If *domains* is a scalar
        (i.e., not a *list* or *tuple*), each element in the list is
        *domains*. Otherwise, *domains* is returned as is.
    """

    if domains is None:
        dom_name = default_domain
        return nresults * [dom_name]
    elif not isinstance(domains, (list, tuple)):
        dom_name = domains
        return nresults * [dom_name]

    domains = [sym.as_dofdesc(d) for d in domains]
    assert len(domains) == nresults

    return domains


def _prepare_auto_where(auto_where, places=None):
    """
    :arg auto_where: a 2-tuple, single identifier or `None` used as a hint
        to determine the default geometries.
    :arg places: a :class:`GeometryCollection`,
        whose :attr:`GeometryCollection.auto_where` is used by default if
        provided and `auto_where` is `None`.
    :return: a tuple ``(source, target)`` of
        :class:`~pytential.symbolic.primitives.DOFDescriptor`s denoting
        the default source and target geometries.
    """

    if auto_where is None:
        if places is None:
            auto_source = sym.DEFAULT_SOURCE
            auto_target = sym.DEFAULT_TARGET
        else:
            auto_source, auto_target = places.auto_where
    elif isinstance(auto_where, (list, tuple)):
        auto_source, auto_target = auto_where
    else:
        auto_source = auto_where
        auto_target = auto_source

    return (sym.as_dofdesc(auto_source), sym.as_dofdesc(auto_target))


def _prepare_expr(places, expr, auto_where=None):
    """
    :arg places: :class:`~pytential.GeometryCollection`.
    :arg expr: a symbolic expression.
    :return: processed symbolic expressions, tagged with the appropriate
        `where` identifier from places, etc.
    """

    from pytential.source import LayerPotentialSourceBase
    from pytential.symbolic.mappers import (
            ToTargetTagger,
            DerivativeBinder)

    auto_source, auto_target = _prepare_auto_where(auto_where, places=places)
    expr = ToTargetTagger(auto_source, auto_target)(expr)
    expr = DerivativeBinder()(expr)

    for name, place in places.places.items():
        if isinstance(place, LayerPotentialSourceBase):
            expr = place.preprocess_optemplate(name, places, expr)

    from pytential.symbolic.mappers import InterpolationPreprocessor
    expr = InterpolationPreprocessor(places)(expr)

    return expr

# }}}


# {{{ geometry collection

def _is_valid_identifier(name):
    import keyword
    return name.isidentifier() and not keyword.iskeyword(name)


class _GeometryCollectionDiscretizationCacheKey:
    """Serves as a unique key for the discretization cache in
    :meth:`GeometryCollection._get_cache`.
    """


class _GeometryCollectionConnectionCacheKey:
    """Serves as a unique key for the connection cache in
    :meth:`GeometryCollection._get_cache`.
    """


class GeometryCollection:
    """A mapping from symbolic identifiers ("place IDs", typically strings)
    to 'geometries', where a geometry can be a
    :class:`~pytential.source.PotentialSource`, a
    :class:`~pytential.target.TargetBase` or a
    :class:`~meshmode.discretization.Discretization`.

    This class is meant to hold a specific combination of sources and targets
    serve to host caches of information derived from them, e.g. FMM trees
    of subsets of them, as well as related common subexpressions such as
    metric terms.

    Refinement of :class:`pytential.qbx.QBXLayerPotentialSource` entries is
    performed on demand, i.e. on calls to :meth:`get_discretization` with
    a specific *discr_stage*. To perform refinement explicitly, call
    :func:`pytential.qbx.refinement.refine_geometry_collection`,
    which allows more customization of the refinement process through
    parameters.

    .. automethod:: __init__

    .. attribute:: auto_source

        Default :class:`~pytential.symbolic.primitives.DOFDescriptor` for the
        source geometry.

    .. attribute:: auto_target

        Default :class:`~pytential.symbolic.primitives.DOFDescriptor` for the
        target geometry.

    .. automethod:: get_geometry
    .. automethod:: get_discretization
    .. automethod:: get_connection

    .. automethod:: copy
    .. automethod:: merge

    """

    def __init__(self, places, auto_where=None):
        r"""
        :arg places: a scalar, tuple of or mapping of symbolic names to
            geometry objects. Supported objects are
            :class:`~pytential.source.PotentialSource`,
            :class:`~pytential.target.TargetBase` and
            :class:`~meshmode.discretization.Discretization`. If this is
            a mapping, the keys that are strings must be valid Python identifiers.
            The tuple should contain only two entries, denoting the source and
            target geometries for layer potential evaluation, identified by
            *auto_where*.

        :arg auto_where: a single or a tuple of two
            :class:`~pytential.symbolic.primitives.DOFDescriptor`\ s, or values
            that can be converted to one using
            :func:`~pytential.symbolic.primitives.as_dofdesc`. The two
            descriptors are used to define the default source and target
            geometries for layer potential evaluations.
            By default, they are set to
            :class:`~pytential.symbolic.primitives.DEFAULT_SOURCE` and
            :class:`~pytential.symbolic.primitives.DEFAULT_TARGET` for
            sources and targets, respectively.
        """

        from pytential.target import TargetBase
        from pytential.source import PotentialSource
        from pytential.qbx import QBXLayerPotentialSource
        from meshmode.discretization import Discretization

        # {{{ construct dict

        self.places = {}
        self.caches = {}

        auto_source, auto_target = _prepare_auto_where(auto_where)
        if isinstance(places, QBXLayerPotentialSource):
            self.places[auto_source.geometry] = places
            auto_target = auto_source
        elif isinstance(places, TargetBase):
            self.places[auto_target.geometry] = places
            auto_source = auto_target
        if isinstance(places, (Discretization, PotentialSource)):
            self.places[auto_source.geometry] = places
            self.places[auto_target.geometry] = places
        elif isinstance(places, tuple):
            source_discr, target_discr = places
            self.places[auto_source.geometry] = source_discr
            self.places[auto_target.geometry] = target_discr
        else:
            self.places = places

        self.auto_where = (auto_source, auto_target)

        # }}}

        # {{{ validate

        # check auto_where
        if auto_source.geometry not in self.places:
            raise ValueError("'auto_where' source geometry is not in the "
                f"collection: '{auto_source.geometry}'")

        if auto_target.geometry not in self.places:
            raise ValueError("'auto_where' target geometry is not in the "
                f"collection: '{auto_target.geometry}'")

        # check allowed identifiers
        for name in self.places:
            if not isinstance(name, str):
                continue
            if not _is_valid_identifier(name):
                raise ValueError(f"'{name}' is not a valid identifier")

        # check allowed types
        for p in self.places.values():
            if not isinstance(p, (PotentialSource, TargetBase, Discretization)):
                raise TypeError(
                    "Values in 'places' must be discretization, targets "
                    f"or layer potential sources, got '{type(p).__name__}'")

        # check ambient_dim
        from pytools import is_single_valued
        ambient_dims = [p.ambient_dim for p in self.places.values()]
        if not is_single_valued(ambient_dims):
            raise RuntimeError("All 'places' must have the same ambient dimension.")

        self.ambient_dim = ambient_dims[0]

        # }}}

    @property
    def auto_source(self):
        return self.auto_where[0]

    @property
    def auto_target(self):
        return self.auto_where[1]

    # {{{ cache handling

    def _get_cache(self, name):
        return self.caches.setdefault(name, {})

    def _get_discr_from_cache(self, geometry, discr_stage):
        cache = self._get_cache(_GeometryCollectionDiscretizationCacheKey)
        key = (geometry, discr_stage)

        if key not in cache:
            raise KeyError(
                    f"cached discretization does not exist on '{geometry}' "
                    f"for stage '{discr_stage}'")

        return cache[key]

    def _add_discr_to_cache(self, discr, geometry, discr_stage):
        cache = self._get_cache(_GeometryCollectionDiscretizationCacheKey)
        key = (geometry, discr_stage)

        if key in cache:
            raise RuntimeError("trying to overwrite the discretization cache of "
                    f"'{geometry}' for stage '{discr_stage}'")

        cache[key] = discr

    def _get_conn_from_cache(self, geometry, from_stage, to_stage):
        cache = self._get_cache(_GeometryCollectionConnectionCacheKey)
        key = (geometry, from_stage, to_stage)

        if key not in cache:
            raise KeyError("cached connection does not exist on "
                    f"'{geometry}' from stage '{from_stage}' to '{to_stage}'")

        return cache[key]

    def _add_conn_to_cache(self, conn, geometry, from_stage, to_stage):
        cache = self._get_cache(_GeometryCollectionConnectionCacheKey)
        key = (geometry, from_stage, to_stage)

        if key in cache:
            raise RuntimeError("trying to overwrite the connection cache of "
                    f"'{geometry}' from stage '{from_stage}' to '{to_stage}'")

        cache[key] = conn

    def _get_qbx_discretization(self, geometry, discr_stage):
        lpot_source = self.get_geometry(geometry)

        try:
            discr = self._get_discr_from_cache(geometry, discr_stage)
        except KeyError:
            from pytential.qbx.refinement import _refine_for_global_qbx

            # NOTE: this adds the required discretizations to the cache
            dofdesc = sym.DOFDescriptor(geometry, discr_stage)
            _refine_for_global_qbx(self, dofdesc,
                    lpot_source.refiner_code_container.get_wrangler(),
                    _copy_collection=False)

            discr = self._get_discr_from_cache(geometry, discr_stage)

        return discr

    # }}}

    def get_connection(self, from_dd, to_dd):
        """Construct a connection from *from_dd* to *to_dd* geometries.

        :param from_dd: a :class:`~pytential.symbolic.primitives.DOFDescriptor`
            or a value that can be converted to one using
            :func:`~pytential.symbolic.primitives.as_dofdesc`.
        :param to_dd: as *from_dd*.

        :returns: an object compatible with the
            :class:`~meshmode.discretization.connection.DiscretizationConnection`
            interface.
        """

        from pytential.symbolic.dof_connection import connection_from_dds
        return connection_from_dds(self, from_dd, to_dd)

    def get_discretization(self, geometry, discr_stage=None):
        """Get the geometry or discretization in the collection.

        If a specific QBX stage discretization is requested, refinement is
        performed on demand and cached for subsequent calls.

        :param geometry: the identifier of the geometry in the collection.
        :param discr_stage: if the geometry is a
            :class:`~pytential.source.LayerPotentialSourceBase`, this denotes
            the QBX stage of the returned discretization. Can be one of
            :class:`~pytential.symbolic.primitives.QBX_SOURCE_STAGE1` (default),
            :class:`~pytential.symbolic.primitives.QBX_SOURCE_STAGE2` or
            :class:`~pytential.symbolic.primitives.QBX_SOURCE_QUAD_STAGE2`.

        :returns: a geometry object in the collection or a
            :class:`~meshmode.discretization.Discretization` corresponding to
            *discr_stage*.
        """
        if discr_stage is None:
            discr_stage = sym.QBX_SOURCE_STAGE1
        discr = self.get_geometry(geometry)

        from pytential.qbx import QBXLayerPotentialSource
        from pytential.source import LayerPotentialSourceBase

        if isinstance(discr, QBXLayerPotentialSource):
            return self._get_qbx_discretization(geometry, discr_stage)
        elif isinstance(discr, LayerPotentialSourceBase):
            return discr.density_discr
        else:
            return discr

    def get_geometry(self, geometry):
        """
        :param geometry: the identifier of the geometry in the collection.
        """

        try:
            return self.places[geometry]
        except KeyError:
            raise KeyError(f"geometry not in the collection: '{geometry}'")

    def copy(self, places=None, auto_where=None):
        """Get a shallow copy of the geometry collection."""

        places = self.places if places is None else places
        return type(self)(
                places=places.copy(),
                auto_where=self.auto_where if auto_where is None else auto_where)

    def merge(self, places):
        """Merges two geometry collections and returns the new collection.

        :param places: a :class:`dict` or :class:`GeometryCollection` to
            merge with the current collection. If it is empty, a copy of the
            current collection is returned.
        """

        new_places = self.places.copy()
        if places:
            if isinstance(places, GeometryCollection):
                places = places.places
            new_places.update(places)

        return self.copy(places=new_places)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.places)})"

    def __str__(self):
        return f"{type(self).__name__}({repr(self.places)})"

# }}}


# {{{ bound expression

def _find_array_context_from_args_in_context(context, supplied_array_context=None):
    from arraycontext import PyOpenCLArrayContext
    array_contexts = []
    if supplied_array_context is not None:
        if not isinstance(supplied_array_context, PyOpenCLArrayContext):
            raise TypeError(
                    "first argument (if supplied) must be a PyOpenCLArrayContext, "
                    f"got '{type(supplied_array_context).__name__}'")

        array_contexts.append(supplied_array_context)
    del supplied_array_context

    def look_for_array_contexts(ary):
        if isinstance(ary, DOFArray):
            if ary.array_context is not None:
                array_contexts.append(ary.array_context)
        elif isinstance(ary, np.ndarray) and ary.dtype.char == "O":
            for idx in np.ndindex(ary.shape):
                look_for_array_contexts(ary[idx])
        else:
            pass

    for val in context.values():
        look_for_array_contexts(val)

    if array_contexts:
        from pytools import is_single_valued
        if not is_single_valued(array_contexts):
            raise ValueError("arguments do not agree on an array context")

        array_context = array_contexts[0]
    else:
        array_context = None

    if not isinstance(array_context, PyOpenCLArrayContext):
        raise TypeError(
                "array context (derived from arguments) is not a "
                f"PyOpenCLArrayContext: '{type(array_context).__name__}'")

    return array_context


class BoundExpression:
    """An expression readied for evaluation by binding it to a
    :class:`~pytential.GeometryCollection`.

    .. automethod :: cost_per_stage
    .. automethod :: cost_per_box
    .. automethod :: scipy_op
    .. automethod :: eval
    .. automethod :: __call__
    .. attribute :: places

    Created by calling :func:`pytential.bind`.
    """

    def __init__(self, places, sym_op_expr):
        self.places = places
        self.sym_op_expr = sym_op_expr
        self.caches = {}

    @property
    @memoize_method
    def code(self):
        from pytential.symbolic.compiler import OperatorCompiler
        return OperatorCompiler(self.places)(self.sym_op_expr)

    def _get_cache(self, name):
        return self.caches.setdefault(name, {})

    def cost_per_stage(self, calibration_params, **kwargs):
        """
        :arg calibration_params: either a :class:`dict` returned by
            `estimate_kernel_specific_calibration_params`, or a :class:`str`
            "constant_one".
        :return: a :class:`dict` mapping from instruction to per-stage cost. Each
            per-stage cost is represented by a :class:`dict` mapping from the stage
            name to the predicted time.
        """
        array_context = _find_array_context_from_args_in_context(kwargs)

        cost_model_mapper = CostModelMapper(
            self, array_context, calibration_params, per_box=False, context=kwargs
        )
        self.code.execute(cost_model_mapper)
        return cost_model_mapper.get_modeled_cost()

    def cost_per_box(self, calibration_params, **kwargs):
        """
        :arg calibration_params: either a :class:`dict` returned by
            `estimate_kernel_specific_calibration_params`, or a :class:`str`
            "constant_one".
        :return: a :class:`dict` mapping from instruction to per-box cost. Each
            per-box cost is represented by a :class:`numpy.ndarray` or
            :class:`pyopencl.array.Array` of shape (nboxes,), where the ith entry
            represents the cost of all stages for box i.
        """
        array_context = _find_array_context_from_args_in_context(kwargs)

        cost_model_mapper = CostModelMapper(
            self, array_context, calibration_params, per_box=True, context=kwargs
        )
        self.code.execute(cost_model_mapper)
        return cost_model_mapper.get_modeled_cost()

    def scipy_op(
            self, actx: PyOpenCLArrayContext, arg_name, dtype,
            domains=None, **extra_args):
        """
        :arg domains: a list of discretization identifiers or
            *None* values indicating the domains on which each component of the
            solution vector lives.  *None* values indicate that the component
            is a scalar.  If *domains* is *None*,
            :class:`~pytential.symbolic.primitives.DEFAULT_TARGET` is required
            to be a key in :attr:`places`.
        :returns: An object that (mostly) satisfies the
            :class:`scipy.sparse.linalg.LinearOperator` protocol, except for
            accepting and returning :class:`pyopencl.array.Array` arrays.
        """

        if isinstance(self.code.result, np.ndarray):
            nresults = len(self.code.result)
        else:
            nresults = 1

        domains = _prepare_domains(nresults,
                self.places, domains, self.places.auto_target)

        total_dofs = 0
        discrs = []
        starts_and_ends = []
        for dom_name in domains:
            if dom_name is None:
                discr = None
                size = 1
            else:
                discr = self.places.get_discretization(
                        dom_name.geometry, dom_name.discr_stage)
                size = discr.ndofs

            discrs.append(discr)
            starts_and_ends.append((total_dofs, total_dofs+size))
            total_dofs += size

        # Hidden assumption: Number of input components
        # equals number of output components. But IMO that's
        # fair, since these operators are usually only used
        # for linear system solving, in which case the assumption
        # has to be true.
        return MatVecOp(self, actx,
                arg_name, dtype, total_dofs, discrs, starts_and_ends, extra_args)

    def eval(self, context=None, timing_data=None,
            array_context: Optional[PyOpenCLArrayContext] = None):
        """Evaluate the expression in *self*, using the
        input variables given in the dictionary *context*.

        :arg timing_data: A dictionary into which timing
            data will be inserted during evaluation.
            (experimental)
        :arg array_context: only needs to be supplied if no instances of
            :class:`~meshmode.dof_array.DOFArray` with a
            :class:`~arraycontext.PyOpenCLArrayContext`
            are supplied as part of *context*.
        :returns: the value of the expression, as a scalar,
            array or an :class:`arraycontext.ArrayContainer` of these.
        """

        if context is None:
            context = {}

        # NOTE: avoid compiling any code if the expression is long lived
        # and already nicely cached in the collection from a previous run
        import pymbolic.primitives as prim
        if isinstance(self.sym_op_expr, prim.CommonSubexpression) \
                and self.sym_op_expr.scope == sym.cse_scope.DISCRETIZATION:
            cache = self.places._get_cache(EvaluationMapperCSECacheKey)

            from numbers import Number
            expr = self.sym_op_expr
            if expr.child in cache:
                value = cache[expr.child]
                if (expr.scope == sym.cse_scope.DISCRETIZATION
                        and not isinstance(value, Number)):
                    value = thaw(value, array_context)

                return value

        array_context = _find_array_context_from_args_in_context(
                context, array_context)

        exec_mapper = EvaluationMapper(
                self, array_context, context, timing_data=timing_data)
        return self.code.execute(exec_mapper)

    def __call__(self, *args, **kwargs):
        """Evaluate the expression in *self*, using the
        input variables given in the dictionary *context*.

        :returns: the value of the expression, as a scalar,
            :class:`meshmode.dof_array.DOFArray`, or an object array of
            these.
        """
        array_context = None
        if len(args) == 1:
            array_context, = args
            if not isinstance(array_context, PyOpenCLArrayContext):
                raise TypeError(
                        "first positional argument (if given) must be a "
                        f"PyOpenCLArrayContext: '{type(array_context).__name__}'")

        elif not args:
            pass

        else:
            raise TypeError("More than one positional argument supplied. "
                    "None or an PyOpenCLArrayContext expected.")

        return self.eval(kwargs, array_context=array_context)


def bind(places, expr, auto_where=None):
    """
    :arg places: a :class:`pytential.GeometryCollection`.
        Alternatively, any list or mapping that is a valid argument for its
        constructor can also be used.
    :arg auto_where: for simple source-to-self or source-to-target
        evaluations, find 'where' attributes automatically.
    :arg expr: one or multiple expressions consisting of primitives
        form :mod:`pytential.symbolic.primitives` (aka ``pytential.sym``).
        Multiple expressions can be combined into one object to pass here
        in the form of a :mod:`numpy` object array
    :returns: a :class:`pytential.symbolic.execution.BoundExpression`
    """
    if not isinstance(places, GeometryCollection):
        places = GeometryCollection(places, auto_where=auto_where)
        auto_where = places.auto_where
    expr = _prepare_expr(places, expr, auto_where=auto_where)

    return BoundExpression(places, expr)

# }}}


# {{{ matrix building

def _bmat(blocks, dtypes):
    from pytools import single_valued
    from pytential.symbolic.matrix import is_zero

    nrows = blocks.shape[0]
    ncolumns = blocks.shape[1]

    # "block row starts"/"block column starts"
    brs = np.cumsum([0]
            + [single_valued(blocks[ibrow, ibcol].shape[0]
                             for ibcol in range(ncolumns)
                             if not is_zero(blocks[ibrow, ibcol]))
             for ibrow in range(nrows)])

    bcs = np.cumsum([0]
            + [single_valued(blocks[ibrow, ibcol].shape[1]
                             for ibrow in range(nrows)
                             if not is_zero(blocks[ibrow, ibcol]))
             for ibcol in range(ncolumns)])

    result = np.zeros((brs[-1], bcs[-1]),
                      dtype=np.find_common_type(dtypes, []))
    for ibcol in range(ncolumns):
        for ibrow in range(nrows):
            result[brs[ibrow]:brs[ibrow + 1], bcs[ibcol]:bcs[ibcol + 1]] = \
                    blocks[ibrow, ibcol]

    return result


def build_matrix(actx, places, exprs, input_exprs, domains=None,
        auto_where=None, context=None):
    """
    :arg actx: a :class:`~arraycontext.PyOpenCLArrayContext`.
    :arg places: a :class:`pytential.GeometryCollection`.
        Alternatively, any list or mapping that is a valid argument for its
        constructor can also be used.
    :arg exprs: an array of expressions corresponding to the output block
        rows of the matrix. May also be a single expression.
    :arg input_exprs: an array of expressions corresponding to the
        input block columns of the matrix. May also be a single expression.
    :arg domains: a list of discretization identifiers (see 'places') or
        *None* values indicating the domains on which each component of the
        solution vector lives.  *None* values indicate that the component
        is a scalar.  If *None*, *auto_where* or, if it is not provided,
        :class:`~pytential.symbolic.primitives.DEFAULT_SOURCE` is required
        to be a key in :attr:`places`.
    :arg auto_where: For simple source-to-self or source-to-target
        evaluations, find 'where' attributes automatically.
    """

    if context is None:
        context = {}

    from pytential import GeometryCollection
    if not isinstance(places, GeometryCollection):
        places = GeometryCollection(places, auto_where=auto_where)
    exprs = _prepare_expr(places, exprs, auto_where=auto_where)

    if not (isinstance(exprs, np.ndarray) and exprs.dtype.char == "O"):
        from pytools.obj_array import make_obj_array
        exprs = make_obj_array([exprs])

    try:
        input_exprs = list(input_exprs)
    except TypeError:
        # not iterable, wrap in a list
        input_exprs = [input_exprs]

    domains = _prepare_domains(len(input_exprs),
            places, domains, places.auto_source)

    from pytential.symbolic.matrix import MatrixBuilder, is_zero
    nblock_rows = len(exprs)
    nblock_columns = len(input_exprs)
    blocks = np.zeros((nblock_rows, nblock_columns), dtype=object)

    dtypes = []
    for ibcol in range(nblock_columns):
        dep_source = places.get_geometry(domains[ibcol].geometry)
        dep_discr = places.get_discretization(
                domains[ibcol].geometry, domains[ibcol].discr_stage)

        mbuilder = MatrixBuilder(
                actx,
                dep_expr=input_exprs[ibcol],
                other_dep_exprs=(input_exprs[:ibcol]
                                 + input_exprs[ibcol + 1:]),
                dep_source=dep_source,
                dep_discr=dep_discr,
                places=places,
                context=context)

        for ibrow in range(nblock_rows):
            block = mbuilder(exprs[ibrow])
            assert is_zero(block) or isinstance(block, np.ndarray)

            blocks[ibrow, ibcol] = block
            if isinstance(block, np.ndarray):
                dtypes.append(block.dtype)

    return actx.from_numpy(_bmat(blocks, dtypes))

# }}}

# vim: foldmethod=marker
