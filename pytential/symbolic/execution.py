from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
from six.moves import zip

from pymbolic.mapper.evaluator import (
        EvaluationMapper as EvaluationMapperBase)
import numpy as np

import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from loopy.version import MOST_RECENT_LANGUAGE_VERSION

from pytools import memoize_in


# FIXME caches: fix up queues

# {{{ evaluation mapper

class EvaluationMapper(EvaluationMapperBase):
    def __init__(self, bound_expr, queue, context={},
            target_geometry=None,
            target_points=None, target_normals=None, target_tangents=None):
        EvaluationMapperBase.__init__(self, context)

        self.bound_expr = bound_expr
        self.queue = queue

    # {{{ map_XXX

    def _map_minmax(self, func, inherited, expr):
        ev_children = [self.rec(ch) for ch in expr.children]
        from functools import reduce, partial
        if any(isinstance(ch, cl.array.Array) for ch in ev_children):
            return reduce(partial(func, queue=self.queue), ev_children)
        else:
            return inherited(expr)

    def map_max(self, expr):
        return self._map_minmax(
                cl.array.maximum,
                super(EvaluationMapper, self).map_max,
                expr)

    def map_min(self, expr):
        return self._map_minmax(
                cl.array.minimum,
                super(EvaluationMapper, self).map_min,
                expr)

    def map_node_sum(self, expr):
        return cl.array.sum(self.rec(expr.operand)).get()[()]

    def map_node_max(self, expr):
        return cl.array.max(self.rec(expr.operand)).get()[()]

    def _map_elementwise_reduction(self, reduction_name, expr):
        @memoize_in(self.bound_expr, "elementwise_"+reduction_name)
        def knl():
            import loopy as lp
            knl = lp.make_kernel(
                "{[el, idof, jdof]: 0<=el<nelements and 0<=idof,jdof<ndofs}",
                "result[el, idof] = %s(jdof, input[el, jdof])" % reduction_name,
                default_offset=lp.auto,
                lang_version=MOST_RECENT_LANGUAGE_VERSION)
            knl = lp.tag_inames(knl, "el:g.0,idof:l.0")
            return knl

        discr = self.bound_expr.get_discretization(expr.where)

        operand = self.rec(expr.operand)

        assert operand.shape == (discr.nnodes,)

        result = cl.array.empty(self.queue, discr.nnodes, operand.dtype)
        for group in discr.groups:
            knl()(self.queue,
                    input=group.view(operand),
                    result=group.view(result))

        return result

    def map_elementwise_sum(self, expr):
        return self._map_elementwise_reduction("sum", expr)

    def map_elementwise_min(self, expr):
        return self._map_elementwise_reduction("min", expr)

    def map_elementwise_max(self, expr):
        return self._map_elementwise_reduction("max", expr)

    def map_ones(self, expr):
        discr = self.bound_expr.get_discretization(expr.where)

        result = (discr
                .empty(queue=self.queue, dtype=discr.real_dtype)
                .with_queue(self.queue))

        result.fill(1)
        return result

    def map_node_coordinate_component(self, expr):
        discr = self.bound_expr.get_discretization(expr.where)
        return discr.nodes()[expr.ambient_axis] \
                .with_queue(self.queue)

    def map_num_reference_derivative(self, expr):
        discr = self.bound_expr.get_discretization(expr.where)

        from pytools import flatten
        ref_axes = flatten([axis] * mult for axis, mult in expr.ref_axes)
        return discr.num_reference_derivative(
                self.queue,
                ref_axes, self.rec(expr.operand)) \
                        .with_queue(self.queue)

    def map_q_weight(self, expr):
        discr = self.bound_expr.get_discretization(expr.where)
        return discr.quad_weights(self.queue) \
                .with_queue(self.queue)

    def map_inverse(self, expr):
        bound_op_cache = self.bound_expr.get_cache("bound_op")

        try:
            bound_op = bound_op_cache[expr]
        except KeyError:
            bound_op = bind(
                    expr.expression,
                    self.bound_expr.places[expr.where],
                    self.bound_expr.iprec)
            bound_op_cache[expr] = bound_op

        scipy_op = bound_op.scipy_op(expr.variable_name, expr.where,
                **dict((var_name, self.rec(var_expr))
                    for var_name, var_expr in six.iteritems(expr.extra_vars)))

        from pytential.solve import gmres
        rhs = self.rec(expr.rhs)
        result = gmres(scipy_op, rhs, debug=False)
        return result

    def map_quad_kernel_op(self, expr):
        source = self.bound_expr.places[expr.source]
        return source.map_quad_kernel_op(expr, self.bound_expr, self.rec)

    # }}}

    def exec_assign(self, queue, insn, bound_expr, evaluate):
        return [(name, evaluate(expr))
                for name, expr in zip(insn.names, insn.exprs)], []

    # {{{ functions

    def apply_real(self, args):
        from pytools.obj_array import is_obj_array
        arg, = args
        result = self.rec(arg)
        assert not is_obj_array(result)  # numpy bug with obj_array.imag
        return result.real

    def apply_imag(self, args):
        from pytools.obj_array import is_obj_array
        arg, = args
        result = self.rec(arg)
        assert not is_obj_array(result)  # numpy bug with obj_array.imag
        return result.imag

    def apply_conj(self, args):
        arg, = args
        return self.rec(arg).conj()

    def apply_abs(self, args):
        arg, = args
        return abs(self.rec(arg))

    # }}}

    def map_call(self, expr):
        from pytential.symbolic.primitives import EvalMapperFunction, CLMathFunction

        if isinstance(expr.function, EvalMapperFunction):
            return getattr(self, "apply_"+expr.function.name)(expr.parameters)
        elif isinstance(expr.function, CLMathFunction):
            args = [self.rec(arg) for arg in expr.parameters]
            from numbers import Number
            if all(isinstance(arg, Number) for arg in args):
                return getattr(np, expr.function.name)(*args)
            else:
                return getattr(cl.clmath, expr.function.name)(
                        *args, queue=self.queue)

        else:
            return EvaluationMapperBase.map_call(self, expr)

# }}}


# {{{ scipy-like mat-vec op

class MatVecOp:
    """A :class:`scipy.sparse.linalg.LinearOperator` work-alike.
    Exposes a :mod:`pytential` operator as a generic matrix operation,
    i.e. given :math:`x`, compute :math:`Ax`.
    """

    def __init__(self,
            bound_expr, queue, arg_name, dtype, total_dofs,
            starts_and_ends, extra_args):
        self.bound_expr = bound_expr
        self.queue = queue
        self.arg_name = arg_name
        self.dtype = dtype
        self.total_dofs = total_dofs
        self.starts_and_ends = starts_and_ends
        self.extra_args = extra_args

    @property
    def shape(self):
        return (self.total_dofs, self.total_dofs)

    def matvec(self, x):
        if isinstance(x, np.ndarray):
            x = cl.array.to_device(self.queue, x)
            out_host = True
        else:
            out_host = False

        do_split = len(self.starts_and_ends) > 1
        from pytools.obj_array import make_obj_array

        if do_split:
            x = make_obj_array(
                    [x[start:end] for start, end in self.starts_and_ends])

        args = self.extra_args.copy()
        args[self.arg_name] = x
        result = self.bound_expr(self.queue, **args)

        if do_split:
            # re-join what was split
            joined_result = cl.array.empty(self.queue, self.total_dofs,
                    self.dtype)
            for res_i, (start, end) in zip(result, self.starts_and_ends):
                joined_result[start:end] = res_i
            result = joined_result

        if out_host:
            result = result.get()

        return result

# }}}


# {{{ default for 'domains' parameter

def _domains_default(nresults, places, domains, default_val):
    if domains is None:
        if default_val not in places:
            raise RuntimeError("'domains is None' requires "
                    "default domain to be defined")
        dom_name = default_val
        return nresults*[dom_name]

    elif not isinstance(domains, (list, tuple)):
        dom_name = domains
        return nresults*[dom_name]

    else:
        return domains

# }}}


# {{{ bound expression

class BoundExpression:
    def __init__(self, optemplate, places):
        self.optemplate = optemplate
        self.places = places

        self.caches = {}

        from pytential.symbolic.compiler import OperatorCompiler
        self.code = OperatorCompiler(self.places)(optemplate)

    def get_cache(self, name):
        return self.caches.setdefault(name, {})

    def get_discretization(self, where):
        from pytential.symbolic.primitives import _QBXSourceStage2
        if isinstance(where, _QBXSourceStage2):
            lpot_source = self.places[where.where]
            return lpot_source.stage2_density_discr

        discr = self.places[where]

        from pytential.source import LayerPotentialSourceBase
        if isinstance(discr, LayerPotentialSourceBase):
            discr = discr.density_discr

        return discr

    def scipy_op(self, queue, arg_name, dtype, domains=None, **extra_args):
        """
        :arg domains: a list of discretization identifiers or
            *None* values indicating the domains on which each component of the
            solution vector lives.  *None* values indicate that the component
            is a scalar.  If *domains* is *None*,
            :class:`pytential.symbolic.primitives.DEFAULT_TARGET`, is required
            to be a key in :attr:`places`.
        """

        from pytools.obj_array import is_obj_array
        if is_obj_array(self.code.result):
            nresults = len(self.code.result)
        else:
            nresults = 1

        from pytential.symbolic.primitives import DEFAULT_TARGET
        domains = _domains_default(nresults, self.places, domains,
                DEFAULT_TARGET)

        total_dofs = 0
        starts_and_ends = []
        for dom_name in domains:
            if dom_name is None:
                size = 1
            else:
                size = self.places[dom_name].nnodes

            starts_and_ends.append((total_dofs, total_dofs+size))
            total_dofs += size

        # Hidden assumption: Number of input components
        # equals number of output compoments. But IMO that's
        # fair, since these operators are usually only used
        # for linear system solving, in which case the assumption
        # has to be true.
        return MatVecOp(self, queue,
                arg_name, dtype, total_dofs, starts_and_ends, extra_args)

    def __call__(self, queue, **args):
        exec_mapper = EvaluationMapper(self, queue, args)
        return self.code.execute(exec_mapper)

# }}}


# {{{ expression prep

def prepare_places(places):
    from pytential.symbolic.primitives import DEFAULT_SOURCE, DEFAULT_TARGET
    from meshmode.discretization import Discretization
    from pytential.source import LayerPotentialSourceBase
    from pytential.target import TargetBase

    if isinstance(places, LayerPotentialSourceBase):
        places = {
                DEFAULT_SOURCE: places,
                DEFAULT_TARGET: places.density_discr,
                }
    elif isinstance(places, (Discretization, TargetBase)):
        places = {
                DEFAULT_TARGET: places,
                }

    elif isinstance(places, tuple):
        source_discr, target_discr = places
        places = {
                DEFAULT_SOURCE: source_discr,
                DEFAULT_TARGET: target_discr,
                }
        del source_discr
        del target_discr

    def cast_to_place(discr):
        from pytential.target import TargetBase
        from pytential.source import PotentialSource
        if not isinstance(discr, (Discretization, TargetBase, PotentialSource)):
            raise TypeError("must pass discretizations, "
                    "layer potential sources or targets as 'places'")
        return discr

    return dict(
            (key, cast_to_place(value))
            for key, value in six.iteritems(places))


def prepare_expr(places, expr, auto_where=None):
    """
    :arg places: result of :func:`prepare_places`
    :arg auto_where: For simple source-to-self or source-to-target
        evaluations, find 'where' attributes automatically.
    """

    from pytential.symbolic.primitives import DEFAULT_SOURCE, DEFAULT_TARGET
    from pytential.source import LayerPotentialSourceBase

    from pytential.symbolic.mappers import (
            ToTargetTagger,
            DerivativeBinder,
            )

    if auto_where is None:
        if DEFAULT_TARGET in places:
            auto_where = DEFAULT_SOURCE, DEFAULT_TARGET
        else:
            auto_where = DEFAULT_SOURCE, DEFAULT_SOURCE

    if auto_where:
        expr = ToTargetTagger(*auto_where)(expr)

    expr = DerivativeBinder()(expr)

    for name, place in six.iteritems(places):
        if isinstance(place, LayerPotentialSourceBase):
            expr = place.preprocess_optemplate(name, places, expr)

    return expr

# }}}


def bind(places, expr, auto_where=None):
    """
    :arg places: a mapping of symbolic names to
        :class:`pytential.discretization.Discretization` objects or a subclass
        of :class:`pytential.discretization.target.TargetBase`.
    :arg auto_where: For simple source-to-self or source-to-target
        evaluations, find 'where' attributes automatically.
    """

    places = prepare_places(places)
    expr = prepare_expr(places, expr, auto_where=auto_where)
    return BoundExpression(expr, places)


# {{{ matrix building

def build_matrix(queue, places, expr, input_exprs, domains=None,
        auto_where=None, context=None):
    """
    :arg queue: a :class:`pyopencl.CommandQueue` used to synchronize
        the calculation.
    :arg places: a mapping of symbolic names to
        :class:`pytential.discretization.Discretization` objects or a subclass
        of :class:`pytential.discretization.target.TargetBase`.
    :arg input_exprs: An object array of expressions corresponding to the
        input block columns of the matrix.

        May also be a single expression.
    :arg domains: a list of discretization identifiers (see 'places') or
        *None* values indicating the domains on which each component of the
        solution vector lives.  *None* values indicate that the component
        is a scalar.  If *None*,
        :class:`pytential.symbolic.primitives.DEFAULT_TARGET`, is required
        to be a key in :attr:`places`.
    :arg auto_where: For simple source-to-self or source-to-target
        evaluations, find 'where' attributes automatically.
    """

    if context is None:
        context = {}

    places = prepare_places(places)
    expr = prepare_expr(places, expr, auto_where=auto_where)

    from pytools.obj_array import is_obj_array, make_obj_array
    if not is_obj_array(expr):
        expr = make_obj_array([expr])
    try:
        input_exprs = list(input_exprs)
    except TypeError:
        # not iterable, wrap in a list
        input_exprs = [input_exprs]

    from pytential.symbolic.primitives import DEFAULT_SOURCE
    domains = _domains_default(len(input_exprs), places, domains,
            DEFAULT_SOURCE)

    nblock_rows = len(expr)
    nblock_columns = len(input_exprs)

    blocks = np.zeros((nblock_rows, nblock_columns), dtype=np.object)

    from pytential.symbolic.matrix import MatrixBuilder, is_zero

    dtypes = []

    for ibcol in range(nblock_columns):
        mbuilder = MatrixBuilder(
                queue,
                dep_expr=input_exprs[ibcol],
                other_dep_exprs=(
                    input_exprs[:ibcol]
                    +
                    input_exprs[ibcol+1:]),
                dep_source=places[domains[ibcol]],
                places=places,
                context=context)

        for ibrow in range(nblock_rows):
            block = mbuilder(expr[ibrow])

            assert (
                    is_zero(block)
                    or isinstance(block, np.ndarray))
            if isinstance(block, np.ndarray):
                dtypes.append(block.dtype)

            blocks[ibrow, ibcol] = block

            if isinstance(block, cl.array.Array):
                dtypes.append(block.dtype)

    from pytools import single_valued

    block_row_counts = [
            single_valued(
                blocks[ibrow, ibcol].shape[0]
                for ibcol in range(nblock_columns)
                if not is_zero(blocks[ibrow, ibcol]))
            for ibrow in range(nblock_rows)]

    block_col_counts = [
            places[domains[ibcol]].density_discr.nnodes
            for ibcol in range(nblock_columns)]

    # "block row starts"/"block column starts"
    brs = np.cumsum([0] + block_row_counts)
    bcs = np.cumsum([0] + block_col_counts)

    result = np.zeros(
            (sum(block_row_counts), sum(block_col_counts)),
            dtype=np.find_common_type(dtypes, []))

    for ibcol in range(nblock_columns):
        for ibrow in range(nblock_rows):
            result[brs[ibrow]:brs[ibrow+1], bcs[ibcol]:bcs[ibcol+1]] = \
                    blocks[ibrow, ibcol]

    return cl.array.to_device(queue, result)

# }}}

# vim: foldmethod=marker
