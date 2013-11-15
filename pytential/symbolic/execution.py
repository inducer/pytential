from __future__ import division

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


from pymbolic.mapper.evaluator import (
        EvaluationMapper as EvaluationMapperBase)
import numpy as np

import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa


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

    def map_node_sum(self, expr):
        return cl.array.sum(self.rec(expr.operand)).get()

    def map_ones(self, expr):
        discr = self.bound_expr.discretizations[expr.where]
        result = (discr
                .empty(discr.real_dtype, queue=self.queue)
                .with_queue(self.queue))

        result.fill(1)
        return result

    def map_node_coordinate_component(self, expr):
        discr = self.bound_expr.discretizations[expr.where]
        return discr.nodes()[expr.ambient_axis] \
                .with_queue(self.queue)

    def map_num_reference_derivative(self, expr):
        discr = self.bound_expr.discretizations[expr.where]
        return discr.num_reference_derivative(
                self.queue,
                expr.ref_axes, self.rec(expr.operand)) \
                        .with_queue(self.queue)

    def map_q_weight(self, expr):
        discr = self.bound_expr.discretizations[expr.where]
        return discr.quad_weights(self.queue) \
                .with_queue(self.queue)

    def map_inverse(self, expr):
        bound_op_cache = self.bound_expr.get_cache("bound_op")

        try:
            bound_op = bound_op_cache[expr]
        except KeyError:
            bound_op = bind(
                    expr.expression,
                    self.bound_expr.discretizations[expr.where],
                    self.bound_expr.iprec)
            bound_op_cache[expr] = bound_op

        scipy_op = bound_op.scipy_op(expr.variable_name, expr.where,
                **dict((var_name, self.rec(var_expr))
                    for var_name, var_expr in expr.extra_vars.iteritems()))

        from pytential.gmres import solve_lin_op
        rhs = self.rec(expr.rhs)
        result = solve_lin_op(scipy_op, rhs, debug=False)
        return result

    def map_quad_kernel_op(self, expr):
        source = self.bound_expr.discretizations[expr.source]
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

    def apply_sqrt(self, args):
        arg, = args
        return cl.clmath.sqrt(self.rec(arg))

    def apply_abs(self, args):
        arg, = args
        return abs(self.rec(arg))

    def apply_conj(self, args):
        arg, = args
        return self.rec(arg).conj()

    # }}}

    def map_call(self, expr):
        from pytential.symbolic.primitives import Function
        if isinstance(expr.function, Function):
            return getattr(self, "apply_"+expr.function.name)(expr.parameters)
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
            bound_expr, queue, arg_name, total_dofs,
            starts_and_ends, extra_args):
        self.bound_expr = bound_expr
        self.queue = queue
        self.arg_name = arg_name
        self.total_dofs = total_dofs
        self.starts_and_ends = starts_and_ends
        self.extra_args = extra_args

    @property
    def shape(self):
        return (self.total_dofs, self.total_dofs)

    dtype = np.dtype(np.complex128)  # FIXME

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
                    np.complex128)  # FIXME
            for res_i, (start, end) in zip(result, self.starts_and_ends):
                joined_result[start:end] = res_i
            result = joined_result

        if out_host:
            result = result.get()

        return result

# }}}


# {{{ bound expression

class BoundExpression:
    def __init__(self, optemplate, discretizations):
        self.optemplate = optemplate
        self.discretizations = discretizations

        self.caches = {}

        from pytential.symbolic.compiler import OperatorCompiler
        self.code = OperatorCompiler(self.discretizations)(optemplate)

    def get_cache(self, name):
        return self.caches.setdefault(name, {})

    def scipy_op(self, queue, arg_name, domains=None, **extra_args):
        """
        :arg domains: a list of discretization identifiers or
            *None* values indicating the domains on which each component of the
            solution vector lives.  *None* values indicate that the component
            is a scalar.  If *None*,
            :class:`pytential.symbolic.primitives.DEFAULT_TARGET`, is required
            to be a key in :attr:`discretizations`.
        """

        from pytools.obj_array import is_obj_array

        if domains is None:
            from pytential.symbolic.primitives import DEFAULT_TARGET
            if DEFAULT_TARGET not in self.discretizations:
                raise RuntimeError("'domains is None' requires "
                        "DEFAULT_TARGET to be defined")
            dom_name = DEFAULT_TARGET
            if is_obj_array(self.code.result):
                domains = len(self.code.result)*[dom_name]
            else:
                domains = [dom_name]
        elif not isinstance(domains, list):
            dom_name = domains
            if is_obj_array(self.code.result):
                domains = len(self.code.result)*[dom_name]
            else:
                domains = [dom_name]

        total_dofs = 0
        starts_and_ends = []
        for dom_name in domains:
            if dom_name is None:
                size = 1
            else:
                size = self.discretizations[dom_name].nnodes

            starts_and_ends.append((total_dofs, total_dofs+size))
            total_dofs += size

        # Hidden assumption: Number of input components
        # equals number of output compoments. But IMO that's
        # fair, since these operators are usually only used
        # for linear system solving, in which case the assumption
        # has to be true.
        return MatVecOp(self, queue,
                arg_name, total_dofs, starts_and_ends, extra_args)

    def __call__(self, queue, **args):
        exec_mapper = EvaluationMapper(self, queue, args)
        return self.code.execute(exec_mapper)

# }}}


# {{{ bind

def bind(discretizations, expr, auto_where=None):
    """
    :arg discretizations: a mapping of symbolic names to
        :class:`pytential.discretization.Discretization` objects or a subclass
        of :class:`pytential.discretization.target.TargetBase`.
    :arg auto_where: For simple source-to-self or source-to-target
        evaluations, find 'where' attributes automatically.
    """

    from pytential.symbolic.primitives import DEFAULT_SOURCE, DEFAULT_TARGET
    from pytential.discretization import Discretization
    if isinstance(discretizations, Discretization):
        discretizations = {
                DEFAULT_SOURCE: discretizations,
                DEFAULT_TARGET: discretizations,
                }

    elif isinstance(discretizations, tuple):
        source_discr, target_discr = discretizations
        discretizations = {
                DEFAULT_SOURCE: source_discr,
                DEFAULT_TARGET: target_discr,
                }
        del source_discr
        del target_discr

    def cast_to_discretization(discr):
        from pytential.discretization.target import TargetBase
        if not isinstance(discr, (Discretization, TargetBase)):
            raise TypeError("must pass discretization or target to bind()")
        return discr

    discretizations = dict(
            (key, cast_to_discretization(value))
            for key, value in discretizations.iteritems())

    from pytential.symbolic.mappers import (
            ToTargetTagger,
            Dimensionalizer,
            DerivativeBinder,
            )

    if auto_where is None:
        if DEFAULT_TARGET in discretizations:
            auto_where = DEFAULT_SOURCE, DEFAULT_TARGET
        else:
            auto_where = DEFAULT_SOURCE, DEFAULT_SOURCE

    if auto_where:
        expr = ToTargetTagger(*auto_where)(expr)

    # Dimensionalize so that preprocessing only has to deal with
    # dimension-specific layer potentials.

    expr = Dimensionalizer(discretizations)(expr)

    expr = DerivativeBinder()(expr)

    for name, discr in discretizations.iteritems():
        expr = discr.preprocess_optemplate(name, discretizations, expr)

    # Dimensionalize again, in case the preprocessor spit out
    # dimension-independent stuff.
    expr = Dimensionalizer(discretizations)(expr)

    return BoundExpression(expr, discretizations)

# }}}

# vim: foldmethod=marker
