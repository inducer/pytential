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




from pymbolic.mapper.evaluator import EvaluationMapper
import scipy.sparse.linalg as sla
import numpy as np

import pyopencl as cl
import pyopencl.array
import pyopencl.clmath



# {{{ evaluator

class ExecutionMapper(EvaluationMapper):
    def __init__(self, executor, context={},
            target_geometry=None,
            target_points=None, target_normals=None, target_tangents=None):
        EvaluationMapper.__init__(self, context)

        self.executor = executor

    # {{{ map_XXX

    def map_node_sum(self, expr):
        discr = self.executor.discretizations[expr.where]
        return cl.sum(self.rec(expr.operand))

    def map_ones(self, expr):
        result = cl.array.empty(
                len(self.executor.discretizations[expr.where]), 
                dtype=np.complex128)
        result.fill(1)
        return result

    # def map_line_integral(self, expr):
    #     line_quad_rule = self.executor.lines[expr.where]
    #     return line_quad_rule(self.rec(expr.operand))

    def map_parametrization_derivative_component(self, expr):
        discr = self.executor.discretizations[expr.where]
        return discr.get_parametrization_derivative_component(
                self.executor.queue,
                expr.ambient_axis, expr.ref_axis)

    def map_q_weights(self, expr):
        discr = self.executor.discretizations[expr.where]
        return discr.get_quad_weights()

    def map_inverse(self, expr):
        bound_op_cache = self.executor.get_cache("bound_op")

        try:
            bound_op = bound_op_cache[expr]
        except KeyError:
            bound_op = bind(expr.expression, self.executor.discretizations[expr.where],
                    self.executor.iprec)
            bound_op_cache[expr] = bound_op

        scipy_op = bound_op.scipy_op(expr.variable_name, expr.where,
                **dict((var_name, self.rec(var_expr))
                    for var_name, var_expr in expr.extra_vars.iteritems()))

        from pytential.gmres import solve_lin_op
        rhs = self.rec(expr.rhs)
        result = solve_lin_op(scipy_op, rhs, debug=False)
        return result

    def map_quad_kernel_op(self, expr):
        source = self.executor.discretizations[expr.source]
        return source.map_quad_kernel_op(expr, self.executor, self.rec)

    # }}}

    def exec_assign(self, insn, executor, evaluate):
        return [(name, evaluate(expr))
                for name, expr in zip(insn.names, insn.exprs)], []

    # {{{ functions

    def apply_real(self, args):
        from pytools.obj_array import is_obj_array
        arg, = args
        result = self.rec(arg)
        assert not is_obj_array(result) # numpy bug with obj_array.imag
        return result.real

    def apply_imag(self, args):
        from pytools.obj_array import is_obj_array
        arg, = args
        result = self.rec(arg)
        assert not is_obj_array(result) # numpy bug with obj_array.imag
        return result.imag

    def apply_sqrt(self, args):
        arg, = args
        return cl.clmath.sqrt(self.rec(arg))

    # }}}

    def map_call(self, expr):
        from pytential.symbolic.primitives import Function
        if isinstance(expr.function, Function):
            return getattr(self, "apply_"+expr.function.name)(expr.parameters)
        else:
            return EvaluationMapper.map_call(self, expr)

# }}}




class MatVecOp:
    def __init__(self, executor, arg_name, total_dofs, starts_and_ends, extra_args):
        self.executor = executor
        self.arg_name = arg_name
        self.total_dofs = total_dofs
        self.starts_and_ends = starts_and_ends
        self.extra_args = extra_args

    def matvec(self, x):
        do_split = len(self.starts_and_ends) > 1
        from pytools.obj_array import make_obj_array

        if do_split:
            x = make_obj_array(
                    [x[start:end] for start, end in self.starts_and_ends])

        args = self.extra_args.copy()
        args[self.arg_name] = x
        result = self.executor(**args)

        if do_split:
            # re-join what was split
            joined_result = np.empty(self.total_dofs, np.complex128)
            for res_i, (start, end) in zip(result, self.starts_and_ends):
                joined_result[start:end] = res_i
            return joined_result
        else:
            return result




class Executor:
    def __init__(self, optemplate, discretizations):
        self.optemplate = optemplate
        self.discretizations = discretizations

        self.caches = {}

        from pytential.symbolic.compiler import OperatorCompiler
        self.code = OperatorCompiler(self.discretizations)(optemplate)

    def get_cache(self, name):
        return self.caches.setdefault(name, {})

    def scipy_op(self, arg_name, domains=None, **extra_args):
        """
        :arg domains: a list of geometry names or None values indicating the domains
          on which each component of the solution vector lives. *None* values indicate
          that the component is a scalar.
        """

        from pytools.obj_array import is_obj_array

        if domains is None:
            from pytools import single_valued
            dom_name = single_valued(self.discretizations.iterkeys())
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
                size = len(self.discretizations[dom_name])

            starts_and_ends.append((total_dofs, total_dofs+size))
            total_dofs += size

        # Hidden assumption: Number of input components
        # equals number of output compoments. But IMO that's
        # fair, since these operators are usually only used
        # for linear system solving, in which case the assumption
        # has to be true.
        matvec = MatVecOp(self, arg_name, total_dofs, starts_and_ends, extra_args)

        return sla.LinearOperator((total_dofs, total_dofs),
                matvec=matvec.matvec, dtype=np.complex128)

    def __call__(self, **args):
        exec_mapper = ExecutionMapper(self, args)
        return self.code.execute_dynamic(exec_mapper)









def bind(discretizations, op, auto_where=None):
    """
    :arg discretizations: a mapping of symbolic names to 
      :class:`pytential.discretization.Discretization` objects
      or a subclass of :class:`pytential.discretization.target.TargetBase`.
    :arg auto_where: For simple source-to-self or source-to-target
      evaluations, find 'where' attributes automatically.
    """

    from pytential.symbolic.primitives import DEFAULT_SOURCE, DEFAULT_TARGET
    from pytential.discretization import Discretization
    if not isinstance(discretizations, (dict, tuple)):
        discretizations = { DEFAULT_SOURCE: discretizations }

    if isinstance(discretizations, tuple):
        discretizations = {
                DEFAULT_SOURCE: discretizations[0],
                DEFAULT_TARGET: discretizations[1],
                }

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
            DiscretizationPluggerInner,
            AutoUpsampler
            )

    if auto_where is None:
        if DEFAULT_TARGET in discretizations:
            auto_where = DEFAULT_SOURCE, DEFAULT_TARGET
        else:
            auto_where = DEFAULT_SOURCE, DEFAULT_SOURCE

    if auto_where:
        op = ToTargetTagger(*auto_where)(op)

    op = DiscretizationPluggerInner(discretizations)(op)
    op = AutoUpsampler()(op)

    return Executor(op, discretizations)

# vim: foldmethod=marker
