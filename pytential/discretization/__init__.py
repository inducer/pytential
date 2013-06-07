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


from pytools import Record
from pytential.symbolic.compiler import Instruction


# {{{ layer pot instruction

class LayerPotentialOutput(Record):
    """
    .. attribute:: name

        the name of the variable to which the result is assigned

    .. attribute:: kernel_index

    .. attribute:: target_name

    .. attribute:: qbx_forced_limit
    """


class LayerPotentialInstruction(Instruction):
    """
    .. attribute:: outputs

        A list of :class:`LayerPotentialOutput` instances
        The entries in the list correspond to :attr:`names`.

    .. attribute:: kernels

        a list of :class:`sumpy.kernel.Kernel` instances, indexed by
        :attr:`LayerPotentialOutput.kernel_index`.

    .. attribute:: base_kernel

        The common base kernel among :attr:`kernels`, with all the
        layer potentials removed.

    .. attribute:: density
    .. attribute:: source

    .. attribute:: priority
    """

    def get_assignees(self):
        return set(o.name for o in self.outputs)

    def get_dependencies(self):
        dep_mapper = self.dep_mapper_factory()

        result = dep_mapper(self.density)

        from pytential.symbolic.mappers import (
                ExpressionKernelCombineMapper, KernelEvalArgumentCollector)
        ekdm = ExpressionKernelCombineMapper(dep_mapper)
        keac = KernelEvalArgumentCollector()

        from pymbolic import var
        for kernel in self.kernels:
            result.update(var(arg.name) for arg in kernel.get_args())
            result.update(ekdm(kernel))

            for karg in keac(kernel):
                if var(karg) in result:
                    result.remove(var(karg))

        return result

    def __str__(self):
        args = ["density=%s" % self.density,
                "source=%s" % self.source]

        from pytential.symbolic.mappers import StringifyMapper, stringify_where
        strify = StringifyMapper()

        lines = []
        for o in self.outputs:
            if o.target_name != self.source:
                tgt_str = " @ %s" % stringify_where(o.target_name)
            else:
                tgt_str = ""

            if o.qbx_forced_limit == 1:
                limit_str = "[+] "
            elif o.qbx_forced_limit == -1:
                limit_str = "[-] "
            elif o.qbx_forced_limit == 0:
                limit_str = "[0] "
            elif o.qbx_forced_limit is None:
                limit_str = ""
            else:
                raise ValueError("unrecognized limit value: %s" % o.qbx_forced_limit)

            line = "%s%s <- %s%s" % (o.name, tgt_str, limit_str,
                    self.kernels[o.kernel_index])

            lines.append(line)

        from pytential.symbolic.mappers import KernelEvalArgumentCollector
        keac = KernelEvalArgumentCollector()

        arg_names_to_exprs = {}
        for kernel in self.kernels:
            arg_names_to_exprs.update(keac(kernel))

        for arg_name, arg_expr in arg_names_to_exprs.iteritems():
            arg_expr_lines = strify(arg_expr).split("\n")
            lines.append("  %s = %s" % (
                arg_name, arg_expr_lines[0]))
            lines.extend("  " + s for s in arg_expr_lines[1:])

        return "{ /* Pot(%s) */\n  %s\n}" % (
                ", ".join(args), "\n  ".join(lines))

    def get_exec_function(self, exec_mapper):
        source = exec_mapper.bound_expr.discretizations[self.source]
        return source.exec_layer_potential_insn

# }}}


class Discretization(object):
    """Abstract interface for discretizations.

    .. attribute:: mesh

    .. attribute:: dim

    .. attribute:: ambient_dim

    .. method:: nodes()

        shape: ``(ambient_dim, nnodes)``

    .. method:: num_reference_derivative(queue, ref_axes, vec)

    .. method:: quad_weights(queue)

        shape: ``(nnodes)``

    .. rubric:: Layer potential source discretizations only

    .. method:: preprocess_optemplate(name, expr)

    .. method:: op_group_features(expr)

        Return a characteristic tuple by which operators that can be
        executed together can be grouped.

        *expr* is a subclass of
        :class:`pymbolic.primitives.LayerPotentialOperatorBase`.

    .. method:: exec_layer_pot_insn(insn, bound_expr, evaluate)
    """

# vim: fdm=marker
