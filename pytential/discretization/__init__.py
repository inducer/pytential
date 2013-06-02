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


from pytential.symbolic.compiler import Instruction


# {{{ layer pot instruction

class LayerPotentialInstruction(Instruction):
    """
    .. attribute:: names

        the names of variables to assign to

    .. attribute:: kernels_and_targets

        list of tuples ``(kernel, target)``, where
        kernel is a :class:`sumpy.kernel.Kernel` instance
        and target is a symbolic name.

        The entries in the list correspond to :attr:`names`.

    .. attribute:: density
    .. attribute:: source
    .. attribute:: dsource

        *None*, or an expression containing
        :class:`pytential.symbolic.primitives.NablaComponent`
        as placeholders for the source derivative components.

    .. attriubte:: priority
    """

    def get_assignees(self):
        return set(self.names)

    def get_dependencies(self, each_vector=False):
        return self.dep_mapper_factory(each_vector)(self.density)

    def __str__(self):
        args = ["kernel=%s" % self.kernel, "density=%s" % self.density,
                "source=%s" % self.source]
        if self.ds_direction is not None:
            if self.ds_direction == "n":
                args.append("ds=normal")
            else:
                args.append("ds=%s" % self.ds_direction)

        lines = []
        for name, (tgt, what, idx) in zip(self.names, self.return_values):
            if idx != ():
                line = "%s <- %s[%s]" % (name, what, idx)
            else:
                line = "%s <- %s" % (name, what)
            if tgt != self.source:
                line += " @ %s" % tgt
            lines.append(line)

        return "{ /* Quad(%s) */\n  %s\n}" % (
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

    .. attribute:: nnodes

    .. method:: nodes(queue)

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

    .. method:: gen_instruction_for_layer_pot_from_src( \
            compiler, tgt_discr, expr, field_var)

    .. method:: exec_layer_pot_insn(insn, bound_expr, evaluate)
    """

# vim: fdm=marker
