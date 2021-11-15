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

from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Dict, Hashable, List, Optional, Set

import numpy as np

from pymbolic.primitives import cse_scope, Expression, Variable
from pytools import memoize_method
from sumpy.kernel import Kernel

from pytential.symbolic.primitives import DOFDescriptor, IntG
from pytential.symbolic.mappers import IdentityMapper, DependencyMapper


# {{{ instructions

@dataclass(frozen=True, eq=False)
class Instruction:
    """
    .. attribute:: names
    .. attribute:: exprs
    .. attribute:: priority
    .. attribute:: dep_mapper_factory
    """
    names: List[str]
    exprs: List[Expression]
    dep_mapper_factory: Callable[[], DependencyMapper]
    priority: int

    def get_assignees(self) -> Set[str]:
        raise NotImplementedError(
                f"get_assignees for '{self.__class__.__name__}'")

    def get_dependencies(self) -> Set[Expression]:
        raise NotImplementedError(
                f"get_dependencies for '{self.__class__.__name__}'")

    def __str__(self):
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Assign(Instruction):
    """
    .. attribute:: do_not_return

        A list of bools indicating whether the corresponding entry in
        :attr:`Instructio.names` and :attr:`Instruction.exprs` describes an
        expression that is not needed beyond this assignment.
    """

    do_not_return: Optional[List[bool]] = None
    comment: str = ""

    def __post_init__(self):
        if self.do_not_return is None:
            object.__setattr__(self, "do_not_return", [False] * len(self.names))

    def get_assignees(self):
        return set(self.names)

    @memoize_method
    def get_dependencies(self):
        # arg is include_subscripts
        dep_mapper = self.dep_mapper_factory()

        from operator import or_
        deps = reduce(or_, (dep_mapper(expr) for expr in self.exprs))

        from pymbolic.primitives import Variable
        deps -= {Variable(name) for name in self.names}

        return deps

    def __str__(self):
        comment = self.comment

        if len(self.names) == 1:
            if comment:
                comment = f"/* {comment} */ "

            return "{} <- {}{}".format(self.names[0], comment, self.exprs[0])
        else:
            if comment:
                comment = f" /* {comment} */"

            lines = []
            lines.append("{" + comment)
            for n, e, dnr in zip(self.names, self.exprs, self.do_not_return):
                if dnr:
                    dnr_indicator = "-#"
                else:
                    dnr_indicator = ""

                lines.append(f"  {n} <{dnr_indicator}- {e}")
            lines.append("}")

            return "\n".join(lines)

    def __hash__(self):
        return id(self)

# }}}


# {{{ layer pot instruction

@dataclass(frozen=True)
class PotentialOutput:
    """
    .. attribute:: name

        the name of the variable to which the result is assigned

    .. attribute:: target_kernel_index

    .. attribute:: target_name

    .. attribute:: qbx_forced_limit

        ``+1`` if the output is required to originate from a QBX center on the
        "+" side of the boundary. ``-1`` for the other side. ``0`` if either
        side of center (or no center at all) is acceptable.
    """

    name: str
    target_kernel_index: int
    target_name: DOFDescriptor
    qbx_forced_limit: int


@dataclass(frozen=True, eq=False)
class ComputePotentialInstruction(Instruction):
    """
    .. attribute:: outputs

        A list of :class:`PotentialOutput` instances
        The entries in the list correspond to :attr:`Instruction.names`.

    .. attribute:: target_kernels

        A list of :class:`sumpy.kernel.Kernel` instances, indexed by
        :attr:`PotentialOutput.target_kernel_index`.

    .. attribute:: kernel_arguments

        A dictionary mapping arg names to kernel arguments

    .. attribute:: source_kernels

        A list of :class:`sumpy.kernel.Kernel` instances with only source
        derivatives and no target derivatives. See
        :class:`pytential.symbolic.primitives.IntG` docstring for details.

    .. attribute:: densities

        A list of densities with the same number of entries as
        :attr:`source_kernels`. See the :class:`pytential.symbolic.primitives.IntG`
        docstring for details.

    .. attribute:: source
    """

    outputs: List[PotentialOutput]
    target_kernels: List[Kernel]
    kernel_arguments: Dict[str, Any]
    source_kernels: List[Kernel]
    densities: List[Expression]
    source: DOFDescriptor

    def get_assignees(self):
        return {o.name for o in self.outputs}

    def get_dependencies(self):
        dep_mapper = self.dep_mapper_factory()

        result = dep_mapper(self.densities[0])
        for density in self.densities[1:]:
            result.update(dep_mapper(density))

        for arg_expr in self.kernel_arguments.values():
            result.update(dep_mapper(arg_expr))

        return result

    def __str__(self):
        args = [f"source={self.source}"]
        for i, density in enumerate(self.densities):
            args.append(f"density{i}={density}")

        from pytential.symbolic.mappers import StringifyMapper, stringify_where
        strify = StringifyMapper()

        lines = []
        for o in self.outputs:
            if o.target_name != self.source:
                tgt_str = " @ {}".format(stringify_where(o.target_name))
            else:
                tgt_str = ""

            if o.qbx_forced_limit == 1:
                limit_str = "[+] "
            elif o.qbx_forced_limit == -1:
                limit_str = "[-] "
            elif o.qbx_forced_limit == 2:
                limit_str = "[(+)] "
            elif o.qbx_forced_limit == -2:
                limit_str = "[(-)] "
            elif o.qbx_forced_limit == "avg":
                limit_str = "[avg] "
            elif o.qbx_forced_limit is None:
                limit_str = ""
            else:
                raise ValueError(f"unrecognized limit value: {o.qbx_forced_limit}")

            source_kernels_str = " + ".join([
                f"density{i} * {source_kernel}" for i, source_kernel in
                enumerate(self.source_kernels)
            ])
            target_kernel = self.target_kernels[o.target_kernel_index]
            target_kernel_str = str(target_kernel)
            base_kernel_str = str(target_kernel.get_base_kernel())
            kernel_str = target_kernel_str.replace(base_kernel_str,
                f"({source_kernels_str})")

            line = "{}{} <- {}{}".format(
                    o.name, tgt_str, limit_str, kernel_str)

            lines.append(line)

        for arg_name, arg_expr in self.kernel_arguments.items():
            arg_expr_lines = strify(arg_expr).split("\n")
            lines.append("  {} = {}".format(arg_name, arg_expr_lines[0]))
            lines.extend("  " + s for s in arg_expr_lines[1:])

        return "{{ /* Pot({}) */\n  {}\n}}".format(
                ", ".join(args), "\n  ".join(lines))

    def __hash__(self):
        return id(self)

# }}}


# {{{ graphviz/dot dataflow graph drawing

def dot_dataflow_graph(code: "Code",
        max_node_label_length: int = 30,
        label_wrap_width: int = 50) -> str:
    origins = {}
    node_names = {}

    result = [
            'initial [label="initial"]'
            'result [label="result"]']

    for num, insn in enumerate(code.instructions):
        node_name = f"node{num}"
        node_names[insn] = node_name
        node_label = str(insn)

        if max_node_label_length is not None:
            node_label = node_label[:max_node_label_length]

        if label_wrap_width is not None:
            from pytools import word_wrap
            node_label = word_wrap(node_label, label_wrap_width,
                    wrap_using="\n      ")

        node_label = node_label.replace("\n", "\\l") + "\\l"

        result.append(f"{node_name} [ "
                f'label="p{insn.priority}: {node_label}" shape=box ];')

        for assignee in insn.get_assignees():
            origins[assignee] = node_name

    def get_orig_node(expr):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return origins.get(expr.name, "initial")
        else:
            return "initial"

    def gen_expr_arrow(expr, target_node):
        orig_node = get_orig_node(expr)
        result.append(f'{orig_node} -> {target_node} [label="{expr}"];')

    for insn in code.instructions:
        for dep in insn.get_dependencies():
            gen_expr_arrow(dep, node_names[insn])

    code_res = code.result

    if isinstance(code_res, np.ndarray) and code_res.dtype.char == "O":
        for subexp in code_res:
            gen_expr_arrow(subexp, "result")
    else:
        gen_expr_arrow(code_res, "result")

    return "digraph dataflow {\n%s\n}\n" % "\n".join(result)

# }}}


# {{{ code representation

class Code:
    def __init__(self, instructions: List[Instruction], result: Variable) -> None:
        self.instructions = instructions
        self.result = result
        self.last_schedule = None

    def dump_dataflow_graph(self):
        from pytools.debug import open_unique_debug_file

        open_unique_debug_file("dataflow", ".dot")[0]\
                .write(dot_dataflow_graph(self, max_node_label_length=None))

    def __str__(self):
        lines = []
        for insn in self.instructions:
            lines.extend(str(insn).split("\n"))
        lines.append("RESULT: " + str(self.result))

        return "\n".join(lines)

    # {{{ dynamic scheduler

    class NoInstructionAvailable(Exception):
        pass

    @memoize_method
    def get_next_step(self, available_names, done_insns):
        from pytools import argmax2
        available_insns = [
                (insn, insn.priority) for insn in self.instructions
                if insn not in done_insns
                and all(dep.name in available_names
                    for dep in insn.get_dependencies())]

        if not available_insns:
            raise self.NoInstructionAvailable

        needed_vars = {
            dep.name
            for insn in self.instructions
            if insn not in done_insns
            for dep in insn.get_dependencies()
            }
        discardable_vars = set(available_names) - needed_vars

        # {{{ make sure results do not get discarded
        from pytools.obj_array import obj_array_vectorize

        from pytential.symbolic.mappers import DependencyMapper
        dm = DependencyMapper(composite_leaves=False)

        def remove_result_variable(result_expr):
            # The extra dependency mapper run is necessary
            # because, for instance, subscripts can make it
            # into the result expression, which then does
            # not consist of just variables.

            for var in dm(result_expr):
                assert isinstance(var, Variable)
                discardable_vars.discard(var.name)

        obj_array_vectorize(remove_result_variable, self.result)
        # }}}

        return argmax2(available_insns), discardable_vars

    @staticmethod
    def get_exec_function(insn, exec_mapper):
        if isinstance(insn, Assign):
            return exec_mapper.exec_assign
        if isinstance(insn, ComputePotentialInstruction):
            return exec_mapper.exec_compute_potential_insn
        raise ValueError(f"unknown instruction class: {type(insn)}")

    def execute(self, exec_mapper, pre_assign_check=None):
        """Execute the instruction stream, make all scheduling decisions
        dynamically.
        """

        context = exec_mapper.context

        done_insns = set()

        while True:
            discardable_vars = []
            insn = None

            try:
                insn, discardable_vars = self.get_next_step(
                        frozenset(context.keys()),
                        frozenset(done_insns))

            except self.NoInstructionAvailable:
                # no available instructions: we're done
                break
            else:
                for name in discardable_vars:
                    del context[name]

                done_insns.add(insn)
                assignments = (
                        self.get_exec_function(insn, exec_mapper)(
                            exec_mapper.array_context,
                            insn, exec_mapper.bound_expr, exec_mapper))

                assignees = insn.get_assignees()
                for target, value in assignments:
                    if pre_assign_check is not None:
                        pre_assign_check(target, value)

                    assert target in assignees
                    context[target] = value

        if len(done_insns) < len(self.instructions):
            print("Unreachable instructions:")
            for insn in set(self.instructions) - done_insns:
                print("    ", str(insn).replace("\n", "\n     "))
                from pymbolic import var
                print("     missing: ", ", ".join(
                        str(s) for s in
                        set(insn.get_dependencies())
                        - {var(v) for v in context.keys()}))

            raise RuntimeError("not all instructions are reachable"
                    "--did you forget to pass a value for a placeholder?")

        from pytools.obj_array import obj_array_vectorize
        return obj_array_vectorize(exec_mapper, self.result)

    # }}}

# }}}


# {{{ compiler

class OperatorCompiler(IdentityMapper):
    def __init__(self,
            places,
            prefix: str = "_expr",
            max_vectors_in_batch_expr: Optional[int] = None) -> None:
        super().__init__()

        self.places = places
        self.prefix = prefix
        self.max_vectors_in_batch_expr = max_vectors_in_batch_expr

        self.code: List[Instruction] = []
        self.expr_to_var: Dict[Expression, Variable] = {}
        self.assigned_names: Set[str] = set()
        self.group_to_operators: Dict[Hashable, Set[IntG]] = {}

    def op_group_features(self, expr) -> Hashable:
        from pytential.symbolic.primitives import hashable_kernel_args
        lpot_source = self.places.get_geometry(expr.source.geometry)
        return (
                lpot_source.op_group_features(expr)
                + hashable_kernel_args(expr.kernel_arguments))

    @memoize_method
    def dep_mapper_factory(self, include_subscripts: bool = False):
        return DependencyMapper(
                # include_operator_bindings=False,
                include_lookups=False,
                include_subscripts=include_subscripts,
                include_calls="descend_args")

    @property
    def dep_mapper(self):
        return self.dep_mapper_factory()

    # {{{ top-level driver

    def __call__(self, expr):
        # {{{ collect operators by operand

        from pytential.symbolic.mappers import OperatorCollector
        operators = [
                op
                for op in OperatorCollector()(expr)
                if isinstance(op, IntG)]

        self.group_to_operators = {}
        for op in operators:
            features = self.op_group_features(op)
            self.group_to_operators.setdefault(features, set()).add(op)

        # }}}

        # Traverse the expression, generate code.

        result = super().__call__(expr)

        # Put the toplevel expressions into variables as well.

        from pytools.obj_array import obj_array_vectorize
        result = obj_array_vectorize(self.assign_to_new_var, result)

        return Code(self.code, result)

    # }}}

    # {{{ variables and names

    def get_var_name(self, prefix: Optional[str] = None) -> str:
        def generate_suffixes() -> str:
            yield ""
            i = 2
            while True:
                yield f"_{i}"
                i += 1

        def generate_plain_names() -> str:
            i = 0
            while True:
                yield f"{self.prefix}{i}"
                i += 1

        if prefix is None:
            for name in generate_plain_names():
                if name not in self.assigned_names:
                    break
        else:
            for suffix in generate_suffixes():
                name = f"{prefix}{suffix}"
                if name not in self.assigned_names:
                    break

        self.assigned_names.add(name)
        return name

    def make_assign(
            self, name: str, expr: Expression, priority: int,
            ) -> Assign:
        return Assign(
                names=[name], exprs=[expr],
                dep_mapper_factory=self.dep_mapper_factory,
                priority=priority)

    def assign_to_new_var(
            self, expr: Expression, priority: int = 0, prefix: Optional[str] = None,
            ) -> Variable:
        from pymbolic.primitives import Subscript

        # Observe that the only things that can be legally subscripted
        # are variables. All other expressions are broken down into
        # their scalar components.
        if isinstance(expr, (Variable, Subscript)):
            return expr

        new_name = self.get_var_name(prefix)
        self.code.append(self.make_assign(new_name, expr, priority))

        return Variable(new_name)

    # }}}

    # {{{ map_xxx routines

    def map_common_subexpression(self, expr):
        # NOTE: EXPRESSION and DISCRETIZATION scopes are handled in
        # execution.py::EvaluationMapperBase so that they can be cached
        # with a longer lifetime
        if expr.scope != cse_scope.EVALUATION:
            return expr

        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            priority = getattr(expr, "priority", 0)

            from pytential.symbolic.primitives import IntG
            if isinstance(expr.child, IntG):
                # We need to catch operators here and
                # treat them specially. They get assigned to their
                # own variable by default, which would mean the
                # CSE prefix would be omitted.

                rec_child = self.rec(expr.child, name_hint=expr.prefix)
            else:
                rec_child = self.rec(expr.child)

            cse_var = self.assign_to_new_var(rec_child,
                    priority=priority, prefix=expr.prefix)

            self.expr_to_var[expr.child] = cse_var
            return cse_var

    def map_int_g(self, expr, name_hint=None):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            from pytential.utils import sort_arrays_together
            source_kernels, densities = \
                sort_arrays_together(expr.source_kernels, expr.densities, key=str)
            # make sure operator assignments stand alone and don't get muddled
            # up in vector arithmetic
            density_vars = [self.assign_to_new_var(self.rec(density)) for
                density in densities]

            group = self.group_to_operators[self.op_group_features(expr)]
            names = [self.get_var_name() for op in group]

            sorted_ops = sorted(group, key=lambda op: repr(op.target_kernel))
            target_kernels = [op.target_kernel for op in sorted_ops]

            target_kernel_to_index = \
                {kernel: i for i, kernel in enumerate(target_kernels)}

            for op in group:
                assert op.qbx_forced_limit in [-2, -1, None, 1, 2]

            kernel_arguments = {
                    arg_name: self.rec(arg_val)
                    for arg_name, arg_val in expr.kernel_arguments.items()}

            outputs = [
                PotentialOutput(
                    name=name,
                    target_kernel_index=target_kernel_to_index[op.target_kernel],
                    target_name=op.target,
                    qbx_forced_limit=op.qbx_forced_limit,
                    )
                for name, op in zip(names, group)
                ]

            self.code.append(
                    ComputePotentialInstruction(
                        # NOTE: these are set to None because they are deduced
                        # from `outputs` in `get_assignees` and `get_dependencies`
                        names=None,
                        exprs=None,
                        outputs=outputs,
                        target_kernels=tuple(target_kernels),
                        kernel_arguments=kernel_arguments,
                        source_kernels=source_kernels,
                        densities=density_vars,
                        source=expr.source,
                        priority=max(getattr(op, "priority", 0) for op in group),
                        dep_mapper_factory=self.dep_mapper_factory))

            from pymbolic.primitives import Variable
            for name, group_expr in zip(names, group):
                self.expr_to_var[group_expr] = Variable(name)

            return self.expr_to_var[expr]

    def map_int_g_ds(self, op):
        raise AssertionError()

    # }}}

# }}}

# vim: foldmethod=marker
