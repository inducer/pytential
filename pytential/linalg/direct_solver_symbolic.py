__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

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

from pytential.symbolic.mappers import (
        IdentityMapper, OperatorCollector, LocationTagger)

__doc__ = """
.. autoclass:: KernelTransformationRemover
.. autoclass:: IntGTermCollector
.. autoclass:: DOFDescriptorReplacer
"""


# {{{ KernelTransformationRemover

class KernelTransformationRemover(IdentityMapper):
    r"""A mapper that removes the transformations from the kernel of all
    :class:`~pytential.symbolic.primitives.IntG`\ s in the expression.

    This includes source and target derivatives and other such transformations.
    Any unnecessary kernel arguments are also removed from
    :attr:`~pytential.symbolic.primitives.IntG.kernel_arguments`.

    This mapper is meant to be used in the directs solver for proxy interaction,
    where it is not possible to evaluate source or target directional derivatives.
    """

    def __init__(self):
        from sumpy.kernel import (
                TargetTransformationRemover,
                SourceTransformationRemover)
        self.sxr = SourceTransformationRemover()
        self.txr = TargetTransformationRemover()

    def map_int_g(self, expr):
        target_kernel = self.txr(expr.target_kernel)
        source_kernels = tuple([self.sxr(kernel) for kernel in expr.source_kernels])
        if (target_kernel == expr.target_kernel
                and source_kernels == expr.source_kernels):
            return expr

        # remove all args that come from the source transformations
        source_args = {
            arg.name for kernel in expr.source_kernels
            for arg in kernel.get_source_args()}
        kernel_arguments = {
            name: self.rec(arg) for name, arg in expr.kernel_arguments.items()
            if name not in source_args
        }

        return expr.copy(target_kernel=target_kernel,
                         source_kernels=source_kernels,
                         densities=self.rec(expr.densities),
                         kernel_arguments=kernel_arguments)

# }}}


# {{{ IntGTermCollector

class IntGTermCollector(IdentityMapper):
    r"""A mapper that removes all non-:class:`~pytential.symbolic.primitives.IntG`
    terms from the expression and all their non-constant factors.

    In particular, an expression of the type

    .. math::

        \sum_{i = 0}^N f_i(\mathbf{x}, \sigma)
        + \sum_{i = 0}^M c_i g_i(\mathbf{x}) \mathrm{IntG}_i(\mathbf{x})

    is reduced to

    .. math::

        \sum_{i = 0}^M c_i \mathrm{IntG}_i(\mathbf{x}).

    The intended used of this transformation is to allow the evaluation of
    the proxy interactions in the direct solver for a given expression
    meant for self-evaluation.
    """

    def map_sum(self, expr):
        collector = OperatorCollector()

        children = []
        for child in expr.children:
            rec_child = self.rec(child)
            if collector(rec_child):
                children.append(rec_child)

        from pymbolic.primitives import flattened_sum
        return flattened_sum(children)

    def map_product(self, expr):

        collector = OperatorCollector()

        from pymbolic.primitives import is_constant
        children_const = []
        children_int_g = []
        for child in expr.children:
            if is_constant(child):
                children_const.append(child)
            else:
                rec_child = self.rec(child)
                if collector(rec_child):
                    if children_int_g:
                        raise RuntimeError(
                                f"{type(self).__name__}.map_product does not "
                                "support products of IntGs")

                    children_int_g.append(rec_child)

        from pymbolic.primitives import flattened_product
        return flattened_product(children_const + children_int_g)

    def map_int_g(self, expr):
        return expr

# }}}


# {{{ DOFDescriptorReplacer

class _LocationReplacer(LocationTagger):
    """Unlike :class:`LocationTagger`, this mapper removes the heuristic for
    target and source tagging and forcefully replaces existing
    :class:`~pytential.symbolic.dof_desc.DOFDescriptor` in the expression.
    """

    def _default_dofdesc(self, dofdesc):
        return self.default_target

    def map_int_g(self, expr):
        return type(expr)(
                expr.target_kernel, expr.source_kernels,
                densities=self.operand_rec(expr.densities),
                qbx_forced_limit=expr.qbx_forced_limit,
                source=self.default_source, target=self.default_target,
                kernel_arguments={
                    name: self.operand_rec(arg_expr)
                    for name, arg_expr in expr.kernel_arguments.items()
                    }
                )


class DOFDescriptorReplacer(_LocationReplacer):
    r"""Mapper that replaces all the
    :class:`~pytential.symbolic.dof_desc.DOFDescriptor`\ s in the expression
    with the given ones.

    This mapper is meant to allow for evaluation of proxy interactions in
    the direct solver when the given expression is already partially
    (or fully) tagged.

    .. automethod:: __init__
    """

    def __init__(self, source, target):
        """
        :param source: a descriptor for all expressions to be evaluated on
            the source geometry.
        :param target: a descriptor for all expressions to be evaluate on
            the target geometry.
        """
        super().__init__(target, default_source=source)
        self.operand_rec = _LocationReplacer(source, default_source=source)

# }}}
