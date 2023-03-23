__copyright__ = "Copyright (C) 2020 Isuru Fernando"

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

import numpy as np

from pymbolic.mapper.coefficient import CoefficientCollector
from pymbolic.geometric_algebra.mapper import WalkMapper
from pymbolic.mapper import CombineMapper

from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative,
    KernelWrapper, TargetPointMultiplier,
    DirectionalSourceDerivative, DirectionalDerivative)

from pytential.symbolic.primitives import (
    hashable_kernel_args, hashable_kernel_arg_value)
from pytential.symbolic.mappers import IdentityMapper
from .reduce import reduce_number_of_fmms

from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

__all__ = (
    "merge_int_g_exprs",
)

__doc__ = """
.. autofunction:: merge_int_g_exprs
"""


# {{{ merge_int_g_exprs

def merge_int_g_exprs(exprs, source_dependent_variables=None):
    """
    Merge expressions involving :class:`~pytential.symbolic.primitives.IntG`
    objects.
    Several techniques are used for merging and reducing number of FMMs
       * :class:`sumpy.kernel.AxisTargetDerivative` instances are converted
         to :class:`sumpy.kernel.AxisSourceDerivative` instances.
         (by flipping signs, assuming translation-invariance).
         Target derivatives will be brought back by the syzygy module
         construction below if beneficial.
         (For example, `D + d/dx(S)` can be re-written as `D - d/dy(S)` which can be
         done in one FMM)
       * If there is a sum of two *IntG* s with same target derivative and different
         source derivatives of the same kernel, they are merged into one FMM.
       * Reduce the number of FMMs by converting the *IntG* expression to
         a matrix and factoring the matrix where the left operand matrix represents
         a transformation at target and the right matrix represents a transformation
         at source. For this to work, we need to know which variables depend on
         source so that they do not end up in the left operand. User needs to supply
         this as the argument *source_dependent_variable*. This is done by the
         call to :func:`pytential.symbolic.pde.systems.reduce_number_of_fmms`.
    :arg base_kernel: A :class:`sumpy.kernel.Kernel` object if given will be used
        for converting a :class:`~pytential.symbolic.primitives.IntG` to a linear
        expression of same type with the kernel replaced by base_kernel and its
        derivatives
    :arg source_dependent_variable: When merging expressions, consider only these
        variables as dependent on source. This is important when reducing the
        number of FMMs needed for the output.
    """
    # Using a dictionary instead of a set because sets are unordered
    all_source_group_identifiers = {}

    result = np.array([0 for _ in exprs], dtype=object)

    int_g_cc = IntGCoefficientCollector()
    int_gs_by_source_group = defaultdict(list)

    def add_int_gs_in_expr(expr):
        for int_g in get_int_g_s([expr]):
            source_group_identifier = get_int_g_source_group_identifier(int_g)
            int_gs_by_source_group[source_group_identifier].append(int_g)
            for density in int_g.densities:
                add_int_gs_in_expr(density)

    for i, expr in enumerate(exprs):
        int_gs_by_group = {}
        try:
            int_g_coeff_map = int_g_cc(expr)
        except (RuntimeError, AssertionError):
            # Don't touch this expression, because it's not linear.
            # FIXME: if there's ever any use case, then we can extract
            # some IntGs from them.
            logger.debug("%s is not linear", expr)
            result[i] += expr
            add_int_gs_in_expr(expr)
            continue
        for int_g, coeff in int_g_coeff_map.items():
            if int_g == 1:
                # coeff map may have some constant terms, add them to
                result[i] += coeff
                continue

            # convert DirectionalSourceDerivative to AxisSourceDerivative
            # as kernel arguments need to be the same for merging
            int_g = convert_directional_source_to_axis_source(int_g)
            # convert TargetDerivative to source before checking the group
            # as the target kernel has to be the same for merging
            int_g = convert_target_deriv_to_source(int_g)
            if not is_expr_target_dependent(coeff):
                # move the coefficient inside
                int_g = int_g.copy(densities=[density*coeff for density in
                    int_g.densities])
                coeff = 1

            source_group_identifier = get_int_g_source_group_identifier(int_g)
            target_group_identifier = get_int_g_target_group_identifier(int_g)
            group = (source_group_identifier, target_group_identifier, coeff)

            all_source_group_identifiers[source_group_identifier] = 1

            if group not in int_gs_by_group:
                new_int_g = int_g
            else:
                prev_int_g = int_gs_by_group[group]
                # Let's merge IntGs with the same group
                new_int_g = merge_two_int_gs(int_g, prev_int_g)
            int_gs_by_group[group] = new_int_g

        # Do some simplifications after merging. Not stricty necessary
        for (_, _, coeff), int_g in int_gs_by_group.items():
            # replace an IntG with d axis source derivatives to an IntG
            # with one directional source derivative
            # TODO: reenable this later
            # result_int_g = convert_axis_source_to_directional_source(int_g)
            # simplify the densities as they may become large due to pymbolic
            # not doing automatic simplifications unlike sympy/symengine
            result_int_g = int_g.copy(
                    densities=simplify_densities(int_g.densities))
            result[i] += result_int_g * coeff
            add_int_gs_in_expr(result_int_g)

    # No IntGs found
    if all(not int_gs for int_gs in int_gs_by_source_group):
        return exprs

    # Do the calculation for each source_group_identifier separately
    # and assemble them
    replacements = {}
    for int_gs in int_gs_by_source_group.values():
        # For each output, we now have a sum of int_gs with
        # different target attributes.
        # for eg: {+}S + {-}D (where {x} is the QBX limit).
        # We can't merge them together, because merging implies
        # that everything happens at the source level and therefore
        # require same target attributes.
        #
        # To handle this case, we can treat them separately as in
        # different source base kernels, but that would imply more
        # FMMs than necessary.
        #
        # Take the following example,
        #
        # [ {+}(S + D), {-}S + {avg}D, {avg}S + {-}D]
        #
        # If we treated the target attributes separately, then we
        # will be reducing [{+}(S + D), 0, 0], [0, {-}S, {-}D],
        # [0, {avg}D, {avg}S] separately which results in
        # [{+}(S + D)], [{-}S, {-}D], [{avg}S, {avg}D] as
        # the reduced FMMs and pytential will calculate
        # [S + D, S, D] as three separate FMMs and then assemble
        # the three outputs by applying target attributes.
        #
        # Instead, we can do S, D as two separate FMMs and get the
        # result for all three outputs. To do that, we will first
        # get all five expressions in the example
        # [ {+}(S + D), {-}S, {avg}D, {avg}S, {-}D]
        # and then remove the target attributes to get,
        # [S + D, S, D]. We will reduce these and restore the target
        # attributes at the end

        targetless_int_g_mapping = defaultdict(list)
        for int_g in int_gs:
            common_int_g = remove_target_attributes(int_g)
            targetless_int_g_mapping[common_int_g].append(int_g)

        insns_to_reduce = list(targetless_int_g_mapping.keys())
        reduced_insns = reduce_number_of_fmms(insns_to_reduce,
                source_dependent_variables)

        for insn, reduced_insn in zip(insns_to_reduce, reduced_insns):
            for int_g in targetless_int_g_mapping[insn]:
                replacements[int_g] = restore_target_attributes(reduced_insn, int_g)

    mapper = IntGSubstitutor(replacements)
    result = [mapper(expr) for expr in result]

    orig_count = get_number_of_fmms(exprs)
    new_count = get_number_of_fmms(result)
    if orig_count < new_count:
        raise RuntimeError("merge_int_g_exprs failed. "
                           "Please open an issue in pytential bug tracker.")

    return result


class IntGCoefficientCollector(CoefficientCollector):
    def __init__(self):
        super().__init__({})

    def map_int_g(self, expr):
        return {expr: 1}

    def map_algebraic_leaf(self, expr, *args, **kwargs):
        return {1: expr}

    handle_unsupported_expression = map_algebraic_leaf


def get_hashable_kernel_argument(arg):
    if hasattr(arg, "__iter__"):
        try:
            return tuple(arg)
        except TypeError:
            pass
    return arg


def get_normal_vector_names(kernel):
    """Return the normal vector names in a kernel
    """
    normal_vectors = set()
    while isinstance(kernel, KernelWrapper):
        if isinstance(kernel, DirectionalDerivative):
            normal_vectors.add(kernel.dir_vec_name)
        kernel = kernel.inner_kernel
    return normal_vectors


def get_int_g_source_group_identifier(int_g):
    """Return a identifier for a group for the *int_g* so that all elements in that
    group have the same source attributes.
    """
    target_arg_names = get_normal_vector_names(int_g.target_kernel)
    args = {k: v for k, v in sorted(
        int_g.kernel_arguments.items()) if k not in target_arg_names}
    return (int_g.source, hashable_kernel_args(args),
            int_g.target_kernel.get_base_kernel())


def get_int_g_target_group_identifier(int_g):
    """Return a identifier for a group for the *int_g* so that all elements in that
    group have the same target attributes.
    """
    target_arg_names = get_normal_vector_names(int_g.target_kernel)
    args = {k: v for k, v in sorted(
        int_g.kernel_arguments.items()) if k in target_arg_names}
    return (int_g.target, int_g.qbx_forced_limit, int_g.target_kernel,
            hashable_kernel_args(args))


def filter_kernel_arguments(knls, kernel_arguments):
    """From a dictionary of kernel arguments, filter out arguments
    that are not needed for the kernels given as a list and return a new
    dictionary.
    """
    kernel_arg_names = set()

    for kernel in knls:
        for karg in (kernel.get_args() + kernel.get_source_args()):
            kernel_arg_names.add(karg.loopy_arg.name)

    return {k: v for (k, v) in kernel_arguments.items() if k in kernel_arg_names}


def convert_directional_source_to_axis_source(int_g):
    """Convert an IntG with a DirectionalSourceDerivative instance
    to an IntG with d AxisSourceDerivative instances.
    """
    source_kernels = []
    densities = []
    for source_kernel, density in zip(int_g.source_kernels, int_g.densities):
        knl_result = _convert_directional_source_knl_to_axis_source(source_kernel,
                        int_g.kernel_arguments)
        for knl, coeff in knl_result:
            source_kernels.append(knl)
            densities.append(coeff * density)

    kernel_arguments = filter_kernel_arguments(
        list(source_kernels) + [int_g.target_kernel], int_g.kernel_arguments)
    return int_g.copy(source_kernels=tuple(source_kernels),
            densities=tuple(densities), kernel_arguments=kernel_arguments)


def _convert_directional_source_knl_to_axis_source(knl, knl_arguments):
    if isinstance(knl, DirectionalSourceDerivative):
        dim = knl.dim
        dir_vec = knl_arguments[knl.dir_vec_name]

        res = []
        inner_result = _convert_directional_source_knl_to_axis_source(
                knl.inner_kernel, knl_arguments)
        for inner_knl, coeff in inner_result:
            for d in range(dim):
                res.append((AxisSourceDerivative(d, inner_knl), coeff*dir_vec[d]))
        return res
    elif isinstance(knl, KernelWrapper):
        inner_result = _convert_directional_source_knl_to_axis_source(
                knl.inner_kernel, knl_arguments)
        return [(knl.replace_inner_kernel(inner_knl), coeff) for
                    inner_knl, coeff in inner_result]
    else:
        return [(knl, 1)]


def convert_target_deriv_to_source(int_g):
    """Converts AxisTargetDerivatives to AxisSourceDerivative instances
    from an IntG. If there are outer TargetPointMultiplier transformations
    they are preserved.
    """
    knl = int_g.target_kernel
    source_kernels = list(int_g.source_kernels)
    coeff = 1
    multipliers = []
    while isinstance(knl, TargetPointMultiplier):
        multipliers.append(knl.axis)
        knl = knl.inner_kernel

    while isinstance(knl, AxisTargetDerivative):
        coeff *= -1
        source_kernels = [AxisSourceDerivative(knl.axis, source_knl) for
                source_knl in source_kernels]
        knl = knl.inner_kernel

    # TargetPointMultiplier has to be the outermost kernel
    # If it is the inner kernel, return early
    if isinstance(knl, TargetPointMultiplier):
        return int_g

    for axis in reversed(multipliers):
        knl = TargetPointMultiplier(axis, knl)

    new_densities = tuple(density*coeff for density in int_g.densities)
    return int_g.copy(target_kernel=knl,
                      densities=new_densities,
                      source_kernels=tuple(source_kernels))


class IsExprTargetDependent(CombineMapper):
    def combine(self, values):
        import operator
        from functools import reduce
        return reduce(operator.or_, values, False)

    def map_constant(self, expr):
        return False

    map_variable = map_constant
    map_wildcard = map_constant
    map_function_symbol = map_constant

    def map_common_subexpression(self, expr):
        return self.rec(expr.child)

    def map_coordinate_component(self, expr):
        return True

    def map_num_reference_derivative(self, expr):
        return True

    def map_q_weight(self, expr):
        return True


def is_expr_target_dependent(expr):
    mapper = IsExprTargetDependent()
    return mapper(expr)


def merge_kernel_arguments(x, y):
    """merge two kernel argument dictionaries and raise a ValueError if
    the two dictionaries do not agree for duplicate keys.
    """
    res = x.copy()
    for k, v in y.items():
        if k in res:
            if hashable_kernel_arg_value(res[k]) \
                    != hashable_kernel_arg_value(v):
                raise ValueError(f"Error merging values for {k}."
                    f"values were {res[k]} and {v}")
        else:
            res[k] = v
    return


def merge_two_int_gs(int_g_1, int_g_2):
    kernel_arguments = merge_kernel_arguments(int_g_1.kernel_arguments,
            int_g_2.kernel_arguments)
    source_kernels = int_g_1.source_kernels + int_g_2.source_kernels
    densities = int_g_1.densities + int_g_2.densities

    return int_g_1.copy(
        source_kernels=tuple(source_kernels),
        densities=tuple(densities),
        kernel_arguments=kernel_arguments,
    )


def simplify_densities(densities):
    """Simplify densities by converting to sympy and converting back
    to trigger sympy's automatic simplification routines.
    """
    from sumpy.symbolic import (SympyToPymbolicMapper, PymbolicToSympyMapper)
    from pymbolic.mapper import UnsupportedExpressionError
    to_sympy = PymbolicToSympyMapper()
    to_pymbolic = SympyToPymbolicMapper()
    result = []
    for density in densities:
        try:
            result.append(to_pymbolic(to_sympy(density)))
        except (ValueError, NotImplementedError, UnsupportedExpressionError):
            logger.debug("%s cannot be simplified", density)
            result.append(density)
    return tuple(result)


def remove_target_attributes(int_g):
    """Remove target attributes from *int_g* and return an expression
    that is common to all expression in the same source group.
    """
    normals = get_normal_vector_names(int_g.target_kernel)
    kernel_arguments = {k: v for k, v in int_g.kernel_arguments.items() if
                        k not in normals}
    return int_g.copy(target=None, qbx_forced_limit=None,
            target_kernel=int_g.target_kernel.get_base_kernel(),
            kernel_arguments=kernel_arguments)


class IntGSubstitutor(IdentityMapper):
    """Replaces IntGs with pymbolic expression given by the
    replacements dictionary
    """
    def __init__(self, replacements):
        self.replacements = replacements

    def map_int_g(self, expr):
        if expr in self.replacements:
            new_expr = self.replacements[expr]
            if new_expr != expr:
                return self.rec(new_expr)
            else:
                expr = new_expr

        densities = [self.rec(density) for density in expr.densities]
        return expr.copy(densities=tuple(densities))


class GetIntGs(WalkMapper):
    """A Mapper that walks expressions and collects
    :class:`~pytential.symbolic.primitives.IntG` objects
    """
    def __init__(self):
        self.int_g_s = set()

    def map_int_g(self, expr):
        self.int_g_s.add(expr)

    def map_constant(self, expr):
        pass

    map_variable = map_constant
    handle_unsupported_expression = map_constant


def get_int_g_s(exprs):
    """Returns all :class:`~pytential.symbolic.primitives.IntG` objects
    in a list of :mod:`pymbolic` expressions.
    """
    get_int_g_mapper = GetIntGs()
    [get_int_g_mapper(expr) for expr in exprs]
    return get_int_g_mapper.int_g_s


def restore_target_attributes(expr, orig_int_g):
    """Restore target attributes from *orig_int_g* to all the
    :class:`~pytential.symbolic.primitives.IntG` objects in the
    input *expr*.
    """
    int_gs = get_int_g_s([expr])

    replacements = {
        int_g: int_g.copy(target=orig_int_g.target,
                qbx_forced_limit=orig_int_g.qbx_forced_limit,
                target_kernel=orig_int_g.target_kernel.replace_base_kernel(
                    int_g.target_kernel),
                kernel_arguments=orig_int_g.kernel_arguments)
        for int_g in int_gs}

    substitutor = IntGSubstitutor(replacements)
    return substitutor(expr)


def get_number_of_fmms(exprs):
    fmms = set()
    for int_g in get_int_g_s(exprs):
        fmms.add(remove_target_attributes(int_g))
    return len(fmms)

# }}}
