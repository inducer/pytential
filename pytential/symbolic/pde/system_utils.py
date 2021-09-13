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

from sumpy.symbolic import make_sym_vector, sym, SympyToPymbolicMapper
from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative,
    DirectionalSourceDerivative, ExpressionKernel,
    KernelWrapper, TargetPointMultiplier)
from pytools import (memoize_on_first_arg,
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)
from collections import defaultdict

from pymbolic.mapper import WalkMapper
from pymbolic.mapper.coefficient import CoefficientCollector
from pymbolic.primitives import Variable
from pytential.symbolic.primitives import IntG, NodeCoordinateComponent
from pytential.symbolic.mappers import IdentityMapper
from pytential.utils import chop, lu_solve_with_expand
import pytential

from pytential.symbolic.pde.reduce_fmms import reduce_number_of_fmms

__all__ = (
    "merge_int_g_exprs",
    "get_deriv_relation",
    )


def merge_int_g_exprs(exprs, base_kernel=None, verbose=False,
        source_dependent_variables=None):
    """
    Merge expressions involving :class:`~pytential.symbolic.primitives.IntG`
    objects.

    Several techniques are used for merging and reducing number of FMMs

       * When `base_kernel` is given an `IntG` is rewritten using `base_kernel`
         and its derivatives.

       * :class:`sumpy.kernel.AxisTargetDerivative` instances are converted
         to :class:`sumpy.kernel.AxisSourceDerivative` instances.

       * If there is a sum of two `IntG`s with same target derivative and different
         source derivatives of the same kernel, they are merged into one FMM.

       * If possible, convert :class:`sumpy.kernel.AxisSourceDerivative` to
         :class:`sumpy.kernel.DirectionalSourceDerivative`.

       * Reduce the number of FMMs by converting the `IntG` expression to
         a matrix and factoring the matrix where the left operand matrix represents
         a transformation at target and the right matrix represents a transformation
         at source. For this to work, we need to know which variables depend on
         source so that they do not end up in the left operand. User needs to supply
         this as the argument `source_dependent_variable`.

    :arg base_kernel: A :class:`sumpy.kernel.Kernel` object if given will be used
        for converting a :class:`~pytential.symbolic.primitives.IntG` to a linear
        expression of same type with the kernel replaced by base_kernel and its
        derivatives

    :arg verbose: increase verbosity of merging

    :arg source_dependent_variable: When merging expressions, consider only these
        variables as dependent on source. Otherwise consider all variables
        as source dependent. This is important when reducing the number of FMMs
        needed for the output.
    """

    int_g_s = get_int_g_s(exprs)

    if base_kernel is not None:
        replacements = {}

        # Iterate all the IntGs in the expressions and create a dictionary
        # of replacements for the IntGs
        for int_g in int_g_s:
            # First convert IntGs with target derivatives to source derivatives
            new_int_g = convert_target_deriv_to_source(int_g)
            # Convert IntGs with TargetMultiplier to a sum of IntGs without
            # TargetMultipliers
            new_int_g_s = convert_target_multiplier_to_source(new_int_g)
            # Convert IntGs with different kernels to expressions containing
            # IntGs with base_kernel or its derivatives
            replacements[int_g] = sum(convert_int_g_to_base(new_int_g,
                base_kernel, verbose=verbose) for new_int_g in new_int_g_s)

        # replace the IntGs in the expressions
        substitutor = IntGSubstitutor(replacements)
        exprs = [substitutor(expr) for expr in exprs]

    differing_kernel_arguments = set()
    kernel_arguments = {}
    for int_g in int_g_s:
        for k, v in int_g.kernel_arguments.items():
            if k not in kernel_arguments:
                kernel_arguments[k] = v
            else:
                if kernel_arguments[k] != v:
                    differing_kernel_arguments.add(k)

    # Using a dictionary instead of a set because sets are unordered
    all_source_groups = dict()

    result = np.array([0 for _ in exprs], dtype=object)

    int_gs_by_group_for_index = []

    int_g_cc = IntGCoefficientCollector()
    for i, expr in enumerate(exprs):
        try:
            int_g_coeff_map = int_g_cc(expr)
        except (RuntimeError, AssertionError):
            # Don't touch this expression, because it's not linear.
            # FIXME: if there's ever any use case, then we can extract
            # some IntGs from them.
            result[i] += expr
            continue
        int_gs_by_group = {}
        for int_g, coeff in int_g_coeff_map.items():
            if int_g == 1:
                # coeff map may have some constant terms, add them to
                result[i] += coeff
                continue

            # convert DirectionalSourceDerivative to AxisSourceDerivative
            # as kernel arguments need to be the same for merging
            if source_dependent_variables is not None:
                int_g = convert_directional_source_to_axis_source(int_g,
                    source_dependent_variables)
            # convert TargetDerivative to source before checking the group
            # as the target kernel has to be the same for merging
            int_g = convert_target_deriv_to_source(int_g)
            # move the coefficient inside
            int_g = int_g.copy(densities=[density*coeff for density in
                    int_g.densities])

            source_group = get_int_g_source_group(int_g, differing_kernel_arguments)
            target_group = get_int_g_target_group(int_g)
            group = (source_group, target_group)

            all_source_groups[source_group] = 1

            if group not in int_gs_by_group:
                new_int_g = int_g
            else:
                prev_int_g = int_gs_by_group[group]
                # Let's merge IntGs with the same group
                new_int_g = merge_two_int_gs(int_g, prev_int_g)
            int_gs_by_group[group] = new_int_g

        # Do some simplifications after merging. Not stricty necessary
        for group, int_g in int_gs_by_group.items():
            # replace an IntG with d axis source derivatives to an IntG
            # with one directional source derivative
            # TODO: reenable this later
            # result_int_g = convert_axis_source_to_directional_source(int_g)
            # simplify the densities as they may become large due to pymbolic
            # not doing automatic simplifications unlike sympy/symengine
            result_int_g = int_g.copy(
                    densities=simplify_densities(int_g.densities))
            int_gs_by_group[group] = result_int_g

        int_gs_by_group_for_index.append(int_gs_by_group)

    # If source_dependent_variables (sdv) is not given return early.
    # Check for (sdv is None) instead of just (sdv) because
    # it can be an empty list.
    if source_dependent_variables is None:
        for iexpr, int_gs_by_group in enumerate(int_gs_by_group_for_index):
            for int_g in int_gs_by_group.values():
                result[iexpr] += int_g
        return result

    # Do the calculation for each source_group separately and assemble them
    for source_group in all_source_groups.keys():
        insn_to_idx_mapping = defaultdict(list)
        for idx, int_gs_by_group in enumerate(int_gs_by_group_for_index):
            for group, int_g in int_gs_by_group.items():
                if group[0] != source_group:
                    continue
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
                common_int_g = remove_target_attributes(int_g)
                insn_to_idx_mapping[common_int_g].append((idx, int_g))

        insns_to_reduce = list(insn_to_idx_mapping.keys())
        reduced_insns = reduce_number_of_fmms(insns_to_reduce,
                source_dependent_variables)

        for insn, reduced_insn in zip(insns_to_reduce, reduced_insns):
            for idx, int_g in insn_to_idx_mapping[insn]:
                result[idx] += restore_target_attributes(reduced_insn, int_g)
    return result


def get_int_g_source_group(int_g, differing_kernel_arguments):
    """Return a group for the *int_g* with so that all elements in that
    group have the same source attributes.
    """
    def get_hashable(arg):
        if hasattr(arg, "__iter__"):
            return tuple(arg)
        return arg

    args = tuple((k, get_hashable(v)) for k, v in sorted(
        int_g.kernel_arguments.items()) if k in
                differing_kernel_arguments)
    return (int_g.source, args, int_g.target_kernel.get_base_kernel())


def get_int_g_target_group(int_g):
    """Return a group for the *int_g* with so that all elements in that
    group have the same source attributes.
    """
    return (int_g.target, int_g.qbx_forced_limit, int_g.target_kernel)


def remove_target_attributes(int_g):
    """Remove target attributes from *int_g* and return an expression
    that is common to all expression in the same source group.
    """
    return int_g.copy(target=None, qbx_forced_limit=None,
            target_kernel=int_g.target_kernel.get_base_kernel())


def restore_target_attributes(expr, orig_int_g):
    """Restore target attributes from *orig_int_g* to all the
    :class:`~pytential.symbolic.primitives.IntG` objects in the
    input *expr*.
    """
    int_gs = get_int_g_s([expr])

    replacements = {
        int_g: int_g.copy(target=orig_int_g.target,
                qbx_forced_limit=orig_int_g.qbx_forced_limit,
                target_kernel=int_g.target_kernel.replace_base_kernel(
                    orig_int_g.target_kernel))
        for int_g in int_gs}

    substitutor = IntGSubstitutor(replacements)
    return substitutor(expr)


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


class IntGSubstitutor(IdentityMapper):
    """Replaces IntGs with pymbolic expression given by the
    replacements dictionary
    """
    def __init__(self, replacements):
        self.replacements = replacements

    def map_int_g(self, expr):
        return self.replacements.get(expr, expr)


class IntGCoefficientCollector(CoefficientCollector):
    def __init__(self):
        super().__init__({})

    def map_int_g(self, expr):
        return {expr: 1}

    def handle_unsupported_expression(self, expr, *args, **kwargs):
        return {1: expr}


def evalf(expr, prec=100):
    """evaluate an expression numerically using ``prec``
    number of bits.
    """
    from sumpy.symbolic import USE_SYMENGINE
    if USE_SYMENGINE:
        return expr.n(prec=prec)
    else:
        import sympy
        dps = int(sympy.log(2**prec, 10))
        return expr.n(n=dps)


@memoize_on_first_arg
def _get_base_kernel_matrix(base_kernel, order=None, verbose=False):
    dim = base_kernel.dim

    pde = base_kernel.get_pde_as_diff_op()
    if order is None:
        order = pde.order

    if order > pde.order:
        raise NotImplementedError(f"order ({order}) cannot be greater than the order"
                         f"of the PDE ({pde.order}) yet.")

    mis = sorted(gnitstam(order, dim), key=sum)
    # (-1, -1, -1) represent a constant
    mis.append((-1, -1, -1))

    if order == pde.order:
        pde_mis = [ident.mi for eq in pde.eqs for ident in eq.keys()]
        pde_mis = [mi for mi in pde_mis if sum(mi) == order]
        if verbose:
            print(f"Removing {pde_mis[-1]} to avoid linear dependent mis")
        mis.remove(pde_mis[-1])

    rand = np.random.randint(1, 100, (dim, len(mis)))
    sym_vec = make_sym_vector("d", dim)

    base_expr = base_kernel.get_expression(sym_vec)

    mat = []
    for rand_vec_idx in range(rand.shape[1]):
        row = []
        for mi in mis[:-1]:
            expr = base_expr
            for var_idx, nderivs in enumerate(mi):
                if nderivs == 0:
                    continue
                expr = expr.diff(sym_vec[var_idx], nderivs)
            replace_dict = dict(
                (k, v) for k, v in zip(sym_vec, rand[:, rand_vec_idx])
            )
            eval_expr = evalf(expr.xreplace(replace_dict))
            row.append(eval_expr)
        row.append(1)
        mat.append(row)

    mat = sym.Matrix(mat)
    L, U, perm = mat.LUdecomposition()
    return (L, U, perm), rand, mis


@memoize_on_first_arg
def get_deriv_relation_kernel(kernel, base_kernel, tol=1e-8, order=None,
        verbose=False):
    (L, U, perm), rand, mis = _get_base_kernel_matrix(base_kernel, order=order,
            verbose=verbose)
    dim = base_kernel.dim
    sym_vec = make_sym_vector("d", dim)
    sympy_conv = SympyToPymbolicMapper()

    expr = kernel.get_expression(sym_vec)
    vec = []
    for i in range(len(mis)):
        vec.append(evalf(expr.xreplace(dict((k, v) for
            k, v in zip(sym_vec, rand[:, i])))))
    vec = sym.Matrix(vec)
    result = []
    const = 0
    if verbose:
        print(kernel, end=" = ", flush=True)

    sol = lu_solve_with_expand(L, U, perm, vec)
    for i, coeff in enumerate(sol):
        coeff = chop(coeff, tol)
        if coeff == 0:
            continue
        if mis[i] != (-1, -1, -1):
            coeff *= kernel.get_global_scaling_const()
            coeff /= base_kernel.get_global_scaling_const()
            result.append((mis[i], sympy_conv(coeff)))
            if verbose:
                print(f"{base_kernel}.diff({mis[i]})*{coeff}", end=" + ")
        else:
            const = sympy_conv(coeff * kernel.get_global_scaling_const())
    if verbose:
        print(const)
    return (const, result)


def get_deriv_relation(kernels, base_kernel, tol=1e-10, order=None, verbose=False):
    res = []
    for knl in kernels:
        res.append(get_deriv_relation_kernel(knl, base_kernel, tol, order, verbose))
    return res


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


def have_int_g_s(expr):
    """Checks if a :mod:`pymbolic` expression has
    :class:`~pytential.symbolic.primitives.IntG` objects
    """
    mapper = GetIntGs()
    mapper(expr)
    return bool(mapper.int_g_s)


def get_int_g_s(exprs):
    """Returns all :class:`~pytential.symbolic.primitives.IntG` objects
    in a list of :mod:`pymbolic` expressions.
    """
    get_int_g_mapper = GetIntGs()
    [get_int_g_mapper(expr) for expr in exprs]
    return get_int_g_mapper.int_g_s


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


def convert_int_g_to_base(int_g, base_kernel, verbose=False):
    result = 0
    for knl, density in zip(int_g.source_kernels, int_g.densities):
        result += _convert_int_g_to_base(
                int_g.copy(source_kernels=(knl,), densities=(density,)),
                base_kernel, verbose)
    return result


def _convert_int_g_to_base(int_g, base_kernel, verbose=False):
    tgt_knl = int_g.target_kernel
    dim = tgt_knl.dim
    if tgt_knl != tgt_knl.get_base_kernel():
        return int_g

    if not len(int_g.densities) == 1:
        raise ValueError
    density = int_g.densities[0]
    source_kernel = int_g.source_kernels[0]

    deriv_relation = get_deriv_relation_kernel(source_kernel.get_base_kernel(),
        base_kernel, verbose=verbose)

    const = deriv_relation[0]
    # NOTE: we set a dofdesc here to force the evaluation of this integral
    # on the source instead of the target when using automatic tagging
    # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
    dd = pytential.sym.DOFDescriptor(None,
            discr_stage=pytential.sym.QBX_SOURCE_STAGE1)
    const *= pytential.sym.integral(dim, dim-1, density, dofdesc=dd)

    result = 0
    if source_kernel == source_kernel.get_base_kernel():
        result += const

    new_kernel_args = filter_kernel_arguments([base_kernel], int_g.kernel_arguments)

    for mi, c in deriv_relation[1]:
        knl = source_kernel.replace_base_kernel(base_kernel)
        for d, val in enumerate(mi):
            for _ in range(val):
                knl = AxisSourceDerivative(d, knl)
                c *= -1
        result += int_g.copy(target_kernel=base_kernel, source_kernels=(knl,),
                densities=(density,), kernel_arguments=new_kernel_args) * c
    return result


def convert_target_multiplier_to_source(int_g):
    """Convert an IntG with TargetMultiplier to an sum of IntGs without
    TargetMultiplier and only source dependent transformations
    """
    from sumpy.symbolic import SympyToPymbolicMapper
    tgt_knl = int_g.target_kernel
    if not isinstance(tgt_knl, TargetPointMultiplier):
        return [int_g]
    if isinstance(tgt_knl.inner_kernel, KernelWrapper):
        return [int_g]

    new_kernel_args = filter_kernel_arguments([tgt_knl], int_g.kernel_arguments)
    result = []
    # If the kernel is G, source is y and target is x,
    # x G = y*G + (x - y)*G
    # For y*G, absorb y into a density
    new_densities = [density*NodeCoordinateComponent(tgt_knl.axis)
            for density in int_g.densities]
    result.append(int_g.copy(target_kernel=tgt_knl.inner_kernel,
                densities=tuple(new_densities), kernel_arguments=new_kernel_args))

    # create a new expression kernel for (x - y)*G
    sym_d = make_sym_vector("d", tgt_knl.dim)
    conv = SympyToPymbolicMapper()

    for knl, density in zip(int_g.source_kernels, int_g.densities):
        new_expr = conv(knl.postprocess_at_source(knl.get_expression(sym_d), sym_d)
                * sym_d[tgt_knl.axis])
        new_knl = ExpressionKernel(knl.dim, new_expr,
                knl.get_base_kernel().global_scaling_const,
                knl.is_complex_valued)
        result.append(int_g.copy(target_kernel=new_knl,
            densities=(density,),
            source_kernels=(new_knl,), kernel_arguments=new_kernel_args))
    return result


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
        return 1, int_g

    for axis in reversed(multipliers):
        knl = TargetPointMultiplier(axis, knl)

    new_densities = tuple(density*coeff for density in int_g.densities)
    return int_g.copy(target_kernel=knl,
                      densities=new_densities,
                      source_kernels=tuple(source_kernels))


def convert_directional_source_to_axis_source(int_g, source_dependent_variables):
    """Convert an IntG with a DirectionalSourceDerivative instance
    to an IntG with d AxisSourceDerivative instances.
    """
    source_kernels = []
    densities = []
    for source_kernel, density in zip(int_g.source_kernels, int_g.densities):
        knl_result = _convert_directional_source_knl_to_axis_source(source_kernel,
                source_dependent_variables)
        for knl, coeff in knl_result:
            source_kernels.append(knl)
            densities.append(coeff * density)
    return int_g.copy(source_kernels=tuple(source_kernels),
            densities=tuple(densities))


def _convert_directional_source_knl_to_axis_source(knl, source_dependent_variables):
    if isinstance(knl, DirectionalSourceDerivative):
        dim = knl.dim
        from pymbolic import make_sym_vector
        dir_vec = make_sym_vector(knl.dir_vec_name, dim)
        if Variable(knl.dir_vec_name) not in source_dependent_variables:
            raise ValueError(f"{knl.dir_vec_name} not given in "
                "source_dependent_variables, but is dependent on source")

        res = []
        inner_result = _convert_directional_source_knl_to_axis_source(
                knl.inner_kernel, source_dependent_variables)
        for inner_knl, coeff in inner_result:
            for d in range(dim):
                res.append((AxisSourceDerivative(d, inner_knl), coeff*dir_vec[d]))
        return res
    elif isinstance(knl, KernelWrapper):
        inner_result = _convert_directional_source_knl_to_axis_source(
                knl.inner_kernel, source_dependent_variables)
        return [(knl.replace_inner_kernel(inner_knl), coeff) for
                    inner_knl, coeff in inner_result]
    else:
        return [(knl, 1)]


def convert_axis_source_to_directional_source(int_g):
    """Convert an IntG with d AxisSourceDerivative instances to
    an IntG with one DirectionalSourceDerivative instance.
    """
    from pytential.symbolic.primitives import _DIR_VEC_NAME
    if not isinstance(int_g, IntG):
        return int_g
    knls = list(int_g.source_kernels)
    dim = knls[0].dim
    if len(knls) != dim:
        return int_g
    if not any(isinstance(knl, AxisSourceDerivative) for knl in knls):
        return int_g
    # TODO: sort
    axes = [knl.axis for knl in knls]
    if axes != list(range(dim)):
        return int_g
    base_knls = set(knl.inner_kernel for knl in knls)
    if len(base_knls) > 1:
        return int_g
    base_knl = base_knls.pop()
    kernel_arguments = int_g.kernel_arguments.copy()

    name = _DIR_VEC_NAME
    count = 0
    while name in kernel_arguments:
        name = _DIR_VEC_NAME + f"_{count}"
        count += 1

    kernel_arguments[name] = \
            np.array(int_g.densities, dtype=np.object)
    res = int_g.copy(
            source_kernels=(
                DirectionalSourceDerivative(base_knl, dir_vec_name=name),),
            densities=(1,),
            kernel_arguments=kernel_arguments)
    return res


def merge_kernel_arguments(x, y):
    """merge two kernel argument dictionaries and raise a ValueError if
    the two dictionaries do not agree for duplicate keys.
    """
    res = x.copy()
    for k, v in y.items():
        if k in res:
            if not res[k] == v:
                raise ValueError
        else:
            res[k] = v
    return res


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
            result.append(density)
    return tuple(result)


if __name__ == "__main__":
    from sumpy.kernel import (StokesletKernel, BiharmonicKernel,  # noqa:F401
        StressletKernel, ElasticityKernel, LaplaceKernel)
    base_kernel = BiharmonicKernel(3)
    #base_kernel = LaplaceKernel(3)
    kernels = [StokesletKernel(3, 0, 2), StokesletKernel(3, 0, 0)]
    kernels += [StressletKernel(3, 0, 0, 0), StressletKernel(3, 0, 0, 1),
            StressletKernel(3, 0, 0, 2), StressletKernel(3, 0, 1, 2)]

    sym_d = make_sym_vector("d", base_kernel.dim)
    sym_r = sym.sqrt(sum(a**2 for a in sym_d))
    conv = SympyToPymbolicMapper()
    expression_knl = ExpressionKernel(3, conv(sym_d[0]*sym_d[1]/sym_r**3), 1, False)
    expression_knl2 = ExpressionKernel(3, conv(1/sym_r + sym_d[0]*sym_d[0]/sym_r**3),
        1, False)
    kernels = [expression_knl, expression_knl2]
    #kernels += [ElasticityKernel(3, 0, 1, poisson_ratio="0.4"),
    #        ElasticityKernel(3, 0, 0, poisson_ratio="0.4")]
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=4, verbose=True)
    density = pytential.sym.make_sym_vector("d", 1)[0]
    from pytential.symbolic.primitives import int_g_vec
    int_g_1 = int_g_vec(TargetPointMultiplier(2, AxisTargetDerivative(2,
            AxisSourceDerivative(1, AxisSourceDerivative(0,
                LaplaceKernel(3))))), density, qbx_forced_limit=1)
    int_g_2 = int_g_vec(TargetPointMultiplier(0, AxisTargetDerivative(0,
        AxisSourceDerivative(0, AxisSourceDerivative(0,
            LaplaceKernel(3))))), density, qbx_forced_limit=1)
    print(merge_int_g_exprs([int_g_1, int_g_2],
        base_kernel=BiharmonicKernel(3), verbose=True)[0])
