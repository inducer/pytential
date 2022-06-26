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

from sumpy.symbolic import make_sym_vector, SympyToPymbolicMapper
import sumpy.symbolic as sym
from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative,
    DirectionalSourceDerivative, ExpressionKernel,
    KernelWrapper, TargetPointMultiplier, DirectionalDerivative)
from pytools import (memoize_on_first_arg,
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)
from collections import defaultdict

from pymbolic.geometric_algebra.mapper import WalkMapper
from pymbolic.mapper import CombineMapper
from pymbolic.mapper.coefficient import CoefficientCollector
from pytential.symbolic.primitives import (IntG, NodeCoordinateComponent,
    hashable_kernel_args, hashable_kernel_arg_value)
from pytential.symbolic.mappers import IdentityMapper
from pytential.utils import chop, lu_solve_with_expand
import pytential

from pytential.symbolic.pde.reduce_fmms import reduce_number_of_fmms

import logging
logger = logging.getLogger(__name__)

__all__ = (
    "merge_int_g_exprs",
    "get_deriv_relation",
    )

__doc__ = """
.. autofunction:: rewrite_using_base_kernel
.. autofunction:: merge_int_g_exprs
.. autofunction:: get_deriv_relation
"""


# {{{ rewrite_using_base_kernel

_NO_ARG_SENTINEL = object()


def rewrite_using_base_kernel(exprs, base_kernel=_NO_ARG_SENTINEL):
    """Rewrites an expression with :class:`~pytential.symbolic.primitives.IntG`
    objects using *base_kernel*.

    For example, if *base_kernel* is the Biharmonic kernel, and a Laplace kernel
    is encountered, this will (forcibly) rewrite the Laplace kernel in terms of
    derivatives of the Biharmonic kernel.

    The routine will fail if this process cannot be completed.
    """
    if base_kernel is None:
        return list(exprs)
    mapper = RewriteUsingBaseKernelMapper(base_kernel)
    return [mapper(expr) for expr in exprs]


class RewriteUsingBaseKernelMapper(IdentityMapper):
    """Rewrites IntGs using the base kernel. First this method replaces
    IntGs with :class:`sumpy.kernel.AxisTargetDerivative` to IntGs
    :class:`sumpy.kernel.AxisSourceDerivative` and then replaces
    IntGs with :class:`sumpy.kernel.TargetPointMultiplier` to IntGs
    without them using :class:`sumpy.kernel.ExpressionKernel`
    and then finally converts them to the base kernel by finding
    a relationship between the derivatives.
    """
    def __init__(self, base_kernel):
        self.base_kernel = base_kernel

    def map_int_g(self, expr):
        # First convert IntGs with target derivatives to source derivatives
        expr = convert_target_deriv_to_source(expr)
        # Convert IntGs with TargetMultiplier to a sum of IntGs without
        # TargetMultipliers
        new_int_gs = convert_target_multiplier_to_source(expr)
        # Convert IntGs with different kernels to expressions containing
        # IntGs with base_kernel or its derivatives
        return sum(convert_int_g_to_base(new_int_g,
            self.base_kernel) for new_int_g in new_int_gs)


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


def _get_kernel_expression(expr, kernel_arguments):
    from pymbolic.mapper.substitutor import substitute
    from sumpy.symbolic import PymbolicToSympyMapperWithSymbols

    pymbolic_expr = substitute(expr, kernel_arguments)

    res = PymbolicToSympyMapperWithSymbols()(pymbolic_expr)
    return res


def _monom_to_expr(monom, variables):
    prod = 1
    for i, nrepeats in enumerate(monom):
        for _ in range(nrepeats):
            prod *= variables[i]
    return prod


def convert_target_multiplier_to_source(int_g):
    """Convert an IntG with TargetMultiplier to a sum of IntGs without
    TargetMultiplier and only source dependent transformations
    """
    import sympy
    import sumpy.symbolic as sym
    from sumpy.symbolic import SympyToPymbolicMapper
    conv = SympyToPymbolicMapper()

    knl = int_g.target_kernel
    # we use a symbol for d = (x - y)
    ds = sympy.symbols(f"d0:{knl.dim}")
    sources = sympy.symbols(f"y0:{knl.dim}")
    # instead of just x, we use x = (d + y)
    targets = [d + source for d, source in zip(ds, sources)]
    orig_expr = sympy.Function("f")(*ds)
    expr = orig_expr
    found = False
    while isinstance(knl, KernelWrapper):
        if isinstance(knl, TargetPointMultiplier):
            expr = targets[knl.axis] * expr
            found = True
        elif isinstance(knl, AxisTargetDerivative):
            # sympy can't differentiate w.r.t target because
            # it's not a symbol, but d/d(x) = d/d(d)
            expr = expr.diff(ds[knl.axis])
        else:
            return [int_g]
        knl = knl.inner_kernel

    if not found:
        return [int_g]

    sources_pymbolic = [NodeCoordinateComponent(i) for i in range(knl.dim)]
    expr = expr.expand()
    # Now the expr is an Add and looks like
    # u''(d, s)*d[0] + u(d, s)
    assert isinstance(expr, sympy.Add)
    result = []

    for arg in expr.args:
        deriv_terms = arg.atoms(sympy.Derivative)
        if len(deriv_terms) == 1:
            deriv_term = deriv_terms.pop()
            rest_terms = sympy.Poly(arg.xreplace({deriv_term: 1}), *ds, *sources)
            derivatives = deriv_term.args[1:]
        elif len(deriv_terms) == 0:
            rest_terms = sympy.Poly(arg.xreplace({orig_expr: 1}), *ds, *sources)
            derivatives = [(d, 0) for d in ds]
        else:
            raise AssertionError("impossible condition")
        assert len(rest_terms.terms()) == 1
        monom, coeff = rest_terms.terms()[0]
        expr_multiplier = _monom_to_expr(monom[:len(ds)], ds)
        density_multiplier = _monom_to_expr(monom[len(ds):], sources_pymbolic) \
                * conv(coeff)

        new_int_gs = _multiply_int_g(int_g, sym.sympify(expr_multiplier),
                density_multiplier)
        for new_int_g in new_int_gs:
            knl = new_int_g.target_kernel
            for axis_var, nrepeats in derivatives:
                axis = ds.index(axis_var)
                for _ in range(nrepeats):
                    knl = AxisTargetDerivative(axis, knl)
            result.append(new_int_g.copy(target_kernel=knl))
    return result


def _multiply_int_g(int_g, expr_multiplier, density_multiplier):
    """Multiply the exprssion in IntG with the *expr_multiplier*
    which is a symbolic expression and multiply the densities
    with *density_multiplier* which is a pymbolic expression.
    """
    from sumpy.symbolic import SympyToPymbolicMapper
    result = []

    base_kernel = int_g.target_kernel.get_base_kernel()
    sym_d = make_sym_vector("d", base_kernel.dim)
    base_kernel_expr = _get_kernel_expression(base_kernel.expression,
            int_g.kernel_arguments)
    conv = SympyToPymbolicMapper()

    for knl, density in zip(int_g.source_kernels, int_g.densities):
        if expr_multiplier == 1:
            new_knl = knl.get_base_kernel()
        else:
            new_expr = conv(knl.postprocess_at_source(base_kernel_expr, sym_d)
                    * expr_multiplier)
            new_knl = ExpressionKernel(knl.dim, new_expr,
                knl.get_base_kernel().global_scaling_const,
                knl.is_complex_valued)
        result.append(int_g.copy(target_kernel=new_knl,
            densities=(density*density_multiplier,),
            source_kernels=(new_knl,)))
    return result


def convert_int_g_to_base(int_g, base_kernel):
    result = 0
    for knl, density in zip(int_g.source_kernels, int_g.densities):
        result += _convert_int_g_to_base(
                int_g.copy(source_kernels=(knl,), densities=(density,)),
                base_kernel)
    return result


def _convert_int_g_to_base(int_g, base_kernel):
    target_kernel = int_g.target_kernel.replace_base_kernel(base_kernel)
    dim = target_kernel.dim

    result = 0
    for density, source_kernel in zip(int_g.densities, int_g.source_kernels):
        deriv_relation = get_deriv_relation_kernel(source_kernel.get_base_kernel(),
            base_kernel, hashable_kernel_arguments=(
                hashable_kernel_args(int_g.kernel_arguments)))

        const = deriv_relation[0]
        # NOTE: we set a dofdesc here to force the evaluation of this integral
        # on the source instead of the target when using automatic tagging
        # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
        dd = pytential.sym.DOFDescriptor(None,
                discr_stage=pytential.sym.QBX_SOURCE_STAGE1)
        const *= pytential.sym.integral(dim, dim-1, density, dofdesc=dd)

        if const != 0 and target_kernel != target_kernel.get_base_kernel():
            # There might be some TargetPointMultipliers hanging around.
            # FIXME: handle them instead of bailing out
            return [int_g]

        if source_kernel != source_kernel.get_base_kernel():
            # We assume that any source transformation is a derivative
            # and the constant when applied becomes zero.
            const = 0

        result += const

        new_kernel_args = filter_kernel_arguments([base_kernel],
                int_g.kernel_arguments)

        for mi, c in deriv_relation[1]:
            knl = source_kernel.replace_base_kernel(base_kernel)
            for d, val in enumerate(mi):
                for _ in range(val):
                    knl = AxisSourceDerivative(d, knl)
                    c *= -1
            result += int_g.copy(source_kernels=(knl,), target_kernel=target_kernel,
                    densities=(density * c,), kernel_arguments=new_kernel_args)
    return result


def get_deriv_relation(kernels, base_kernel, tol=1e-10, order=None,
        kernel_arguments=None):
    res = []
    for knl in kernels:
        res.append(get_deriv_relation_kernel(knl, base_kernel, tol, order,
            hashable_kernel_arguments=hashable_kernel_args(kernel_arguments)))
    return res


@memoize_on_first_arg
def get_deriv_relation_kernel(kernel, base_kernel, tol=1e-10, order=None,
        hashable_kernel_arguments=None):
    kernel_arguments = dict(hashable_kernel_arguments)
    (L, U, perm), rand, mis = _get_base_kernel_matrix(base_kernel, order=order)
    dim = base_kernel.dim
    sym_vec = make_sym_vector("d", dim)
    sympy_conv = SympyToPymbolicMapper()

    expr = _get_kernel_expression(kernel.expression, kernel_arguments)
    vec = []
    for i in range(len(mis)):
        vec.append(evalf(expr.xreplace(dict((k, v) for
            k, v in zip(sym_vec, rand[:, i])))))
    vec = sym.Matrix(vec)
    result = []
    const = 0
    logger.debug("%s = ", kernel)

    sol = lu_solve_with_expand(L, U, perm, vec)
    for i, coeff in enumerate(sol):
        coeff = chop(coeff, tol)
        if coeff == 0:
            continue
        if mis[i] != (-1, -1, -1):
            coeff *= _get_kernel_expression(kernel.global_scaling_const,
                    kernel_arguments)
            coeff /= _get_kernel_expression(base_kernel.global_scaling_const,
                    kernel_arguments)
            result.append((mis[i], sympy_conv(coeff)))
            logger.debug("  + %s.diff(%s)*%s", base_kernel, mis[i], coeff)
        else:
            const = sympy_conv(coeff * _get_kernel_expression(
                kernel.global_scaling_const, kernel_arguments))
    logger.debug("  + %s", const)
    return (const, result)


@memoize_on_first_arg
def _get_base_kernel_matrix(base_kernel, order=None, retries=3,
        kernel_arguments=None):
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
        logger.debug(f"Removing {pde_mis[-1]} to avoid linear dependent mis")
        mis.remove(pde_mis[-1])

    rand = np.random.randint(1, 10**15, (dim, len(mis)))
    rand = rand.astype(object)
    for i in range(rand.shape[0]):
        for j in range(rand.shape[1]):
            rand[i, j] = sym.sympify(rand[i, j])/10**15
    sym_vec = make_sym_vector("d", dim)

    base_expr = _get_kernel_expression(base_kernel.expression, kernel_arguments)

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
    failed = False
    try:
        L, U, perm = mat.LUdecomposition()
    except RuntimeError:
        # symengine throws an error when rank deficient
        # and sympy returns U with last row zero
        failed = True

    if not sym.USE_SYMENGINE and all(expr == 0 for expr in U[-1, :]):
        failed = True

    if failed:
        if retries == 0:
            raise RuntimeError("Failed to find a base kernel")
        return _get_base_kernel_matrix(
            base_kernel=base_kernel,
            order=order,
            retries=retries-1,
        )

    return (L, U, perm), rand, mis


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

# }}}


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
         call to :func:`pytential.symbolic.pde.reduce_fmms.reduce_number_of_fmms`.

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


def get_number_of_fmms(exprs):
    fmms = set()
    for int_g in get_int_g_s(exprs):
        fmms.add(remove_target_attributes(int_g))
    return len(fmms)


class IntGCoefficientCollector(CoefficientCollector):
    def __init__(self):
        super().__init__({})

    def map_int_g(self, expr):
        return {expr: 1}

    def map_algebraic_leaf(self, expr, *args, **kwargs):
        return {1: expr}

    handle_unsupported_expression = map_algebraic_leaf


def is_expr_target_dependent(expr):
    mapper = IsExprTargetDependent()
    return mapper(expr)


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


def get_hashable_kernel_argument(arg):
    if hasattr(arg, "__iter__"):
        try:
            return tuple(arg)
        except TypeError:
            pass
    return arg


def get_int_g_source_group_identifier(int_g):
    """Return a identifier for a group for the *int_g* so that all elements in that
    group have the same source attributes.
    """
    target_arg_names = get_normal_vector_names(int_g.target_kernel)
    args = dict((k, v) for k, v in sorted(
        int_g.kernel_arguments.items()) if k not in target_arg_names)
    return (int_g.source, hashable_kernel_args(args),
            int_g.target_kernel.get_base_kernel())


def get_int_g_target_group_identifier(int_g):
    """Return a identifier for a group for the *int_g* so that all elements in that
    group have the same target attributes.
    """
    target_arg_names = get_normal_vector_names(int_g.target_kernel)
    args = dict((k, v) for k, v in sorted(
        int_g.kernel_arguments.items()) if k in target_arg_names)
    return (int_g.target, int_g.qbx_forced_limit, int_g.target_kernel,
            hashable_kernel_args(args))


def get_normal_vector_names(kernel):
    """Return the normal vector names in a kernel
    """
    normal_vectors = set()
    while isinstance(kernel, KernelWrapper):
        if isinstance(kernel, DirectionalDerivative):
            normal_vectors.add(kernel.dir_vec_name)
        kernel = kernel.inner_kernel
    return normal_vectors


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
            if hashable_kernel_arg_value(res[k]) \
                    != hashable_kernel_arg_value(v):
                raise ValueError(f"Error merging values for {k}."
                    f"values were {res[k]} and {v}")
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
            logger.debug("%s cannot be simplified", density)
            result.append(density)
    return tuple(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
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
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=4)
