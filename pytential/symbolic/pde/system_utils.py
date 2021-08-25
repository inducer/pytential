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

from pymbolic.mapper import WalkMapper, Mapper
from pymbolic.primitives import Sum, Product, Quotient
from pytential.symbolic.primitives import IntG, NodeCoordinateComponent
import pytential

from collections import defaultdict


def _chop(expr, tol):
    nums = expr.atoms(sym.Number)
    replace_dict = {}
    for num in nums:
        if float(abs(num)) < tol:
            replace_dict[num] = 0
        else:
            new_num = float(num)
            if abs((int(new_num) - new_num)/new_num) < tol:
                new_num = int(new_num)
            replace_dict[num] = new_num
    return expr.xreplace(replace_dict)


def _n(expr):
    from sumpy.symbolic import USE_SYMENGINE
    if USE_SYMENGINE:
        # 100 bits
        return expr.n(prec=100)
    else:
        # 30 decimal places
        return expr.n(n=30)


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
            eval_expr = _n(expr.xreplace(replace_dict))
            row.append(eval_expr)
        row.append(1)
        mat.append(row)

    mat = sym.Matrix(mat)
    L, U, perm = mat.LUdecomposition()
    return (L, U, perm), rand, mis


def _LUsolve_with_expand(L, U, perm, b):
    def forward_substitution(L, b):
        n = len(b)
        res = sym.Matrix(b)
        for i in range(n):
            for j in range(i):
                res[i] -= L[i, j]*res[j]
            res[i] = (res[i] / L[i, i]).expand()
        return res

    def backward_substitution(U, b):
        n = len(b)
        res = sym.Matrix(b)
        for i in range(n-1, -1, -1):
            for j in range(n - 1, i, -1):
                res[i] -= U[i, j]*res[j]
            res[i] = (res[i] / U[i, i]).expand()
        return res

    def permuteFwd(b, perm):
        res = sym.Matrix(b)
        for p, q in perm:
            res[p], res[q] = res[q], res[p]
        return res

    return backward_substitution(U,
            forward_substitution(L, permuteFwd(b, perm)))


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
        vec.append(_n(expr.xreplace(dict((k, v) for
            k, v in zip(sym_vec, rand[:, i])))))
    vec = sym.Matrix(vec)
    result = []
    const = 0
    if verbose:
        print(kernel, end=" = ", flush=True)

    sol = _LUsolve_with_expand(L, U, perm, vec)
    for i, coeff in enumerate(sol):
        coeff = _chop(coeff, tol)
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
    def __init__(self):
        self.int_g_s = set()

    def map_int_g(self, expr):
        self.int_g_s.add(expr)

    def map_constant(self, expr):
        pass

    map_variable = map_constant
    handle_unsupported_expression = map_constant


def have_int_g_s(expr):
    mapper = GetIntGs()
    mapper(expr)
    return bool(mapper.int_g_s)


def convert_int_g_to_base(int_g, base_kernel, verbose=False):
    result = 0
    for knl, density in zip(int_g.source_kernels, int_g.densities):
        result += _convert_int_g_to_base(
                int_g.copy(source_kernels=(knl,), densities=(density,)),
                base_kernel, verbose)
    return result


def _filter_kernel_arguments(knls, kernel_arguments):
    kernel_arg_names = set()

    for kernel in knls:
        for karg in (kernel.get_args() + kernel.get_source_args()):
            kernel_arg_names.add(karg.loopy_arg.name)

    return {k: v for (k, v) in kernel_arguments.items() if k in kernel_arg_names}


def _convert_int_g_to_base(int_g, base_kernel, verbose=False):
    tgt_knl = int_g.target_kernel
    dim = tgt_knl.dim
    if tgt_knl != tgt_knl.get_base_kernel():
        return int_g

    assert len(int_g.densities) == 1
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

    new_kernel_args = _filter_kernel_arguments([base_kernel], int_g.kernel_arguments)

    for mi, c in deriv_relation[1]:
        knl = source_kernel.replace_base_kernel(base_kernel)
        for d, val in enumerate(mi):
            for _ in range(val):
                knl = AxisSourceDerivative(d, knl)
                c *= -1
        result += int_g.copy(target_kernel=base_kernel, source_kernels=(knl,),
                densities=(density,), kernel_arguments=new_kernel_args) * c
    return result


def merge_int_g_exprs(exprs, base_kernel=None, verbose=False,
        source_dependent_variables=None):
    replacements = {}

    if base_kernel is not None:
        mapper = GetIntGs()
        [mapper(expr) for expr in exprs]
        int_g_s = mapper.int_g_s
        for int_g in int_g_s:
            new_int_g = _convert_target_deriv_to_source(int_g)
            tgt_knl = new_int_g.target_kernel
            if isinstance(tgt_knl, TargetPointMultiplier) \
                    and not isinstance(tgt_knl.inner_kernel, KernelWrapper):
                new_int_g_s = _convert_target_multiplier_to_source(new_int_g)
            else:
                new_int_g_s = [new_int_g]
            replacements[int_g] = sum(convert_int_g_to_base(new_int_g,
                base_kernel, verbose=verbose) for new_int_g in new_int_g_s)

    result_coeffs = []
    result_int_gs = []

    for expr in exprs:
        if not have_int_g_s(expr):
            result_coeffs.append(expr)
            result_int_gs.append(0)
        try:
            result_coeff, result_int_g = _merge_int_g_expr(expr, replacements)
            result_int_g = _convert_axis_source_to_directional_source(result_int_g)
            result_int_g = result_int_g.copy(
                    densities=_simplify_densities(result_int_g.densities))
            result_coeffs.append(result_coeff)
            result_int_gs.append(result_int_g)
        except AssertionError:
            result_coeffs.append(expr)
            result_int_gs.append(0)

    if source_dependent_variables is not None:
        result_int_gs = _reduce_number_of_fmms(result_int_gs,
                source_dependent_variables)
    result = [coeff + int_g for coeff, int_g in zip(result_coeffs, result_int_gs)]
    return np.array(result, dtype=object)


def _convert_kernel_to_poly(kernel, axis_vars):
    if isinstance(kernel, AxisTargetDerivative):
        poly = _convert_kernel_to_poly(kernel.inner_kernel, axis_vars)
        return axis_vars[kernel.axis]*poly
    elif isinstance(kernel, AxisSourceDerivative):
        poly = _convert_kernel_to_poly(kernel.inner_kernel, axis_vars)
        return -axis_vars[kernel.axis]*poly
    return 1


def _convert_source_poly_to_int_g(poly, orig_int_g, axis_vars):
    from pymbolic.interop.sympy import SympyToPymbolicMapper
    to_pymbolic = SympyToPymbolicMapper()

    orig_kernel = orig_int_g.source_kernels[0]
    source_kernels = []
    densities = []
    for monom, coeff in poly.terms():
        kernel = orig_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisSourceDerivative(idim, kernel)
        source_kernels.append(kernel)
        densities.append(to_pymbolic(coeff) * (-1)**sum(monom))
    return orig_int_g.copy(source_kernels=tuple(source_kernels),
            densities=tuple(densities))


def _convert_target_poly_to_int_g(poly, orig_int_g, rhs_int_g):
    from pymbolic.interop.sympy import SympyToPymbolicMapper
    to_pymbolic = SympyToPymbolicMapper()

    result = 0
    for monom, coeff in poly.terms():
        kernel = orig_int_g.target_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisTargetDerivative(idim, kernel)
        result += orig_int_g.copy(target_kernel=kernel,
                source_kernels=rhs_int_g.source_kernels,
                densities=rhs_int_g.densities) * to_pymbolic(coeff)

    return result


def _syzygy_module(m, gens):
    import sympy
    from sympy.polys.orderings import grevlex

    def _convert_to_matrix(module, *gens):
        import sympy
        result = []
        for syzygy in module:
            row = []
            for dmp in syzygy.data:
                row.append(sympy.Poly(dmp.to_dict(), *gens,
                    domain=sympy.EX).as_expr())
            result.append(row)
        return sympy.Matrix(result)

    ring = sympy.EX.old_poly_ring(*gens, order=grevlex)
    column_ideals = [ring.free_module(1).submodule(*m[:, i].tolist(), order=grevlex)
                for i in range(m.shape[1])]
    column_syzygy_modules = [ideal.syzygy_module() for ideal in column_ideals]

    intersection = column_syzygy_modules[0]
    for i in range(1, len(column_syzygy_modules)):
        intersection = intersection.intersect(column_syzygy_modules[i])

    m2 = intersection._groebner_vec()
    m3 = _convert_to_matrix(m2, *gens)
    return m3


def _factor_left(mat, axis_vars):
    return _syzygy_module(_syzygy_module(mat, axis_vars).T, axis_vars).T


def _factor_right(mat, factor_left):
    import sympy
    ys = sympy.symbols(f"_y{{0:{factor_left.shape[1]}}}")
    factor_right = []
    for i in range(mat.shape[1]):
        aug_mat = sympy.zeros(factor_left.shape[0], factor_left.shape[1] + 1)
        aug_mat[:, :factor_left.shape[1]] = factor_left
        aug_mat[:, factor_left.shape[1]] = mat[:, i]
        res_map = sympy.solve_linear_system(aug_mat, *ys)
        row = []
        for y in ys:
            row.append(res_map[y])
        factor_right.append(row)
    factor_right = sympy.Matrix(factor_right).T
    return factor_right


def _check_int_gs_common(int_gs):
    base_kernel = int_gs[0].source_kernels[0].get_base_kernel()
    common_int_g = int_gs[0].copy(target_kernel=base_kernel,
            source_kernels=(base_kernel,), densities=(1,))
    for int_g in int_gs:
        for source_kernel in int_g.source_kernels:
            if source_kernel.get_base_kernel() != base_kernel:
                return False
        if common_int_g != int_g.copy(target_kernel=base_kernel,
                source_kernels=(base_kernel,), densities=(1,)):
            return False
    return True


def _reduce_number_of_fmms(int_gs, source_dependent_variables):
    from pymbolic.interop.sympy import PymbolicToSympyMapper
    import sympy

    source_exprs = []
    mapper = ConvertDensityToSourceExprCoeffMap(source_dependent_variables)
    matrix = []
    dim = int_gs[0].target_kernel.dim
    axis_vars = sympy.symbols(f"_x0:{dim}")
    to_sympy = PymbolicToSympyMapper()

    _check_int_gs_common(int_gs)

    for int_g in int_gs:
        row = [0]*len(source_exprs)
        for density, source_kernel in zip(int_g.densities, int_g.source_kernels):
            try:
                d = mapper(density)
            except ImportError:
                return int_gs
            for source_expr, coeff in d.items():
                if source_expr not in source_exprs:
                    source_exprs.append(source_expr)
                    row += [0]
                poly = _convert_kernel_to_poly(source_kernel, axis_vars)
                row[source_exprs.index(source_expr)] += poly * to_sympy(coeff)
        matrix.append(row)

    for row in matrix:
        row += [0]*(len(source_exprs) - len(row))

    mat = sympy.Matrix(matrix)
    lhs_mat = _factor_left(mat, axis_vars)
    rhs_mat = _factor_right(mat, lhs_mat)

    if rhs_mat.shape[0] >= mat.shape[0]:
        return int_gs

    rhs_mat = rhs_mat.applyfunc(lambda x: x.as_poly(*axis_vars, domain=sympy.EX))
    lhs_mat = lhs_mat.applyfunc(lambda x: x.as_poly(*axis_vars, domain=sympy.EX))

    base_kernel = int_gs[0].source_kernels[0].get_base_kernel()
    base_int_g = int_gs[0].copy(target_kernel=base_kernel,
            source_kernels=(base_kernel,), densities=(1,))
    rhs_mat_int_gs = [[_convert_source_poly_to_int_g(poly, base_int_g, axis_vars)
            for poly in row] for row in rhs_mat.tolist()]

    rhs_int_gs = []
    for i in range(rhs_mat.shape[0]):
        source_kernels = []
        densities = []
        for j in range(rhs_mat.shape[1]):
            new_densities = [density * source_exprs[j] for density in
                    rhs_mat_int_gs[i][j].densities]
            source_kernels.extend(rhs_mat_int_gs[i][j].source_kernels)
            densities.extend(new_densities)
        rhs_int_gs.append(rhs_mat_int_gs[i][0].copy(
            source_kernels=tuple(source_kernels), densities=tuple(densities)))

    res = [0]*lhs_mat.shape[0]
    for i in range(lhs_mat.shape[0]):
        for j in range(lhs_mat.shape[1]):
            res[i] += _convert_target_poly_to_int_g(lhs_mat[i, j],
                    int_gs[i], rhs_int_gs[j])

    return res


class ConvertDensityToSourceExprCoeffMap(Mapper):
    def __init__(self, source_dependent_variables):
        self.source_dependent_variables = source_dependent_variables

    def __call__(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        try:
            return super().__call__(expr)
        except NotImplementedError:
            return {1: expr}

    rec = __call__

    def map_sum(self, expr):
        d = defaultdict(lambda: 0)
        for child in expr.children:
            d_child = self.rec(child)
            for k, v in d_child.items():
                d[k] += v
        return dict(d)

    def map_product(self, expr):
        if len(expr.children) > 2:
            left = Product(tuple(expr.children[:2]))
            right = Product(tuple(expr.children[2:]))
            new_prod = Product((left, right))
            return self.rec(new_prod)
        elif len(expr.children) == 1:
            return self.rec(expr.children[0])
        elif len(expr.children) == 0:
            return {1: 1}
        left, right = expr.children
        d_left = self.rec(left)
        d_right = self.rec(right)
        d = defaultdict(lambda: 0)
        for k_left, v_left in d_left.items():
            for k_right, v_right in d_right.items():
                d[k_left*k_right] += v_left*v_right
        return dict(d)

    def map_power(self, expr):
        d_base = self.rec(expr.base)
        d_exponent = self.rec(expr.exponent)
        if len(d_exponent) > 1:
            raise ValueError
        exp_k, exp_v = list(d_exponent.items())[0]
        if exp_k != 1:
            raise ValueError
        for k in d_base.keys():
            d_base[k] = d_base[k]**exp_v
        return d_base

    def map_quotient(self, expr):
        d_num = self.rec(expr.numerator)
        d_den = self.rec(expr.denominator)
        if len(d_den) > 1:
            raise ValueError
        den_k, den_v = list(d_den.items())[0]
        if den_k != 1:
            raise ValueError
        for k in d_num.keys():
            d_num[k] /= den_v
        return d_num


def _convert_target_multiplier_to_source(int_g):
    from sumpy.symbolic import SympyToPymbolicMapper
    tgt_knl = int_g.target_kernel
    if not isinstance(tgt_knl, TargetPointMultiplier):
        return int_g
    if isinstance(tgt_knl.inner_kernel, KernelWrapper):
        return int_g

    new_kernel_args = _filter_kernel_arguments([tgt_knl], int_g.kernel_arguments)
    result = []
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


def _convert_target_deriv_to_source(int_g):
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


def _convert_axis_source_to_directional_source(int_g):
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
    name = "generated_dir_vec"
    kernel_arguments[name] = \
            np.array(int_g.densities, dtype=np.object)
    res = int_g.copy(
            source_kernels=(
                DirectionalSourceDerivative(base_knl, dir_vec_name=name),),
            densities=(1,),
            kernel_arguments=kernel_arguments)
    return res


def _merge_source_kernel_duplicates(source_kernels, densities):
    new_source_kernels = []
    new_densities = []
    for knl, density in zip(source_kernels, densities):
        if knl not in new_source_kernels:
            new_source_kernels.append(knl)
            new_densities.append(density)
        else:
            idx = new_source_kernels.index(knl)
            new_densities[idx] += density
    return new_source_kernels, new_densities


def _merge_kernel_arguments(x, y):
    res = x.copy()
    for k, v in y.items():
        if k in res:
            assert res[k] == v
        else:
            res[k] = v
    return res


def _simplify_densities(densities):
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


def _merge_int_g_expr(expr, replacements):
    if isinstance(expr, Sum):
        result_coeff = 0
        result_int_g = 0
        for c in expr.children:
            coeff, int_g = _merge_int_g_expr(c, replacements)
            result_coeff += coeff
            if int_g == 0:
                continue
            if result_int_g == 0:
                result_int_g = int_g
                continue
            assert result_int_g.source == int_g.source
            assert result_int_g.target == int_g.target
            assert result_int_g.qbx_forced_limit == int_g.qbx_forced_limit
            assert result_int_g.target_kernel == int_g.target_kernel
            kernel_arguments = _merge_kernel_arguments(result_int_g.kernel_arguments,
                    int_g.kernel_arguments)
            source_kernels = result_int_g.source_kernels + int_g.source_kernels
            densities = result_int_g.densities + int_g.densities
            new_source_kernels, new_densities = source_kernels, densities
            # new_source_kernels, new_densities = \
            #        _merge_source_kernel_duplicates(source_kernels, densities)
            result_int_g = result_int_g.copy(
                source_kernels=tuple(new_source_kernels),
                densities=tuple(new_densities),
                kernel_arguments=kernel_arguments,
            )
        return result_coeff, result_int_g
    elif isinstance(expr, Product):
        mult = 1
        found_int_g = None
        for c in expr.children:
            if not have_int_g_s(c):
                mult *= c
            elif found_int_g:
                raise RuntimeError("Not a linear expression.")
            else:
                found_int_g = c
        if not found_int_g:
            return expr, 0
        else:
            coeff, new_int_g = _merge_int_g_expr(found_int_g, replacements)
            new_densities = (density * mult for density in new_int_g.densities)
            return coeff*mult, new_int_g.copy(densities=new_densities)
    elif isinstance(expr, IntG):
        new_expr = replacements.get(expr, expr)
        if new_expr == expr:
            new_int_g = _convert_target_deriv_to_source(expr)
            return 0, new_int_g
        else:
            return _merge_int_g_expr(new_expr, replacements)
    elif isinstance(expr, Quotient):
        mult = 1/expr.denominator
        coeff, new_int_g = _merge_int_g_expr(expr.numerator, replacements)
        new_densities = (density * mult for density in new_int_g.densities)
        return coeff * mult, new_int_g.copy(densities=new_densities)
    else:
        return expr, 0


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
    """density = pytential.sym.make_sym_vector("d", 1)[0]
    int_g_1 = int_g_vec(TargetPointMultiplier(2, AxisTargetDerivative(2,
            AxisSourceDerivative(1, AxisSourceDerivative(0,
                LaplaceKernel(3))))), density, qbx_forced_limit=1)
    int_g_2 = int_g_vec(TargetPointMultiplier(0, AxisTargetDerivative(0,
        AxisSourceDerivative(0, AxisSourceDerivative(0,
            LaplaceKernel(3))))), density, qbx_forced_limit=1)
    print(merge_int_g_exprs([int_g_1, int_g_2],
        base_kernel=BiharmonicKernel(3), verbose=True)[0])"""
