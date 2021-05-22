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

from pymbolic.mapper import WalkMapper
from pymbolic.primitives import Sum, Product, Quotient
from pytential.symbolic.primitives import IntG, NodeCoordinateComponent, int_g_vec
import pytential


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

    if order == pde.degree:
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

    for mi, c in deriv_relation[1]:
        knl = source_kernel.replace_base_kernel(base_kernel)
        for d, val in enumerate(mi):
            for _ in range(val):
                knl = AxisSourceDerivative(d, knl)
                c *= -1
        result += int_g.copy(target_kernel=base_kernel, source_kernels=(knl,),
                densities=(density,)) * c
    return result


def merge_int_g_exprs(exprs, base_kernel=None, verbose=False):
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

    return np.array([merge_int_g_expr(expr, replacements) for expr in exprs])


def _convert_target_multiplier_to_source(int_g):
    from sumpy.symbolic import SympyToPymbolicMapper
    tgt_knl = int_g.target_kernel
    if not isinstance(tgt_knl, TargetPointMultiplier):
        return int_g
    if isinstance(tgt_knl.inner_kernel, KernelWrapper):
        return int_g
    result = []
    # x G = y*G + (x - y)*G
    # For y*G, absorb y into a density
    new_densities = [density*NodeCoordinateComponent(tgt_knl.axis)
            for density in int_g.densities]
    result.append(int_g.copy(target_kernel=tgt_knl.inner_kernel,
                densities=tuple(new_densities)))

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
            source_kernels=(new_knl,)))
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


def merge_int_g_expr(expr, replacements):
    if not have_int_g_s(expr):
        return expr
    try:
        result_coeff, result_int_g = _merge_int_g_expr(expr, replacements)
        result_int_g = _convert_axis_source_to_directional_source(result_int_g)
        result_int_g = result_int_g.copy(
                densities=_simplify_densities(result_int_g.densities))
        return result_coeff + result_int_g
    except AssertionError:
        return expr


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
            new_source_kernels, new_densities = \
                    _merge_source_kernel_duplicates(source_kernels, densities)
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
    from sumpy.kernel import (StokesletKernel, BiharmonicKernel, StressletKernel,
            ElasticityKernel, LaplaceKernel)
    base_kernel = BiharmonicKernel(3)
    kernels = [StokesletKernel(3, 0, 1), StokesletKernel(3, 0, 0)]
    kernels += [StressletKernel(3, 0, 1, 0), StressletKernel(3, 0, 0, 0),
            StressletKernel(3, 0, 1, 2)]
    kernels += [ElasticityKernel(3, 0, 1, poisson_ratio="0.4"),
            ElasticityKernel(3, 0, 0, poisson_ratio="0.4")]
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=2, verbose=True)
    density = pytential.sym.make_sym_vector("d", 1)[0]
    int_g_1 = int_g_vec(TargetPointMultiplier(2, AxisTargetDerivative(2,
            AxisSourceDerivative(1, AxisSourceDerivative(0,
                LaplaceKernel(3))))), density, qbx_forced_limit=1)
    int_g_2 = int_g_vec(TargetPointMultiplier(0, AxisTargetDerivative(0,
        AxisSourceDerivative(0, AxisSourceDerivative(0,
            LaplaceKernel(3))))), density, qbx_forced_limit=1)
    print(merge_int_g_exprs([int_g_1, int_g_2],
        base_kernel=BiharmonicKernel(3), verbose=True)[0])
