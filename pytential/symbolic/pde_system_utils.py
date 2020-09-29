import numpy as np

from sumpy.symbolic import Matrix, make_sym_vector, sym

from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)


def _chop(expr, tol):
    nums = expr.atoms(sym.Number)
    replace_dict = {}
    for num in nums:
        if abs(num) < tol:
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


def get_deriv_relation(kernels, base_kernel, tol=1e-10, order=4, verbose=False):
    dim = base_kernel.dim

    mis = sorted(gnitstam(order, dim), key=sum)

    # (-1, -1, -1) represent a constant
    mis.append((-1, -1, -1))

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

    mat = Matrix(mat)
    res = []

    for kernel in kernels:
        expr = kernel.get_expression(sym_vec)
        vec = []
        for a in rand.T:
            vec.append(_n(expr.xreplace(dict((k, v) for k, v in zip(sym_vec, a)))))
        vec = Matrix(vec)
        result = []
        const = 0
        if verbose:
            print(kernel, end=" = ")
        for i, coeff in enumerate(mat.solve(vec)):
            coeff = _chop(coeff, tol)
            if coeff == 0:
                continue
            if mis[i] != (-1, -1, -1):
                coeff *= kernel.get_global_scaling_const()
                coeff /= base_kernel.get_global_scaling_const()
                result.append((mis[i], coeff))
                if verbose:
                    print(f"{base_kernel}.diff({mis[i]})*{coeff}", end=" + ")
            else:
                const = coeff * kernel.get_global_scaling_const()
        if verbose:
            print(const)
        res.append((const, result))

    return res


if __name__ == "__main__":
    from sumpy.kernel import StokesletKernel, BiharmonicKernel, StressletKernel
    base_kernel = BiharmonicKernel(2)
    kernels = [StokesletKernel(2, 0, 1), StokesletKernel(2, 0, 0)]
    kernels = [StressletKernel(2, 0, 1, 0)]
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=3)
