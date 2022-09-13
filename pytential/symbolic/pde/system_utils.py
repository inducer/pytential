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

import sumpy.symbolic as sym
import pymbolic
from sumpy.kernel import (AxisTargetDerivative, AxisSourceDerivative,
    ExpressionKernel, KernelWrapper, TargetPointMultiplier,
    DirectionalSourceDerivative)
from pytools import (memoize_on_first_arg,
    generate_nonnegative_integer_tuples_summing_to_at_most
    as gnitstam)

from pytential.symbolic.primitives import (NodeCoordinateComponent,
    hashable_kernel_args, IntG)
from pytential.symbolic.mappers import IdentityMapper
from pytential.utils import chop, lu_solve_with_expand
import pytential

from typing import List, Mapping, Text, Any, Union, Tuple, Optional
from pytential.symbolic.typing import ExpressionT

import logging
logger = logging.getLogger(__name__)

__all__ = (
    "rewrite_using_base_kernel",
    "get_deriv_relation",
    )

__doc__ = """
.. autofunction:: rewrite_using_base_kernel
.. autofunction:: get_deriv_relation
"""


# {{{ rewrite_using_base_kernel

_NO_ARG_SENTINEL = object()


def rewrite_using_base_kernel(exprs: List[ExpressionT],
        base_kernel=_NO_ARG_SENTINEL) -> List[ExpressionT]:
    """
    Rewrites a list of expressions with :class:`~pytential.symbolic.primitives.IntG`
    objects using *base_kernel*.

    For example, if *base_kernel* is the biharmonic kernel, and a Laplace kernel
    is encountered, this will (forcibly) rewrite the Laplace kernel in terms of
    derivatives of the Biharmonic kernel.

    If *base_kernel* is *None*, the expression list is returned as is.
    If *base_kernel* is not given, the method will try to find a base kernel.
    This is currently not implemented and will raise a ``NotImplementedError``.

    The routine will fail with a ``RuntimeError`` when *base_kernel* is given, but
    method fails to find a way to rewrite.
    """
    if base_kernel is None:
        return list(exprs)
    if base_kernel == _NO_ARG_SENTINEL:
        raise NotImplementedError
    mapper = RewriteUsingBaseKernelMapper(base_kernel)
    return [mapper(expr) for expr in exprs]


class RewriteUsingBaseKernelMapper(IdentityMapper):
    """Rewrites ``IntG``s using the base kernel. First this method replaces
    ``IntG``s with :class:`sumpy.kernel.AxisTargetDerivative` to ``IntG``s
    :class:`sumpy.kernel.AxisSourceDerivative` and
    ``IntG``s with :class:`sumpy.kernel.TargetPointMultiplier` to ``IntG``s
    without them using :class:`sumpy.kernel.ExpressionKernel`
    and then converts them to the base kernel by finding
    a relationship between the derivatives.
    """
    def __init__(self, base_kernel):
        self.base_kernel = base_kernel

    def map_int_g(self, expr):
        # Convert IntGs with TargetPointMultiplier/AxisTargetDerivative to a sum of
        # IntGs without TargetPointMultipliers
        new_int_gs = convert_target_transformation_to_source(expr)
        # Convert IntGs with different kernels to expressions containing
        # IntGs with base_kernel or its derivatives
        return sum(convert_int_g_to_base(new_int_g,
            self.base_kernel) for new_int_g in new_int_gs)


def _get_sympy_kernel_expression(expr: ExpressionT,
        kernel_arguments: Mapping[Text, Any]) -> sym.Basic:
    """Convert a :mod:`pymbolic` expression to :mod:`sympy` expression
    after substituting kernel arguments.

    For eg: `exp(I*k*r)/r` with `{k: 1}` is converted to the sympy expression
    `exp(I*r)/r`
    """
    from pymbolic.mapper.substitutor import substitute
    from sumpy.symbolic import PymbolicToSympyMapperWithSymbols

    pymbolic_expr = substitute(expr, kernel_arguments)

    res = PymbolicToSympyMapperWithSymbols()(pymbolic_expr)
    return res


def _monom_to_expr(monom: List[int],
        variables: List[Union[sym.Basic, ExpressionT]]) \
        -> Union[sym.Basic, ExpressionT]:
    """Convert a monomial to an expression using given variables.

    For eg: [3, 2, 1] with variables [x, y, z] is converted to x^3 y^2 z.
    """
    prod = 1
    for i, nrepeats in enumerate(monom):
        for _ in range(nrepeats):
            prod *= variables[i]
    return prod


def convert_target_transformation_to_source(int_g: IntG) -> List[IntG]:
    """Convert an ``IntG`` with :class:`sumpy.kernel.AxisTargetDerivative`
    or :class:`sumpy.kernel.TargetPointMultiplier` to a list
    of ``IntG`` s without them and only source dependent transformations.
    The sum of the list returned is equivalent to the input *int_g*.

    For example::

       IntG(d/dx r, sigma) -> [IntG(d/dy r, -sigma)]
       IntG(x*r, sigma) -> [IntG(r, sigma*y), IntG(r*(x -y), sigma)]
    """
    import sympy
    from pymbolic.interop.sympy import SympyToPymbolicMapper
    conv = SympyToPymbolicMapper()

    knl = int_g.target_kernel
    # we use a symbol for d = (x - y)
    ds = sympy.symbols(f"d0:{knl.dim}")
    sources = sympy.symbols(f"y0:{knl.dim}")
    # instead of just x, we use x = (d + y)
    targets = [d + source for d, source in zip(ds, sources)]
    orig_expr = sympy.Function("f")(*ds)  # pylint: disable=not-callable
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
            found = True
        else:
            return [int_g]
        knl = knl.inner_kernel

    if not found:
        return [int_g]

    int_g = int_g.copy(target_kernel=knl)

    sources_pymbolic = [NodeCoordinateComponent(i) for i in range(knl.dim)]
    expr = expr.expand()
    # Now the expr is an Add and looks like
    # u_{d[0], d[1]}(d, y)*d[0]*y[1] + u(d, y) * d[1]
    # or a single term like u(d, y) * d[1]
    if isinstance(expr, sympy.Add):
        args = expr.args
    else:
        args = [expr]

    result = []
    for arg in args:
        deriv_terms = arg.atoms(sympy.Derivative)
        if len(deriv_terms) == 1:
            # for eg: we have a term like u_{d[0], d[1]}(d, y) * d[0] * y[1]
            # deriv_term is u_{d[0], d[1]}
            deriv_term = deriv_terms.pop()
            # eg: rest_terms is d[0] * y[1]
            rest_terms = sympy.Poly(arg.xreplace({deriv_term: 1}), *ds, *sources)
            # eg: derivatives is (d[0], 1), (d[1], 1)
            derivatives = deriv_term.args[1:]
        elif len(deriv_terms) == 0:
            # for eg: we have a term like u(d, y) * d[1]
            # rest_terms = d[1]
            rest_terms = sympy.Poly(arg.xreplace({orig_expr: 1}), *ds, *sources)
            derivatives = [(d, 0) for d in ds]
        else:
            raise AssertionError("impossible condition")

        # apply the derivatives
        new_source_kernels = []
        for source_kernel in int_g.source_kernels:
            knl = source_kernel
            for axis_var, nrepeats in derivatives:
                axis = ds.index(axis_var)
                for _ in range(nrepeats):
                    knl = AxisSourceDerivative(axis, knl)
            new_source_kernels.append(knl)
        new_int_g = int_g.copy(source_kernels=new_source_kernels)

        (monom, coeff,) = rest_terms.terms()[0]
        # Now from d[0]*y[1], we separate the two terms
        # d terms end up in the expression and y terms end up in the density
        d_terms, y_terms = monom[:len(ds)], monom[len(ds):]
        expr_multiplier = _monom_to_expr(d_terms, ds)
        density_multiplier = _monom_to_expr(y_terms, sources_pymbolic) \
                * conv(coeff)
        # since d/d(d) = - d/d(y), we multiply by -1 to get source derivatives
        density_multiplier *= (-1)**int(sum(nrepeats for _, nrepeats in derivatives))
        new_int_gs = _multiply_int_g(new_int_g, sym.sympify(expr_multiplier),
                density_multiplier)
        result.extend(new_int_gs)
    return result


def _multiply_int_g(int_g: IntG, expr_multiplier: sym.Basic,
        density_multiplier: ExpressionT) -> List[IntG]:
    """Multiply the expression in ``IntG`` with the *expr_multiplier*
    which is a symbolic (:mod:`sympy` or :mod:`symengine`) expression and
    multiply the densities with *density_multiplier* which is a :mod:`pymbolic`
    expression.
    """
    result = []

    base_kernel = int_g.target_kernel.get_base_kernel()
    sym_d = sym.make_sym_vector("d", base_kernel.dim)
    base_kernel_expr = _get_sympy_kernel_expression(base_kernel.expression,
            int_g.kernel_arguments)
    subst = {pymbolic.var(f"d{i}"): pymbolic.var("d")[i] for i in
            range(base_kernel.dim)}
    conv = sym.SympyToPymbolicMapper()

    if expr_multiplier == 1:
        # if there's no expr_multiplier, only multiply the densities
        return [int_g.copy(densities=tuple(density*density_multiplier
            for density in int_g.densities))]

    for knl, density in zip(int_g.source_kernels, int_g.densities):
        if expr_multiplier == 1:
            new_knl = knl.get_base_kernel()
        else:
            new_expr = conv(knl.postprocess_at_source(base_kernel_expr, sym_d)
                    * expr_multiplier)
            new_expr = pymbolic.substitute(new_expr, subst)
            new_knl = ExpressionKernel(knl.dim, new_expr,
                knl.get_base_kernel().global_scaling_const,
                knl.is_complex_valued)
        result.append(int_g.copy(target_kernel=new_knl,
            densities=(density*density_multiplier,),
            source_kernels=(new_knl,)))
    return result


def convert_int_g_to_base(int_g: IntG, base_kernel: ExpressionKernel) \
        -> ExpressionT:
    """Converts an *IntG* to an expression with *IntG*s having the
    base kernel *base_kernel*.
    """
    result = 0
    for knl, density in zip(int_g.source_kernels, int_g.densities):
        result += _convert_int_g_to_base(
                int_g.copy(source_kernels=(knl,), densities=(density,)),
                base_kernel)
    return result


def _convert_int_g_to_base(int_g: IntG, base_kernel: ExpressionKernel) \
        -> ExpressionT:
    """Converts an *IntG* with only one source kernel to an expression with *IntG*s
    having the base kernel *base_kernel*.
    """
    target_kernel = int_g.target_kernel.replace_base_kernel(base_kernel)
    dim = target_kernel.dim

    result = 0

    density, = int_g.densities
    source_kernel, = int_g.source_kernels
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
        return int_g

    if const != 0 and source_kernel != source_kernel.get_base_kernel():
        # We only handle the case where any source transformation is a derivative
        # and the constant when applied becomes zero. We bail out if not
        knl = source_kernel
        while isinstance(knl, KernelWrapper):
            if not isinstance(knl,
                              (AxisSourceDerivative, DirectionalSourceDerivative)):
                return int_g
            knl = knl.inner_kernel
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


def get_deriv_relation(kernels: List[ExpressionKernel],
        base_kernel: ExpressionKernel,
        kernel_arguments: Mapping[Text, Any],
        tol: float = 1e-10,
        order: Optional[int] = None) \
        -> List[Tuple[ExpressionT, List[Tuple[Tuple[int], ExpressionT]]]]:
    res = []
    for knl in kernels:
        res.append(get_deriv_relation_kernel(knl, base_kernel,
            hashable_kernel_arguments=hashable_kernel_args(kernel_arguments),
            tol=tol, order=order))
    return res


@memoize_on_first_arg
def get_deriv_relation_kernel(kernel: ExpressionKernel,
        base_kernel: ExpressionKernel,
        hashable_kernel_arguments: Tuple[Tuple[Text, Any], ...],
        tol: float = 1e-10,
        order: Optional[int] = None) \
        -> Tuple[ExpressionT, List[Tuple[Tuple[int, ...], ExpressionT]]]:
    """Takes a *kernel* and a base_kernel* as input and re-writes the
    *kernel* as a linear combination of derivatives of *base_kernel* up-to
    order *order* and a constant. *tol* is an upper limit for small numbers that
    are replaced with zero in the numerical procedure.

    :returns: the constant and a list of (multi-index, coeff) to represent the
              linear combination of derivatives
    """
    kernel_arguments = dict(hashable_kernel_arguments)
    (L, U, perm), rand, mis = _get_base_kernel_matrix(base_kernel, order=order,
            hashable_kernel_arguments=hashable_kernel_arguments)
    dim = base_kernel.dim
    sym_vec = sym.make_sym_vector("d", dim)
    sympy_conv = sym.SympyToPymbolicMapper()

    expr = _get_sympy_kernel_expression(kernel.expression, kernel_arguments)
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
            coeff *= _get_sympy_kernel_expression(kernel.global_scaling_const,
                    kernel_arguments)
            coeff /= _get_sympy_kernel_expression(base_kernel.global_scaling_const,
                    kernel_arguments)
            result.append((mis[i], sympy_conv(coeff)))
            logger.debug("  + %s.diff(%s)*%s", base_kernel, mis[i], coeff)
        else:
            const = sympy_conv(coeff * _get_sympy_kernel_expression(
                kernel.global_scaling_const, kernel_arguments))
    logger.debug("  + %s", const)
    return (const, result)


@memoize_on_first_arg
def _get_base_kernel_matrix(base_kernel: ExpressionKernel,
        hashable_kernel_arguments: Tuple[Tuple[Text, Any], ...],
        order: Optional[int] = None, retries: int = 3) \
        -> Tuple[Tuple[sym.Matrix, sym.Matrix, List[Tuple[int, int]]],
            np.ndarray, List[Tuple[int, ...]]]:
    dim = base_kernel.dim

    pde = base_kernel.get_pde_as_diff_op()
    if order is None:
        order = pde.order

    if order > pde.order:
        raise RuntimeError(f"order ({order}) cannot be greater than the order"
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
    sym_vec = sym.make_sym_vector("d", dim)

    base_expr = _get_sympy_kernel_expression(base_kernel.expression,
        dict(hashable_kernel_arguments))

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
            hashable_kernel_arguments=hashable_kernel_arguments,
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from sumpy.kernel import (StokesletKernel, BiharmonicKernel,  # noqa:F401
        StressletKernel, ElasticityKernel, LaplaceKernel)
    base_kernel = BiharmonicKernel(2)
    #base_kernel = LaplaceKernel(3)
    kernels = [StokesletKernel(2, 0, 1), StokesletKernel(2, 0, 0)]
    kernels += [StressletKernel(2, 0, 0, 0), StressletKernel(2, 0, 0, 1),
            StressletKernel(2, 0, 0, 1), StressletKernel(2, 0, 1, 1)]
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=4, kernel_arguments={})

    base_kernel = BiharmonicKernel(3)
    sym_d = sym.make_sym_vector("d", base_kernel.dim)
    sym_r = sym.sqrt(sum(a**2 for a in sym_d))
    conv = sym.SympyToPymbolicMapper()
    expression_knl = ExpressionKernel(3, conv(sym_d[0]*sym_d[1]/sym_r**3), 1, False)
    expression_knl2 = ExpressionKernel(3, conv(1/sym_r + sym_d[0]*sym_d[0]/sym_r**3),
        1, False)
    kernels = [expression_knl, expression_knl2]
    get_deriv_relation(kernels, base_kernel, tol=1e-10, order=4, kernel_arguments={})
