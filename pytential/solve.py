__copyright__ = "Copyright (C) 2012-2013 Andreas Kloeckner"

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

__doc__ = """

.. autofunction:: gmres

.. autoclass:: GMRESResult()

.. autoexception:: GMRESError

.. autoclass:: ResidualPrinter
"""


import numpy as np
from numbers import Number
import pyopencl as cl
import pyopencl.array  # noqa
from meshmode.dof_array import obj_or_dof_array_vectorize_n_args, DOFArray
from pytools.obj_array import obj_array_vectorize_n_args


def structured_vdot(x, y):
    # vdot() implementation that is aware of scalars and host or
    # PyOpenCL arrays. It also recurses down nested object arrays.
    if (isinstance(x, Number)
            or (isinstance(x, np.ndarray) and x.dtype.char != "O")):
        return np.vdot(x, y)
    elif isinstance(x, cl.array.Array):
        return cl.array.vdot(x, y).get()
    elif isinstance(x, np.ndarray) and x.dtype.char == "O":
        return np.sum(obj_array_vectorize_n_args(structured_vdot, x, y))
    elif isinstance(x, DOFArray):
        return sum(obj_or_dof_array_vectorize_n_args(structured_vdot, x, y))


# {{{ gmres

# Modified Python port of ./Apps/Acoustics/root/matlab/gmres_restart.m
# from hellskitchen.
# Necessary because SciPy gmres is not reentrant and thus does
# not allow recursive solves.

from pytools import Record


class GMRESError(RuntimeError):
    pass


# {{{ main routine

class GMRESResult(Record):
    """
    .. attribute:: solution
    .. attribute:: residual_norms
    .. attribute:: iteration_count
    .. attribute:: success

        a :class:`bool` indicating whether the iteration succeeded

    .. attribute:: state

        a verbal description of the outcome of the iteration
    """


def _gmres(A, b, restart=None, tol=None, x0=None, dot=None,  # noqa
        maxiter=None, hard_failure=None, require_monotonicity=True,
        no_progress_factor=None, stall_iterations=None,
        callback=None):

    # {{{ input processing

    n, _ = A.shape

    if not callable(A):
        a_call = A.matvec
    else:
        a_call = A

    if restart is None:
        restart = min(n, 20)

    if tol is None:
        tol = 1e-5

    if maxiter is None:
        maxiter = 2*n

    if hard_failure is None:
        hard_failure = True

    if stall_iterations is None:
        stall_iterations = 10
    if no_progress_factor is None:
        no_progress_factor = 1.25

    # }}}

    def norm(x):
        return np.sqrt(abs(dot(x, x)))

    if x0 is None:
        x = 0*b
        r = b
        recalc_r = False
    else:
        x = x0
        del x0
        recalc_r = True

    Ae = [None]*restart  # noqa
    e = [None]*restart

    k = 0

    norm_b = norm(b)
    last_resid_norm = None
    residual_norms = []

    for iteration in range(maxiter):
        # restart if required
        if k == restart:
            k = 0
            orth_count = restart
        else:
            orth_count = k

        # recalculate residual every 10 steps
        if recalc_r:
            r = b - a_call(x)

        norm_r = norm(r)
        residual_norms.append(norm_r)

        if callback is not None:
            callback(r)

        if norm_r < tol*norm_b or norm_r == 0:
            return GMRESResult(solution=x,
                    residual_norms=residual_norms,
                    iteration_count=iteration, success=True,
                    state="success")
        if last_resid_norm is not None:
            if norm_r > 1.25*last_resid_norm:
                state = "non-monotonic residuals"
                if require_monotonicity:
                    if hard_failure:
                        raise GMRESError(state)
                    else:
                        return GMRESResult(solution=x,
                                residual_norms=residual_norms,
                                iteration_count=iteration, success=False,
                                state=state)
                else:
                    print("*** WARNING: non-monotonic residuals in GMRES")

            if (stall_iterations
                    and len(residual_norms) > stall_iterations
                    and norm_r > (
                        residual_norms[-stall_iterations]  # noqa pylint:disable=invalid-unary-operand-type
                        / no_progress_factor)):

                state = "stalled"
                if hard_failure:
                    raise GMRESError(state)
                else:
                    return GMRESResult(solution=x,
                            residual_norms=residual_norms,
                            iteration_count=iteration, success=False,
                            state=state)

        last_resid_norm = norm_r

        # initial new direction guess
        w = a_call(r)

        # {{{ double-orthogonalize the new direction against preceding ones

        rp = r

        for _orth_trips in range(2):
            for j in range(0, orth_count):
                d = dot(Ae[j], w)
                w = w - d * Ae[j]
                rp = rp - d * e[j]

            # normalize
            d = 1/norm(w)
            w = d*w
            rp = d*rp

        # }}}

        Ae[k] = w
        e[k] = rp

        # update the residual and solution
        d = dot(Ae[k], r)

        recalc_r = (iteration+1) % 10 == 0
        if not recalc_r:
            r = r - d*Ae[k]

        x = x + d*e[k]

        k += 1

    state = "max iterations"
    if hard_failure:
        raise GMRESError(state)
    else:
        return GMRESResult(solution=x,
                residual_norms=residual_norms,
                iteration_count=iteration, success=False,
                state=state)

# }}}


# {{{ progress reporting

class ResidualPrinter:
    def __init__(self, inner_product=structured_vdot):
        self.count = 0
        self.inner_product = inner_product

    def __call__(self, resid):
        import sys
        if resid is not None:
            norm = np.sqrt(self.inner_product(resid, resid))
            sys.stdout.write(f"IT {self.count:8d} {abs(norm):.8e}\n")
        else:
            sys.stdout.write(f"IT {self.count:8d}\n")
        self.count += 1
        sys.stdout.flush()

# }}}


# {{{ entrypoint

def gmres(op, rhs, restart=None, tol=None, x0=None,
        inner_product=structured_vdot,
        maxiter=None, hard_failure=None,
        no_progress_factor=None, stall_iterations=None,
        callback=None, progress=False, require_monotonicity=True):
    """Solve a linear system Ax=b by means of GMRES
    with restarts.

    :arg op: a callable to evaluate A(x)
    :arg b: the right hand side
    :arg restart: the maximum number of iteration after
       which GMRES algorithm needs to be restarted
    :arg tol: the required decrease in residual norm
    :arg inner_product: Must have an interface compatible with
        :func:`numpy.vdot`. Must return a host scalar.
    :arg maxiter: the maximum number of iteration permitted
    :arg hard_failure: If True, raise :exc:`GMRESError` in case of failure.
    :arg stall_iterations: Number of iterations with residual decrease
        below *no_progress_factor* indicates stall. Set to 0 to disable
        stall detection.

    :return: a :class:`GMRESResult`
    """
    if callback is None:
        if progress:
            callback = ResidualPrinter(inner_product)
        else:
            callback = None

    result = _gmres(op, rhs, restart=restart, tol=tol, x0=x0,
            dot=inner_product,
            maxiter=maxiter, hard_failure=hard_failure,
            no_progress_factor=no_progress_factor,
            stall_iterations=stall_iterations, callback=callback,
            require_monotonicity=require_monotonicity)

    return result

# }}}

# }}}


# {{{ direct solve

def lu(op, rhs, show_spectrum=False):
    import numpy.linalg as la

    from sumpy.tools import build_matrix
    mat = build_matrix(op)

    print(f"condition number: {la.cond(mat)}")
    if show_spectrum:
        ev = la.eigvals(mat)
        import matplotlib.pyplot as pt
        pt.plot(ev.real, ev.imag, "o")
        pt.show()

    return la.solve(mat, rhs)

# }}}

# vim: fdm=marker
