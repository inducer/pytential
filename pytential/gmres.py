from __future__ import division

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


# Modified Python port of ./Apps/Acoustics/root/matlab/gmres_restart.m
# from hellskitchen.
# Necessary because SciPy gmres is not reentrant and thus does
# not allow recursive solves.

import numpy as np
from pytools import Record


class GMRESError(RuntimeError):
    pass


# {{{ main routine

class GMRESResult(Record):
    pass


def _gmres(A, b, restart=None, tol=None, x0=None, inner_product=None,
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

    if inner_product is None:
        dot = np.vdot
    else:
        dot = inner_product
    del inner_product

    def norm(x):
        return np.sqrt(abs(dot(x, x)))

    if x0 is None:
        x = np.zeros(n, A.dtype)
        r = b
        recalc_r = False
    else:
        x = x0
        del x0
        recalc_r = True

    Ae = np.empty((n, restart), A.dtype, order="f")
    e = np.empty((n, restart), A.dtype, order="f")

    k = 0

    norm_b = norm(b)
    last_resid_norm = None
    stall_count = 0
    residual_norms = []

    for iteration in xrange(maxiter):
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

        if abs(norm_r) < tol*norm_b:
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
                    print "*** WARNING: non-monotonic residuals in GMRES"

            if stall_iterations and \
                    norm_r > last_resid_norm/no_progress_factor:
                stall_count += 1

                if stall_count >= stall_iterations:
                    state = "stalled"
                    if hard_failure:
                        raise GMRESError(state)
                    else:
                        return GMRESResult(solution=x,
                                residual_norms=residual_norms,
                                iteration_count=iteration, success=False,
                                state=state)
            else:
                stall_count = 0

        last_resid_norm = norm_r

        # initial new direction guess
        w = a_call(r)

        # {{{ double-orthogonalize the new direction against preceding ones

        rp = r

        for orth_trips in range(2):
            for j in range(0, orth_count):
                d = dot(Ae[:, j], w)
                w = w - d * Ae[:, j]
                rp = rp - d * e[:, j]

            # normalize
            d = 1/norm(w)
            w = d*w
            rp = d*rp

        # }}}

        Ae[:, k] = w
        e[:, k] = rp

        # update the residual and solution
        d = dot(Ae[:, k], r)

        recalc_r = (iteration+1) % 10 == 0
        if not recalc_r:
            r = r - d*Ae[:, k]

        x = x + d*e[:, k]

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
    def __init__(self, inner_product=None):
        self.count = 0
        if inner_product is None:
            inner_product = np.vdot

        self.inner_product = inner_product

    def __call__(self, resid):
        import sys
        if resid is not None:
            norm = np.sqrt(self.inner_product(resid, resid))
            sys.stdout.write("IT %8d %g\n" % (
                self.count, abs(norm)))
        else:
            sys.stdout.write("IT %8d\n" % self.count)
        self.count += 1
        sys.stdout.flush()

# }}}


# {{{ entrypoint

def gmres(op, rhs, restart=None, tol=None, x0=None,
        inner_product=None,
        maxiter=None, hard_failure=None,
        no_progress_factor=None, stall_iterations=None,
        callback=None, progress=False):
    """Solve a linear system Ax=b by means of GMRES
    with restarts.

    :arg A: a callable to evaluate A(x)
    :arg b: the right hand side
    :arg restart: the maximum number of iteration after
       which GMRES algorithm needs to be restarted
    :arg tol: the required decrease in residual norm
    :arg maxiter: the maximum number of iteration permitted
    :arg hard_failure: If True, raise exceptions in case of failure.
    :arg stall_iterations: Number of iterations with residual decrease
        below *no_progress_factor* indicates stall. Set to 0 to disable
        stall detection.

    :return: An object containing attribute *solution*, *residual_norms*,
        *iteration_count*, *success*, *state*, where *state* is a string
        explaining the result.
    """
    from pytools.obj_array import is_obj_array, make_obj_array
    if is_obj_array(rhs):
        stacked_rhs = np.hstack(rhs)

        slices = []
        num_dofs = 0
        for entry in rhs:
            if isinstance(entry, np.ndarray):
                length = len(entry)
            else:
                length = 1

            slices.append(slice(num_dofs, num_dofs+length))
            num_dofs += length

    else:
        stacked_rhs = rhs

    if callback is None:
        if progress:
            callback = ResidualPrinter(inner_product)
        else:
            callback = None

    result = _gmres(op, stacked_rhs, restart=restart, tol=tol, x0=x0,
            inner_product=inner_product,
            maxiter=maxiter, hard_failure=hard_failure,
            no_progress_factor=no_progress_factor,
            stall_iterations=stall_iterations, callback=callback)

    if is_obj_array(rhs):
        return result.copy(
                solution=make_obj_array([result.solution[slc]
                    for slc in slices]))
    else:
        return result

# }}}

# vim: fdm=marker
