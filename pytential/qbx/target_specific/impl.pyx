# cython: warn.unused=True, warn.unused_arg=True, warn.unreachable=True
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# cython: embedsignature=True, language_level=3

"""Copyright (C) 2018 Matt Wala

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
import cython
import cython.parallel

from libc.math cimport sqrt
from libc.stdio cimport printf, fprintf, stderr
from libc.stdlib cimport abort

cimport openmp


cdef extern from "complex.h" nogil:
    double cabs(double complex)


cdef extern from "helmholtz_utils.h" nogil:
    int jfuns3d_(int *ier, int *nterms, double complex * z, double *scale,
                 double complex *fjs, int *ifder, double complex *fjder,
                 int *lwfjs, int *iscale, int *ntop);
    int h3dall_(int *nterms, double complex *z, double *scale,
		double complex *hvec, int *ifder, double complex *hder);


cdef extern from "impl.h" nogil:
    const int BUFSIZE
    const int PADDING


# {{{ (externally visible) wrappers for bessel / hankel functions

def jfuns3d_wrapper(nterms, z, scale, fjs, fjder):
    """Evaluate spherical Bessel functions.

    Arguments:
        nterms: Highest order to be computed
        z: Argument
        scale: Output scaling factor (recommended: min(abs(z), 1))
        fjs: Output array of complex doubles
        fjder: *None*, or output array of complex double derivatives
    """
    cdef:
        double complex[BUFSIZE] fjstemp
        double complex[BUFSIZE] fjdertmp
        int[BUFSIZE] iscale
        int ier, ifder, lwfjs, ntop, i, nterms_
        double scale_
        double complex z_

    if nterms <= 0:
        raise ValueError("nterms should be positive")

    nterms_ = nterms
    z_ = z
    scale_ = scale
    ifder = fjder is not None
    lwfjs = BUFSIZE

    jfuns3d_(&ier, &nterms_, &z_, &scale_, fjstemp, &ifder, fjdertmp, &lwfjs,
             iscale, &ntop)

    if ier:
        raise ValueError("jfuns3d_ returned error code %d" % ier)

    for i in range(1 + nterms):
        fjs[i] = fjstemp[i]
        if ifder:
            fjder[i] = fjdertmp[i]


def h3dall_wrapper(nterms, z, scale, hs, hders):
    """Evaluate spherical Hankel functions.

    Arguments:
        nterms: Highest order to be computed
        z: Argument
        scale: Output scaling factor (recommended: min(abs(z), 1))
        hs: Output array of complex doubles
        hders: *None*, or output array of complex double derivatives
    """
    cdef:
        int nterms_, ifder
        double scale_
        double complex z_
        double complex[:] hvec = np.empty(1 + nterms, np.complex)
        double complex[:] hdervec = np.empty(1 + nterms, np.complex)

    ifder = hders is not None

    if nterms <= 0:
        raise ValueError("nterms should be positive")

    z_ = z
    scale_ = scale
    nterms_ = nterms

    h3dall_(&nterms_, &z_, &scale_, &hvec[0], &ifder, &hdervec[0])

    hs[:1 + nterms] = hvec[:]
    if ifder:
        hders[:1 + nterms] = hdervec[:]

# }}}


# {{{ helpers

cdef void legvals(double x, int n, double[] vals, double[] derivs) nogil:
    """Compute the values of the Legendre polynomial up to order n at x.
    Optionally, if derivs is non-NULL, compute the values of the derivative too.

    Borrowed from fmmlib.
    """
    cdef:
        double pj, derj, pjm2, pjm1, derjm2, derjm1
        int j

    pjm2 = 1
    pjm1 = x

    vals[0] = 1
    if derivs != NULL:
        derivs[0] = 0
        derjm2 = 0
        derjm1 = 1

    if n == 0:
        return

    vals[1] = x
    if derivs != NULL:
        derivs[1] = 1

    if n == 1:
        return

    for j in range(2, n + 1):
        pj = ( (2*j-1)*x*pjm1-(j-1)*pjm2 ) / j
        vals[j] = pj

        if derivs != NULL:
            derj = (2*j-1)*(pjm1+x*derjm1)-(j-1)*derjm2
            derj = derj / j
            derivs[j] = derj
            derjm2 = derjm1
            derjm1 = derj

        pjm2 = pjm1
        pjm1 = pj


cdef double dist(double[3] a, double[3] b) nogil:
    """Calculate the Euclidean distance between a and b."""
    return sqrt(
            (a[0] - b[0]) * (a[0] - b[0]) +
            (a[1] - b[1]) * (a[1] - b[1]) +
            (a[2] - b[2]) * (a[2] - b[2]))


cdef void ts_helmholtz_precompute(
        double[3] center,
        double[3] target,
        int order,
        int ifder,
        double complex k,
        double complex[] jvals,
        double complex[] jderivs,
        double *jscale) nogil:
    """Evaluate the source-invariant Bessel terms of the Helmholtz target-specific
    expansion."""

    cdef:
        double complex z
        double tc_d
        int ier, ntop, lwfjs
        int[BUFSIZE] iscale

    tc_d = dist(target, center)
    jscale[0] = cabs(k * tc_d) if (cabs(k * tc_d) < 1) else 1

    # Evaluate the spherical Bessel terms.
    z = k * tc_d
    lwfjs = BUFSIZE
    # jfuns3d_ only supports order > 0 (goes out of bounds if order = 0)
    order = max(1, order)
    jfuns3d_(&ier, &order, &z, jscale, jvals, &ifder, jderivs, &lwfjs, iscale,
             &ntop)
    if ier:
        # This could in theory fail.
        fprintf(stderr, "array passed to jfuns3d_ was too small\n")
        abort()

# }}}


# {{{ Laplace S

cdef double complex ts_laplace_s(
        double[3] source,
        double[3] center,
        double[3] target,
        double complex charge,
        int order) nogil:
    """Evaluate the target-specific expansion of the Laplace single-layer kernel."""

    cdef:
        double j
        double result, r, sc_d, tc_d, cos_angle
        # Legendre recurrence values
        double pj, pjm1, pjm2

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    cos_angle = ((
            (target[0] - center[0]) * (source[0] - center[0]) +
            (target[1] - center[1]) * (source[1] - center[1]) +
            (target[2] - center[2]) * (source[2] - center[2]))
            / (tc_d * sc_d))

    if order == 0:
        return charge / sc_d

    pjm2 = 1
    pjm1 = cos_angle

    result = 1 / sc_d + (cos_angle * tc_d) / (sc_d * sc_d)

    r = (tc_d * tc_d) / (sc_d * sc_d * sc_d)

    # Invariant: j matches loop counter. Using a double-precision version of the
    # loop counter avoids an int-to-double conversion inside the loop.
    j = 2.

    for _ in range(2, order + 1):
        pj = ( (2.*j-1.)*cos_angle*pjm1-(j-1.)*pjm2 ) / j
        result += pj * r

        r *= (tc_d / sc_d)
        j += 1
        pjm2 = pjm1
        pjm1 = pj

    return charge * result

# }}}


# {{{ Laplace grad(S)

cdef void ts_laplace_sp(
        double complex[3] grad,
        double[3] source,
        double[3] center,
        double[3] target,
        double complex charge,
        int order) nogil:
    """Evaluate the target-specific expansion of the gradient of the Laplace
    single-layer kernel."""

    cdef:
        double[3] grad_tmp
        double sc_d, tc_d, cos_angle, Rn
        double[BUFSIZE] lvals, lderivs
        double[3] smc, tmc
        int n

    for m in range(3):
        smc[m] = source[m] - center[m]
        tmc[m] = target[m] - center[m]
        grad_tmp[m] = 0

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    cos_angle = (tmc[0] * smc[0] + tmc[1] * smc[1] + tmc[2] * smc[2]) / (tc_d * sc_d)
    legvals(cos_angle, order, lvals, lderivs)

    # Invariant: Rn = tc_d ** (n - 1) / sc_d ** (n + 1)
    Rn = 1 / (sc_d * sc_d)

    for n in range(1, 1 + order):
        for m in range(3):
            grad_tmp[m] += Rn * (
                n * (tmc[m] / tc_d) * lvals[n]
                + (smc[m] / sc_d - cos_angle * tmc[m] / tc_d) * lderivs[n])
        Rn *= tc_d / sc_d

    for m in range(3):
        grad[m] += charge * grad_tmp[m]

# }}}


# {{{ Laplace D

cdef double complex ts_laplace_d(
        double[3] source,
        double[3] center,
        double[3] target,
        double[3] dipole,
        double complex dipstr,
        int order) nogil:
    """Evaluate the target-specific expansion of the Laplace double-layer kernel."""

    cdef:
        int n, m
        double sc_d, tc_d, cos_angle, Rn
        double[BUFSIZE] lvals, lderivs
        double[3] smc, tmc, grad

    for m in range(3):
        smc[m] = source[m] - center[m]
        tmc[m] = target[m] - center[m]
        grad[m] = 0

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    cos_angle = (tmc[0] * smc[0] + tmc[1] * smc[1] + tmc[2] * smc[2]) / (tc_d * sc_d)
    legvals(cos_angle, order, lvals, lderivs)

    # Invariant: Rn = (tc_d ** n / sc_d ** (n + 2))
    Rn = 1 / (sc_d * sc_d)

    for n in range(0, order + 1):
        for m in range(3):
            grad[m] += Rn * (
                    -(n + 1) * (smc[m] / sc_d) * lvals[n]
                    + (tmc[m] / tc_d - cos_angle * smc[m] / sc_d) * lderivs[n])
        Rn *= (tc_d / sc_d)

    return dipstr * (
            dipole[0] * grad[0] + dipole[1] * grad[1] + dipole[2] * grad[2])

# }}}


# {{{ Helmholtz S

cdef double complex ts_helmholtz_s(
        double[3] source,
        double[3] center,
        double[3] target,
        double complex charge,
        int order,
        double complex k,
        double complex[] jvals,
        double jscale) nogil:
    """Evaluate the target-specific expansion of the Helmholtz single-layer
    kernel."""

    cdef:
        int n, ifder
        double sc_d, tc_d, cos_angle
        double[BUFSIZE] lvals
        double complex[BUFSIZE] hvals
        double hscale, unscale
        double complex z, result

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    cos_angle = ((
            (target[0] - center[0]) * (source[0] - center[0]) +
            (target[1] - center[1]) * (source[1] - center[1]) +
            (target[2] - center[2]) * (source[2] - center[2]))
            / (tc_d * sc_d))

    # Evaluate the Legendre terms.
    legvals(cos_angle, order, lvals, NULL)

    # Scaling magic for Hankel terms.
    # These values are taken from the fmmlib documentation.
    hscale = cabs(k * sc_d) if (cabs(k * sc_d) < 1) else 1
    # unscale = (jscale / hscale) ** n
    # Multiply against unscale to remove the scaling.
    unscale = 1

    # Evaluate the spherical Hankel terms.
    z = k * sc_d
    ifder = 0
    h3dall_(&order, &z, &hscale, hvals, &ifder, NULL)

    result = 0

    for n in range(1 + order):
        result += (2 * n + 1) * unscale * (jvals[n] * hvals[n] * lvals[n])
        unscale *= jscale / hscale

    return 1j * k * charge * result

# }}}


# {{{ Helmholtz grad(S)

cdef void ts_helmholtz_sp(
        double complex[3] grad,
        double[3] source,
        double[3] center,
        double[3] target,
        double complex charge,
        int order,
        double complex k,
        double complex[] jvals,
        double complex[] jderivs,
        double jscale) nogil:
    """Evaluate the target-specific expansion of the gradient of the Helmholtz
    single-layer kernel."""

    cdef:
        int n, m
        int ifder
        double sc_d, tc_d, cos_angle
        double[3] smc, tmc
        double complex[3] grad_tmp
        double[BUFSIZE] lvals, lderivs
        double complex z
        double complex [BUFSIZE] hvals
        double hscale, unscale

    for m in range(3):
        smc[m] = source[m] - center[m]
        tmc[m] = target[m] - center[m]
        grad_tmp[m] = 0

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    # Evaluate the Legendre terms.
    cos_angle = (tmc[0] * smc[0] + tmc[1] * smc[1] + tmc[2] * smc[2]) / (tc_d * sc_d)
    legvals(cos_angle, order, lvals, lderivs)

    # Scaling magic for Hankel terms.
    # These values are taken from the fmmlib documentation.
    hscale = cabs(k * sc_d) if (cabs(k * sc_d) < 1) else 1
    # unscale = (jscale / hscale) ** n
    # Multiply against unscale to remove the scaling.
    unscale = 1

    # Evaluate the spherical Hankel terms.
    z = k * sc_d
    ifder = 0
    h3dall_(&order, &z, &hscale, hvals, &ifder, NULL)

    #
    # This is a mess, but amounts to the t-gradient of:
    #
    #      __ order
    #  ik \         (2n  +  1) j (k |t - c|) h (k |s - c|) P (cos θ)
    #     /__ n = 0             n             n             n
    #
    #
    for n in range(0, order + 1):
        for m in range(3):
            grad_tmp[m] += (2 * n + 1) * unscale * hvals[n] / tc_d * (
                    k * jderivs[n] * lvals[n] * tmc[m]
                    + (smc[m] / sc_d - cos_angle * tmc[m] / tc_d)
                    * jvals[n] * lderivs[n])
        unscale *= jscale / hscale

    for m in range(3):
        grad[m] += 1j * k * charge * grad_tmp[m]

# }}}


# {{{ Helmholtz D

cdef double complex ts_helmholtz_d(
        double[3] source,
        double[3] center,
        double[3] target,
        double[3] dipole,
        double complex dipstr,
        int order,
        double complex k,
        double complex[] jvals,
        double jscale) nogil:
    """Evaluate the target-specific expansion of the Helmholtz double-layer
    kernel."""

    cdef:
        int n, m
        int ifder
        double sc_d, tc_d, cos_angle
        double[3] smc, tmc
        double complex[3] grad
        double[BUFSIZE] lvals, lderivs
        double complex z
        double complex [BUFSIZE] hvals, hderivs
        double hscale, unscale

    for m in range(3):
        smc[m] = source[m] - center[m]
        tmc[m] = target[m] - center[m]
        grad[m] = 0

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    cos_angle = (tmc[0] * smc[0] + tmc[1] * smc[1] + tmc[2] * smc[2]) / (tc_d * sc_d)

    # Evaluate the Legendre terms.
    legvals(cos_angle, order, lvals, lderivs)

    # Scaling magic for Hankel terms.
    # These values are taken from the fmmlib documentation.
    hscale = cabs(k * sc_d) if (cabs(k * sc_d) < 1) else 1
    # unscale = (jscale / hscale) ** n
    # Multiply against unscale to remove the scaling.
    unscale = 1

    # Evaluate the spherical Hankel terms.
    z = k * sc_d
    ifder = 1
    h3dall_(&order, &z, &hscale, hvals, &ifder, hderivs)

    #
    # This is a mess, but amounts to the s-gradient of:
    #
    #      __ order
    #  ik \         (2n  +  1) j (k |t - c|) h (k |s - c|) P (cos θ)
    #     /__ n = 0             n             n             n
    #
    #
    for n in range(0, order + 1):
        for m in range(3):
            grad[m] += (2 * n + 1) * unscale * jvals[n] / sc_d * (
                    k * smc[m] * hderivs[n] * lvals[n]
                    + (tmc[m] / tc_d - cos_angle * smc[m] / sc_d)
                    * hvals[n] * lderivs[n])
        unscale *= jscale / hscale

    return 1j * k * dipstr * (
            grad[0] * dipole[0] + grad[1] * dipole[1] + grad[2] * dipole[2])

# }}}


def eval_target_specific_qbx_locals(
        int ifpot,
        int ifgrad,
        int ifcharge,
        int ifdipole,
        int order,
        double[:,:] sources,
        double[:,:] targets,
        double[:,:] centers,
        int[:] qbx_centers,
        int[:] qbx_center_to_target_box,
        int[:] center_to_target_starts, int[:] center_to_target_lists,
        int[:] source_box_starts, int[:] source_box_lists,
        int[:] box_source_starts, int[:] box_source_counts_nonchild,
        double complex helmholtz_k,
        double complex[:] charge,
        double complex[:] dipstr,
        double[:,:] dipvec,
        double complex[:] pot,
        double complex[:,:] grad):
    """TSQBX entry point.

    Arguments:
        ifpot: Flag indicating whether to evaluate the potential
        ifgrad: Flag indicating whether to evaluate the gradient of the potential
        ifcharge: Flag indicating whether to include monopole sources
        ifdipole: Flag indicating whether to include dipole sources
        order: Expansion order
        sources: Array of sources of shape (3, *nsrcs*)
        targets: Array of targets of shape (3, *ntgts*)
        centers: Array of centers of shape (3, *nctrs*)
        qbx_centers: Array of subset of indices into *centers* which are QBX centers
        qbx_center_to_target_box: Array mapping centers to target box numbers
        center_to_target_starts: "Start" indices for center-to-target CSR list
        center_to_target_lists: Center-to-target CSR list
        source_box_starts: "Start" indices for target-box-to-source-box CSR list
        source_box_lists: Target-box-to-source-box CSR list
        box_source_starts: "Start" indices for sources for each box
        box_source_counts_nonchild: Number of sources per box
        helmholtz_k: Helmholtz parameter (Pass 0 for Laplace)
        charge: (Complex) Source strengths, shape (*nsrcs*,), or *None*
        dipstr: (Complex) Dipole source strengths, shape (*nsrcs*,) or *None*
        dipvec: (Real) Dipole source orientations, shape (3, *nsrcs*), or *None*
        pot: (Complex) Output potential, shape (*ngts*,), or *None*
        grad: (Complex) Output gradient, shape (3, *ntgts*), or *None*
    """

    cdef:
        int tgt, ictr, ctr
        int itgt, itgt_start, itgt_end
        int tgt_box, src_ibox
        int isrc_box, isrc_box_start, isrc_box_end
        int isrc, isrc_start, isrc_end
        int tid, m
        double jscale
        double complex result
        double[:,:] source, center, target, dipole
        double complex[:,:] result_grad, jvals, jderivs
        int laplace_s, helmholtz_s, laplace_sp, helmholtz_sp, laplace_d, helmholtz_d

    # {{{ process arguments

    if ifcharge:
        if charge is None:
            raise ValueError("Missing charge")

    if ifdipole:
        if dipstr is None:
            raise ValueError("Missing dipstr")
        if dipvec is None:
            raise ValueError("Missing dipvec")

    if ifdipole and ifgrad:
        raise ValueError("Does not support computing gradient of dipole sources")

    laplace_s = laplace_sp = laplace_d = 0
    helmholtz_s = helmholtz_sp = helmholtz_d = 0

    if helmholtz_k == 0:
        if ifpot:
            laplace_s = ifcharge
            laplace_d = ifdipole

        if ifgrad:
            laplace_sp = ifcharge

    else:
        if ifpot:
            helmholtz_s = ifcharge
            helmholtz_d = ifdipole

        if ifgrad:
            helmholtz_sp = ifcharge

    # }}}

    if not any([
            laplace_s, laplace_sp, laplace_d, helmholtz_s, helmholtz_sp,
            helmholtz_d]):
        return

    if qbx_centers.shape[0] == 0:
        return

    # {{{ set up thread-local storage

    # Hack to obtain thread-local storage
    maxthreads = openmp.omp_get_max_threads()

    # Prevent false sharing by padding the thread-local buffers
    source = np.zeros((maxthreads, PADDING))
    target = np.zeros((maxthreads, PADDING))
    center = np.zeros((maxthreads, PADDING))
    dipole = np.zeros((maxthreads, PADDING))
    result_grad = np.zeros((maxthreads, PADDING), dtype=np.complex)
    jvals = np.zeros((maxthreads, BUFSIZE + PADDING), dtype=np.complex)
    jderivs = np.zeros((maxthreads, BUFSIZE + PADDING), dtype=np.complex)

    # TODO: Check that the order is not too high, since temporary
    # arrays in this module that are limited by BUFSIZE may overflow
    # if that is the case

    # }}}

    for ictr in cython.parallel.prange(0, qbx_centers.shape[0],
                                       nogil=True, schedule="static",
                                       chunksize=128):
        # Assign to jscale so Cython marks it as private
        jscale = 0
        ctr = qbx_centers[ictr]
        itgt_start = center_to_target_starts[ctr]
        itgt_end = center_to_target_starts[ctr + 1]
        tgt_box = qbx_center_to_target_box[ctr]
        tid = cython.parallel.threadid()

        for m in range(3):
            center[tid, m] = centers[m, ctr]

        for itgt in range(itgt_start, itgt_end):
            result = 0
            tgt = center_to_target_lists[itgt]

            for m in range(3):
                target[tid, m] = targets[m, tgt]
                if ifgrad:
                    result_grad[tid, m] = 0

            if helmholtz_s or helmholtz_sp or helmholtz_d:
                # Precompute source-invariant Helmholtz terms.
                ts_helmholtz_precompute(
                        &center[tid, 0], &target[tid, 0],
                        order, ifgrad, helmholtz_k, &jvals[tid, 0],
                        &jderivs[tid, 0], &jscale)

            isrc_box_start = source_box_starts[tgt_box]
            isrc_box_end = source_box_starts[tgt_box + 1]

            for isrc_box in range(isrc_box_start, isrc_box_end):
                src_ibox = source_box_lists[isrc_box]
                isrc_start = box_source_starts[src_ibox]
                isrc_end = isrc_start + box_source_counts_nonchild[src_ibox]

                for isrc in range(isrc_start, isrc_end):

                    for m in range(3):
                        source[tid, m] = sources[m, isrc]
                        if ifdipole:
                            dipole[tid, m] = dipvec[m, isrc]

                    # NOTE: Don't use +=, since that makes Cython think we are
                    # doing an OpenMP reduction.

                    # {{{ evaluate potentials

                    if laplace_s:
                        result = result + (
                                ts_laplace_s(
                                    &source[tid, 0], &center[tid, 0], &target[tid, 0],
                                    charge[isrc], order))

                    if laplace_sp:
                        ts_laplace_sp(
                                &result_grad[tid, 0],
                                &source[tid, 0], &center[tid, 0], &target[tid, 0],
                                charge[isrc], order)

                    if laplace_d:
                        result = result + (
                                ts_laplace_d(
                                    &source[tid, 0], &center[tid, 0], &target[tid, 0],
                                    &dipole[tid, 0], dipstr[isrc], order))

                    if helmholtz_s:
                        result = result + (
                                ts_helmholtz_s(&source[tid, 0], &center[tid, 0],
                                    &target[tid, 0], charge[isrc], order, helmholtz_k,
                                    &jvals[tid, 0], jscale))

                    if helmholtz_sp:
                        ts_helmholtz_sp(
                                &result_grad[tid, 0],
                                &source[tid, 0], &center[tid, 0], &target[tid, 0],
                                charge[isrc], order, helmholtz_k,
                                &jvals[tid, 0], &jderivs[tid, 0], jscale)

                    if helmholtz_d:
                        result = result + (
                                ts_helmholtz_d(
                                    &source[tid, 0], &center[tid, 0], &target[tid, 0],
                                    &dipole[tid, 0], dipstr[isrc], order, helmholtz_k,
                                    &jvals[tid, 0], jscale))

                    # }}}

            if ifpot:
                pot[tgt] = result

            if ifgrad:
                for m in range(3):
                    grad[m, tgt] = result_grad[tid, m]
