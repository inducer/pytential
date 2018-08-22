#!python
#cython: warn.unused=True, warn.unused_arg=True, warn.unreachable=True, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True

import numpy as np
import cython
import cython.parallel

from libc.math cimport sqrt
from libc.stdio cimport printf, fprintf, stderr
from libc.stdlib cimport abort

cimport openmp


cdef extern from "complex.h" nogil:
    double cabs(double complex)


cdef extern from "_helmholtz_utils.h" nogil:
    int jfuns3d_(int *ier, int *nterms, double complex * z, double *scale,
                 double complex *fjs, int *ifder, double complex *fjder,
                 int *lwfjs, int *iscale, int *ntop);
    int h3dall_(int *nterms, double complex *z, double *scale,
		double complex *hvec, int *ifder, double complex *hder);


cdef extern from "_internal.h" nogil:
    const int BUFSIZE


def jfuns3d_wrapper(nterms, z, scale, fjs, fjder):
    """Evaluate spherical Bessel functions.

    Arguments:
        nterms: Number of terms to evaluate
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

    nterms_ = nterms
    z_ = z
    scale_ = scale
    ifder = fjder is not None
    lwfjs = BUFSIZE

    jfuns3d_(&ier, &nterms_, &z_, &scale_, fjstemp, &ifder, fjdertmp, &lwfjs,
             iscale, &ntop)

    if ier:
        raise ValueError("jfuns3d_ returned error code %d" % ier)

    for i in range(nterms):
        fjs[i] = fjstemp[i]
        if ifder:
            fjder[i] = fjdertmp[i]


def h3dall_wrapper(nterms, z, scale, hs, hders):
    """Evaluate spherical Hankel functions.

    Arguments:
        nterms: Number of terms to evaluate
        z: Argument
        scale: Output scaling factor (recommended: min(abs(z), 1))
        hs: Output array of complex doubles
        hders: *None*, or output array of complex double derivatives
    """
    cdef:
        int nterms_, ifder
        double scale_
        double complex z_
        double complex[:] hvec = np.empty(nterms, np.complex)
        double complex[:] hdervec = np.empty(nterms, np.complex)

    ifder = hders is not None

    if nterms == 0:
        return

    nterms_ = nterms - 1
    z_ = z
    scale_ = scale

    h3dall_(&nterms_, &z_, &scale_, &hvec[0], &ifder, &hdervec[0])

    hs[:nterms] = hvec[:]
    if ifder:
        hders[:nterms] = hdervec[:]


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
    return sqrt(
            (a[0] - b[0]) * (a[0] - b[0]) +
            (a[1] - b[1]) * (a[1] - b[1]) +
            (a[2] - b[2]) * (a[2] - b[2]))


cdef void tsqbx_laplace_dlp(
        double[3] source,
        double[3] center,
        double[3] target,
        double[3] grad,
        int order) nogil:
    cdef:
        int j, m
        double sc_d, tc_d, cos_angle, alpha, Rj
        double[BUFSIZE] lvals, lderivs
        double[3] cms, tmc, grad_tmp

    for m in range(3):
        cms[m] = center[m] - source[m]
        tmc[m] = target[m] - center[m]
        grad[m] = 0

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    alpha = (
            (target[0] - center[0]) * (source[0] - center[0]) +
            (target[1] - center[1]) * (source[1] - center[1]) +
            (target[2] - center[2]) * (source[2] - center[2]))

    cos_angle = alpha / (tc_d * sc_d)

    # Evaluate the Legendre terms.
    legvals(cos_angle, order, lvals, lderivs)

    # Invariant: Rj = (t_cd ** j / sc_d ** (j + 2))    
    Rj = 1 / (sc_d * sc_d)

    for j in range(0, order + 1):
        for m in range(3):
            grad_tmp[m] = (j + 1) * (cms[m] / sc_d) * lvals[j]
        for m in range(3):
            # Siegel and Tornberg has a sign flip here :(
            grad_tmp[m] += (tmc[m] / tc_d + cos_angle * cms[m] / sc_d) * lderivs[j]
        for m in range(3):
            grad[m] += Rj * grad_tmp[m]

        Rj *= (tc_d / sc_d)

    return


cdef void tsqbx_helmholtz_dlp(
        double[3] source,
        double[3] center,
        double[3] target,
        double complex[3] grad,
        int order,
        double complex k) nogil:
    cdef:
        int n, m
        int ier, ntop, ifder, lwfjs
        double sc_d, tc_d, cos_angle, alpha
        double[3] cms, tmc
        double complex[3] grad_tmp
        double[BUFSIZE] lvals, lderivs
        double complex z
        double complex[BUFSIZE] jvals, hvals, hderivs
        int[BUFSIZE] iscale
        double jscale, hscale, unscale

    for m in range(3):
        cms[m] = center[m] - source[m]
        tmc[m] = target[m] - center[m]
        grad[m] = 0

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    alpha = (
            (target[0] - center[0]) * (source[0] - center[0]) +
            (target[1] - center[1]) * (source[1] - center[1]) +
            (target[2] - center[2]) * (source[2] - center[2]))

    cos_angle = alpha / (tc_d * sc_d)

    # Evaluate the Legendre terms.
    legvals(cos_angle, order, lvals, lderivs)

    # Scaling magic for Bessel and Hankel terms.
    # These values are taken from the fmmlib documentation.
    jscale = cabs(k * tc_d) if (cabs(k * tc_d) < 1) else 1
    hscale = cabs(k * sc_d) if (cabs(k * sc_d) < 1) else 1
    # unscale = (jscale / hscale) ** n
    # Multiply against unscale to remove the scaling.
    unscale = 1

    # Evaluate the spherical Bessel terms.
    z = k * tc_d
    ifder = 0
    lwfjs = BUFSIZE
    jfuns3d_(&ier, &order, &z, &jscale, jvals, &ifder, NULL, &lwfjs, iscale,
             &ntop)
    if ier:
        # This could in theory fail.
        fprintf(stderr, "array passed to jfuns3d was too small\n")
        abort()

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
            grad_tmp[m] = -hderivs[n] * k * cms[m] * lvals[n] / sc_d
        for m in range(3):
            grad_tmp[m] += hvals[n] * (
                    tmc[m] / (tc_d * sc_d) +
                    alpha * cms[m] / (tc_d * sc_d * sc_d * sc_d)) * lderivs[n]
        for m in range(3):
            grad[m] += (2 * n + 1) * unscale * (grad_tmp[m] * jvals[n])
        unscale *= jscale / hscale

    for m in range(3):
        grad[m] *= 1j * k

    return


cdef double tsqbx_laplace_slp(
        double[3] source,
        double[3] center,
        double[3] target,
        int order) nogil:
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
        return 1 / sc_d

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

    return result


cdef double complex tsqbx_helmholtz_slp(
        double[3] source,
        double[3] center,
        double[3] target,
        int order,
        double complex k) nogil:
    cdef:
        int n, ntop, ier, ifder, lwfjs
        double sc_d, tc_d, cos_angle
        double[BUFSIZE] lvals
        double complex[BUFSIZE] jvals, hvals
        int[BUFSIZE] iscale
        double jscale, hscale, unscale
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

    # Scaling magic for Bessel and Hankel terms.
    # These values are taken from the fmmlib documentation.
    jscale = cabs(k * tc_d) if (cabs(k * tc_d) < 1) else 1
    hscale = cabs(k * sc_d) if (cabs(k * sc_d) < 1) else 1
    # unscale = (jscale / hscale) ** n
    # Multiply against unscale to remove the scaling.
    unscale = 1

    # Evaluate the spherical Bessel terms.
    z = k * tc_d
    ifder = 0
    lwfjs = BUFSIZE
    jfuns3d_(&ier, &order, &z, &jscale, jvals, &ifder, NULL, &lwfjs, iscale,
             &ntop)
    if ier:
        # This could in theory fail.
        fprintf(stderr, "array passed to jfuns3d was too small\n")
        abort()

    # Evaluate the spherical Hankel terms.
    z = k * sc_d
    h3dall_(&order, &z, &hscale, hvals, &ifder, NULL)

    result = 0

    for n in range(1 + order):
        result += (2 * n + 1) * unscale * (jvals[n] * hvals[n] * lvals[n])
        unscale *= jscale / hscale

    return result * 1j * k


def eval_target_specific_qbx_locals(
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
        double[:] charge,
        double[:] dipstr,
        double[:,:] dipvec,
        double complex[:] pot):
    """TSQBX entry point.

    Arguments:
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
        charge: Source strengths, shape (*nsrcs*,) or *None*
        dipstr: Dipole source strengths, shape (*nsrcs*,) or *None*
        dipvec: Dipole source orientations, shape (3, *nsrcs*), or *None*
        pot: Output potential, shape (*ngts*,)
    """

    cdef:
        int tgt, ictr, ctr
        int itgt, itgt_start, itgt_end
        int tgt_box, src_ibox
        int isrc_box, isrc_box_start, isrc_box_end
        int isrc, isrc_start, isrc_end
        int m, tid
        double complex result
        double[:,:] source, center, target, grad
        double complex[:,:] grad_complex
        int laplace_slp, helmholtz_slp, laplace_dlp, helmholtz_dlp

    if charge is None and (dipstr is None or dipvec is None):
        raise ValueError("must specify either charge, or both dipstr and dipvec")

    if charge is not None and (dipstr is not None or dipvec is not None):
        raise ValueError("does not support simultaneous monopoles and dipoles")

    laplace_slp = (helmholtz_k == 0) and (dipvec is None)
    laplace_dlp = (helmholtz_k == 0) and (dipvec is not None)
    helmholtz_slp = (helmholtz_k != 0) and (dipvec is None)
    helmholtz_dlp = (helmholtz_k != 0) and (dipvec is not None)

    assert laplace_slp or laplace_dlp or helmholtz_slp or helmholtz_dlp

    if qbx_centers.shape[0] == 0:
        return

    # Hack to obtain thread-local storage
    maxthreads = openmp.omp_get_max_threads()

    # Prevent false sharing by over-allocating the buffers
    source = np.zeros((maxthreads, 65))
    target = np.zeros((maxthreads, 65))
    center = np.zeros((maxthreads, 65))
    grad = np.zeros((maxthreads, 65))
    grad_complex = np.zeros((maxthreads, 65), dtype=np.complex)

    # TODO: Check that the order is not too high, since some temporary arrays
    # used above might overflow if that is the case.

    for ictr in cython.parallel.prange(0, qbx_centers.shape[0],
                                       nogil=True, schedule="static",
                                       chunksize=128):
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

            isrc_box_start = source_box_starts[tgt_box]
            isrc_box_end = source_box_starts[tgt_box + 1]

            for isrc_box in range(isrc_box_start, isrc_box_end):
                src_ibox = source_box_lists[isrc_box]
                isrc_start = box_source_starts[src_ibox]
                isrc_end = isrc_start + box_source_counts_nonchild[src_ibox]

                for isrc in range(isrc_start, isrc_end):
                    for m in range(3):
                        source[tid, m] = sources[m, isrc]

                    # NOTE: Don't use +=, since that makes Cython think we are
                    # doing an OpenMP reduction.

                    if laplace_slp:
                        result = result + charge[isrc] * (
                                tsqbx_laplace_slp(&source[tid, 0], &center[tid, 0],
                                                  &target[tid, 0], order))

                    elif helmholtz_slp:
                        result = result + charge[isrc] * (
                                tsqbx_helmholtz_slp(&source[tid, 0], &center[tid, 0],
                                                    &target[tid, 0], order,
                                                    helmholtz_k))

                    elif laplace_dlp:
                        tsqbx_laplace_dlp(&source[tid, 0], &center[tid, 0],
                                          &target[tid, 0], &grad[tid, 0], order)

                        result = result + dipstr[isrc] * (
                                grad[tid, 0] * dipvec[0, isrc] +
                                grad[tid, 1] * dipvec[1, isrc] +
                                grad[tid, 2] * dipvec[2, isrc])

                    elif helmholtz_dlp:
                        tsqbx_helmholtz_dlp(&source[tid, 0], &center[tid, 0],
                                            &target[tid, 0], &grad_complex[tid, 0],
                                            order, helmholtz_k)

                        result = result + dipstr[isrc] * (
                                grad_complex[tid, 0] * dipvec[0, isrc] +
                                grad_complex[tid, 1] * dipvec[1, isrc] +
                                grad_complex[tid, 2] * dipvec[2, isrc])

            pot[tgt] = pot[tgt] + result

        # The Cython-generated OpenMP loop marks these variables as lastprivate.
        # Due to this GCC warns that these could be used without being initialized.
        # Initialize them here to suppress the warning.
        result = 0
        tid = 0
        ctr = 0
        src_ibox = tgt_box = 0
        tgt = itgt = itgt_start = itgt_end = 0
        isrc = isrc_box = isrc_start = isrc_end = isrc_box_start = isrc_box_end = 0
