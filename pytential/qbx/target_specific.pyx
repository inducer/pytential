#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
import cython
import cython.parallel

from libc.math cimport sqrt
from libc.stdio cimport printf

cimport openmp


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
        pjm2 = pjm1
        pjm1 = pj

        if derivs != NULL:
            derj = (2*j-1)*(pjm1+x*derjm1)-(j-1)*derjm2
            derj = derj / j
            derivs[j] = derj
            derjm2 = derjm1
            derjm1 = derj


cdef double dist(double[3] a, double[3] b) nogil:
    return sqrt(
            (a[0] - b[0]) * (a[0] - b[0]) +
            (a[1] - b[1]) * (a[1] - b[1]) +
            (a[2] - b[2]) * (a[2] - b[2]))


"""
cdef void tsqbx_grad_from_source(
        double[3] source,
        double[3] center,
        double[3] target,
        double[3] grad,    
        int order,
        double[] tmp) nogil:
    cdef:
        int i
        double result, r, sc_d, tc_d, cos_angle, alpha
        double *derivs

    derivs = &tmp[order + 1]

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    alpha = (
            (target[0] - center[0]) * (source[0] - center[0]) +
            (target[1] - center[1]) * (source[1] - center[1]) +
            (target[2] - center[2]) * (source[2] - center[2])

    cos_angle = alpha / (tc_d * sc_d)

    legvals(cos_angle, order, tmp, derivs)

    result = 0
    r = 1 / sc_d

    for i in range(0, order + 1):
        result = result + tmp[i] * r
        r = r * (tc_d / sc_d)

    return result
"""


cdef double tsqbx_from_source(
        double[3] source,
        double[3] center,
        double[3] target,
        int order,
        double[] tmp) nogil:
    cdef:
        int i
        double result, r, sc_d, tc_d, cos_angle

    tc_d = dist(target, center)
    sc_d = dist(source, center)

    cos_angle = ((
            (target[0] - center[0]) * (source[0] - center[0]) +
            (target[1] - center[1]) * (source[1] - center[1]) +
            (target[2] - center[2]) * (source[2] - center[2]))
            / (tc_d * sc_d))

    legvals(cos_angle, order, tmp, NULL)

    result = 0
    r = 1 / sc_d

    for i in range(0, order + 1):
        result = result + tmp[i] * r
        r = r * (tc_d / sc_d)

    return result


def eval_target_specific_global_qbx_locals(
        int order,
        double[:,:] sources,
        double[:,:] targets,
        double[:,:] centers,
        int[:] global_qbx_centers,
        int[:] qbx_center_to_target_box,
        int[:] center_to_target_starts, int[:] center_to_target_lists,
        int[:] source_box_starts, int[:] source_box_lists,
        int[:] box_source_starts, int[:] box_source_counts_nonchild,
        double[:] src_weights,
        double complex[:] pot):

    cdef:
        int tgt, ictr, ctr
        int itgt, itgt_start, itgt_end
        int tgt_box, src_ibox
        int isrc_box, isrc_box_start, isrc_box_end
        int isrc, isrc_start, isrc_end
        int i, tid
        double result
        double[:,:] source, center, target, tmp

    # Yucky thread-local hack
    maxthreads = openmp.omp_get_max_threads()

    source = np.zeros((1 + maxthreads, 3))
    target = np.zeros((1 + maxthreads, 3))
    center = np.zeros((1 + maxthreads, 3))
    tmp = np.zeros((1 + maxthreads, 256))

    # TODO: Check if order > 256

    for ictr in cython.parallel.prange(0, global_qbx_centers.shape[0],
                                       nogil=True, schedule="dynamic",
                                       chunksize=10):
        ctr = global_qbx_centers[ictr]
        itgt_start = center_to_target_starts[ctr]
        itgt_end = center_to_target_starts[ctr + 1]
        tgt_box = qbx_center_to_target_box[ctr]
        tid = cython.parallel.threadid()

        for i in range(3):
            center[tid, i] = centers[i, ctr]

        for itgt in range(itgt_start, itgt_end):
            result = 0
            tgt = center_to_target_lists[itgt]

            for i in range(3):
                target[tid, i] = targets[i, tgt]

            isrc_box_start = source_box_starts[tgt_box]
            isrc_box_end = source_box_starts[tgt_box + 1]

            for isrc_box in range(isrc_box_start, isrc_box_end):
                src_ibox = source_box_lists[isrc_box]
                isrc_start = box_source_starts[src_ibox]
                isrc_end = isrc_start + box_source_counts_nonchild[src_ibox]

                for isrc in range(isrc_start, isrc_end):
                    for i in range(3):
                        source[tid, i] = sources[i, isrc]

                    result = result + src_weights[isrc] * (
                        tsqbx_from_source(&source[tid, 0], &center[tid, 0],
                                          &target[tid, 0], order, &tmp[tid, 0]))

            pot[tgt] = pot[tgt] + result
