#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
import cython
import cython.parallel

from libc.math cimport sqrt
from libc.stdio cimport printf

cimport openmp


cdef double legendre(double x, int n, double[] coeffs) nogil:
    """Evaluate the Legendre series of order n at x.

    Taken from SciPy.
    """
    cdef:
        double c0, c1, tmp
        int nd, i

    if n == 0:
        c0 = coeffs[0]
        c1 = 0
    elif n == 1:
        c0 = coeffs[0]
        c1 = coeffs[1]
    else:
        nd = n + 1
        c0 = coeffs[n - 1]
        c1 = coeffs[n]

        for i in range(3, n + 2):
            tmp = c0
            nd = nd - 1
            c0 = coeffs[1+n-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x


cdef double dist(double[3] a, double[3] b) nogil:
    return sqrt(
            (a[0] - b[0]) * (a[0] - b[0]) +
            (a[1] - b[1]) * (a[1] - b[1]) +
            (a[2] - b[2]) * (a[2] - b[2]))


cdef double tsqbx_from_source(
        double[3] source,
        double[3] center,
        double[3] target,
        int order,
        double[] tmp) nogil:
    cdef:
        int i
        double r, sc_d, tc_d
        double cos_angle

    tc_d = dist(target, center)
    sc_d = dist(source, center)
    r = tc_d / sc_d
    tmp[0] = 1 / sc_d

    for i in range(1, order + 1):
        tmp[i] = tmp[i - 1] * r

    cos_angle = ((
            (target[0] - center[0]) * (source[0] - center[0]) +
            (target[1] - center[1]) * (source[1] - center[1]) +
            (target[2] - center[2]) * (source[2] - center[2]))
            / (tc_d * sc_d))

    return legendre(cos_angle, order, tmp)


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
