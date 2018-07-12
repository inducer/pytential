# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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

import numpy as np  # noqa: F401
import pyopencl as cl  # noqa: F401
import six
from pytools import memoize_method


__doc__ = """
.. autoclass:: PotentialSource
.. autoclass:: PointPotentialSource
.. autoclass:: LayerPotentialSourceBase
"""


class PotentialSource(object):
    """
    .. method:: preprocess_optemplate(name, expr)

    .. method:: op_group_features(expr)

        Return a characteristic tuple by which operators that can be
        executed together can be grouped.

        *expr* is a subclass of
        :class:`pytential.symbolic.primitives.IntG`.
    """


# {{{ point potential source

class PointPotentialSource(PotentialSource):
    """
    .. attribute:: points

        An :class:`pyopencl.array.Array` of shape ``[ambient_dim, npoints]``.

    .. attribute:: nnodes
    """

    def __init__(self, cl_context, points):
        self.cl_context = cl_context
        self.points = points

    @property
    def real_dtype(self):
        return self.points.dtype

    @property
    def nnodes(self):
        return self.points.shape[-1]

    @property
    def complex_dtype(self):
        return {
                np.float32: np.complex64,
                np.float64: np.complex128
                }[self.real_dtype.type]

    @property
    def ambient_dim(self):
        return self.points.shape[0]

    def op_group_features(self, expr):
        from sumpy.kernel import AxisTargetDerivativeRemover
        result = (
                expr.source, expr.density,
                AxisTargetDerivativeRemover()(expr.kernel),
                )

        return result

    @memoize_method
    def get_p2p(self, kernels):
        # needs to be separate method for caching

        from pytools import any
        if any(knl.is_complex_valued for knl in kernels):
            value_dtype = self.complex_dtype
        else:
            value_dtype = self.real_dtype

        from sumpy.p2p import P2P
        p2p = P2P(self.cl_context,
                    kernels, exclude_self=False, value_dtypes=value_dtype)

        return p2p

    def perf_model_compute_potential_insn(self, queue, insn, bound_expr,
                                          evaluate, costs):
        raise NotImplementedError

    def exec_compute_potential_insn(self, queue, insn, bound_expr, evaluate):
        p2p = None

        kernel_args = {}
        for arg_name, arg_expr in six.iteritems(insn.kernel_arguments):
            kernel_args[arg_name] = evaluate(arg_expr)

        strengths = (evaluate(insn.density).with_queue(queue)
                * self.weights_and_area_elements())

        # FIXME: Do this all at once
        result = []
        for o in insn.outputs:
            target_discr = bound_expr.get_discretization(o.target_name)

            # no on-disk kernel caching
            if p2p is None:
                p2p = self.get_p2p(insn.kernels)

            evt, output_for_each_kernel = p2p(queue,
                    target_discr.nodes(), self.points,
                    [strengths], **kernel_args)

            result.append((o.name, output_for_each_kernel[o.kernel_index]))

        timing_data = {}
        new_futures = []
        return result, new_futures, timing_data

    @memoize_method
    def weights_and_area_elements(self):
        with cl.CommandQueue(self.cl_context) as queue:
            result = cl.array.empty(queue, self.points.shape[-1],
                    dtype=self.real_dtype)
            result.fill(1)

        return result.with_queue(None)

# }}}


# {{{ layer potential source

class LayerPotentialSourceBase(PotentialSource):
    """A discretization of a layer potential using panel-based geometry, with
    support for refinement and upsampling.

    .. rubric:: Discretizations

    .. attribute:: density_discr
    .. attribute:: stage2_density_discr
    .. attribute:: quad_stage2_density_discr
    .. attribute:: resampler
    .. method:: with_refinement

    .. rubric:: Discretization data

    .. attribute:: cl_context
    .. attribute:: ambient_dim
    .. attribute:: dim
    .. attribute:: real_dtype
    .. attribute:: complex_dtype
    .. attribute:: h_max

    .. rubric:: Execution

    .. method:: weights_and_area_elements
    .. method:: perf_model_compute_potential_insn
    .. method:: exec_compute_potential_insn
    """

    @property
    def ambient_dim(self):
        return self.density_discr.ambient_dim

    @property
    def dim(self):
        return self.density_discr.dim

    @property
    def cl_context(self):
        return self.density_discr.cl_context

    @property
    def real_dtype(self):
        return self.density_discr.real_dtype

    @property
    def complex_dtype(self):
        return self.density_discr.complex_dtype

    @memoize_method
    def get_p2p(self, kernels):
        # needs to be separate method for caching

        from pytools import any
        if any(knl.is_complex_valued for knl in kernels):
            value_dtype = self.density_discr.complex_dtype
        else:
            value_dtype = self.density_discr.real_dtype

        from sumpy.p2p import P2P
        p2p = P2P(self.cl_context,
                  kernels, exclude_self=False, value_dtypes=value_dtype)

        return p2p

    # {{{ fmm setup helpers

    def get_fmm_kernel(self, kernels):
        fmm_kernel = None

        from sumpy.kernel import AxisTargetDerivativeRemover
        for knl in kernels:
            candidate_fmm_kernel = AxisTargetDerivativeRemover()(knl)

            if fmm_kernel is None:
                fmm_kernel = candidate_fmm_kernel
            else:
                assert fmm_kernel == candidate_fmm_kernel

        return fmm_kernel

    def get_fmm_output_and_expansion_dtype(self, base_kernel, strengths):
        if base_kernel.is_complex_valued or strengths.dtype.kind == "c":
            return self.complex_dtype
        else:
            return self.real_dtype

    def get_fmm_expansion_wrangler_extra_kwargs(
            self, queue, out_kernels, tree_user_source_ids, arguments, evaluator):
        # This contains things like the Helmholtz parameter k or
        # the normal directions for double layers.

        def reorder_sources(source_array):
            if isinstance(source_array, cl.array.Array):
                return (source_array
                        .with_queue(queue)
                        [tree_user_source_ids]
                        .with_queue(None))
            else:
                return source_array

        kernel_extra_kwargs = {}
        source_extra_kwargs = {}

        from sumpy.tools import gather_arguments, gather_source_arguments
        from pytools.obj_array import with_object_array_or_scalar
        for func, var_dict in [
                (gather_arguments, kernel_extra_kwargs),
                (gather_source_arguments, source_extra_kwargs),
                ]:
            for arg in func(out_kernels):
                var_dict[arg.name] = with_object_array_or_scalar(
                        reorder_sources,
                        evaluator(arguments[arg.name]))

        return kernel_extra_kwargs, source_extra_kwargs

    # }}}

# }}}
