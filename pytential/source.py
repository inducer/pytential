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

import numpy as np
import pyopencl as cl

from pytools import memoize_in
from arraycontext import thaw, flatten, unflatten

from sumpy.fmm import UnableToCollectTimingData


__doc__ = """
.. autoclass:: PotentialSource
.. autoclass:: PointPotentialSource
.. autoclass:: LayerPotentialSourceBase
"""


class PotentialSource:
    """
    .. automethod:: preprocess_optemplate

    .. method:: op_group_features(expr)

        Return a characteristic tuple by which operators that can be
        executed together can be grouped.

        *expr* is a subclass of
        :class:`pytential.symbolic.primitives.IntG`.
    """

    def preprocess_optemplate(self, name, discretizations, expr):
        return expr

    @property
    def real_dtype(self):
        raise NotImplementedError

    @property
    def complex_dtype(self):
        raise NotImplementedError

    def get_p2p(self, actx, kernels):
        raise NotImplementedError


class _SumpyP2PMixin:

    def get_p2p(self, actx, target_kernels, source_kernels=None):
        @memoize_in(actx, (_SumpyP2PMixin, "p2p"))
        def p2p(target_kernels, source_kernels):
            if any(knl.is_complex_valued for knl in target_kernels):
                value_dtype = self.complex_dtype
            else:
                value_dtype = self.real_dtype

            from sumpy.p2p import P2P
            return P2P(actx.context,
                    target_kernels, exclude_self=False, value_dtypes=value_dtype,
                    source_kernels=source_kernels)

        return p2p(target_kernels, source_kernels)


# {{{ point potential source

def evaluate_kernel_arguments(actx, evaluate, kernel_arguments, flat=True):
    from meshmode.dof_array import DOFArray

    kernel_args = {}
    for arg_name, arg_expr in kernel_arguments.items():
        value = evaluate(arg_expr)

        if flat:
            if isinstance(value, np.ndarray):
                value = actx.np.reshape(flatten(value, actx), (value.size, -1))
            elif isinstance(value, DOFArray):
                value = flatten(value, actx)
            else:
                pass

        kernel_args[arg_name] = value

    return kernel_args


class PointPotentialSource(_SumpyP2PMixin, PotentialSource):
    """
    .. attribute:: nodes

        An :class:`pyopencl.array.Array` of shape ``[ambient_dim, ndofs]``.

    .. attribute:: ndofs

    .. automethod:: cost_model_compute_potential_insn
    .. automethod:: exec_compute_potential_insn
    """

    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def points(self):
        from warnings import warn
        warn("'points' has been renamed to nodes().",
             DeprecationWarning, stacklevel=2)

        return self._nodes

    def nodes(self):
        return self._nodes

    @property
    def real_dtype(self):
        return self._nodes.dtype

    @property
    def ndofs(self):
        for coord_ary in self._nodes:
            return coord_ary.shape[0]

    @property
    def complex_dtype(self):
        return {
                np.float32: np.complex64,
                np.float64: np.complex128
                }[self.real_dtype.type]

    @property
    def ambient_dim(self):
        return self._nodes.shape[0]

    def op_group_features(self, expr):
        from pytential.utils import sort_arrays_together
        # since IntGs with the same source kernels and densities calculations
        # for P2E and E2E are the same and only differs in E2P depending on the
        # target kernel, we group all IntGs with same source kernels and densities.
        # sorting is done to avoid duplicates as the order of the sum of source
        # kernels does not matter.
        result = (
                expr.source,
                *sort_arrays_together(expr.source_kernels, expr.densities, key=str),
                expr.target_kernel,
                )

        return result

    def cost_model_compute_potential_insn(self, actx, insn, bound_expr,
                                          evaluate, costs):
        raise NotImplementedError

    def exec_compute_potential_insn(self, actx, insn, bound_expr, evaluate,
            return_timing_data):
        if return_timing_data:
            from warnings import warn
            warn(
                   "Timing data collection not supported.",
                   category=UnableToCollectTimingData)

        p2p = None

        kernel_args = evaluate_kernel_arguments(
                actx, evaluate, insn.kernel_arguments, flat=False)
        strengths = [evaluate(density) for density in insn.densities]

        # FIXME: Do this all at once
        results = []
        for o in insn.outputs:
            target_discr = bound_expr.places.get_discretization(
                    o.target_name.geometry, o.target_name.discr_stage)

            # no on-disk kernel caching
            if p2p is None:
                p2p = self.get_p2p(actx, source_kernels=insn.source_kernels,
                target_kernels=insn.target_kernels)

            evt, output_for_each_kernel = p2p(actx.queue,
                    targets=actx.np.reshape(
                        flatten(target_discr.nodes(), actx),
                        (target_discr.ambient_dim, -1)),
                    sources=self._nodes,
                    strength=strengths, **kernel_args)

            from meshmode.discretization import Discretization
            result = output_for_each_kernel[o.target_kernel_index]
            if isinstance(target_discr, Discretization):
                template_ary = thaw(target_discr.nodes()[0], actx)
                result = unflatten(template_ary, result, actx)

            results.append((o.name, result))

        timing_data = {}
        return results, timing_data

# }}}


# {{{ layer potential source

def _entry_dtype(ary):
    from meshmode.dof_array import DOFArray
    if isinstance(ary, DOFArray):
        # the "normal case"
        return ary.entry_dtype
    elif isinstance(ary, np.ndarray):
        if ary.dtype.char == "O":
            from pytools import single_valued
            return single_valued(_entry_dtype(entry) for entry in ary.flat)
        else:
            return ary.dtype
    elif isinstance(ary, cl.array.Array):
        # for "unregularized" layer potential sources
        return ary.dtype
    else:
        raise TypeError(f"unexpected type '{type(ary)}' in _entry_dtype")


class LayerPotentialSourceBase(_SumpyP2PMixin, PotentialSource):
    """A discretization of a layer potential using panel-based geometry, with
    support for refinement and upsampling.

    .. rubric:: Discretization data

    .. attribute:: density_discr
    .. attribute:: cl_context
    .. attribute:: ambient_dim
    .. attribute:: dim
    .. attribute:: real_dtype
    .. attribute:: complex_dtype

    """

    def __init__(self, density_discr):
        self.density_discr = density_discr

    @property
    def ambient_dim(self):
        return self.density_discr.ambient_dim

    @property
    def _setup_actx(self):
        return self.density_discr._setup_actx

    @property
    def dim(self):
        return self.density_discr.dim

    @property
    def cl_context(self):
        return self._setup_actx.context

    @property
    def real_dtype(self):
        return self.density_discr.real_dtype

    @property
    def complex_dtype(self):
        return self.density_discr.complex_dtype

    # {{{ fmm setup helpers

    def get_fmm_kernel(self, kernels):
        fmm_kernel = None

        from sumpy.kernel import TargetTransformationRemover
        for knl in kernels:
            candidate_fmm_kernel = TargetTransformationRemover()(knl)

            if fmm_kernel is None:
                fmm_kernel = candidate_fmm_kernel
            else:
                assert fmm_kernel == candidate_fmm_kernel

        return fmm_kernel

    def get_fmm_output_and_expansion_dtype(self, kernels, strengths):
        if any(knl.is_complex_valued for knl in kernels) or \
                _entry_dtype(strengths).kind == "c":
            return self.complex_dtype
        else:
            return self.real_dtype

    def get_fmm_expansion_wrangler_extra_kwargs(
            self, actx, target_kernels, tree_user_source_ids, arguments, evaluator):
        # This contains things like the Helmholtz parameter k or
        # the normal directions for double layers.

        def reorder_sources(source_array):
            if isinstance(source_array, cl.array.Array):
                return source_array[tree_user_source_ids].with_queue(None)
            else:
                return source_array

        kernel_extra_kwargs = {}
        source_extra_kwargs = {}

        from sumpy.tools import gather_arguments, gather_source_arguments
        from arraycontext import map_array_container
        from meshmode.dof_array import DOFArray

        for func, var_dict in [
                (gather_arguments, kernel_extra_kwargs),
                (gather_source_arguments, source_extra_kwargs),
                ]:
            for arg in func(target_kernels):
                value = evaluator(arguments[arg.name])
                if isinstance(value, np.ndarray):
                    value = map_array_container(
                            lambda ary: reorder_sources(flatten(value, actx)),
                            value)
                elif isinstance(value, DOFArray):
                    value = reorder_sources(flatten(value, actx))
                else:
                    pass

                var_dict[arg.name] = value

        return kernel_extra_kwargs, source_extra_kwargs

    # }}}

# }}}

# vim: foldmethod=marker
