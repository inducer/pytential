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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Hashable, Optional, Tuple

import numpy as np
from arraycontext import PyOpenCLArrayContext, flatten, unflatten
from meshmode.dof_array import DOFArray
from pytools import T, memoize_in
from sumpy.fmm import UnableToCollectTimingData
from sumpy.kernel import Kernel
from sumpy.p2p import P2PBase

from pytential import sym

if TYPE_CHECKING:
    from pytential.collection import GeometryCollection

__doc__ = """
.. autoclass:: PotentialSource
.. autoclass:: PointPotentialSource
.. autoclass:: LayerPotentialSourceBase
"""


class PotentialSource(ABC):
    """
    .. autoproperty:: ambient_dim
    .. autoproperty:: ndofs

    .. autoproperty:: real_dtype
    .. autoproperty:: complex_dtype

    .. automethod:: op_group_features
    .. automethod:: get_p2p
    .. automethod:: preprocess_optemplate
    """

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Ambient dimension of the points in the source geometry."""

    @property
    @abstractmethod
    def ndofs(self) -> int:
        """Number of points (DOFs) in the source geometry."""

    @property
    @abstractmethod
    def real_dtype(self):
        """:class:`~numpy.dtype` of real data living on the source geometry."""

    @property
    @abstractmethod
    def complex_dtype(self):
        """:class:`~numpy.dtype` of complex data living on the source geometry."""

    @abstractmethod
    def op_group_features(self, expr: sym.IntG) -> Tuple[Hashable, ...]:
        """
        :arg expr: a subclass of :class:`~pytential.symbolic.primitives.IntG`.
        :returns: a characteristic tuple by which operators that can be
            executed together can be grouped.
        """

    @abstractmethod
    def get_p2p(self,
                actx: PyOpenCLArrayContext,
                target_kernels: Tuple[Kernel, ...],
                source_kernels: Optional[Tuple[Kernel, ...]] = None) -> P2PBase:
        """
        :returns: a subclass of :class:`~sumpy.p2p.P2PBase` for evaluating
            the *target_kernels* and the *source_kernels* on the source geometry.
        """

    def preprocess_optemplate(self,
                name: str,
                discretizations: "GeometryCollection",
                expr: T) -> T:
        """
        :returns: a processed *expr*, where each
            :class:`~pytential.symbolic.primitives.IntG` operator has been
            modified to work with the current source geometry.
        """
        return expr


class _SumpyP2PMixin:

    def get_p2p(self,
                actx: PyOpenCLArrayContext,
                target_kernels: Tuple[Kernel, ...],
                source_kernels: Optional[Tuple[Kernel, ...]] = None) -> P2PBase:
        @memoize_in(actx, (_SumpyP2PMixin, "p2p"))
        def p2p(target_kernels: Tuple[Kernel, ...],
                source_kernels: Optional[Tuple[Kernel, ...]]) -> P2PBase:
            if any(knl.is_complex_valued for knl in target_kernels):
                value_dtype = self.complex_dtype    # type: ignore[attr-defined]
            else:
                value_dtype = self.real_dtype       # type: ignore[attr-defined]

            from sumpy.p2p import P2P
            return P2P(actx.context,
                    target_kernels, exclude_self=False, value_dtypes=value_dtype,
                    source_kernels=source_kernels)

        return p2p(target_kernels, source_kernels)


# {{{ point potential source

def evaluate_kernel_arguments(actx, evaluate, kernel_arguments, flat=True):
    kernel_args = {}
    for arg_name, arg_expr in kernel_arguments.items():
        value = evaluate(arg_expr)

        if flat:
            value = flatten(value, actx, leaf_class=DOFArray)
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
                expr.target_kernel.get_base_kernel(),
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

            _, output_for_each_kernel = p2p(actx.queue,
                    targets=flatten(target_discr.nodes(), actx, leaf_class=DOFArray),
                    sources=self._nodes,
                    strength=strengths, **kernel_args)

            from meshmode.discretization import Discretization
            result = output_for_each_kernel[o.target_kernel_index]
            if isinstance(target_discr, Discretization):
                template_ary = actx.thaw(target_discr.nodes()[0])
                result = unflatten(template_ary, result, actx, strict=False)

            results.append((o.name, result))

        timing_data = {}
        return results, timing_data

# }}}


# {{{ layer potential source

def _entry_dtype(actx, ary):
    from meshmode.dof_array import DOFArray

    if isinstance(ary, DOFArray):
        # the "normal case"
        return ary.entry_dtype
    elif isinstance(ary, np.ndarray):
        if ary.dtype.char == "O":
            from pytools import single_valued
            return single_valued(_entry_dtype(actx, entry) for entry in ary.flat)
        else:
            return ary.dtype
    elif isinstance(ary, actx.array_types):
        # for "unregularized" layer potential sources
        return ary.dtype
    else:
        raise TypeError(f"unexpected type: '{type(ary).__name__}'")


class LayerPotentialSourceBase(_SumpyP2PMixin, PotentialSource):
    """A discretization of a layer potential using element-based geometry, with
    support for refinement and upsampling.

    Inherits from :class:`PotentialSource`.

    .. attribute:: density_discr
    """

    def __init__(self, density_discr):
        self.density_discr = density_discr

    @property
    def _setup_actx(self):
        return self.density_discr._setup_actx

    @property
    def cl_context(self):
        return self._setup_actx.context

    @property
    def ambient_dim(self):
        return self.density_discr.ambient_dim

    @property
    def dim(self):
        return self.density_discr.dim

    @property
    def ndofs(self):
        return self.density_discr.ndofs

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
                _entry_dtype(self._setup_actx, strengths).kind == "c":
            return self.complex_dtype
        else:
            return self.real_dtype

    def get_fmm_expansion_wrangler_extra_kwargs(
            self, actx, target_kernels, tree_user_source_ids, arguments, evaluator):
        # This contains things like the Helmholtz parameter k or
        # the normal directions for double layers.

        def flatten_and_reorder_sources(source_array):
            if isinstance(source_array, DOFArray):
                source_array = flatten(source_array, actx)

            if isinstance(source_array, actx.array_types):
                return actx.freeze(
                        actx.thaw(source_array)[tree_user_source_ids]
                        )
            else:
                return source_array

        kernel_extra_kwargs = {}
        source_extra_kwargs = {}

        from sumpy.tools import gather_arguments, gather_source_arguments
        from arraycontext import rec_map_array_container

        for func, var_dict in [
                (gather_arguments, kernel_extra_kwargs),
                (gather_source_arguments, source_extra_kwargs),
                ]:
            for arg in func(target_kernels):
                var_dict[arg.name] = rec_map_array_container(
                        flatten_and_reorder_sources,
                        evaluator(arguments[arg.name]),
                        leaf_class=DOFArray)

        return kernel_extra_kwargs, source_extra_kwargs

    # }}}

# }}}

# vim: foldmethod=marker
