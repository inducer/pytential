from __future__ import annotations


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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from typing_extensions import override

from arraycontext import (
    Array,
    ArrayOrContainerOrScalar,
    PyOpenCLArrayContext,
    flatten,
    unflatten,
)
from meshmode.dof_array import DOFArray
from pytools import T, memoize_in
from sumpy.kernel import Kernel


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    import pyopencl as cl
    from meshmode.discretization import Discretization
    from sumpy.kernel import Kernel
    from sumpy.p2p import P2P

    from pytential.collection import GeometryCollection
    from pytential.symbolic.compiler import ComputePotential
    from pytential.symbolic.execution import BoundExpression, EvaluationMapperBase
    from pytential.symbolic.primitives import IntG, KernelArgumentMapping, Operand

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
    def real_dtype(self) -> np.dtype[np.floating]:
        """:class:`~numpy.dtype` of real data living on the source geometry."""

    @property
    @abstractmethod
    def complex_dtype(self) -> np.dtype[np.complexfloating]:
        """:class:`~numpy.dtype` of complex data living on the source geometry."""

    @abstractmethod
    def op_group_features(self, expr: IntG) -> tuple[Hashable, ...]:
        """
        :arg expr: a subclass of :class:`~pytential.symbolic.primitives.IntG`.
        :returns: a characteristic tuple by which operators that can be
            executed together can be grouped.
        """

    def get_p2p(self,
                actx: PyOpenCLArrayContext,
                target_kernels: tuple[Kernel, ...],
                source_kernels: tuple[Kernel, ...] | None = None) -> P2P:
        """
        :returns: a subclass of :class:`~sumpy.p2p.P2P` for evaluating
            the *target_kernels* and the *source_kernels* on the source geometry.
        """

        @memoize_in(actx, (PotentialSource, "p2p"))
        def p2p(target_kernels: tuple[Kernel, ...],
                source_kernels: tuple[Kernel, ...] | None) -> P2P:
            if any(knl.is_complex_valued for knl in target_kernels):
                value_dtype = self.complex_dtype
            else:
                value_dtype = self.real_dtype

            from sumpy.p2p import P2P
            return P2P(target_kernels, exclude_self=False, value_dtypes=value_dtype,
                       source_kernels=source_kernels)

        return p2p(target_kernels, source_kernels)

    def preprocess_optemplate(self,
                name: Hashable,
                discretizations: GeometryCollection,
                expr: T) -> T:
        """
        :returns: a processed *expr*, where each
            :class:`~pytential.symbolic.primitives.IntG` operator has been
            modified to work with the current source geometry.
        """
        return expr


# {{{ point potential source

def evaluate_kernel_arguments(
        actx: PyOpenCLArrayContext,
        evaluate: EvaluationMapperBase,
        kernel_arguments: KernelArgumentMapping,
        flat: bool = True,
    ) -> dict[str, ArrayOrContainerOrScalar]:
    from arraycontext.typing import is_scalar_like

    kernel_args: dict[str, ArrayOrContainerOrScalar] = {}
    for arg_name, arg_expr in kernel_arguments.items():
        value = evaluate(arg_expr)

        if flat and not is_scalar_like(value):
            value = flatten(value, actx, leaf_class=DOFArray)

        kernel_args[arg_name] = value

    return kernel_args


class PointPotentialSource(PotentialSource):
    """
    .. attribute:: ndofs

    .. automethod:: nodes
    .. automethod:: cost_model_compute_potential_insn
    .. automethod:: exec_compute_potential_insn
    """

    def __init__(self, nodes: Array) -> None:
        self._nodes: Array = nodes

    @property
    def points(self) -> Array:
        from warnings import warn
        warn("'points' has been renamed to nodes(). It will be removed in 2026.",
             DeprecationWarning, stacklevel=2)

        return self._nodes

    def nodes(self) -> Array:
        """
        :returns: an :class:`~arraycontext.Array` of shape ``[ambient_dim, ndofs]``.
        """
        return self._nodes

    @property
    @override
    def real_dtype(self) -> np.dtype[np.floating]:
        return self._nodes.dtype

    @property
    @override
    def ndofs(self) -> int:
        for coord_ary in self._nodes:
            axis = coord_ary.shape[0]
            assert isinstance(axis, int)

            return axis

        raise AttributeError(
            f"type object '{type(self).__name__}' has no attribute 'ndofs'")

    @property
    @override
    def complex_dtype(self) -> np.dtype[np.complexfloating]:
        return {
                np.float32: np.dtype(np.complex64),
                np.float64: np.dtype(np.complex128)
                }[self.real_dtype.type]

    @property
    @override
    def ambient_dim(self) -> int:
        dim = self._nodes.shape[0]
        assert isinstance(dim, int)

        return dim

    @override
    def op_group_features(self, expr: IntG) -> tuple[Hashable, ...]:
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

    def cost_model_compute_potential_insn(
            self,
            actx: PyOpenCLArrayContext,
            insn: ComputePotential,
            bound_expr: BoundExpression[Any],
            evaluate: EvaluationMapperBase,
            calibration_params: dict[str, float],
            per_box: bool,
        ) -> list[tuple[str, DOFArray]]:
        raise NotImplementedError

    def exec_compute_potential_insn(
            self,
            actx: PyOpenCLArrayContext,
            insn: ComputePotential,
            bound_expr: BoundExpression[Any],
            evaluate: EvaluationMapperBase,
        ) -> list[tuple[str, ArrayOrContainerOrScalar]]:
        p2p = None
        kernel_args = evaluate_kernel_arguments(
                actx, evaluate, insn.kernel_arguments, flat=False)
        strengths = [cast("Array", evaluate(density)) for density in insn.densities]

        from meshmode.discretization import Discretization

        # FIXME: Do this all at once
        results: list[tuple[str, ArrayOrContainerOrScalar]] = []
        for o in insn.outputs:
            target_or_discr = bound_expr.places.get_target_or_discretization(
                    o.target_name.geometry, o.target_name.discr_stage)

            # no on-disk kernel caching
            if p2p is None:
                p2p = self.get_p2p(actx, source_kernels=insn.source_kernels,
                target_kernels=insn.target_kernels)

            output_for_each_kernel = p2p(actx,
                    targets=flatten(target_or_discr.nodes(), actx, leaf_class=DOFArray),
                    sources=self._nodes,
                    strength=strengths, **kernel_args)

            result = output_for_each_kernel[o.target_kernel_index]
            if isinstance(target_or_discr, Discretization):
                template_ary = actx.thaw(target_or_discr.nodes()[0])
                result = unflatten(template_ary, result, actx, strict=False)

            results.append((o.name, result))

        return results

# }}}


# {{{ layer potential source

def _entry_dtype(actx: PyOpenCLArrayContext,
                 ary: ArrayOrContainerOrScalar) -> np.dtype[Any]:
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
    elif actx.is_array_type(ary):
        # for "unregularized" layer potential sources
        return ary.dtype
    else:
        raise TypeError(f"unexpected type: '{type(ary).__name__}'")


class LayerPotentialSourceBase(PotentialSource, ABC):
    """A discretization of a layer potential using element-based geometry, with
    support for refinement and upsampling.

    Inherits from :class:`PotentialSource`.

    .. attribute:: density_discr

    .. attribute:: ambient_dim
    .. attribute:: dim
    .. attribute:: real_dtype
    .. attribute:: complex_dtype
    """

    def __init__(self, density_discr: Discretization):
        self.density_discr: Discretization = density_discr

    @property
    def _setup_actx(self) -> PyOpenCLArrayContext:
        actx = self.density_discr._setup_actx
        assert isinstance(actx, PyOpenCLArrayContext)
        return actx

    @property
    def cl_context(self) -> cl.Context:
        return self._setup_actx.context

    @property
    @override
    def ambient_dim(self) -> int:
        return self.density_discr.ambient_dim

    @property
    def dim(self) -> int:
        return self.density_discr.dim

    @property
    @override
    def ndofs(self) -> int:
        return self.density_discr.ndofs

    @property
    @override
    def real_dtype(self) -> np.dtype[np.floating]:
        return self.density_discr.real_dtype

    @property
    @override
    def complex_dtype(self) -> np.dtype[np.complexfloating]:
        return self.density_discr.complex_dtype

    # {{{ fmm setup helpers

    def get_fmm_kernel(self, kernels: Iterable[Kernel]) -> Kernel | None:
        fmm_kernel = None

        from sumpy.kernel import TargetTransformationRemover
        for knl in kernels:
            candidate_fmm_kernel = TargetTransformationRemover()(knl)

            if fmm_kernel is None:
                fmm_kernel = candidate_fmm_kernel
            else:
                assert fmm_kernel == candidate_fmm_kernel

        return fmm_kernel

    def get_fmm_output_and_expansion_dtype(
            self,
            kernels: Iterable[Kernel],
            strengths: ArrayOrContainerOrScalar) -> np.dtype[Any]:
        if (
                any(knl.is_complex_valued for knl in kernels)
                or _entry_dtype(self._setup_actx, strengths).kind == "c"):
            return self.complex_dtype
        else:
            return self.real_dtype

    def get_fmm_expansion_wrangler_extra_kwargs(
            self,
            actx: PyOpenCLArrayContext,
            target_kernels: tuple[Kernel, ...],
            tree_user_source_ids: Array,
            arguments: KernelArgumentMapping,
            evaluator: Callable[[Operand], ArrayOrContainerOrScalar],
        ) -> tuple[dict[str, ArrayOrContainerOrScalar],
                   dict[str, ArrayOrContainerOrScalar]]:
        # This contains things like the Helmholtz parameter k or
        # the normal directions for double layers.

        def flatten_and_reorder_sources(
                source_array: ArrayOrContainerOrScalar,
            ) -> ArrayOrContainerOrScalar:
            if isinstance(source_array, DOFArray):
                source_array = flatten(source_array, actx)

            if actx.is_array_type(source_array):
                return actx.freeze(
                        actx.thaw(source_array)[tree_user_source_ids]
                        )
            else:
                return source_array

        kernel_extra_kwargs: dict[str, ArrayOrContainerOrScalar] = {}
        source_extra_kwargs: dict[str, ArrayOrContainerOrScalar] = {}

        from arraycontext import rec_map_array_container
        from sumpy.tools import gather_arguments, gather_source_arguments

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
