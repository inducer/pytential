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

from pytools import memoize_in, memoize_method
from arraycontext import flatten, unflatten
from meshmode.dof_array import DOFArray
from meshmode.discretization import Discretization
from arraycontext import ArrayContext

from sumpy.fmm import (SumpyTimingFuture,
    SumpyTreeIndependentDataForWrangler, SumpyExpansionWrangler)
from sumpy.expansion import DefaultExpansionFactory

from functools import partial
from collections import defaultdict
from typing import Optional, Mapping, Union, Callable


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
    kernel_args = {}
    for arg_name, arg_expr in kernel_arguments.items():
        value = evaluate(arg_expr)

        if flat:
            value = flatten(value, actx, leaf_class=DOFArray)
        kernel_args[arg_name] = value

    return kernel_args


default_expansion_factory = DefaultExpansionFactory()


class PointPotentialSource(_SumpyP2PMixin, PotentialSource):
    """
    .. method:: nodes

        An :class:`pyopencl.array.Array` of shape ``[ambient_dim, ndofs]``.

    .. attribute:: ndofs

    .. automethod:: cost_model_compute_potential_insn
    .. automethod:: exec_compute_potential_insn
    """

    def __init__(self, nodes, *,
            fmm_order: Optional[int] = False,
            fmm_level_to_order: Optional[Union[bool, Callable[..., int]]] = None,
            expansion_factory: Optional[DefaultExpansionFactory]
                    = default_expansion_factory,
            tree_build_kwargs: Optional[Mapping] = None,
            trav_build_kwargs: Optional[Mapping] = None,
            setup_actx: Optional[ArrayContext] = None):
        """
        :arg nodes: The point potential source given as a
               :class:`pyopencl.array.Array`
        :arg fmm_order: The order of the FMM for all levels if *fmm_order* is not
               *False*. Mutually exclusive with argument *fmm_level_to_order*.
               If both arguments are not given a direct point-to-point calculation
               is used.
        :arg fmm_level_to_order: An optional callable that returns the FMM order
               to use for a given level. Mutually exclusive with *fmm_order*
               argument.
        :arg expansion_factory: An expansion factory to get the expansion objects
               when an FMM is used.
        :arg tree_build_kwargs: Keyword arguments to be passed when building the
               tree for an FMM.
        :arg trav_build_kwargs: Keyword arguments to be passed when building a
               traversal for an FMM.
        :arg setup_actx: An array context to be used when building a tree
              for an FMM.
        """

        if fmm_order is not False and fmm_level_to_order is not None:
            raise TypeError("may not specify both fmm_order and fmm_level_to_order")

        if fmm_level_to_order is None:
            if fmm_order is not False:
                def fmm_level_to_order(kernel, kernel_args, tree, level):  # noqa pylint:disable=function-redefined
                    return fmm_order
            else:
                fmm_level_to_order = False
        self.fmm_level_to_order = fmm_level_to_order
        self.expansion_factory = expansion_factory
        self.tree_build_kwargs = tree_build_kwargs if tree_build_kwargs else {}
        self.trav_build_kwargs = trav_build_kwargs if trav_build_kwargs else {}
        self._setup_actx = setup_actx
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

    def copy(self, *, nodes=None, fmm_order=None, fmm_level_to_order=None,
             expansion_factory=None, tree_build_kwargs=None, trav_build_kwargs=None,
             setup_actx=None):
        if nodes is None:
            nodes = self._nodes
        if setup_actx is None:
            setup_actx = self._setup_actx
        if fmm_level_to_order is None and fmm_order is None:
            fmm_level_to_order = self.fmm_level_to_order
        if expansion_factory is None:
            expansion_factory = self.expansion_factory
        if tree_build_kwargs is None:
            tree_build_kwargs = self.tree_build_kwargs
        if trav_build_kwargs is None:
            trav_build_kwargs = self.trav_build_kwargs

        return type(self)(
            nodes=nodes,
            fmm_order=fmm_order,
            fmm_level_to_order=fmm_level_to_order,
            expansion_factory=expansion_factory,
            tree_build_kwargs=tree_build_kwargs,
            trav_build_kwargs=trav_build_kwargs,
            setup_actx=setup_actx,
        )

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

    @memoize_method
    def _get_tree(self, target_discr):
        """Builds a tree for targets given by *target_discr* and caches the
        result. Needed only when an FMM is used.
        """
        from boxtree import TreeBuilder
        from boxtree.traversal import FMMTraversalBuilder

        actx = self._setup_actx
        sources = self._nodes
        targets = flatten(target_discr.nodes(), actx, leaf_class=DOFArray)
        tree_build = TreeBuilder(actx.context)
        trav_build = FMMTraversalBuilder(actx.context,
                **self.trav_build_kwargs)
        tree, _ = tree_build(actx.queue, sources, targets=targets,
                **self.tree_build_kwargs)
        trav, _ = trav_build(actx.queue, tree)
        return tree, trav

    @memoize_method
    def _get_exec_insn_func(self, source_kernels, target_kernels, target_discr):
        if self.fmm_level_to_order is False:
            def exec_insn(actx, strengths, kernel_args, dtype, return_timing_data):
                sources = self._nodes
                targets = flatten(target_discr.nodes(), actx, leaf_class=DOFArray)
                p2p = self.get_p2p(actx, source_kernels=source_kernels,
                    target_kernels=target_kernels)

                evt, output = p2p(actx.queue,
                    targets=targets,
                    sources=sources,
                    strength=strengths, **kernel_args)

                if return_timing_data:
                    timing_data = {"eval_direct":
                        SumpyTimingFuture(actx.queue, [evt]).result()}
                else:
                    timing_data = None
                return timing_data, output
        else:
            from boxtree.fmm import drive_fmm

            kernel = target_kernels[0].get_base_kernel()
            local_expansion_factory = \
                self.expansion_factory.get_local_expansion_class(kernel)
            local_expansion_factory = partial(local_expansion_factory, kernel)
            mpole_expansion_factory = \
                self.expansion_factory.get_multipole_expansion_class(kernel)
            mpole_expansion_factory = partial(mpole_expansion_factory, kernel)

            tree, trav = self._get_tree(target_discr)

            def exec_insn(actx, strengths, kernel_args, dtype, return_timing_data):
                tree_indep = SumpyTreeIndependentDataForWrangler(
                    actx.context,
                    mpole_expansion_factory,
                    local_expansion_factory,
                    target_kernels=target_kernels,
                    source_kernels=source_kernels,
                )
                wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
                    fmm_level_to_order=self.fmm_level_to_order,
                    kernel_extra_kwargs=kernel_args)
                timing_data = {} if return_timing_data else None
                output = drive_fmm(wrangler, strengths, timing_data=timing_data)
                return timing_data, output

        return exec_insn

    def exec_compute_potential_insn(self, actx, insn, bound_expr, evaluate,
            return_timing_data):
        kernel_args = evaluate_kernel_arguments(
                actx, evaluate, insn.kernel_arguments, flat=False)
        strengths = [evaluate(density) for density in insn.densities]

        if any(knl.is_complex_valued for knl in insn.target_kernels) or \
                any(_entry_dtype(actx, strength).kind == "c" for
                    strength in strengths):
            dtype = self.complex_dtype
        else:
            dtype = self.real_dtype

        outputs_grouped_by_target = defaultdict(list)
        for o in insn.outputs:
            outputs_grouped_by_target[o.target_name].append(o)

        results = []
        timing_data_arr = []
        for target_name, output_group in outputs_grouped_by_target.items():
            target_discr = bound_expr.places.get_discretization(
                target_name.geometry, target_name.discr_stage)

            exec_insn = self._get_exec_insn_func(
                source_kernels=insn.source_kernels,
                target_kernels=insn.target_kernels,
                target_discr=target_discr,
            )

            timing_data, output_for_each_kernel = \
                exec_insn(actx, strengths, kernel_args,
                          dtype, return_timing_data)
            timing_data_arr.append(timing_data)

            for o in output_group:
                result = output_for_each_kernel[o.target_kernel_index]
                if isinstance(target_discr, Discretization):
                    template_ary = actx.thaw(target_discr.nodes()[0])
                    result = unflatten(template_ary, result, actx, strict=False)

                results.append((o.name, result))

        timing_data = defaultdict(list)
        if return_timing_data and timing_data_arr:
            for timing_data in timing_data_arr:
                for description, result in timing_data.items():
                    timing_data[description].merge(result)

        return results, dict(timing_data)

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
