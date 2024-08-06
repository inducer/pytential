__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

from functools import partial
from typing import Callable, Optional, Union

import numpy as np

from arraycontext import PyOpenCLArrayContext, flatten, unflatten
from meshmode.discretization import Discretization
from meshmode.dof_array import DOFArray
from pytools import memoize_method, memoize_in, single_valued
from sumpy.expansion import DefaultExpansionFactory as DefaultExpansionFactoryBase

from pytential.qbx.cost import AbstractQBXCostModel
from pytential.qbx.target_assoc import QBXTargetAssociationFailedException
from pytential.source import LayerPotentialSourceBase

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: QBXLayerPotentialSource

.. autoclass:: QBXTargetAssociationFailedException

.. autoclass:: DefaultExpansionFactory

.. autoclass:: NonFFTExpansionFactory
"""


# {{{ QBX layer potential source

class DefaultExpansionFactory(DefaultExpansionFactoryBase):
    """A expansion factory to create QBX local, local and multipole expansions
    """
    def get_qbx_local_expansion_class(self, kernel):
        local_expn_class = DefaultExpansionFactoryBase.get_local_expansion_class(
                self, kernel)
        from sumpy.expansion.m2l import NonFFTM2LTranslationClassFactory
        factory = NonFFTM2LTranslationClassFactory()
        m2l_translation = factory.get_m2l_translation_class(kernel,
            local_expn_class)()
        return partial(local_expn_class, m2l_translation=m2l_translation)


class NonFFTExpansionFactory(DefaultExpansionFactoryBase):
    """A expansion factory to create QBX local, local and multipole expansions
    with no FFT for multipole-to-local translations
    """
    get_local_expansion_class = DefaultExpansionFactory.get_qbx_local_expansion_class


class _not_provided:  # noqa: N801
    pass


class _LevelToOrderWrapper:
    """
    Helper functor to convert a constant integer fmm order into a pickable and
    callable object.
    """
    def __init__(self, fmm_order):
        self.fmm_order = fmm_order

    def __call__(self, kernel, kernel_args, tree, level):
        return self.fmm_order


class QBXLayerPotentialSource(LayerPotentialSourceBase):
    """A source discretization for a QBX layer potential.

    .. attribute :: qbx_order
    .. attribute :: fmm_order

    .. automethod :: __init__
    .. automethod :: copy

    See :ref:`qbxguts` for some information on the inner workings of this.
    """

    # {{{ constructor / copy

    def __init__(
            self,
            density_discr: Discretization,
            fine_order: Optional[int],
            qbx_order: Optional[int] = None,
            fmm_order: Optional[Union[bool, int]] = None,
            fmm_level_to_order: Optional[
                Union[bool, Callable[..., int]]
                ] = None,
            expansion_factory: Optional[DefaultExpansionFactoryBase] = None,
            target_association_tolerance: Optional[
                float] = _not_provided,  # type: ignore[assignment]

            # begin experimental arguments
            # FIXME default debug=False once everything has matured
            debug: bool = True,
            _disable_refinement: bool = False,
            _expansions_in_tree_have_extent: bool = True,
            _expansion_stick_out_factor: float = 0.5,
            _max_leaf_refine_weight: Optional[int] = None,
            _box_extent_norm: Optional[str] = None,
            _tree_kind: str = "adaptive",
            _well_sep_is_n_away: int = 2,
            _from_sep_smaller_crit: Optional[str] = None,
            _from_sep_smaller_min_nsources_cumul: Optional[int] = None,
            _use_target_specific_qbx: Optional[bool] = None,
            geometry_data_inspector: Optional[Callable[..., bool]] = None,
            cost_model: Optional[AbstractQBXCostModel] = None,
            fmm_backend: str = "sumpy",
            ) -> None:
        """
        :arg fine_order: The total degree to which the (upsampled)
            underlying quadrature is exact.
        :arg fmm_order: Use *False* for direct calculation or an integer
            otherwise. May not be given (i.e. left as *None*) if
            *fmm_level_to_order* is given.
        :arg fmm_level_to_order: A callable that takes arguments of
            *(kernel, kernel_args, tree, level)* and returns the expansion
            order to be used on a given *level* of *tree* with *kernel*, where
            *kernel* is the :class:`sumpy.kernel.Kernel` being evaluated, and
            *kernel_args* is a set of *(key, value)* tuples with evaluated
            kernel arguments. May not be given if *fmm_order* is given. If used in
            the distributed setting, this argument must be pickable.
        :arg fmm_backend: a string denoting the desired FMM backend to use,
            either `"sumpy"` or `"fmmlib"`. Only used if *fmm_order* or
            *fmm_level_to_order* are provided.
        :arg expansion_factory: used to get local and multipole expansions for
            the FMM evaluations.
        :arg target_association_tolerance: passed on to
            :func:`pytential.qbx.target_assoc.associate_targets_to_qbx_centers`.

        Experimental arguments without a promise of forward compatibility:

        :arg _expansions_in_tree_have_extent: if *True*, target radii are passed
            to the tree build, see :meth:`boxtree.TreeBuilder.__call__`.
        :arg _expansion_stick_out_factor: passed on to the tree builder, see
            :attr:`boxtree.Tree.stick_out_factor` for meaning.
        :arg _max_leaf_refine_weight: passed on to the tree builder, see
            :meth:`boxtree.TreeBuilder.__call__`.
        :arg _box_extent_norm: passed on to the tree builder, see
            :meth:`boxtree.TreeBuilder.__call__`.
        :arg _tree_kind: passed on to the tree builder, see
            :meth:`boxtree.TreeBuilder.__call__`.

        :arg _well_sep_is_n_away: see
            :class:`boxtree.traversal.FMMTraversalBuilder`
        :arg _from_sep_smaller_crit: see
            :class:`boxtree.traversal.FMMTraversalBuilder`.
        :arg _from_sep_smaller_min_nsources_cumul: see
            :meth:`boxtree.traversal.FMMTraversalBuilder.__call__`.
        :arg _use_target_specific_qbx: Whether to use target-specific
            acceleration by default if possible. *None* means "use if possible".

        :arg cost_model: Either *None* or an object implementing the
             :class:`~pytential.qbx.cost.AbstractQBXCostModel` interface, used for
             gathering modeled costs if provided (experimental).
        """

        # {{{ argument processing

        if fine_order is None:
            raise ValueError("'fine_order' must be provided.")
        assert isinstance(fine_order, int)

        if qbx_order is None:
            raise ValueError("'qbx_order' must be provided.")
        assert isinstance(qbx_order, int)

        if target_association_tolerance is _not_provided:
            target_association_tolerance = (
                1.0e+3 * float(np.finfo(density_discr.real_dtype).eps))
        assert isinstance(target_association_tolerance, float)

        if (
                (fmm_order is not None and fmm_level_to_order is not None)
                or (fmm_order is None and fmm_level_to_order is None)):
            raise TypeError(
                "must specify exactly one of 'fmm_order' or 'fmm_level_to_order'.")

        if _box_extent_norm is None:
            _box_extent_norm = "l2"
        assert isinstance(_box_extent_norm, str)

        if _from_sep_smaller_crit is None:
            # This seems to win no matter what the box extent norm is
            # https://gitlab.tiker.net/papers/2017-qbx-fmm-3d/issues/10
            _from_sep_smaller_crit = "precise_linf"
        assert isinstance(_from_sep_smaller_crit, str)

        if fmm_level_to_order is None:
            if fmm_order is False:
                fmm_level_to_order = False
            else:
                assert isinstance(fmm_order, int) and not isinstance(fmm_order, bool)

                fmm_level_to_order = _LevelToOrderWrapper(fmm_order)

        assert isinstance(fmm_level_to_order, bool) or callable(fmm_level_to_order)

        if _max_leaf_refine_weight is None:
            if density_discr.ambient_dim == 2:
                # FIXME: This should be verified now that l^2 is the default.
                _max_leaf_refine_weight = 64
            elif density_discr.ambient_dim == 3:
                # FIXME: this is likely no longer up to date as the translation
                # operators have changed (as of 07-07-2022)
                # For static_linf/linf (private url):
                #   https://gitlab.tiker.net/papers/2017-qbx-fmm-3d/issues/8#note_25009
                # For static_l2/l2 (private url):
                #   https://gitlab.tiker.net/papers/2017-qbx-fmm-3d/issues/12
                _max_leaf_refine_weight = 512
            else:
                # Just guessing...
                _max_leaf_refine_weight = 64
        assert isinstance(_max_leaf_refine_weight, int)

        if _from_sep_smaller_min_nsources_cumul is None:
            # See here for the comment thread that led to these defaults:
            # https://gitlab.tiker.net/inducer/boxtree/merge_requests/28#note_18661
            if density_discr.dim == 1:
                _from_sep_smaller_min_nsources_cumul = 15
            else:
                _from_sep_smaller_min_nsources_cumul = 30
        assert isinstance(_from_sep_smaller_min_nsources_cumul, int)

        if expansion_factory is None:
            expansion_factory = DefaultExpansionFactory()

        if cost_model is None:
            from pytential.qbx.cost import QBXCostModel
            cost_model = QBXCostModel()

        # }}}

        if density_discr.dim != density_discr.ambient_dim - 1:
            raise RuntimeError("QBX requires geometry with codimension one. "
                    f"Got: dim={density_discr.dim} and "
                    f"ambient_dim={density_discr.ambient_dim}.")

        super().__init__(density_discr)

        self.fine_order = fine_order
        self.qbx_order = qbx_order
        self.fmm_level_to_order = fmm_level_to_order
        self.fmm_backend = fmm_backend

        self.expansion_factory = expansion_factory
        self.target_association_tolerance = target_association_tolerance

        self.debug = debug
        self._disable_refinement = _disable_refinement
        self._expansions_in_tree_have_extent = \
                _expansions_in_tree_have_extent
        self._expansion_stick_out_factor = _expansion_stick_out_factor
        self._well_sep_is_n_away = _well_sep_is_n_away
        self._max_leaf_refine_weight = _max_leaf_refine_weight
        self._box_extent_norm = _box_extent_norm
        self._from_sep_smaller_crit = _from_sep_smaller_crit
        self._from_sep_smaller_min_nsources_cumul = \
                _from_sep_smaller_min_nsources_cumul
        self._tree_kind = _tree_kind
        self._use_target_specific_qbx = _use_target_specific_qbx

        self.geometry_data_inspector = geometry_data_inspector
        self.cost_model = cost_model

        # /!\ *All* parameters set here must also be set by copy() below,
        # otherwise they will be reset to their default values behind your
        # back if the layer potential source is ever copied. (such as
        # during refinement)

    def copy(
            self,
            density_discr=None,
            fine_order=None,
            qbx_order=None,
            fmm_order=_not_provided,
            fmm_level_to_order=_not_provided,
            expansion_factory=None,
            target_association_tolerance=_not_provided,
            _expansions_in_tree_have_extent=_not_provided,
            _expansion_stick_out_factor=_not_provided,
            _max_leaf_refine_weight=None,
            _box_extent_norm=None,
            _from_sep_smaller_crit=None,
            _tree_kind=None,
            _use_target_specific_qbx=_not_provided,
            geometry_data_inspector=None,
            cost_model=_not_provided,
            fmm_backend=None,

            debug=_not_provided,
            _disable_refinement=_not_provided,
            ):
        if target_association_tolerance is _not_provided:
            target_association_tolerance = self.target_association_tolerance

        kwargs = {}

        if (fmm_order is not _not_provided
                and fmm_level_to_order is not _not_provided):
            raise TypeError(
                "may not specify both 'fmm_order' and 'fmm_level_to_order'")
        elif fmm_order is not _not_provided:
            kwargs["fmm_order"] = fmm_order
        elif fmm_level_to_order is not _not_provided:
            kwargs["fmm_level_to_order"] = fmm_level_to_order
        else:
            kwargs["fmm_level_to_order"] = self.fmm_level_to_order

        # FIXME Could/should share wrangler and geometry kernels
        # if no relevant changes have been made.
        return type(self)(
                density_discr=density_discr or self.density_discr,
                fine_order=(
                    fine_order if fine_order is not None else self.fine_order),
                qbx_order=qbx_order if qbx_order is not None else self.qbx_order,

                target_association_tolerance=target_association_tolerance,
                expansion_factory=(
                    expansion_factory or self.expansion_factory),

                debug=(
                    # False is a valid value here
                    debug if debug is not _not_provided else self.debug),
                _disable_refinement=(
                    # False is a valid value here
                    _disable_refinement
                    if _disable_refinement is not _not_provided
                    else self._disable_refinement),
                _expansions_in_tree_have_extent=(
                    # False is a valid value here
                    _expansions_in_tree_have_extent
                    if _expansions_in_tree_have_extent is not _not_provided
                    else self._expansions_in_tree_have_extent),
                _expansion_stick_out_factor=(
                    # 0 is a valid value here
                    _expansion_stick_out_factor
                    if _expansion_stick_out_factor is not _not_provided
                    else self._expansion_stick_out_factor),
                _well_sep_is_n_away=self._well_sep_is_n_away,
                _max_leaf_refine_weight=(
                    _max_leaf_refine_weight or self._max_leaf_refine_weight),
                _box_extent_norm=(_box_extent_norm or self._box_extent_norm),
                _from_sep_smaller_crit=(
                    _from_sep_smaller_crit or self._from_sep_smaller_crit),
                _from_sep_smaller_min_nsources_cumul=(
                    self._from_sep_smaller_min_nsources_cumul),
                _tree_kind=_tree_kind or self._tree_kind,
                _use_target_specific_qbx=(_use_target_specific_qbx
                    if _use_target_specific_qbx is not _not_provided
                    else self._use_target_specific_qbx),
                geometry_data_inspector=(
                    geometry_data_inspector or self.geometry_data_inspector),
                cost_model=(
                    # None is a valid value here
                    cost_model
                    if cost_model is not _not_provided
                    else self.cost_model),
                fmm_backend=fmm_backend or self.fmm_backend,
                **kwargs)

    # }}}

    # {{{ internal API

    @memoize_method
    def qbx_fmm_geometry_data(self, places, name,
            target_discrs_and_qbx_sides):
        """
        :arg target_discrs_and_qbx_sides:
            a tuple of *(discr, qbx_forced_limit)*
            tuples, where *discr* is a
            :class:`meshmode.discretization.Discretization`
            or
            :class:`pytential.target.TargetBase`
            instance
        """
        from pytential.qbx.geometry import qbx_fmm_geometry_data_code_container
        code_container = qbx_fmm_geometry_data_code_container(
                self._setup_actx, self.ambient_dim,
                debug=self.debug,
                well_sep_is_n_away=self._well_sep_is_n_away,
                from_sep_smaller_crit=self._from_sep_smaller_crit)

        from pytential.qbx.geometry import QBXFMMGeometryData
        return QBXFMMGeometryData(
                places, name, code_container, target_discrs_and_qbx_sides,
                target_association_tolerance=self.target_association_tolerance,
                tree_kind=self._tree_kind,
                debug=self.debug)

    # }}}

    # {{{ helpers for symbolic operator processing

    def preprocess_optemplate(self, name, discretizations, expr):
        """
        :arg name: The symbolic name for *self*, which the preprocessor
            should use to find which expressions it is allowed to modify.
        """
        from pytential.symbolic.mappers import QBXPreprocessor
        return QBXPreprocessor(name, discretizations)(expr)

    def op_group_features(self, expr):
        from pytential.utils import sort_arrays_together
        result = (
                expr.source, *sort_arrays_together(expr.source_kernels,
                expr.densities, key=str)
                )

        return result

    # }}}

    # {{{ internal functionality for execution

    def exec_compute_potential_insn(self, actx, insn, bound_expr, evaluate,
            return_timing_data):
        extra_args = {}

        if self.fmm_level_to_order is False:
            func = self.exec_compute_potential_insn_direct
            extra_args["return_timing_data"] = return_timing_data

        else:
            func = self.exec_compute_potential_insn_fmm

            def drive_fmm(wrangler, strengths, geo_data, kernel, kernel_arguments):
                del geo_data, kernel, kernel_arguments
                from pytential.qbx.fmm import drive_fmm
                if return_timing_data:
                    timing_data = {}
                else:
                    timing_data = None
                return drive_fmm(wrangler, strengths, timing_data), timing_data

            extra_args["fmm_driver"] = drive_fmm

        return self._dispatch_compute_potential_insn(
                actx, insn, bound_expr, evaluate, func, extra_args)

    def cost_model_compute_potential_insn(self, actx, insn, bound_expr, evaluate,
                                          calibration_params, per_box):
        """Using :attr:`cost_model`, evaluate the cost of executing *insn*.
        Cost model results are gathered in
        :attr:`pytential.symbolic.execution.BoundExpression.modeled_cost`
        along the way.

        :arg calibration_params: a :class:`dict` of calibration parameters, mapping
            from parameter names to calibration values.
        :arg per_box: if *true*, cost model result will be a :class:`numpy.ndarray`
            or :class:`pyopencl.array.Array` with shape of the number of boxes, where
            the ith entry is the sum of the cost of all stages for box i. If *false*,
            cost model result will be a :class:`dict`, mapping from the stage name to
            predicted cost of the stage for all boxes.

        :returns: whatever :meth:`exec_compute_potential_insn_fmm` returns.
        """
        if self.fmm_level_to_order is False:
            raise NotImplementedError("perf modeling direct evaluations")

        def drive_cost_model(
                    wrangler, strengths, geo_data, kernel, kernel_arguments):

            if per_box:
                cost_model_result, metadata = self.cost_model.qbx_cost_per_box(
                    actx.queue, geo_data, kernel, kernel_arguments,
                    calibration_params
                )
            else:
                cost_model_result, metadata = self.cost_model.qbx_cost_per_stage(
                    actx.queue, geo_data, kernel, kernel_arguments,
                    calibration_params
                )

            from pytools.obj_array import obj_array_vectorize
            from functools import partial
            return (
                    obj_array_vectorize(
                        partial(wrangler.finalize_potentials,
                            template_ary=strengths[0]),
                        wrangler.full_output_zeros(strengths[0])),
                    (cost_model_result, metadata))

        return self._dispatch_compute_potential_insn(
            actx, insn, bound_expr, evaluate,
            self.exec_compute_potential_insn_fmm,
            extra_args={"fmm_driver": drive_cost_model}
        )

    def _dispatch_compute_potential_insn(self, actx, insn, bound_expr,
            evaluate, func, extra_args=None):
        if self._disable_refinement:
            from warnings import warn
            warn(
                    "Executing global QBX without refinement. "
                    "This is unlikely to work.")

        if extra_args is None:
            extra_args = {}

        return func(actx, insn, bound_expr, evaluate, **extra_args)

    # {{{ fmm-based execution

    @memoize_method
    def _tree_indep_data_for_wrangler(self, source_kernels, target_kernels):
        from functools import partial
        base_kernel = single_valued(kernel.get_base_kernel() for
            kernel in source_kernels)
        mpole_expn_class = \
                self.expansion_factory.get_multipole_expansion_class(base_kernel)
        local_expn_class = \
                self.expansion_factory.get_local_expansion_class(base_kernel)

        try:
            qbx_local_expn_class = \
                self.expansion_factory.get_qbx_local_expansion_class(base_kernel)
        except AttributeError:
            qbx_local_expn_class = local_expn_class

        fmm_mpole_factory = partial(mpole_expn_class, base_kernel)
        fmm_local_factory = partial(local_expn_class, base_kernel)
        qbx_local_factory = partial(qbx_local_expn_class, base_kernel)

        if self.fmm_backend == "sumpy":
            from pytential.qbx.fmm import \
                    QBXSumpyTreeIndependentDataForWrangler
            return QBXSumpyTreeIndependentDataForWrangler(
                    self.cl_context,
                    fmm_mpole_factory, fmm_local_factory, qbx_local_factory,
                    target_kernels=target_kernels, source_kernels=source_kernels)

        elif self.fmm_backend == "fmmlib":
            source_kernel, = source_kernels
            target_kernels_new = [
                target_kernel.replace_base_kernel(source_kernel) for
                target_kernel in target_kernels
            ]
            from pytential.qbx.fmmlib import \
                    QBXFMMLibTreeIndependentDataForWrangler
            return QBXFMMLibTreeIndependentDataForWrangler(
                    self.cl_context,
                    multipole_expansion_factory=fmm_mpole_factory,
                    local_expansion_factory=fmm_local_factory,
                    qbx_local_expansion_factory=qbx_local_factory,
                    target_kernels=target_kernels_new,
                    _use_target_specific_qbx=self._use_target_specific_qbx)

        else:
            raise ValueError(f"invalid FMM backend: {self.fmm_backend}")

    def get_target_discrs_and_qbx_sides(self, insn, bound_expr):
        """Build the list of unique target discretizations used by the
        provided instruction.
        """
        # map (name, qbx_side) to number in list
        target_name_and_side_to_number = {}
        # list of tuples (discr, qbx_side)
        target_discrs_and_qbx_sides = []

        for o in insn.outputs:
            key = (o.target_name, o.qbx_forced_limit)
            if key not in target_name_and_side_to_number:
                target_name_and_side_to_number[key] = \
                        len(target_discrs_and_qbx_sides)

                target_discr = bound_expr.places.get_discretization(
                        o.target_name.geometry, o.target_name.discr_stage)
                if isinstance(target_discr, LayerPotentialSourceBase):
                    target_discr = target_discr.density_discr

                qbx_forced_limit = o.qbx_forced_limit
                if qbx_forced_limit is None:
                    qbx_forced_limit = 0

                target_discrs_and_qbx_sides.append(
                        (target_discr, qbx_forced_limit))

        return target_name_and_side_to_number, tuple(target_discrs_and_qbx_sides)

    def exec_compute_potential_insn_fmm(self, actx: PyOpenCLArrayContext,
            insn, bound_expr, evaluate, fmm_driver):
        """
        :arg fmm_driver: A function that accepts four arguments:
            *wrangler*, *strength*, *geo_data*, *kernel*, *kernel_arguments*
        :returns: a tuple ``(assignments, extra_outputs)``, where *assignments*
            is a list of tuples containing pairs ``(name, value)`` representing
            assignments to be performed in the evaluation context.
            *extra_outputs* is data that *fmm_driver* may return
            (such as timing data), passed through unmodified.
        """
        target_name_and_side_to_number, target_discrs_and_qbx_sides = (
                self.get_target_discrs_and_qbx_sides(insn, bound_expr))

        geo_data = self.qbx_fmm_geometry_data(
                bound_expr.places,
                insn.source.geometry,
                target_discrs_and_qbx_sides)

        # FIXME Exert more positive control over geo_data attribute lifetimes using
        # geo_data.<method>.clear_cache(geo_data).

        # FIXME Synthesize "bad centers" around corners and edges that have
        # inadequate QBX coverage.

        # FIXME don't compute *all* output kernels on all targets--respect that
        # some target discretizations may only be asking for derivatives (e.g.)

        flat_strengths = get_flat_strengths_from_densities(
                actx, bound_expr.places, evaluate, insn.densities,
                dofdesc=insn.source)

        base_kernel = single_valued(knl.get_base_kernel() for
            knl in insn.source_kernels)

        output_and_expansion_dtype = (
                self.get_fmm_output_and_expansion_dtype(insn.source_kernels,
                    flat_strengths[0]))
        kernel_extra_kwargs, source_extra_kwargs = (
                self.get_fmm_expansion_wrangler_extra_kwargs(
                    actx, insn.target_kernels + insn.source_kernels,
                    geo_data.tree().user_source_ids,
                    insn.kernel_arguments, evaluate))

        tree_indep = self._tree_indep_data_for_wrangler(
                target_kernels=insn.target_kernels,
                source_kernels=insn.source_kernels)

        wrangler = tree_indep.wrangler_cls(
                        tree_indep, geo_data, output_and_expansion_dtype,
                        self.qbx_order,
                        self.fmm_level_to_order,
                        source_extra_kwargs=source_extra_kwargs,
                        kernel_extra_kwargs=kernel_extra_kwargs,
                        _use_target_specific_qbx=self._use_target_specific_qbx,
                        )

        from pytential.qbx.geometry import target_state
        if actx.to_numpy(actx.np.any(
                actx.thaw(geo_data.user_target_to_center())
                == target_state.FAILED)):
            raise RuntimeError("geometry has failed targets")

        # {{{ geometry data inspection hook

        if self.geometry_data_inspector is not None:
            perform_fmm = self.geometry_data_inspector(insn, bound_expr, geo_data)
            if not perform_fmm:
                return [(o.name, 0) for o in insn.outputs]

        # }}}

        # Execute global QBX.
        all_potentials_on_every_target, extra_outputs = (
                fmm_driver(
                    wrangler, flat_strengths, geo_data,
                    base_kernel, kernel_extra_kwargs))

        results = []

        for o in insn.outputs:
            target_side_number = target_name_and_side_to_number[
                    o.target_name, o.qbx_forced_limit]
            target_discr, _ = target_discrs_and_qbx_sides[target_side_number]
            target_slice = slice(*geo_data.target_info().target_discr_starts[
                    target_side_number:target_side_number+2])

            result = \
                all_potentials_on_every_target[o.target_kernel_index][target_slice]

            if isinstance(target_discr, Discretization):
                template_ary = actx.thaw(target_discr.nodes()[0])
                result = unflatten(template_ary, result, actx, strict=False)

            results.append((o.name, result))

        return results, extra_outputs

    # }}}

    # {{{ direct execution

    @memoize_method
    def get_expansion_for_qbx_direct_eval(self, base_kernel, target_kernels):
        from sumpy.expansion.local import LineTaylorLocalExpansion
        from sumpy.kernel import TargetDerivativeRemover

        # line Taylor cannot support target derivatives
        txr = TargetDerivativeRemover()
        if any(knl != txr(knl) for knl in target_kernels):
            return self.expansion_factory.get_local_expansion_class(
                    base_kernel)(base_kernel, self.qbx_order)
        else:
            return LineTaylorLocalExpansion(base_kernel, self.qbx_order)

    @memoize_method
    def get_lpot_applier(self, target_kernels, source_kernels):
        # needs to be separate method for caching

        if any(knl.is_complex_valued for knl in target_kernels):
            value_dtype = self.density_discr.complex_dtype
        else:
            value_dtype = self.density_discr.real_dtype

        base_kernel = single_valued(knl.get_base_kernel() for knl in source_kernels)

        from sumpy.qbx import LayerPotential
        return LayerPotential(self.cl_context,
                    expansion=self.get_expansion_for_qbx_direct_eval(
                        base_kernel, target_kernels),
                    target_kernels=target_kernels, source_kernels=source_kernels,
                    value_dtypes=value_dtype)

    @memoize_method
    def get_lpot_applier_on_tgt_subset(self, target_kernels, source_kernels):
        # needs to be separate method for caching

        if any(knl.is_complex_valued for knl in target_kernels):
            value_dtype = self.density_discr.complex_dtype
        else:
            value_dtype = self.density_discr.real_dtype

        base_kernel = single_valued(knl.get_base_kernel() for knl in source_kernels)

        from pytential.qbx.direct import LayerPotentialOnTargetAndCenterSubset
        from sumpy.expansion.local import VolumeTaylorLocalExpansion
        return LayerPotentialOnTargetAndCenterSubset(
                self.cl_context,
                expansion=VolumeTaylorLocalExpansion(base_kernel, self.qbx_order),
                target_kernels=target_kernels, source_kernels=source_kernels,
                value_dtypes=value_dtype)

    @memoize_method
    def get_qbx_target_numberer(self, dtype):
        assert dtype == np.int32
        from pyopencl.scan import GenericScanKernel
        return GenericScanKernel(
                self.cl_context, np.int32,
                arguments="int *tgt_to_qbx_center, int *qbx_tgt_number, int *count",
                input_expr="tgt_to_qbx_center[i] >= 0 ? 1 : 0",
                scan_expr="a+b", neutral="0",
                output_statement="""
                    if (item != prev_item)
                        qbx_tgt_number[item-1] = i;

                    if (i+1 == N)
                        *count = item;
                    """)

    def exec_compute_potential_insn_direct(self, actx, insn, bound_expr, evaluate,
            return_timing_data):
        from pytential import bind, sym
        from meshmode.discretization import Discretization

        if return_timing_data:
            from pytential.source import UnableToCollectTimingData
            from warnings import warn
            warn(
                    "Timing data collection not supported.",
                    category=UnableToCollectTimingData)

        # {{{ evaluate and flatten inputs

        @memoize_in(bound_expr.places,
                (QBXLayerPotentialSource, "flat_nodes"))
        def _flat_nodes(dofdesc):
            discr = bound_expr.places.get_discretization(
                    dofdesc.geometry, dofdesc.discr_stage)
            return actx.freeze(flatten(discr.nodes(), actx, leaf_class=DOFArray))

        @memoize_in(bound_expr.places,
                (QBXLayerPotentialSource, "flat_expansion_radii"))
        def _flat_expansion_radii(dofdesc):
            radii = bind(
                    bound_expr.places,
                    sym.expansion_radii(self.ambient_dim, dofdesc=dofdesc),
                    )(actx)
            return actx.freeze(flatten(radii, actx))

        @memoize_in(bound_expr.places,
                (QBXLayerPotentialSource, "flat_centers"))
        def _flat_centers(dofdesc, qbx_forced_limit):
            centers = bind(bound_expr.places,
                    sym.expansion_centers(
                        self.ambient_dim, qbx_forced_limit, dofdesc=dofdesc),
                    )(actx)
            return actx.freeze(flatten(centers, actx, leaf_class=DOFArray))

        from pytential.source import evaluate_kernel_arguments
        flat_kernel_args = evaluate_kernel_arguments(
                actx, evaluate, insn.kernel_arguments, flat=True)
        flat_strengths = get_flat_strengths_from_densities(
                actx, bound_expr.places, evaluate, insn.densities,
                dofdesc=insn.source)

        flat_source_nodes = _flat_nodes(insn.source)

        # }}}

        # {{{ partition interactions in target kernels

        from collections import defaultdict
        self_outputs = defaultdict(list)
        other_outputs = defaultdict(list)

        for i, o in enumerate(insn.outputs):
            # For purposes of figuring out whether this is a self-interaction,
            # disregard discr_stage.
            source_dd = insn.source.copy(discr_stage=o.target_name.discr_stage)

            target_discr = bound_expr.places.get_discretization(
                    o.target_name.geometry, o.target_name.discr_stage)
            density_discr = bound_expr.places.get_discretization(
                    source_dd.geometry, source_dd.discr_stage)

            if target_discr is density_discr:
                # NOTE: QBXPreprocessor is supposed to have taken care of this
                assert o.qbx_forced_limit is not None
                assert abs(o.qbx_forced_limit) > 0

                self_outputs[(o.target_name, o.qbx_forced_limit)].append((i, o))
            else:
                qbx_forced_limit = o.qbx_forced_limit
                if qbx_forced_limit is None:
                    qbx_forced_limit = 0

                other_outputs[(o.target_name, qbx_forced_limit)].append((i, o))

        queue = actx.queue
        results = [None] * len(insn.outputs)

        # }}}

        # {{{ self interactions

        # FIXME: Do this all at once

        lpot_applier = self.get_lpot_applier(
                insn.target_kernels, insn.source_kernels)

        for (target_name, qbx_forced_limit), outputs in self_outputs.items():
            target_discr = bound_expr.places.get_discretization(
                    target_name.geometry, target_name.discr_stage)
            flat_target_nodes = _flat_nodes(target_name)

            _, output_for_each_kernel = lpot_applier(queue,
                    targets=flat_target_nodes,
                    sources=flat_source_nodes,
                    centers=_flat_centers(target_name, qbx_forced_limit),
                    strengths=flat_strengths,
                    expansion_radii=_flat_expansion_radii(target_name),
                    **flat_kernel_args)

            for i, o in outputs:
                result = output_for_each_kernel[o.target_kernel_index]
                if isinstance(target_discr, Discretization):
                    template_ary = actx.thaw(target_discr.nodes()[0])
                    result = unflatten(template_ary, result, actx, strict=False)

                results[i] = (o.name, result)

        # }}}

        # {{{ off-surface interactions

        if other_outputs:
            p2p = self.get_p2p(actx, insn.target_kernels, insn.source_kernels)
            lpot_applier_on_tgt_subset = self.get_lpot_applier_on_tgt_subset(
                    insn.target_kernels, insn.source_kernels)

        for (target_name, qbx_forced_limit), outputs in other_outputs.items():
            target_discr = bound_expr.places.get_discretization(
                    target_name.geometry, target_name.discr_stage)
            flat_target_nodes = _flat_nodes(target_name)

            # FIXME: (Somewhat wastefully) compute P2P for all targets
            _, output_for_each_kernel = p2p(  # pylint: disable=possibly-used-before-assignment
                  queue,
                  targets=flat_target_nodes,
                  sources=flat_source_nodes,
                  strength=flat_strengths,
                  **flat_kernel_args)

            target_discrs_and_qbx_sides = ((target_discr, qbx_forced_limit),)
            geo_data = self.qbx_fmm_geometry_data(
                    bound_expr.places,
                    insn.source.geometry,
                    target_discrs_and_qbx_sides=target_discrs_and_qbx_sides)

            # center-related info is independent of targets

            # First ncenters targets are the centers
            tgt_to_qbx_center = actx.np.copy(actx.thaw(
                    geo_data.user_target_to_center()[geo_data.ncenters:]
                    ))

            qbx_tgt_numberer = self.get_qbx_target_numberer(
                    tgt_to_qbx_center.dtype)
            qbx_tgt_count = actx.zeros((), np.int32)
            qbx_tgt_numbers = actx.np.zeros_like(tgt_to_qbx_center)

            qbx_tgt_numberer(
                    tgt_to_qbx_center, qbx_tgt_numbers, qbx_tgt_count,
                    queue=queue)

            qbx_tgt_count = int(actx.to_numpy(qbx_tgt_count).item())
            if (abs(qbx_forced_limit) == 1 and qbx_tgt_count < target_discr.ndofs):
                raise RuntimeError(
                        "Did not find a matching QBX center for some targets")

            qbx_tgt_numbers = qbx_tgt_numbers[:qbx_tgt_count]
            qbx_center_numbers = tgt_to_qbx_center[qbx_tgt_numbers]
            qbx_center_numbers.finish()

            tgt_subset_kwargs = flat_kernel_args.copy()
            for i, res_i in enumerate(output_for_each_kernel):
                tgt_subset_kwargs[f"result_{i}"] = res_i

            if qbx_tgt_count:
                lpot_applier_on_tgt_subset(  # pylint: disable=possibly-used-before-assignment
                        queue,
                        targets=flat_target_nodes,
                        sources=flat_source_nodes,
                        centers=geo_data.flat_centers(),
                        expansion_radii=geo_data.flat_expansion_radii(),
                        strengths=flat_strengths,
                        qbx_tgt_numbers=qbx_tgt_numbers,
                        qbx_center_numbers=qbx_center_numbers,
                        **tgt_subset_kwargs)

            for i, o in outputs:
                result = output_for_each_kernel[o.target_kernel_index]
                if isinstance(target_discr, Discretization):
                    template_ary = actx.thaw(target_discr.nodes()[0])
                    result = unflatten(template_ary, result, actx, strict=False)

                results[i] = (o.name, result)

        # }}}

        timing_data = {}
        return results, timing_data

    # }}}

    # }}}


def get_flat_strengths_from_densities(
        actx, places, evaluate, densities, dofdesc=None):
    from pytential import bind, sym
    waa = bind(
            places,
            sym.weights_and_area_elements(places.ambient_dim, dofdesc=dofdesc),
            )(actx)
    density_dofarrays = [evaluate(density) for density in densities]
    for i, ary in enumerate(density_dofarrays):
        if not isinstance(ary, DOFArray):
            raise ValueError(
                f"DOFArray expected for density '{densities[i]}', "
                f"{type(ary)} received instead")

        # FIXME Maybe check shape?

    return [flatten(waa * density_dofarray, actx)
            for density_dofarray in density_dofarrays]

# }}}


__all__ = (
        "QBXLayerPotentialSource",
        "QBXTargetAssociationFailedException",
        )

# vim: fdm=marker
