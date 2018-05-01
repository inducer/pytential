# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

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

import six

import numpy as np
from pytools import memoize_method
from meshmode.discretization import Discretization
from pytential.qbx.target_assoc import QBXTargetAssociationFailedException
from pytential.source import LayerPotentialSourceBase

import pyopencl as cl

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: QBXLayerPotentialSource

.. autoclass:: QBXTargetAssociationFailedException
"""


# {{{ QBX layer potential source

class _not_provided:  # noqa: N801
    pass


class QBXLayerPotentialSource(LayerPotentialSourceBase):
    """A source discretization for a QBX layer potential.

    .. attribute :: qbx_order
    .. attribute :: fmm_order

    See :ref:`qbxguts` for some information on the inner workings of this.
    """

    # {{{ constructor / copy

    def __init__(self,
            density_discr,
            fine_order,
            qbx_order=None,
            fmm_order=None,
            fmm_level_to_order=None,
            to_refined_connection=None,
            expansion_factory=None,
            target_association_tolerance=_not_provided,

            # begin undocumented arguments
            # FIXME default debug=False once everything has matured
            debug=True,
            _refined_for_global_qbx=False,
            _expansions_in_tree_have_extent=True,
            _expansion_stick_out_factor=0.5,
            _well_sep_is_n_away=2,
            _max_leaf_refine_weight=None,
            _box_extent_norm=None,
            _from_sep_smaller_crit=None,
            _from_sep_smaller_min_nsources_cumul=None,
            _tree_kind="adaptive",
            geometry_data_inspector=None,
            fmm_backend="sumpy",
            target_stick_out_factor=_not_provided):
        """
        :arg fine_order: The total degree to which the (upsampled)
             underlying quadrature is exact.
        :arg to_refined_connection: A connection used for resampling from
             *density_discr* the fine density discretization.  It is assumed
             that the fine density discretization given by
             *to_refined_connection.to_discr* is *not* already upsampled. May
             be *None*.
        :arg fmm_order: `False` for direct calculation. May not be given if
            *fmm_level_to_order* is given.
        :arg fmm_level_to_order: A function that takes arguments of
             *(kernel, kernel_args, tree, level)* and returns the expansion
             order to be used on a given *level* of *tree* with *kernel*, where
             *kernel* is the :class:`sumpy.kernel.Kernel` being evaluated, and
             *kernel_args* is a set of *(key, value)* tuples with evaluated
             kernel arguments. May not be given if *fmm_order* is given.
        """

        # {{{ argument processing

        if target_stick_out_factor is not _not_provided:
            from warnings import warn
            warn("target_stick_out_factor has been renamed to "
                    "target_association_tolerance. "
                    "Using target_stick_out_factor is deprecated "
                    "and will stop working in 2018.",
                    DeprecationWarning, stacklevel=2)

            if target_association_tolerance is not _not_provided:
                raise TypeError("May not pass both target_association_tolerance and "
                        "target_stick_out_factor.")

            target_association_tolerance = target_stick_out_factor

        del target_stick_out_factor

        if target_association_tolerance is _not_provided:
            target_association_tolerance = float(
                    np.finfo(density_discr.real_dtype).eps) * 1e3

        if fmm_order is not None and fmm_level_to_order is not None:
            raise TypeError("may not specify both fmm_order and fmm_level_to_order")

        if _box_extent_norm is None:
            _box_extent_norm = "l2"

        if fmm_level_to_order is None:
            if fmm_order is False:
                fmm_level_to_order = False
            else:
                def fmm_level_to_order(kernel, kernel_args, tree, level):
                    return fmm_order

        if _max_leaf_refine_weight is None:
            if density_discr.ambient_dim == 2:
                _max_leaf_refine_weight = 64
            elif density_discr.ambient_dim == 3:
                _max_leaf_refine_weight = 128
            else:
                # Just guessing...
                _max_leaf_refine_weight = 64

        if _from_sep_smaller_min_nsources_cumul is None:
            # See here for the comment thread that led to these defaults:
            # https://gitlab.tiker.net/inducer/boxtree/merge_requests/28#note_18661
            if density_discr.dim == 1:
                _from_sep_smaller_min_nsources_cumul = 15
            else:
                _from_sep_smaller_min_nsources_cumul = 30

        # }}}

        self.fine_order = fine_order
        self.qbx_order = qbx_order
        self.density_discr = density_discr
        self.fmm_level_to_order = fmm_level_to_order

        assert target_association_tolerance is not None

        self.target_association_tolerance = target_association_tolerance
        self.fmm_backend = fmm_backend

        # Default values are lazily provided if these are None
        self._to_refined_connection = to_refined_connection

        if expansion_factory is None:
            from sumpy.expansion import DefaultExpansionFactory
            expansion_factory = DefaultExpansionFactory()
        self.expansion_factory = expansion_factory

        self.debug = debug
        self._refined_for_global_qbx = _refined_for_global_qbx
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
        self.geometry_data_inspector = geometry_data_inspector

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
            to_refined_connection=None,
            target_association_tolerance=_not_provided,
            _expansions_in_tree_have_extent=_not_provided,
            _expansion_stick_out_factor=_not_provided,
            _max_leaf_refine_weight=None,
            _box_extent_norm=None,
            _from_sep_smaller_crit=None,
            _tree_kind=None,
            geometry_data_inspector=None,
            fmm_backend=None,

            debug=_not_provided,
            _refined_for_global_qbx=_not_provided,
            target_stick_out_factor=_not_provided,
            ):

        # {{{ argument processing

        if target_stick_out_factor is not _not_provided:
            from warnings import warn
            warn("target_stick_out_factor has been renamed to "
                    "target_association_tolerance. "
                    "Using target_stick_out_factor is deprecated "
                    "and will stop working in 2018.",
                    DeprecationWarning, stacklevel=2)

            if target_association_tolerance is not _not_provided:
                raise TypeError("May not pass both target_association_tolerance and "
                        "target_stick_out_factor.")

            target_association_tolerance = target_stick_out_factor

        elif target_association_tolerance is _not_provided:
            target_association_tolerance = self.target_association_tolerance

        del target_stick_out_factor

        # }}}

        kwargs = {}

        if (fmm_order is not _not_provided
                and fmm_level_to_order is not _not_provided):
            raise TypeError("may not specify both fmm_order and fmm_level_to_order")
        elif fmm_order is not _not_provided:
            kwargs["fmm_order"] = fmm_order
        elif fmm_level_to_order is not _not_provided:
            kwargs["fmm_level_to_order"] = fmm_level_to_order
        else:
            kwargs["fmm_level_to_order"] = self.fmm_level_to_order

        # FIXME Could/should share wrangler and geometry kernels
        # if no relevant changes have been made.
        return QBXLayerPotentialSource(
                density_discr=density_discr or self.density_discr,
                fine_order=(
                    fine_order if fine_order is not None else self.fine_order),
                qbx_order=qbx_order if qbx_order is not None else self.qbx_order,

                target_association_tolerance=target_association_tolerance,
                to_refined_connection=(
                    to_refined_connection or self._to_refined_connection),

                debug=(
                    # False is a valid value here
                    debug if debug is not _not_provided else self.debug),
                _refined_for_global_qbx=(
                    # False is a valid value here
                    _refined_for_global_qbx
                    if _refined_for_global_qbx is not _not_provided
                    else self._refined_for_global_qbx),
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
                geometry_data_inspector=(
                    geometry_data_inspector or self.geometry_data_inspector),
                fmm_backend=fmm_backend or self.fmm_backend,
                **kwargs)

    # }}}

    @property
    def stage2_density_discr(self):
        """The refined, interpolation-focused density discretization (no oversampling).
        """
        return (self._to_refined_connection.to_discr
                if self._to_refined_connection is not None
                else self.density_discr)

    @property
    @memoize_method
    def refined_interp_to_ovsmp_quad_connection(self):
        from meshmode.discretization.connection import make_same_mesh_connection

        return make_same_mesh_connection(
                self.quad_stage2_density_discr,
                self.stage2_density_discr)

    @property
    @memoize_method
    def quad_stage2_density_discr(self):
        """The refined, quadrature-focused density discretization (with upsampling).
        """
        from meshmode.discretization.poly_element import (
                QuadratureSimplexGroupFactory)

        return Discretization(
            self.density_discr.cl_context, self.stage2_density_discr.mesh,
            QuadratureSimplexGroupFactory(self.fine_order),
            self.real_dtype)

    # {{{ weights and area elements

    @memoize_method
    def weights_and_area_elements(self):
        import pytential.symbolic.primitives as p
        from pytential.symbolic.execution import bind
        with cl.CommandQueue(self.cl_context) as queue:
            # quad_stage2_density_discr is not guaranteed to be usable for
            # interpolation/differentiation. Use density_discr to find
            # area element instead, then upsample that.

            area_element = self.refined_interp_to_ovsmp_quad_connection(
                    queue,
                    bind(
                        self.stage2_density_discr,
                        p.area_element(self.ambient_dim, self.dim)
                        )(queue))

            qweight = bind(self.quad_stage2_density_discr, p.QWeight())(queue)

            return (area_element.with_queue(queue)*qweight).with_queue(None)

    # }}}

    @property
    @memoize_method
    def resampler(self):
        from meshmode.discretization.connection import \
                ChainedDiscretizationConnection

        conn = self.refined_interp_to_ovsmp_quad_connection

        if self._to_refined_connection is not None:
            return ChainedDiscretizationConnection(
                    [self._to_refined_connection, conn])

        return conn

    @property
    @memoize_method
    def tree_code_container(self):
        from pytential.qbx.utils import TreeCodeContainer
        return TreeCodeContainer(self.cl_context)

    @property
    @memoize_method
    def refiner_code_container(self):
        from pytential.qbx.refinement import RefinerCodeContainer
        return RefinerCodeContainer(self.cl_context, self.tree_code_container)

    @property
    @memoize_method
    def target_association_code_container(self):
        from pytential.qbx.target_assoc import TargetAssociationCodeContainer
        return TargetAssociationCodeContainer(
                self.cl_context, self.tree_code_container)

    @memoize_method
    def with_refinement(self, target_order=None, kernel_length_scale=None,
            maxiter=None, visualize=False, refiner=None,
            _expansion_disturbance_tolerance=None,
            _force_stage2_uniform_refinement_rounds=None,
            _scaled_max_curvature_threshold=None):
        """
        :arg refiner: If the mesh underlying :attr:`density_discr`
            is itself the result of refinement, then its
            :class:`meshmode.refinement.Refiner` instance may need to
            be reused for continued refinement. This argument
            provides the opportunity to pass in an existing refiner
            that should be used for continued refinement.
        :returns: a tuple ``(lpot_src, cnx)``, where ``lpot_src`` is a
            :class:`QBXLayerPotentialSource` and ``cnx`` is a
            :class:`meshmode.discretization.connection.DiscretizationConnection`
            from the originally given to the refined geometry.
        """
        from pytential.qbx.refinement import refine_for_global_qbx

        from meshmode.discretization.poly_element import (
                InterpolatoryQuadratureSimplexGroupFactory)

        if target_order is None:
            target_order = self.density_discr.groups[0].order

        with cl.CommandQueue(self.cl_context) as queue:
            lpot, connection = refine_for_global_qbx(
                    self,
                    self.refiner_code_container.get_wrangler(queue),
                    InterpolatoryQuadratureSimplexGroupFactory(target_order),
                    kernel_length_scale=kernel_length_scale,
                    maxiter=maxiter, visualize=visualize,
                    expansion_disturbance_tolerance=_expansion_disturbance_tolerance,
                    force_stage2_uniform_refinement_rounds=(
                        _force_stage2_uniform_refinement_rounds),
                    scaled_max_curvature_threshold=(
                        _scaled_max_curvature_threshold),
                    refiner=refiner)

        return lpot, connection

    @property
    @memoize_method
    def h_max(self):
        with cl.CommandQueue(self.cl_context) as queue:
            quad_res = self._coarsest_quad_resolution("npanels").with_queue(queue)
            return np.asscalar(cl.array.max(quad_res).get())

    # {{{ internal API

    @memoize_method
    def _panel_centers_of_mass(self):
        import pytential.qbx.utils as utils
        return utils.element_centers_of_mass(self.density_discr)

    @memoize_method
    def _stage2_panel_centers_of_mass(self):
        import pytential.qbx.utils as utils
        return utils.element_centers_of_mass(self.stage2_density_discr)

    def _dim_fudge_factor(self):
        if self.density_discr.dim == 2:
            return 0.5
        else:
            return 1

    @memoize_method
    def _expansion_radii(self, last_dim_length):
        with cl.CommandQueue(self.cl_context) as queue:
                return (self._coarsest_quad_resolution(last_dim_length)
                        .with_queue(queue)
                        * 0.5 * self._dim_fudge_factor()).with_queue(None)

    # _expansion_radii should not be needed for the fine discretization

    @memoize_method
    def _source_danger_zone_radii(self, last_dim_length="npanels"):
        # This should be the expression of the expansion radii, but
        #
        # - in reference to the stage 2 discretization
        # - mutliplied by 0.75 because
        #
        #   - Setting this equal to the expansion radii ensures that *every*
        #     stage 2 element will be refined, which is wasteful.
        #     (so this needs to be smaller than that)
        #

        #   - Setting this equal to half the expansion radius will not provide
        #     a refinement 'buffer layer' at a 2x coarsening fringe.

        with cl.CommandQueue(self.cl_context) as queue:
            return (
                    (self._stage2_coarsest_quad_resolution(last_dim_length)
                        .with_queue(queue))
                    * 0.5 * 0.75 * self._dim_fudge_factor()).with_queue(None)

    @memoize_method
    def _close_target_tunnel_radius(self, last_dim_length):
        with cl.CommandQueue(self.cl_context) as queue:
                return (
                        self._expansion_radii(last_dim_length).with_queue(queue)
                        * 0.5
                        ).with_queue(None)

    @memoize_method
    def _coarsest_quad_resolution(self, last_dim_length="npanels"):
        """This measures the quadrature resolution across the
        mesh. In a 1D uniform mesh of uniform 'parametrization speed', it
        should be the same as the panel length.
        """
        import pytential.qbx.utils as utils
        from pytential import sym, bind
        with cl.CommandQueue(self.cl_context) as queue:
            maxstretch = bind(
                    self,
                    sym._simplex_mapping_max_stretch_factor(
                        self.ambient_dim)
                    )(queue)

            maxstretch = utils.to_last_dim_length(
                    self.density_discr, maxstretch, last_dim_length)
            maxstretch = maxstretch.with_queue(None)

        return maxstretch

    @memoize_method
    def _stage2_coarsest_quad_resolution(self, last_dim_length="npanels"):
        """This measures the quadrature resolution across the
        mesh. In a 1D uniform mesh of uniform 'parametrization speed', it
        should be the same as the panel length.
        """
        if last_dim_length != "npanels":
            # Not technically required below, but no need to loosen for now.
            raise NotImplementedError()

        import pytential.qbx.utils as utils
        from pytential import sym, bind
        with cl.CommandQueue(self.cl_context) as queue:
            maxstretch = bind(
                    self, sym._simplex_mapping_max_stretch_factor(
                        self.ambient_dim,
                        where=sym._QBXSourceStage2(sym.DEFAULT_SOURCE))
                    )(queue)
            maxstretch = utils.to_last_dim_length(
                    self.stage2_density_discr, maxstretch, last_dim_length)
            maxstretch = maxstretch.with_queue(None)

        return maxstretch

    @memoize_method
    def qbx_fmm_geometry_data(self, target_discrs_and_qbx_sides):
        """
        :arg target_discrs_and_qbx_sides:
            a tuple of *(discr, qbx_forced_limit)*
            tuples, where *discr* is a
            :class:`meshmode.discretization.Discretization`
            or
            :class:`pytential.target.TargetBase`
            instance
        """
        from pytential.qbx.geometry import QBXFMMGeometryData

        return QBXFMMGeometryData(self.qbx_fmm_code_getter,
                self, target_discrs_and_qbx_sides,
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
        from sumpy.kernel import AxisTargetDerivativeRemover
        result = (
                expr.source, expr.density,
                AxisTargetDerivativeRemover()(expr.kernel),
                )

        return result

    # }}}

    # {{{ internal functionality for execution

    def exec_compute_potential_insn(self, queue, insn, bound_expr, evaluate):
        from pytools.obj_array import with_object_array_or_scalar

        def oversample_nonscalars(vec):
            from numbers import Number
            if isinstance(vec, Number):
                return vec
            else:
                return self.resampler(queue, vec)

        if not self._refined_for_global_qbx:
            from warnings import warn
            warn(
                "Executing global QBX without refinement. "
                "This is unlikely to work.")

        def evaluate_wrapper(expr):
            value = evaluate(expr)
            return with_object_array_or_scalar(oversample_nonscalars, value)

        if self.fmm_level_to_order is False:
            func = self.exec_compute_potential_insn_direct
        else:
            func = self.exec_compute_potential_insn_fmm

        return func(queue, insn, bound_expr, evaluate_wrapper)

    @property
    @memoize_method
    def qbx_fmm_code_getter(self):
        from pytential.qbx.geometry import QBXFMMGeometryCodeGetter
        return QBXFMMGeometryCodeGetter(self.cl_context, self.ambient_dim,
                self.tree_code_container, debug=self.debug,
                _well_sep_is_n_away=self._well_sep_is_n_away,
                _from_sep_smaller_crit=self._from_sep_smaller_crit)

    # {{{ fmm-based execution

    @memoize_method
    def expansion_wrangler_code_container(self, fmm_kernel, out_kernels):
        mpole_expn_class = \
                self.expansion_factory.get_multipole_expansion_class(fmm_kernel)
        local_expn_class = \
                self.expansion_factory.get_local_expansion_class(fmm_kernel)

        from functools import partial
        fmm_mpole_factory = partial(mpole_expn_class, fmm_kernel)
        fmm_local_factory = partial(local_expn_class, fmm_kernel)
        qbx_local_factory = partial(local_expn_class, fmm_kernel)

        if self.fmm_backend == "sumpy":
            from pytential.qbx.fmm import \
                    QBXSumpyExpansionWranglerCodeContainer
            return QBXSumpyExpansionWranglerCodeContainer(
                    self.cl_context,
                    fmm_mpole_factory, fmm_local_factory, qbx_local_factory,
                    out_kernels)

        elif self.fmm_backend == "fmmlib":
            from pytential.qbx.fmmlib import \
                    QBXFMMLibExpansionWranglerCodeContainer
            return QBXFMMLibExpansionWranglerCodeContainer(
                    self.cl_context,
                    fmm_mpole_factory, fmm_local_factory, qbx_local_factory,
                    out_kernels)

        else:
            raise ValueError("invalid FMM backend: %s" % self.fmm_backend)

    def exec_compute_potential_insn_fmm(self, queue, insn, bound_expr, evaluate):
        # {{{ build list of unique target discretizations used

        # map (name, qbx_side) to number in list
        tgt_name_and_side_to_number = {}
        # list of tuples (discr, qbx_side)
        target_discrs_and_qbx_sides = []

        for o in insn.outputs:
            key = (o.target_name, o.qbx_forced_limit)
            if key not in tgt_name_and_side_to_number:
                tgt_name_and_side_to_number[key] = \
                        len(target_discrs_and_qbx_sides)

                target_discr = bound_expr.places[o.target_name]
                if isinstance(target_discr, LayerPotentialSourceBase):
                    target_discr = target_discr.density_discr

                qbx_forced_limit = o.qbx_forced_limit
                if qbx_forced_limit is None:
                    qbx_forced_limit = 0

                target_discrs_and_qbx_sides.append(
                        (target_discr, qbx_forced_limit))

        target_discrs_and_qbx_sides = tuple(target_discrs_and_qbx_sides)

        # }}}

        geo_data = self.qbx_fmm_geometry_data(target_discrs_and_qbx_sides)

        # geo_data.plot()

        # FIXME Exert more positive control over geo_data attribute lifetimes using
        # geo_data.<method>.clear_cache(geo_data).

        # FIXME Synthesize "bad centers" around corners and edges that have
        # inadequate QBX coverage.

        # FIXME don't compute *all* output kernels on all targets--respect that
        # some target discretizations may only be asking for derivatives (e.g.)

        strengths = (evaluate(insn.density).with_queue(queue)
                * self.weights_and_area_elements())

        out_kernels = tuple(knl for knl in insn.kernels)
        fmm_kernel = self.get_fmm_kernel(out_kernels)
        output_and_expansion_dtype = (
                self.get_fmm_output_and_expansion_dtype(fmm_kernel, strengths))
        kernel_extra_kwargs, source_extra_kwargs = (
                self.get_fmm_expansion_wrangler_extra_kwargs(
                    queue, out_kernels, geo_data.tree().user_source_ids,
                    insn.kernel_arguments, evaluate))

        wrangler = self.expansion_wrangler_code_container(
                fmm_kernel, out_kernels).get_wrangler(
                        queue, geo_data, output_and_expansion_dtype,
                        self.qbx_order,
                        self.fmm_level_to_order,
                        source_extra_kwargs=source_extra_kwargs,
                        kernel_extra_kwargs=kernel_extra_kwargs)

        from pytential.qbx.geometry import target_state
        if (geo_data.user_target_to_center().with_queue(queue)
                == target_state.FAILED).any().get():
            raise RuntimeError("geometry has failed targets")

        # {{{ performance data hook

        if self.geometry_data_inspector is not None:
            perform_fmm = self.geometry_data_inspector(insn, bound_expr, geo_data)
            if not perform_fmm:
                return [(o.name, 0) for o in insn.outputs], []

        # }}}

        # {{{ execute global QBX

        from pytential.qbx.fmm import drive_fmm
        all_potentials_on_every_tgt = drive_fmm(wrangler, strengths)

        # }}}

        result = []

        for o in insn.outputs:
            tgt_side_number = tgt_name_and_side_to_number[
                    o.target_name, o.qbx_forced_limit]
            tgt_slice = slice(*geo_data.target_info().target_discr_starts[
                    tgt_side_number:tgt_side_number+2])

            result.append(
                    (o.name,
                        all_potentials_on_every_tgt[o.kernel_index][tgt_slice]))

        return result, []

    # }}}

    # {{{ direct execution

    @memoize_method
    def get_lpot_applier(self, kernels):
        # needs to be separate method for caching

        from pytools import any
        if any(knl.is_complex_valued for knl in kernels):
            value_dtype = self.density_discr.complex_dtype
        else:
            value_dtype = self.density_discr.real_dtype

        from sumpy.qbx import LayerPotential
        from sumpy.expansion.local import LineTaylorLocalExpansion
        return LayerPotential(self.cl_context,
                    [LineTaylorLocalExpansion(knl, self.qbx_order)
                        for knl in kernels],
                    value_dtypes=value_dtype)

    @memoize_method
    def get_lpot_applier_on_tgt_subset(self, kernels):
        # needs to be separate method for caching

        from pytools import any
        if any(knl.is_complex_valued for knl in kernels):
            value_dtype = self.density_discr.complex_dtype
        else:
            value_dtype = self.density_discr.real_dtype

        from pytential.qbx.direct import LayerPotentialOnTargetAndCenterSubset
        from sumpy.expansion.local import VolumeTaylorLocalExpansion
        return LayerPotentialOnTargetAndCenterSubset(
                self.cl_context,
                [VolumeTaylorLocalExpansion(knl, self.qbx_order)
                    for knl in kernels],
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

    def exec_compute_potential_insn_direct(self, queue, insn, bound_expr, evaluate):
        lpot_applier = self.get_lpot_applier(insn.kernels)
        p2p = None
        lpot_applier_on_tgt_subset = None

        kernel_args = {}
        for arg_name, arg_expr in six.iteritems(insn.kernel_arguments):
            kernel_args[arg_name] = evaluate(arg_expr)

        strengths = (evaluate(insn.density).with_queue(queue)
                * self.weights_and_area_elements())

        import pytential.qbx.utils as utils

        # FIXME: Do this all at once
        result = []
        for o in insn.outputs:
            target_discr = bound_expr.get_discretization(o.target_name)

            is_self = self.density_discr is target_discr

            if is_self:
                # QBXPreprocessor is supposed to have taken care of this
                assert o.qbx_forced_limit is not None
                assert abs(o.qbx_forced_limit) > 0

                evt, output_for_each_kernel = lpot_applier(
                        queue, target_discr.nodes(),
                        self.quad_stage2_density_discr.nodes(),
                        utils.get_centers_on_side(self, o.qbx_forced_limit),
                        [strengths],
                        expansion_radii=self._expansion_radii("nsources"),
                        **kernel_args)
                result.append((o.name, output_for_each_kernel[o.kernel_index]))
            else:
                # no on-disk kernel caching
                if p2p is None:
                    p2p = self.get_p2p(insn.kernels)
                if lpot_applier_on_tgt_subset is None:
                    lpot_applier_on_tgt_subset = self.get_lpot_applier_on_tgt_subset(
                            insn.kernels)

                evt, output_for_each_kernel = p2p(queue,
                        target_discr.nodes(),
                        self.quad_stage2_density_discr.nodes(),
                        [strengths], **kernel_args)

                qbx_forced_limit = o.qbx_forced_limit
                if qbx_forced_limit is None:
                    qbx_forced_limit = 0

                geo_data = self.qbx_fmm_geometry_data(
                        target_discrs_and_qbx_sides=(
                            (target_discr, qbx_forced_limit),
                        ))

                # center-related info is independent of targets

                # First ncenters targets are the centers
                tgt_to_qbx_center = (
                        geo_data.user_target_to_center()[geo_data.ncenters:]
                        .copy(queue=queue)
                        .with_queue(queue))

                qbx_tgt_numberer = self.get_qbx_target_numberer(
                        tgt_to_qbx_center.dtype)
                qbx_tgt_count = cl.array.empty(queue, (), np.int32)
                qbx_tgt_numbers = cl.array.empty_like(tgt_to_qbx_center)

                qbx_tgt_numberer(
                        tgt_to_qbx_center, qbx_tgt_numbers, qbx_tgt_count,
                        queue=queue)

                qbx_tgt_count = int(qbx_tgt_count.get())

                if (o.qbx_forced_limit is not None
                        and abs(o.qbx_forced_limit) == 1
                        and qbx_tgt_count < target_discr.nnodes):
                    raise RuntimeError("Did not find a matching QBX center "
                            "for some targets")

                qbx_tgt_numbers = qbx_tgt_numbers[:qbx_tgt_count]
                qbx_center_numbers = tgt_to_qbx_center[qbx_tgt_numbers]
                qbx_center_numbers.finish()

                tgt_subset_kwargs = kernel_args.copy()
                for i, res_i in enumerate(output_for_each_kernel):
                    tgt_subset_kwargs["result_%d" % i] = res_i

                if qbx_tgt_count:
                    lpot_applier_on_tgt_subset(
                            queue,
                            targets=target_discr.nodes(),
                            sources=self.quad_stage2_density_discr.nodes(),
                            centers=geo_data.centers(),
                            expansion_radii=geo_data.expansion_radii(),
                            strengths=[strengths],
                            qbx_tgt_numbers=qbx_tgt_numbers,
                            qbx_center_numbers=qbx_center_numbers,
                            **tgt_subset_kwargs)

                result.append((o.name, output_for_each_kernel[o.kernel_index]))

        return result, []

    # }}}

    # }}}

# }}}


__all__ = (
        QBXLayerPotentialSource,
        QBXTargetAssociationFailedException,
        )

# vim: fdm=marker
