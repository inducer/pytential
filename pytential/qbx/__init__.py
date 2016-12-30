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

import loopy as lp
import numpy as np
from pytools import memoize_method
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory
from pytential.qbx.target_assoc import QBXTargetAssociationFailedException

import pyopencl as cl

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: QBXLayerPotentialSource

.. autoclass:: QBXTargetAssociationFailedException
"""


# {{{ jump term interface helper

class _JumpTermArgumentProvider(object):
    def __init__(self, discr, density, ds_direction, side=None):
        self.discr = discr
        self.density = density
        self.ds_direction = ds_direction
        self.side = side

    @property
    def normal(self):
        return self.discr.curve.normals.reshape(2, -1).T

    @property
    def tangent(self):
        return self.discr.curve.tangents.reshape(2, -1).T

    @property
    def src_derivative_dir(self):
        return self.ds_direction

    @property
    def mean_curvature(self):
        return self.discr.curve.curvature.reshape(-1)

    @property
    def density_0(self):
        return self.density.reshape(-1)

    @property
    @memoize_method
    def density_0_prime(self):
        diff_mat = self.discr.curve.expansion.get_differentiation_matrix()
        return (2 * np.dot(diff_mat, self.density.T).T.reshape(-1)
                / self.discr.curve.speed.reshape(-1))

# }}}


class LayerPotentialSource(object):
    """
    .. method:: preprocess_optemplate(name, expr)

    .. method:: op_group_features(expr)

        Return a characteristic tuple by which operators that can be
        executed together can be grouped.

        *expr* is a subclass of
        :class:`pytential.symbolic.primitives.IntG`.
    """


def get_local_expansion_class(base_kernel):
    # FIXME: Don't hard-code expansion types
    from sumpy.kernel import HelmholtzKernel
    if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
            and base_kernel.dim == 2):
        from sumpy.expansion.local import H2DLocalExpansion
        return H2DLocalExpansion
    else:
        from sumpy.expansion.local import LaplaceConformingVolumeTaylorLocalExpansion
        return LaplaceConformingVolumeTaylorLocalExpansion


def get_multipole_expansion_class(base_kernel):
    # FIXME: Don't hard-code expansion types
    from sumpy.kernel import HelmholtzKernel
    if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
            and base_kernel.dim == 2):
        from sumpy.expansion.multipole import H2DMultipoleExpansion
        return H2DMultipoleExpansion
    else:
        from sumpy.expansion.multipole import (
                LaplaceConformingVolumeTaylorMultipoleExpansion)
        return LaplaceConformingVolumeTaylorMultipoleExpansion


# {{{ QBX layer potential source

class QBXLayerPotentialSource(LayerPotentialSource):
    """A source discretization for a QBX layer potential.

    .. attribute :: density_discr
    .. attribute :: qbx_order
    .. attribute :: fmm_order
    .. attribute :: cl_context
    .. automethod :: centers
    .. automethod :: panel_sizes
    .. automethod :: weights_and_area_elements
    .. automethod :: with_refinement

    See :ref:`qbxguts` for some information on the inner workings of this.
    """
    def __init__(self, density_discr, fine_order,
            qbx_order=None,
            fmm_order=None,
            fmm_level_to_order=None,
            target_stick_out_factor=1e-10,

            # begin undocumented arguments
            # FIXME default debug=False once everything works
            debug=True,
            refined_for_global_qbx=False,
            performance_data_file=None):
        """
        :arg fine_order: The total degree to which the (upsampled)
            underlying quadrature is exact.
        :arg fmm_order: `False` for direct calculation. ``None`` will set
            a reasonable(-ish?) default.
        """

        if fmm_level_to_order is None:
            if fmm_order is None and qbx_order is not None:
                fmm_order = qbx_order + 1

        if fmm_order is not None and fmm_level_to_order is not None:
            raise TypeError("may not specify both fmm_order and fmm_level_to_order")

        if fmm_level_to_order is None:
            if fmm_order is False:
                fmm_level_to_order = False
            else:
                def fmm_level_to_order(level):
                    return fmm_order

        self.fine_order = fine_order
        self.qbx_order = qbx_order
        self.density_discr = density_discr
        self.fmm_level_to_order = fmm_level_to_order
        self.target_stick_out_factor = target_stick_out_factor

        self.debug = debug
        self.refined_for_global_qbx = refined_for_global_qbx
        self.performance_data_file = performance_data_file

    def copy(
            self,
            density_discr=None,
            fine_order=None,
            qbx_order=None,
            fmm_level_to_order=None,
            target_stick_out_factor=None,

            debug=None,
            refined_for_global_qbx=None,
            ):
        # FIXME Could/should share wrangler and geometry kernels
        # if no relevant changes have been made.
        return QBXLayerPotentialSource(
                density_discr=density_discr or self.density_discr,
                fine_order=(
                    fine_order if fine_order is not None else self.fine_order),
                qbx_order=qbx_order if qbx_order is not None else self.qbx_order,
                fmm_level_to_order=(
                    fmm_level_to_order or self.fmm_level_to_order),
                target_stick_out_factor=(
                    target_stick_out_factor or self.target_stick_out_factor),

                debug=(
                    debug if debug is not None else self.debug),
                refined_for_global_qbx=(
                    refined_for_global_qbx if refined_for_global_qbx is not None
                    else self.refined_for_global_qbx),
                performance_data_file=self.performance_data_file)

    @property
    @memoize_method
    def fine_density_discr(self):
        return Discretization(
            self.density_discr.cl_context, self.density_discr.mesh,
            QuadratureSimplexGroupFactory(self.fine_order), self.real_dtype)

    @property
    @memoize_method
    def resampler(self):
        from meshmode.discretization.connection import make_same_mesh_connection
        return make_same_mesh_connection(
                self.fine_density_discr, self.density_discr)

    def el_view(self, discr, group_nr, global_array):
        """Return a view of *global_array* of shape
        ``(..., discr.groups[group_nr].nelements)``
        where *global_array* is of shape ``(..., nelements)``,
        where *nelements* is the global (per-discretization) node count.
        """

        group = discr.groups[group_nr]
        el_nr_base = sum(group.nelements for group in discr.groups[:group_nr])

        return global_array[
            ..., el_nr_base:el_nr_base + group.nelements] \
            .reshape(
                global_array.shape[:-1]
                + (group.nelements,))

    @memoize_method
    def with_refinement(self, target_order=None):
        """
        :returns: a tuple ``(lpot_src, cnx)``, where ``lpot_src`` is a
            :class:`QBXLayerPotentialSource` and ``cnx`` is a
            :class:`meshmode.discretization.connection.DiscretizationConnection`
            from the originally given to the refined geometry.
        """
        from pytential.qbx.refinement import QBXLayerPotentialSourceRefiner
        refiner = QBXLayerPotentialSourceRefiner(self.cl_context)
        from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)
        if target_order is None:
            target_order = self.density_discr.groups[0].order
        lpot, connection = refiner(self,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
        return lpot, connection

    @property
    @memoize_method
    def h_max(self):
        with cl.CommandQueue(self.cl_context) as queue:
            panel_sizes = self.panel_sizes("npanels").with_queue(queue)
            return np.asscalar(cl.array.max(panel_sizes).get())

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
    def panel_centers_of_mass(self):
        knl = lp.make_kernel(
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nunit_nodes}""",
            """
                panels[dim, k] = sum(i, nodes[dim, k, i])/nunit_nodes
                """,
            default_offset=lp.auto, name="find_panel_centers_of_mass")

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.split_iname(knl, "k", 128, inner_tag="l.0", outer_tag="g.0")
        knl = lp.tag_inames(knl, dict(dim="ilp"))

        with cl.CommandQueue(self.cl_context) as queue:
            mesh = self.density_discr.mesh
            panels = cl.array.empty(queue, (mesh.ambient_dim, mesh.nelements),
                                    dtype=self.density_discr.real_dtype)
            for group_nr, group in enumerate(self.density_discr.groups):
                _, (result,) = knl(queue,
                    nelements=group.nelements,
                    nunit_nodes=group.nunit_nodes,
                    nodes=group.view(self.density_discr.nodes()),
                    panels=self.el_view(self.density_discr, group_nr, panels))
            panels.finish()
            panels = panels.with_queue(None)
            return tuple(panels[d, :] for d in range(mesh.ambient_dim))

    @memoize_method
    def panel_sizes(self, last_dim_length="nsources"):
        assert last_dim_length in ("nsources", "ncenters", "npanels")
        # To get the panel size this does the equivalent of ∫ 1 ds.
        # FIXME: Kernel optimizations

        discr = self.density_discr

        if last_dim_length == "nsources" or last_dim_length == "ncenters":
            knl = lp.make_kernel(
                "{[i,j,k]: 0<=i<nelements and 0<=j,k<nunit_nodes}",
                "panel_sizes[i,j] = sum(k, ds[i,k])",
                name="compute_size")

            def panel_size_view(discr, group_nr):
                return discr.groups[group_nr].view

        elif last_dim_length == "npanels":
            knl = lp.make_kernel(
                "{[i,j]: 0<=i<nelements and 0<=j<nunit_nodes}",
                "panel_sizes[i] = sum(j, ds[i,j])",
                name="compute_size")
            from functools import partial

            def panel_size_view(discr, group_nr):
                return partial(self.el_view, discr, group_nr)

        else:
            raise ValueError("unknown dim length specified")

        with cl.CommandQueue(self.cl_context) as queue:
            from pytential import bind, sym
            ds = bind(
                    discr,
                    sym.area_element(ambient_dim=discr.ambient_dim, dim=discr.dim)
                    * sym.QWeight()
                    )(queue)
            panel_sizes = cl.array.empty(
                queue, discr.nnodes
                if last_dim_length in ("nsources", "ncenters")
                else discr.mesh.nelements, discr.real_dtype)
            for group_nr, group in enumerate(discr.groups):
                _, (result,) = knl(queue,
                    nelements=group.nelements,
                    nunit_nodes=group.nunit_nodes,
                    ds=group.view(ds),
                    panel_sizes=panel_size_view(
                        discr, group_nr)(panel_sizes))
            panel_sizes.finish()
            if last_dim_length == "ncenters":
                from pytential.qbx.utils import get_interleaver_kernel
                knl = get_interleaver_kernel(discr.real_dtype)
                _, (panel_sizes,) = knl(queue, dstlen=2*discr.nnodes,
                                        src1=panel_sizes, src2=panel_sizes)
            return panel_sizes.with_queue(None)

    @memoize_method
    def centers(self, sign):
        adim = self.density_discr.ambient_dim
        dim = self.density_discr.dim

        from pytential import sym, bind
        with cl.CommandQueue(self.cl_context) as queue:
            nodes = bind(self.density_discr, sym.nodes(adim))(queue)
            normals = bind(self.density_discr, sym.normal(adim, dim=dim))(queue)
            panel_sizes = self.panel_sizes().with_queue(queue)
            return (nodes + normals * sign * panel_sizes / 2).as_vector(np.object)

    @memoize_method
    def weights_and_area_elements(self):
        import pytential.symbolic.primitives as p
        from pytential.symbolic.execution import bind
        with cl.CommandQueue(self.cl_context) as queue:
            # fine_density_discr is not guaranteed to be usable for
            # interpolation/differentiation. Use density_discr to find
            # area element instead, then upsample that.

            area_element = self.resampler(queue,
                    bind(
                        self.density_discr,
                        p.area_element(self.ambient_dim, self.dim)
                        )(queue))

            qweight = bind(self.fine_density_discr, p.QWeight())(queue)

            return (area_element.with_queue(queue)*qweight).with_queue(None)

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

    def exec_layer_potential_insn(self, queue, insn, bound_expr, evaluate):
        from pytools.obj_array import with_object_array_or_scalar
        from functools import partial
        oversample = partial(self.resampler, queue)

        if not self.refined_for_global_qbx:
            from warnings import warn
            warn(
                "Executing global QBX without refinement. "
                "This is unlikely to work.")

        def evaluate_wrapper(expr):
            value = evaluate(expr)
            return with_object_array_or_scalar(oversample, value)

        if self.fmm_level_to_order is False:
            func = self.exec_layer_potential_insn_direct
        else:
            func = self.exec_layer_potential_insn_fmm

        return func(queue, insn, bound_expr, evaluate_wrapper)

    @property
    @memoize_method
    def qbx_fmm_code_getter(self):
        from pytential.qbx.geometry import QBXFMMGeometryCodeGetter
        return QBXFMMGeometryCodeGetter(self.cl_context, self.ambient_dim,
                debug=self.debug)

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
                target_stick_out_factor=self.target_stick_out_factor,
                debug=self.debug)

    # {{{ fmm-based execution

    @memoize_method
    def expansion_wrangler_code_container(self, base_kernel, out_kernels):
        mpole_expn_class = get_multipole_expansion_class(base_kernel)
        local_expn_class = get_local_expansion_class(base_kernel)

        from functools import partial
        fmm_mpole_factory = partial(mpole_expn_class, base_kernel)
        fmm_local_factory = partial(local_expn_class, base_kernel)
        qbx_local_factory = partial(local_expn_class, base_kernel)

        from pytential.qbx.fmm import \
                QBXExpansionWranglerCodeContainer
        return QBXExpansionWranglerCodeContainer(
                self.cl_context,
                fmm_mpole_factory, fmm_local_factory, qbx_local_factory,
                out_kernels)

    def exec_layer_potential_insn_fmm(self, queue, insn, bound_expr, evaluate):
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
                if isinstance(target_discr, LayerPotentialSource):
                    target_discr = target_discr.density_discr

                qbx_forced_limit = o.qbx_forced_limit
                if qbx_forced_limit is None:
                    qbx_forced_limit = 0

                target_discrs_and_qbx_sides.append(
                        (target_discr, qbx_forced_limit))

        target_discrs_and_qbx_sides = tuple(target_discrs_and_qbx_sides)

        # }}}

        geo_data = self.qbx_fmm_geometry_data(target_discrs_and_qbx_sides)

        # FIXME Exert more positive control over geo_data attribute lifetimes using
        # geo_data.<method>.clear_cache(geo_data).

        # FIXME Synthesize "bad centers" around corners and edges that have
        # inadequate QBX coverage.

        # FIXME don't compute *all* output kernels on all targets--respect that
        # some target discretizations may only be asking for derivatives (e.g.)

        strengths = (evaluate(insn.density).with_queue(queue)
                * self.weights_and_area_elements())

        # {{{ get expansion wrangler

        base_kernel = None
        out_kernels = []

        from sumpy.kernel import AxisTargetDerivativeRemover
        for knl in insn.kernels:
            candidate_base_kernel = AxisTargetDerivativeRemover()(knl)

            if base_kernel is None:
                base_kernel = candidate_base_kernel
            else:
                assert base_kernel == candidate_base_kernel

        out_kernels = tuple(knl for knl in insn.kernels)

        if base_kernel.is_complex_valued or strengths.dtype.kind == "c":
            value_dtype = self.complex_dtype
        else:
            value_dtype = self.real_dtype

        # {{{ build extra_kwargs dictionaries

        # This contains things like the Helmholtz parameter k or
        # the normal directions for double layers.

        def reorder_sources(source_array):
            if isinstance(source_array, cl.array.Array):
                return (source_array
                        .with_queue(queue)
                        [geo_data.tree().user_source_ids]
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
                        evaluate(insn.kernel_arguments[arg.name]))

        # }}}

        wrangler = self.expansion_wrangler_code_container(
                base_kernel, out_kernels).get_wrangler(
                        queue, geo_data, value_dtype,
                        self.qbx_order,
                        self.fmm_level_to_order,
                        source_extra_kwargs=source_extra_kwargs,
                        kernel_extra_kwargs=kernel_extra_kwargs)

        # }}}

        if len(geo_data.global_qbx_centers()) != geo_data.center_info().ncenters:
            raise NotImplementedError("geometry has centers requiring local QBX")

        from pytential.qbx.geometry import target_state
        if (geo_data.user_target_to_center().with_queue(queue)
                == target_state.FAILED).any().get():
            raise RuntimeError("geometry has failed targets")

        if self.performance_data_file is not None:
            from pytential.qbx.fmm import write_performance_model
            with open(self.performance_data_file, "w") as outf:
                write_performance_model(outf, geo_data)

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

    def exec_layer_potential_insn_direct(self, queue, insn, bound_expr, evaluate):
        lpot_applier = self.get_lpot_applier(insn.kernels)
        p2p = None
        lpot_applier_on_tgt_subset = None

        kernel_args = {}
        for arg_name, arg_expr in six.iteritems(insn.kernel_arguments):
            kernel_args[arg_name] = evaluate(arg_expr)

        strengths = (evaluate(insn.density).with_queue(queue)
                * self.weights_and_area_elements())

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
                        self.fine_density_discr.nodes(),
                        self.centers(o.qbx_forced_limit),
                        [strengths], **kernel_args)
                result.append((o.name, output_for_each_kernel[o.kernel_index]))
            else:
                # no on-disk kernel caching
                if p2p is None:
                    p2p = self.get_p2p(insn.kernels)
                if lpot_applier_on_tgt_subset is None:
                    lpot_applier_on_tgt_subset = self.get_lpot_applier_on_tgt_subset(
                            insn.kernels)

                evt, output_for_each_kernel = p2p(queue,
                        target_discr.nodes(), self.fine_density_discr.nodes(),
                        [strengths], **kernel_args)

                qbx_forced_limit = o.qbx_forced_limit
                if qbx_forced_limit is None:
                    qbx_forced_limit = 0

                geo_data = self.qbx_fmm_geometry_data(
                        target_discrs_and_qbx_sides=(
                            (target_discr, qbx_forced_limit),
                        ))

                # center_info is independent of targets
                center_info = geo_data.center_info()

                # First ncenters targets are the centers
                tgt_to_qbx_center = (
                        geo_data.user_target_to_center()[center_info.ncenters:]
                        .copy(queue=queue))

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

                tgt_subset_kwargs = kernel_args.copy()
                for i, res_i in enumerate(output_for_each_kernel):
                    tgt_subset_kwargs["result_%d" % i] = res_i

                if qbx_tgt_count:
                    lpot_applier_on_tgt_subset(
                            queue,
                            targets=target_discr.nodes(),
                            sources=self.fine_density_discr.nodes(),
                            centers=center_info.centers,
                            strengths=[strengths],
                            qbx_tgt_numbers=qbx_tgt_numbers,
                            qbx_center_numbers=qbx_center_numbers,
                            **tgt_subset_kwargs)

                result.append((o.name, output_for_each_kernel[o.kernel_index]))

        return result, []

    # }}}

# }}}


__all__ = (
        QBXLayerPotentialSource,
        QBXTargetAssociationFailedException,
        )

# vim: fdm=marker
