# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2016 Matt Wala
"""

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


import loopy as lp
import numpy as np
from pytools import memoize_method
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory
from pytential.qbx.utils import DiscrPlotterMixin

import pyopencl as cl

import logging
logger = logging.getLogger(__name__)

__doc__ = """
Refinement
^^^^^^^^^^

.. autoclass:: QBXLayerPotentialSourceRefiner
"""


from pytential.qbx import LayerPotentialSource, get_multipole_expansion_class, get_local_expansion_class


# {{{ layer potential source

# FIXME: Move to own file, replace existing QBXLayerPotentialSource when
# finished.
class NewQBXLayerPotentialSource(LayerPotentialSource):
    """A source discretization for a QBX layer potential.

    .. attribute :: density_discr
    .. attribute :: qbx_order
    .. attribute :: fmm_order
    .. attribute :: cl_context
    .. automethod :: centers
    .. automethod :: panel_sizes
    .. automethod :: weights_and_area_elements

    See :ref:`qbxguts` for some information on the inner workings of this.
    """
    def __init__(self, density_discr, fine_order,
            qbx_order=None, fmm_order=None,
            fmm_level_to_order=None,
            # FIXME set debug=False once everything works
            real_dtype=np.float64, debug=True,
            performance_data_file=None):
        """
        :arg fine_order: The total degree to which the (upsampled)
            underlying quadrature is exact.
        :arg fmm_order: `False` for direct calculation. ``None`` will set
            a reasonable(-ish?) default.
        """

        self.fine_density_discr = Discretization(
                density_discr.cl_context, density_discr.mesh,
                QuadratureSimplexGroupFactory(fine_order), real_dtype)

        from meshmode.discretization.connection import make_same_mesh_connection
        self.resampler = make_same_mesh_connection(
                self.fine_density_discr, density_discr)

        if fmm_level_to_order is None:
            if fmm_order is None and qbx_order is not None:
                fmm_order = qbx_order + 1

        if fmm_order is not None and fmm_level_to_order is not None:
            raise TypeError("may not specify both fmm_order an fmm_level_to_order")

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
        self.debug = debug
        self.performance_data_file = performance_data_file

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

    @property
    def ambient_dim(self):
        return self.density_discr.ambient_dim

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
        assert last_dim_length in ["nsources", "ncenters", "npanels"]
        # To get the panel size this does the equivalent of ∫ 1 ds.
        # FIXME: Kernel optimizations

        discr = self.density_discr

        if last_dim_length == "nsources" or last_dim_length == "ncenters":
            knl = lp.make_kernel(
                "{[i,j,k]: 0<=i<nelements and 0<=j,k<nunit_nodes}",
                "panel_sizes[i,j] = sum(k, ds[i,k])",
                name="compute_size")
            knl = lp.tag_inames(knl, dict(i="g.0", j="l.0"))

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
            ds = bind(discr, sym.area_element() * sym.QWeight())(queue)
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
        from pytential import sym, bind
        with cl.CommandQueue(self.cl_context) as queue:
            nodes = bind(self.density_discr, sym.Nodes())(queue)
            normals = bind(self.density_discr, sym.normal())(queue)
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
                    bind(self.density_discr,
                        p.area_element())(queue))

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
        from pytential.symbolic.primitives import IntGdSource
        assert not isinstance(expr, IntGdSource)

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
                self, target_discrs_and_qbx_sides, debug=self.debug)

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

        #geo_data.plot()

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
                        self.centers(target_discr, o.qbx_forced_limit),
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

                geo_data = self.qbx_fmm_geometry_data(
                        target_discrs_and_qbx_sides=[
                            (target_discr, qbx_forced_limit)
                        ])

                # center_info is independent of targets
                center_info = geo_data.center_info()

                qbx_forced_limit = o.qbx_forced_limit
                if qbx_forced_limit is None:
                    qbx_forced_limit = 0

                tgt_to_qbx_center = (
                        geo_data.user_target_to_center()[center_info.ncenters:])

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

from boxtree.area_query import AreaQueryElementwiseTemplate
from pyopencl.elementwise import ElementwiseTemplate
from boxtree.tools import InlineBinarySearch
from pytential.qbx.utils import QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS


unwrap_args = AreaQueryElementwiseTemplate.unwrap_args


# {{{ kernels

TUNNEL_QUERY_DISTANCE_FINDER_TEMPLATE = ElementwiseTemplate(
    arguments=r"""//CL:mako//
        /* input */
        particle_id_t source_offset,
        particle_id_t panel_offset,
        int npanels,
        particle_id_t *panel_to_source_starts,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,

        /* output */
        float *tunnel_query_dists,

        /* input, dim-dependent size */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    operation=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""//CL:mako//
        /* Find my panel. */
        particle_id_t panel = bsearch(panel_to_source_starts, npanels + 1, i);

        /* Compute dist(tunnel region, panel center) */

        coord_vec_t center_of_mass;
        ${load_particle("INDEX_FOR_PANEL_PARTICLE(panel)", "center_of_mass")}

        coord_vec_t center;
        ${load_particle("INDEX_FOR_SOURCE_PARTICLE(i)", "center")}

        coord_t panel_size = panel_sizes[panel];

        coord_t max_dist = 0;

        %for ax in AXIS_NAMES[:dimensions]:
        {
            max_dist = fmax(max_dist,
                distance(center_of_mass.${ax}, center.${ax} + panel_size / 2));
            max_dist = fmax(max_dist,
                distance(center_of_mass.${ax}, center.${ax} - panel_size / 2));
        }
        %endfor

        // The atomic max operation supports only integer types.
        // However, max_dist is of a floating point type.
        // For comparison purposes we reinterpret the bits of max_dist
        // as an integer. The comparison result is the same as for positive
        // IEEE floating point numbers, so long as the float/int endianness
        // matches (fingers crossed).
        atomic_max(
            (volatile __global int *)
                &tunnel_query_dists[panel],
            as_int((float) max_dist));
        """,
    name="find_tunnel_query_distance",
    preamble=str(InlineBinarySearch("particle_id_t")))


# Implements "Algorithm for triggering refinement based on Condition 1"
#
# FIXME: There is probably a better way to do this. For instance, since
# we are not using Newton to compute center-panel distances, we can just
# do an area query of size h_k / 2 around each center.
CENTER_IS_CLOSEST_TO_ORIG_PANEL_REFINER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_panel_starts,
        particle_id_t *box_to_panel_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t *panel_to_center_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
        int npanels,
        coord_t r_max,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        particle_id_t my_panel = bsearch(panel_to_center_starts, npanels + 1, i);

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(i)", ball_center)}
        ${ball_radius} = r_max + panel_sizes[my_panel] / 2;
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        for (particle_id_t panel_idx = box_to_panel_starts[${leaf_box_id}];
             panel_idx < box_to_panel_starts[${leaf_box_id} + 1];
             ++panel_idx)
        {
            particle_id_t panel = box_to_panel_lists[panel_idx];

            // Skip self.
            if (my_panel == panel)
            {
                continue;
            }

            bool is_close = false;

            for (particle_id_t source = panel_to_source_starts[panel];
                 source < panel_to_source_starts[panel + 1];
                 ++source)
            {
                coord_vec_t source_coords;

                ${load_particle(
                    "INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

                is_close |= (
                    distance(${ball_center}, source_coords)
                    <= panel_sizes[my_panel] / 2);
            }

            if (is_close)
            {
                panel_refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="refine_center_closest_to_orig_panel",
    preamble=str(InlineBinarySearch("particle_id_t")))


# Implements "Algorithm for triggering refinement based on Condition 2"
CENTER_IS_FAR_FROM_NONNEIGHBOR_PANEL_REFINER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_panel_starts,
        particle_id_t *box_to_panel_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t *panel_to_center_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t panel_offset,
        particle_id_t *sorted_target_ids,
        coord_t *panel_sizes,
        particle_id_t *panel_adjacency_starts,
        particle_id_t *panel_adjacency_lists,
        int npanels,
        coord_t *tunnel_query_dists,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        particle_id_t my_panel = bsearch(panel_to_center_starts, npanels + 1, i);
        coord_vec_t my_center_coords;

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(i)", "my_center_coords")}
        ${load_particle("INDEX_FOR_PANEL_PARTICLE(my_panel)", ball_center)}
        ${ball_radius} = tunnel_query_dists[my_panel];
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        for (particle_id_t panel_idx = box_to_panel_starts[${leaf_box_id}];
             panel_idx < box_to_panel_starts[${leaf_box_id} + 1];
             ++panel_idx)
        {
            particle_id_t panel = box_to_panel_lists[panel_idx];

            bool is_self_or_adjacent = (my_panel == panel);

            for (particle_id_t adj_panel_idx = panel_adjacency_starts[my_panel];
                 adj_panel_idx < panel_adjacency_starts[my_panel + 1];
                 ++adj_panel_idx)
            {
                is_self_or_adjacent |= (
                    panel_adjacency_lists[adj_panel_idx] == panel);
            }

            // Skip self and adjacent panels.
            if (is_self_or_adjacent)
            {
                continue;
            }

            bool is_close = false;

            for (particle_id_t source = panel_to_source_starts[panel];
                 source < panel_to_source_starts[panel + 1];
                 ++source)
            {
                coord_vec_t source_coords;

                ${load_particle(
                    "INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

                is_close |= (
                    distance(my_center_coords, source_coords)
                    <= panel_sizes[panel] / 2);
            }

            if (is_close)
            {
                panel_refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="refine_center_far_from_nonneighbor_panels",
    preamble=str(InlineBinarySearch("particle_id_t")))

# }}}


# {{{ lpot source refiner

class QBXLayerPotentialSourceRefiner(DiscrPlotterMixin):
    """
    Driver for refining the QBX source grid. Follows [1]_.

    .. [1] Rachh, Manas, Andreas Klöckner, and Michael O'Neil. "Fast
       algorithms for Quadrature by Expansion I: Globally valid expansions."

    .. automethod:: get_refine_flags
    .. automethod:: __call__
    """

    def __init__(self, context):
        self.context = context
        from pytential.qbx.utils import TreeWithQBXMetadataBuilder
        self.tree_builder = TreeWithQBXMetadataBuilder(self.context)
        from boxtree.area_query import PeerListFinder
        self.peer_list_finder = PeerListFinder(self.context)

    # {{{ kernels

    @memoize_method
    def get_tunnel_query_distance_finder(self, dimensions, coord_dtype,
                                         particle_id_dtype):
        from pyopencl.tools import dtype_to_ctype
        from boxtree.tools import AXIS_NAMES
        logger.info("refiner: building tunnel query distance finder kernel")

        knl = TUNNEL_QUERY_DISTANCE_FINDER_TEMPLATE.build(
                self.context,
                type_aliases=(
                    ("particle_id_t", particle_id_dtype),
                    ("coord_t", coord_dtype),
                    ),
                var_values=(
                    ("dimensions", dimensions),
                    ("AXIS_NAMES", AXIS_NAMES),
                    ("coord_dtype", coord_dtype),
                    ("dtype_to_ctype", dtype_to_ctype),
                    ("vec_types", tuple(cl.array.vec.types.items())),
                    ))

        logger.info("refiner: done building tunnel query distance finder kernel")
        return knl

    @memoize_method
    def get_center_is_closest_to_orig_panel_refiner(self, dimensions,
                                                    coord_dtype, box_id_dtype,
                                                    peer_list_idx_dtype,
                                                    particle_id_dtype,
                                                    max_levels):
        return CENTER_IS_CLOSEST_TO_ORIG_PANEL_REFINER.generate(self.context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def get_center_is_far_from_nonneighbor_panel_refiner(self, dimensions,
                                                         coord_dtype,
                                                         box_id_dtype,
                                                         peer_list_idx_dtype,
                                                         particle_id_dtype,
                                                         max_levels):
        return CENTER_IS_FAR_FROM_NONNEIGHBOR_PANEL_REFINER.generate(self.context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def get_2_to_1_panel_ratio_refiner(self):
        knl = lp.make_kernel([
            "{[panel]: 0<=panel<npanels}",
            "{[ineighbor]: neighbor_start<=ineighbor<neighbor_stop}"
            ],
            """
            for panel
                <> neighbor_start = panel_adjacency_starts[panel]
                <> neighbor_stop = panel_adjacency_starts[panel + 1]
                for ineighbor
                    <> neighbor = panel_adjacency_lists[ineighbor]

                    <> oversize = (refine_flags_prev[panel] == 0
                           and (
                               (panel_sizes[panel] > 2 * panel_sizes[neighbor]) or
                               (panel_sizes[panel] > panel_sizes[neighbor] and
                                   refine_flags_prev[neighbor] == 1)))

                    if oversize
                        refine_flags[panel] = 1
                        refine_flags_updated = 1 {id=write_refine_flags_updated}
                    end
                end
            end
            """, [
            lp.GlobalArg("panel_adjacency_lists", shape=None),
            "..."
            ],
            options="return_dict",
            silenced_warnings="write_race(write_refine_flags_updated)",
            name="refine_2_to_1_adj_panel_size_ratio")
        knl = lp.split_iname(knl, "panel", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    @memoize_method
    def get_helmholtz_k_to_panel_ratio_refiner(self):
        knl = lp.make_kernel(
            "{[panel]: 0<=panel<npanels}",
            """
            for panel
                <> oversize = panel_sizes[panel] * helmholtz_k > 5
                refine_flags[panel] = 1 {if=oversize}
                refine_flags_updated = 1 {id=write_refine_flags_updated,if=oversize}
            end
            """,
            options="return_dict",
            silenced_warnings="write_race(write_refine_flags_updated)",
            name="refine_helmholtz_k_to_panel_size_ratio")
        knl = lp.split_iname(knl, "panel", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    # }}}

    # {{{ refinement triggering

    def refinement_check_center_is_closest_to_orig_panel(self, queue, tree,
            lpot_source, peer_lists, tq_dists, refine_flags, debug, wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_center_is_closest_to_orig_panel_refiner(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        logger.info("refiner: checking center is closest to orig panel")

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        found_panel_to_refine = cl.array.zeros(queue, 1, np.int32)
        found_panel_to_refine.finish()

        r_max = cl.array.max(tq_dists).get()

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_panel_starts,
                tree.box_to_qbx_panel_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_panel_to_center_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("npanels"),
                tree.nqbxpanels,
                r_max,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxcenters),
            queue=queue)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking center is closest to orig panel")

        return found_panel_to_refine.get()[0] == 1

    def refinement_check_center_is_far_from_nonneighbor_panels(self, queue,
                tree, lpot_source, peer_lists, tq_dists, refine_flags, debug,
                wait_for=None):
        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = 10 * div_ceil(tree.nlevels, 10)

        knl = self.get_center_is_far_from_nonneighbor_panel_refiner(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        logger.info("refiner: checking center is far from nonneighbor panels")

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        found_panel_to_refine = cl.array.zeros(queue, 1, np.int32)
        found_panel_to_refine.finish()

        adjacency = self.get_adjacency_on_device(queue, lpot_source)

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_panel_starts,
                tree.box_to_qbx_panel_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_panel_to_center_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.qbx_user_panel_slice.start,
                tree.sorted_target_ids,
                lpot_source.panel_sizes("npanels"),
                adjacency.adjacency_starts,
                adjacency.adjacency_lists,
                tree.nqbxpanels,
                tq_dists,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxcenters),
            queue=queue)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking center is far from nonneighbor panels")

        return found_panel_to_refine.get()[0] == 1

    def refinement_check_helmholtz_k_to_panel_size_ratio(self, queue, lpot_source,
                helmholtz_k, refine_flags, debug, wait_for=None):
        knl = self.get_helmholtz_k_to_panel_ratio_refiner()

        logger.info("refiner: checking helmholtz k to panel size ratio")

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        evt, out = knl(queue,
                       panel_sizes=lpot_source.panel_sizes("npanels"),
                       refine_flags=refine_flags,
                       refine_flags_updated=np.array(0),
                       helmholtz_k=np.array(helmholtz_k),
                       wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking helmholtz k to panel size ratio")

        return (out["refine_flags_updated"].get() == 1).all()

    def refinement_check_2_to_1_panel_size_ratio(self, queue, lpot_source,
                refine_flags, debug, wait_for=None):
        knl = self.get_2_to_1_panel_ratio_refiner()
        adjacency = self.get_adjacency_on_device(queue, lpot_source)

        refine_flags_updated = False

        logger.info("refiner: checking 2-to-1 panel size ratio")

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        # Iterative refinement until no more panels can be marked
        while True:
            evt, out = knl(queue,
                           npanels=lpot_source.density_discr.mesh.nelements,
                           panel_sizes=lpot_source.panel_sizes("npanels"),
                           refine_flags=refine_flags,
                           # It's safe to pass this here, as the resulting data
                           # race won't affect the final result of the
                           # computation.
                           refine_flags_prev=refine_flags,
                           refine_flags_updated=np.array(0),
                           panel_adjacency_starts=adjacency.adjacency_starts,
                           panel_adjacency_lists=adjacency.adjacency_lists,
                           wait_for=wait_for)

            cl.wait_for_events([evt])

            if (out["refine_flags_updated"].get() == 1).all():
                refine_flags_updated = True
            else:
                break

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        logger.info("refiner: done checking 2-to-1 panel size ratio")

        return refine_flags_updated

    # }}}

    # {{{ other utilities

    def get_tunnel_query_dists(self, queue, tree, lpot_source):
        """
        Compute radii for the tubular neighborhood around each panel center of mass.
        """
        nqbxpanels = lpot_source.density_discr.mesh.nelements
        # atomic_max only works on float32
        tq_dists = cl.array.zeros(queue, nqbxpanels, np.float32)
        tq_dists.finish()

        knl = self.get_tunnel_query_distance_finder(tree.dimensions,
                tree.coord_dtype, tree.particle_id_dtype)

        evt = knl(tree.qbx_user_source_slice.start,
                  tree.qbx_user_panel_slice.start,
                  nqbxpanels,
                  tree.qbx_panel_to_source_starts,
                  tree.sorted_target_ids,
                  lpot_source.panel_sizes("npanels"),
                  tq_dists,
                  *tree.sources,
                  queue=queue,
                  range=slice(tree.nqbxsources))

        cl.wait_for_events([evt])

        if tree.coord_dtype != tq_dists.dtype:
            tq_dists = tq_dists.astype(tree.coord_dtype)

        return tq_dists, evt

    def get_adjacency_on_device(self, queue, lpot_source):
        """
        Take adjacency information from the mesh and place it onto the device.
        """
        from boxtree.tools import DeviceDataRecord
        adjacency = lpot_source.density_discr.mesh.nodal_adjacency
        adjacency_starts = cl.array.to_device(queue, adjacency.neighbors_starts)
        adjacency_lists = cl.array.to_device(queue, adjacency.neighbors)

        return DeviceDataRecord(
            adjacency_starts=adjacency_starts,
            adjacency_lists=adjacency_lists)

    def get_refine_flags(self, queue, lpot_source):
        """
        Return an array on the device suitable for use as element refine flags.

        :arg queue: An instance of :class:`pyopencl.CommandQueue`.
        :arg lpot_source: An instance of :class:`NewQBXLayerPotentialSource`.

        :returns: An instance of :class:`pyopencl.array.Array` suitable for
            use as refine flags, initialized to zero.
        """
        result = cl.array.zeros(
            queue, lpot_source.density_discr.mesh.nelements, np.int32)
        return result, result.events[0]

    # }}}

    def refine(self, queue, lpot_source, refine_flags, refiner, factory, debug):
        """
        Refine the underlying mesh and discretization.
        """
        if isinstance(refine_flags, cl.array.Array):
            refine_flags = refine_flags.get(queue)
        refine_flags = refine_flags.astype(np.bool)

        logger.info("refiner: calling meshmode")

        refiner.refine(refine_flags)
        from meshmode.discretization.connection import make_refinement_connection

        conn = make_refinement_connection(
                refiner, lpot_source.density_discr,
                factory)

        logger.info("refiner: done calling meshmode")

        new_density_discr = conn.to_discr

        new_lpot_source = NewQBXLayerPotentialSource(
            new_density_discr, lpot_source.fine_order,
            qbx_order=lpot_source.qbx_order,
            fmm_level_to_order=lpot_source.fmm_level_to_order,
            real_dtype=lpot_source.real_dtype, debug=debug)

        return new_lpot_source, conn

    def __call__(self, lpot_source, discr_factory, helmholtz_k=None,
                 # FIXME: Set debug=False once everything works.
                 refine_flags=None, debug=True, maxiter=50):
        """
        Entry point for calling the refiner.

        :arg lpot_source: An instance of :class:`NewQBXLayerPotentialSource`.

        :arg group_factory: An instance of
            :class:`meshmode.mesh.discretization.ElementGroupFactory`. Used for
            discretizing the refined mesh.

        :arg helmholtz_k: The Helmholtz parameter, or `None` if not applicable.

        :arg refine_flags: A :class:`pyopencl.array.Array` indicating which
            panels should get refined initially, or `None` if no initial
            refinement should be done. Should have size equal to the number of
            panels. See also :meth:`get_refine_flags()`.

        :returns: A tuple ``(lpot_source, conns)`` where ``lpot_source`` is the
            refined layer potential source, and ``conns`` is a list of
            :class:`meshmode.discretization.connection.DiscretizationConnection`
            objects going from the original mesh to the refined mesh.
        """
        from meshmode.mesh.refinement import Refiner
        refiner = Refiner(lpot_source.density_discr.mesh)
        connections = []

        with cl.CommandQueue(self.context) as queue:
            if refine_flags:
                lpot_source, conn = self.refine(
                            queue, lpot_source, refine_flags, refiner, discr_factory,
                            debug)
                connections.append(conn)

            done_refining = False
            niter = 0

            while not done_refining:
                niter += 1

                if niter > maxiter:
                    logger.warning(
                        "Max iteration count reached in QBX layer potential source"
                        " refiner.")
                    break

                # Build tree and auxiliary data.
                # FIXME: The tree should not have to be rebuilt at each iteration.
                tree = self.tree_builder(queue, lpot_source)
                wait_for = []

                peer_lists, evt = self.peer_list_finder(queue, tree, wait_for)
                wait_for = [evt]

                refine_flags, evt = self.get_refine_flags(queue, lpot_source)
                wait_for.append(evt)

                tq_dists, evt = self.get_tunnel_query_dists(queue, tree, lpot_source)
                wait_for.append(evt)

                # Run refinement checkers.
                must_refine = False

                must_refine |= \
                        self.refinement_check_center_is_closest_to_orig_panel(
                            queue, tree, lpot_source, peer_lists, tq_dists,
                            refine_flags, debug, wait_for)

                must_refine |= \
                        self.refinement_check_center_is_far_from_nonneighbor_panels(
                            queue, tree, lpot_source, peer_lists, tq_dists,
                            refine_flags, debug, wait_for)

                must_refine |= \
                        self.refinement_check_2_to_1_panel_size_ratio(
                            queue, lpot_source, refine_flags, debug, wait_for)

                if helmholtz_k:
                    must_refine |= \
                            self.refinement_check_helmholtz_k_to_panel_size_ratio(
                                queue, lpot_source, helmholtz_k, refine_flags, debug,
                                wait_for)

                if must_refine:
                    lpot_source, conn = self.refine(
                            queue, lpot_source, refine_flags, refiner, discr_factory,
                            debug)
                    connections.append(conn)

                del tree
                del peer_lists
                del tq_dists
                del refine_flags
                done_refining = not must_refine

        return lpot_source, connections

# }}}

# vim: foldmethod=marker:filetype=pyopencl
