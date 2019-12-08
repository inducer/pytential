# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

__copyright__ = """
Copyright (C) 2017 Matt Wala
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

import six

import numpy as np
import loopy as lp

from boxtree.tools import DeviceDataRecord
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from pytential.source import LayerPotentialSourceBase
from pytools import memoize_method

import pyopencl as cl
import pyopencl.array  # noqa

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: UnregularizedLayerPotentialSource
"""


# {{{ (panel-based) unregularized layer potential source

class UnregularizedLayerPotentialSource(LayerPotentialSourceBase):
    """A source discretization for a layer potential discretized with a Nyström
    method that uses panel-based quadrature and does not modify the kernel.

    .. attribute:: fmm_level_to_order
    """

    def __init__(self, density_discr,
            fmm_order=False,
            fmm_level_to_order=None,
            expansion_factory=None,
            # begin undocumented arguments
            # FIXME default debug=False once everything works
            debug=True):
        """
        :arg fmm_order: `False` for direct calculation.
        """
        LayerPotentialSourceBase.__init__(self, density_discr)
        self.debug = debug

        if fmm_order is not False and fmm_level_to_order is not None:
            raise TypeError("may not specify both fmm_order and fmm_level_to_order")

        if fmm_level_to_order is None:
            if fmm_order is not False:
                def fmm_level_to_order(kernel, kernel_args, tree, level):  # noqa pylint:disable=function-redefined
                    return fmm_order
            else:
                fmm_level_to_order = False

        self.density_discr = density_discr
        self.fmm_level_to_order = fmm_level_to_order

        if expansion_factory is None:
            from sumpy.expansion import DefaultExpansionFactory
            expansion_factory = DefaultExpansionFactory()
        self.expansion_factory = expansion_factory

    @memoize_method
    def weights_and_area_elements(self):
        from pytential import bind, sym
        with cl.CommandQueue(self.cl_context) as queue:
            waa = bind(self,
                    sym.weights_and_area_elements(self.ambient_dim))(queue)

            return waa.with_queue(None)

    def copy(
            self,
            density_discr=None,
            fmm_level_to_order=None,
            debug=None,
            ):
        return type(self)(
                fmm_level_to_order=(
                    fmm_level_to_order or self.fmm_level_to_order),
                density_discr=density_discr or self.density_discr,
                debug=debug if debug is not None else self.debug)

    def exec_compute_potential_insn(self, queue, insn, bound_expr, evaluate,
            return_timing_data):
        if return_timing_data:
            from warnings import warn
            from pytential.source import UnableToCollectTimingData
            warn(
                   "Timing data collection not supported.",
                   category=UnableToCollectTimingData)

        from pytools.obj_array import with_object_array_or_scalar

        def evaluate_wrapper(expr):
            value = evaluate(expr)
            return with_object_array_or_scalar(lambda x: x, value)

        if self.fmm_level_to_order is False:
            func = self.exec_compute_potential_insn_direct
        else:
            func = self.exec_compute_potential_insn_fmm

        return func(queue, insn, bound_expr, evaluate_wrapper)

    def op_group_features(self, expr):
        from sumpy.kernel import AxisTargetDerivativeRemover
        result = (
                expr.source, expr.density,
                AxisTargetDerivativeRemover()(expr.kernel),
                )

        return result

    def preprocess_optemplate(self, name, discretizations, expr):
        """
        :arg name: The symbolic name for *self*, which the preprocessor
            should use to find which expressions it is allowed to modify.
        """
        from pytential.symbolic.mappers import UnregularizedPreprocessor
        return UnregularizedPreprocessor(name, discretizations)(expr)

    def exec_compute_potential_insn_direct(self, queue, insn, bound_expr, evaluate):
        kernel_args = {}

        for arg_name, arg_expr in six.iteritems(insn.kernel_arguments):
            kernel_args[arg_name] = evaluate(arg_expr)

        strengths = (evaluate(insn.density).with_queue(queue)
                * self.weights_and_area_elements())

        result = []
        p2p = None

        for o in insn.outputs:
            target_discr = bound_expr.get_discretization(o.target_name)

            if p2p is None:
                p2p = self.get_p2p(insn.kernels)

            evt, output_for_each_kernel = p2p(queue,
                    target_discr.nodes(),
                    self.density_discr.nodes(),
                    [strengths], **kernel_args)

            result.append((o.name, output_for_each_kernel[o.kernel_index]))

        timing_data = {}
        return result, timing_data

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

        from sumpy.fmm import SumpyExpansionWranglerCodeContainer
        return SumpyExpansionWranglerCodeContainer(
                self.cl_context,
                fmm_mpole_factory,
                fmm_local_factory,
                out_kernels)

    @property
    @memoize_method
    def fmm_geometry_code_container(self):
        return _FMMGeometryCodeContainer(
                self.cl_context, self.ambient_dim, self.debug)

    def fmm_geometry_data(self, targets):
        return _FMMGeometryData(
                self,
                self.fmm_geometry_code_container,
                targets,
                self.debug)

    def exec_compute_potential_insn_fmm(self, queue, insn, bound_expr, evaluate):
        # {{{ gather unique target discretizations used

        target_name_to_index = {}
        targets = []

        for o in insn.outputs:
            assert o.qbx_forced_limit not in (-1, 1)

            if o.target_name in target_name_to_index:
                continue

            target_name_to_index[o.target_name] = len(targets)
            targets.append(bound_expr.places.get_geometry(o.target_name))

        targets = tuple(targets)

        # }}}

        # {{{ get wrangler

        geo_data = self.fmm_geometry_data(targets)

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
                    queue,
                    geo_data.tree(),
                    output_and_expansion_dtype,
                    self.fmm_level_to_order,
                    source_extra_kwargs=source_extra_kwargs,
                    kernel_extra_kwargs=kernel_extra_kwargs)

        # }}}

        from boxtree.fmm import drive_fmm
        all_potentials_on_every_tgt = drive_fmm(
                geo_data.traversal(), wrangler, strengths, timing_data=None)

        # {{{ postprocess fmm

        result = []

        for o in insn.outputs:
            target_index = target_name_to_index[o.target_name]
            target_slice = slice(*geo_data.target_info().target_discr_starts[
                    target_index:target_index+2])

            result.append(
                    (o.name,
                        all_potentials_on_every_tgt[o.kernel_index][target_slice]))

        # }}}

        timing_data = {}
        return result, timing_data

    # }}}

# }}}


# {{{ fmm tools

class _FMMGeometryCodeContainer(object):

    def __init__(self, cl_context, ambient_dim, debug):
        self.cl_context = cl_context
        self.ambient_dim = ambient_dim
        self.debug = debug

    @memoize_method
    def copy_targets_kernel(self):
        knl = lp.make_kernel(
            """{[dim,i]:
                0<=dim<ndims and
                0<=i<npoints}""",
            """
                targets[dim, i] = points[dim, i]
                """,
            default_offset=lp.auto, name="copy_targets",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.fix_parameters(knl, ndims=self.ambient_dim)

        knl = lp.split_iname(knl, "i", 128, inner_tag="l.0", outer_tag="g.0")
        knl = lp.tag_array_axes(knl, "points", "sep, C")

        knl = lp.tag_array_axes(knl, "targets", "stride:auto, stride:1")
        return lp.tag_inames(knl, dict(dim="ilp"))

    @property
    @memoize_method
    def build_tree(self):
        from boxtree import TreeBuilder
        return TreeBuilder(self.cl_context)

    @property
    @memoize_method
    def build_traversal(self):
        from boxtree.traversal import FMMTraversalBuilder
        return FMMTraversalBuilder(self.cl_context)


class _TargetInfo(DeviceDataRecord):
    """
    .. attribute:: targets

        Shape: ``[dim,ntargets]``

    .. attribute:: target_discr_starts

        Shape: ``[ndiscrs+1]``

    .. attribute:: ntargets
    """


class _FMMGeometryData(object):

    def __init__(self, lpot_source, code_getter, target_discrs, debug=True):
        self.lpot_source = lpot_source
        self.code_getter = code_getter
        self.target_discrs = target_discrs
        self.debug = debug

    @property
    def cl_context(self):
        return self.lpot_source.cl_context

    @property
    def coord_dtype(self):
        return self.lpot_source.density_discr.nodes().dtype

    @property
    def ambient_dim(self):
        return self.lpot_source.density_discr.ambient_dim

    @memoize_method
    def traversal(self):
        with cl.CommandQueue(self.cl_context) as queue:
            trav, _ = self.code_getter.build_traversal(queue, self.tree(),
                    debug=self.debug)

            return trav

    @memoize_method
    def tree(self):
        """Build and return a :class:`boxtree.tree.Tree`
        for this source with these targets.

        |cached|
        """

        code_getter = self.code_getter
        lpot_src = self.lpot_source
        target_info = self.target_info()

        with cl.CommandQueue(self.cl_context) as queue:
            nsources = lpot_src.density_discr.nnodes
            nparticles = nsources + target_info.ntargets

            refine_weights = cl.array.zeros(queue, nparticles, dtype=np.int32)
            refine_weights[:nsources] = 1
            refine_weights.finish()

            MAX_LEAF_REFINE_WEIGHT = 32  # noqa

            tree, _ = code_getter.build_tree(queue,
                    particles=lpot_src.density_discr.nodes(),
                    targets=target_info.targets,
                    max_leaf_refine_weight=MAX_LEAF_REFINE_WEIGHT,
                    refine_weights=refine_weights,
                    debug=self.debug,
                    kind="adaptive")

            return tree

    @memoize_method
    def target_info(self):
        code_getter = self.code_getter
        lpot_src = self.lpot_source
        target_discrs = self.target_discrs

        with cl.CommandQueue(self.cl_context) as queue:
            ntargets = 0
            target_discr_starts = []

            for target_discr in target_discrs:
                target_discr_starts.append(ntargets)
                ntargets += target_discr.nnodes

            target_discr_starts.append(ntargets)

            targets = cl.array.empty(
                    self.cl_context,
                    (lpot_src.ambient_dim, ntargets),
                    self.coord_dtype)

            for start, target_discr in zip(target_discr_starts, target_discrs):
                code_getter.copy_targets_kernel()(
                        queue,
                        targets=targets[:, start:start+target_discr.nnodes],
                        points=target_discr.nodes())

            return _TargetInfo(
                    targets=targets,
                    target_discr_starts=target_discr_starts,
                    ntargets=ntargets).with_queue(None)

# }}}


__all__ = (
        "UnregularizedLayerPotentialSource",
        )

# vim: fdm=marker
