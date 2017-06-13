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

from pytential.source import LayerPotentialSourceBase

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: NystromLayerPotentialSource
"""


# {{{ (panel-based) Nystrom layer potential source

class NystromLayerPotentialSource(LayerPotentialSourceBase):
    """A source discretization for a layer potential discretized with a Nyström
    method that uses panel-based quadrature.
    """

    def __init__(self, density_discr,
            # begin undocumented arguments
            # FIXME default debug=False once everything works
            debug=True):
        """
        """
        self.density_discr = density_discr
        self.debug = debug

    @property
    def fine_density_discr(self):
        return self.density_discr

    def resampler(self, queue, f):
        return f

    def with_refinement(self):
        raise NotImplementedError

    def copy(
            self,
            density_discr=None,
            debug=None
            ):
        return type(self)(
                density_discr=density_discr or self.density_discr,
                debug=debug if debug is not None else self.debug)

    def exec_compute_potential_insn(self, queue, insn, bound_expr, evaluate):
        from pytools.obj_array import with_object_array_or_scalar

        def evaluate_wrapper(expr):
            value = evaluate(expr)
            return with_object_array_or_scalar(lambda x: x, value)

        func = self.exec_compute_potential_insn_direct
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
        from pytential.symbolic.mappers import NystromPreprocessor
        return NystromPreprocessor(name, discretizations)(expr)

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
                    target_discr.nodes(), self.fine_density_discr.nodes(),
                    [strengths], **kernel_args)

            result.append((o.name, output_for_each_kernel[o.kernel_index]))

        return result, []

# }}}


__all__ = (
        NystromLayerPotentialSource,
        )

# vim: fdm=marker
