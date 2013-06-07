from __future__ import division

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


import numpy as np
#import numpy.linalg as la
import modepy as mp
from pytools import memoize_method
from pytential.discretization.poly_element import (
        PolynomialElementGroupBase,
        PolynomialElementDiscretizationBase,
        PolynomialElementDiscretization)

import pyopencl as cl


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


# {{{ element group

class QBXElementGroup(PolynomialElementGroupBase):
    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 1:
            return mp.LegendreGaussQuadrature(self.order)
        else:
            return mp.XiaoGimbutasSimplexQuadrature(self.order, dims)

    @property
    @memoize_method
    def unit_nodes(self):
        return self._quadrature_rule().nodes

    @property
    @memoize_method
    def weights(self):
        return self._quadrature_rule().weights

# }}}


# {{{ QBX discretization

class QBXDiscretization(PolynomialElementDiscretizationBase):
    """An (unstructured) grid for discretizing a QBX operator.

    .. attribute :: mesh
    .. attribute :: groups
    .. attribute :: nnodes

    .. autoattribute :: nodes
    """
    def __init__(self, cl_ctx, mesh, exact_order, qbx_order,
            expansion_getter=None, real_dtype=np.float64):
        """
        :arg exact_order: The total degree to which the underlying quadrature
            is exact.
        """

        PolynomialElementDiscretizationBase.__init__(
                self, cl_ctx, mesh, exact_order, real_dtype)

        self.qbx_order = qbx_order

        if expansion_getter is None:
            from sumpy.expansion.local import LineTaylorLocalExpansion
            expansion_getter = LineTaylorLocalExpansion
        self.expansion_getter = expansion_getter

        self._centers_cache = {}

    group_class = QBXElementGroup

    @memoize_method
    def centers(self, target_discr, sign):
        import pytential.symbolic.primitives as p
        from pytential.symbolic.execution import bind
        with cl.CommandQueue(self.cl_context) as queue:
            return bind(target_discr,
                    p.Nodes() + 0.5*p.area_element()*p.normal())(queue) \
                            .as_vector(np.object)

    # {{{ interface with execution

    def preprocess_optemplate(self, name, expr):
        """
        :arg name: The symbolic name for *self*, which the preprocessor
            should use to find which expressions it is allowed to modify.
        """
        from pytential.symbolic.mappers import QBXPreprocessor
        return QBXPreprocessor(name)(expr)

    def op_group_features(self, expr):
        from pytential.symbolic.primitives import IntGdSource
        assert not isinstance(expr, IntGdSource)

        from sumpy.kernel import AxisTargetDerivativeRemover
        result = (expr.source, expr.target, expr.density,
                AxisTargetDerivativeRemover()(expr.kernel))

        return result

    @memoize_method
    def get_lpot_applier_and_arg_names_to_exprs(self, kernels):
        # needs to be separate method for caching

        from pytools import any
        if any(knl.is_complex_valued for knl in kernels):
            value_dtype = self.complex_dtype
        else:
            value_dtype = self.real_dtype

        from sumpy.layerpot import LayerPotential
        lpot_applier = LayerPotential(self.cl_context,
                    [self.expansion_getter(knl, self.qbx_order)
                        for knl in kernels],
                    value_dtypes=value_dtype)

        from pytential.symbolic.mappers import KernelEvalArgumentCollector
        keac = KernelEvalArgumentCollector()
        arg_names_to_exprs = {}
        for k in kernels:
            arg_names_to_exprs.update(keac(k))

        return lpot_applier, arg_names_to_exprs

    def exec_layer_potential_insn(self, queue, insn, bound_expr, evaluate):
        lp_applier, arg_names_to_exprs = \
                self.get_lpot_applier_and_arg_names_to_exprs(insn.kernels)

        kernel_args = {}
        for arg_name, arg_expr in arg_names_to_exprs.iteritems():
            kernel_args[arg_name] = evaluate(arg_expr)

        from pymbolic import var
        kernel_args.update(
                (arg.name, evaluate(var(arg.name)))
                for arg in lp_applier.gather_kernel_arguments()
                if arg.name not in kernel_args)

        # FIXME: Do this all at once
        result = []
        for o in insn.outputs:
            target_discr = bound_expr.discretizations[o.target_name]

            assert abs(o.qbx_forced_limit) > 0

            evt, output_for_each_kernel = lp_applier(queue, target_discr.nodes(),
                    self.nodes(),
                    self.centers(target_discr, o.qbx_forced_limit),
                    [evaluate(insn.density)], **kernel_args)
            result.append((o.name, output_for_each_kernel[o.kernel_index]))

        return result, []

    # }}}

# }}}


def make_upsampling_qbx_discr(cl_ctx, mesh, target_order, qbx_order,
        source_order=None, expansion_getter=None, real_dtype=np.float64):
    if source_order is None:
        # twice as many points in 1D?
        source_order = target_order * 4

    tgt_discr = PolynomialElementDiscretization(
            cl_ctx, mesh, target_order, real_dtype=real_dtype)
    src_discr = QBXDiscretization(
            cl_ctx, mesh, source_order, qbx_order,
            expansion_getter=expansion_getter, real_dtype=real_dtype)

    from pytential.discretization.upsampling import \
            UpsampleToSourceDiscretization
    return UpsampleToSourceDiscretization(tgt_discr, src_discr)


# vim: fdm=marker
