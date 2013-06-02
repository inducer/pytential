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

#import pyopencl as cl


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

    group_class = QBXElementGroup

    # {{{ interface with execution

    def preprocess_optemplate(self, name, expr):
        """
        :arg name: The symbolic name for *self*, which the preprocessor
            should use to find which expressions it is allowed to modify.
        """
        from pytential.symbolic.mappers import QBXOnSurfaceMapper
        return QBXOnSurfaceMapper(name)(expr)

    def op_group_features(self, expr):
        from pytential.symbolic.primitives import IntGdSource
        assert not isinstance(expr, IntGdSource)

        from sumpy.kernel import remove_axis_target_derivatives
        result = (expr.source, expr.target, expr.density,
                remove_axis_target_derivatives(expr.kernel), expr.qbx_forced_limit)

        return result

    def gen_instruction_for_layer_pot_from_src(
            self, compiler, tgt_discr, expr, field_var):
        group = compiler.group_to_operators[compiler.op_group_features(expr)]
        names = [compiler.get_var_name() for op in group]

        from pytential.symbolic.primitives import (
                IntGdSource,
                Variable)

        is_source_derivative = isinstance(expr, IntGdSource)
        if is_source_derivative:
            dsource = expr.dsource
            from pytools import is_single_valued
            assert is_single_valued(op.dsource for op in group)
        else:
            dsource = None

        from pytential.discretization import LayerPotentialInstruction
        compiler.code.append(
                LayerPotentialInstruction(names=names,
                    kernels_and_targets=[(op.kernel, op.target) for op in group],
                    density=field_var,
                    source=expr.source,
                    dsource=dsource,
                    priority=max(getattr(op, "priority", 0) for op in group),
                    dep_mapper_factory=compiler.dep_mapper_factory))

        for name, group_expr in zip(names, group):
            compiler.expr_to_var[group_expr] = Variable(name)

        return compiler.expr_to_var[expr]

    @memoize_method
    def get_lpot_applier(self, kernels):
        # needs to be separate method for caching

        from pytools import any
        if any(knl.is_complex_valued for knl in kernels):
            value_dtype = self.complex_dtype
        else:
            value_dtype = self.real_dtype

        from sumpy.layerpot import LayerPotential
        return LayerPotential(self.cl_context,
                    [self.expansion_getter(knl, self.qbx_order)
                        for knl in kernels],
                    value_dtypes=value_dtype)

    def exec_layer_potential_insn(self, queue, insn, bound_expr, evaluate):
        kernels = tuple(knl for knl, target in insn.kernels_and_targets)
        lp_applier = self.get_lpot_applier(kernels)

        density = evaluate(insn.density)

        if insn.dsource is None:
            dsource = None
        else:
            dsource = evaluate(insn.dsource)

        for kernel, target in insn.kernels_and_targets:
            centers = self.centers(target)

        ovsmp_nodes = self.ovsmp_curve.points.reshape(2, -1).T

        # {{{ compute layer potential

        # FIXME: Don't ignore qbx_forced_limit

        if nderivatives == 1:
            # compute principal-value integrals by two-sided QBX

            both_outputs = []
            for center_side in [-1, 1]:
                centers = self.curve.centers(force_side=center_side) \
                        .reshape(2, -1).T
                evt, outputs = lp_applier(queue, self.nodes,
                        ovsmp_nodes, centers, [ovsmp_density],
                        self.get_speed(),
                        self.get_weights(), **kwargs)
                both_outputs.append(outputs)

            outputs = [0.5*(out1+outm1) for out1, outm1 in zip(*both_outputs)]
        else:
            # compute all other integrals using one-sided QBX

            centers = self.curve.centers().reshape(2, -1).T
            evt, outputs = lp_applier(queue, self.nodes,
                    ovsmp_nodes, centers, [ovsmp_density],
                    self.get_speed(),
                    self.get_weights(), **kwargs)

            # apply jumps
            evt, outputs = jt_applier(queue, outputs,
                    _JumpTermArgumentProvider(self, density, ds_direction,
                        side=self.curve.high_accuracy_center_sides
                        .reshape(-1)))

        # }}}

        return [(name, get_idx(outputs, self.curve.dimensions, what, idx))
                for name, (target, what, idx) in
                zip(insn.names, insn.return_values)], []

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
            cl_ctx, mesh, qbx_order, source_order,
            expansion_getter=expansion_getter, real_dtype=real_dtype)

    from pytential.discretization.upsampling import \
            UpsampleToSourceDiscretization
    return UpsampleToSourceDiscretization(tgt_discr, src_discr)


# vim: fdm=marker
