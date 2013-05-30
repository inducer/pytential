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
from pytential.symbolic.compiler import Instruction
from pytential.discretization.poly_element import (
        PolynomialElementGroupBase,
        PolynomialElementDiscretizationBase)

#import pyopencl as cl


# {{{ QBX layer pot instruction

class QBXLayerPotentialInstruction(Instruction):
    """
    :ivar names: the names of variables to assign to
    :ivar return_values: list of tuples *(what, subscript)*
      For the meaning of *what* see :mod:`hellskitchen.fmm`.
    :ivar density:
    :ivar source:
    :ivar ds_direction: None, "n" for normal, otherwise expression.
    :ivar kernel:
    :ivar priority:
    """

    def get_assignees(self):
        return set(self.names)

    def get_dependencies(self, each_vector=False):
        return self.dep_mapper_factory(each_vector)(self.density)

    def __str__(self):
        args = ["kernel=%s" % self.kernel, "density=%s" % self.density,
                "source=%s" % self.source]
        if self.ds_direction is not None:
            if self.ds_direction == "n":
                args.append("ds=normal")
            else:
                args.append("ds=%s" % self.ds_direction)

        lines = []
        for name, (tgt, what, idx) in zip(self.names, self.return_values):
            if idx != ():
                line = "%s <- %s[%s]" % (name, what, idx)
            else:
                line = "%s <- %s" % (name, what)
            if tgt != self.source:
                line += " @ %s" % tgt
            lines.append(line)

        return "{ /* Quad(%s) */\n  %s\n}" % (
                ", ".join(args), "\n  ".join(lines))

    def get_exec_function(self, exec_mapper):
        source = exec_mapper.executor.discretizations[self.source]
        return source.exec_quad_op

# }}}


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

    @memoize_method
    def weights(self):
        return self._quadrature_rule().weights

# }}}


# {{{ QBX discretization base

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

        PolynomialElementDiscretizationBase.__init__(self, cl_ctx, mesh, exact_order,
                real_dtype)

        self.qbx_order = qbx_order

        if expansion_getter is None:
            from sumpy.expansion.local import LineTaylorLocalExpansion
            expansion_getter = LineTaylorLocalExpansion
        self.expansion_getter = expansion_getter

    group_class = QBXElementGroup

    # {{{ interface with execution

    def extra_op_group_features(self, expr):
        target, what, index = expr.what()
        return (what,)

    def gen_instruction_for_layer_pot_from_src(
            self, compiler, tgt_discr, expr, field_var):
        from pytential.symbolic.operators import (
                SourceDiffLayerPotentialOperatorBase,
                Variable)
        is_source_derivative = isinstance(
                expr, SourceDiffLayerPotentialOperatorBase)
        if is_source_derivative:
            if expr.ds_direction is None:
                ds_direction = "n"
            else:
                ds_direction = expr.ds_direction
        else:
            ds_direction = None

        group = compiler.group_to_operators[compiler.op_group_tuple(expr)]
        names = [compiler.get_var_name() for op in group]

        compiler.code.append(
                QBXLayerPotentialInstruction(names=names,
                    return_values=[op.what() for op in group],
                    density=field_var,
                    source=expr.source,
                    ds_direction=ds_direction,
                    kernel=expr.kernel,
                    priority=max(getattr(op, "priority", 0) for op in group),
                    dep_mapper_factory=compiler.dep_mapper_factory))

        for n, d in zip(names, group):
            compiler.expr_to_var[d] = Variable(n)

        return compiler.expr_to_var[expr]

    def make_source_derivative_vec(self, sdvec_obj_array):
        nelements, nlege_nodes = self.curve.points.shape[1:]

        source_derivative_vec = np.empty((2, nlege_nodes, nelements),
                np.float64, order="F")
        assert sdvec_obj_array.shape == (2,)
        for i in range(2):
            source_derivative_vec[i, :, :] = \
                    sdvec_obj_array[i].reshape(nelements, -1).T
        return source_derivative_vec

    @memoize_method
    def get_sumpy_kernels(self, kernel, what_letter, ds_direction):
        nderivatives = 0

        if ds_direction is not None:
            from sumpy.kernel import SourceDerivative
            kernel = SourceDerivative(kernel)
            nderivatives += 1

        if what_letter == "p":
            return [kernel], nderivatives
        elif what_letter == "g":
            from sumpy.kernel import TargetDerivative
            return [TargetDerivative(i, kernel)
                    for i in range(self.mesh.ambient_dim)], nderivatives+1
        elif what_letter == "h":
            from sumpy.kernel import TargetDerivative
            return [
                    TargetDerivative(j, TargetDerivative(i, kernel))
                    for i, j in [(0, 0), (1, 0), (1, 1)]], nderivatives+2
        else:
            raise RuntimeError("unsupported what_letter")

    @memoize_method
    def get_lpot_and_jump_term_applier(
            self, kernel, what_letter, ds_direction):
        kernels, nderivatives = self.get_sumpy_kernels(
                kernel, what_letter, ds_direction)
        from sumpy.layerpot import LayerPotential, JumpTermApplier
        return (
                LayerPotential(self.cl_context,
                    [self.expansion_getter(knl, self.order)
                        for knl in kernels],
                    value_dtypes=np.complex128),
                JumpTermApplier(self.cl_context, kernels,
                    value_dtypes=np.complex128),
                nderivatives)

    def exec_quad_op(self, insn, executor, evaluate):
        from pytools import single_valued
        target = executor.discretizations[
                single_valued(tgt for tgt, what, index in insn.return_values)]

        if not target is self:
            return self.exec_quad_op_not_self(insn, executor, evaluate)

        what_letter = single_valued(
                what for tgt, what, index in insn.return_values)
        kernel = insn.kernel.evaluate(evaluate)

        hashable_ds_direction = insn.ds_direction
        if isinstance(hashable_ds_direction, np.ndarray):
            hashable_ds_direction = tuple(hashable_ds_direction)

        lp_applier, jt_applier, nderivatives = \
                self.get_lpot_and_jump_term_applier(
                        kernel, what_letter, hashable_ds_direction)

        kwargs = {}
        kwargs.update(kernel.k_kwargs())

        if "zk" in kwargs:
            kwargs["k"] = kwargs["zk"]
            del kwargs["zk"]

        # {{{ get, oversample density

        nelements, nlege_nodes = self.curve.points.shape[1:]
        density = evaluate(insn.density).reshape(nelements, -1)
        ovsmp_density = np.dot(self.get_oversampling_matrix(),
                density.T).T.reshape(-1)

        # }}}

        # {{{ get, oversample source derivative direction

        if insn.ds_direction is None:
            ds_direction = None
        elif isinstance(insn.ds_direction, str) and insn.ds_direction == "n":
            ds_direction = self.curve.normals.reshape(2, -1).T
            ovsmp_ds_direction = self.ovsmp_curve.normals.reshape(2, -1).T
        else:
            # conversion is necessary because ds_direction comes in as an
            # object array
            ev_ds_dir = np.array(list(evaluate(insn.ds_direction)))

            ds_direction = ev_ds_dir.reshape(2, nelements, -1)

            ovsmp_ds_direction = np.tensordot(
                    self.get_oversampling_matrix(),
                    ds_direction, ([1], [2]))

            # now shaped (nnodes, dim, nelements)
            ovsmp_ds_direction = (
                    ovsmp_ds_direction
                    .transpose(2, 0, 1)
                    .copy()
                    .reshape(-1, 2)
                    )
            # now shaped (dim, points)

            ds_direction = ds_direction.reshape(2, -1).T.copy()

        if ds_direction is not None:
            kwargs["src_derivative_dir"] = ovsmp_ds_direction

        # }}}

        ovsmp_nodes = self.ovsmp_curve.points.reshape(2, -1).T

        # {{{ compute layer potential

        import pyopencl as cl
        queue = cl.CommandQueue(self.cl_context)

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

    def exec_quad_op_not_self(self, insn, executor, evaluate):
        from pytools import single_valued
        target = executor.discretizations[
                single_valued(tgt for tgt, what, index in insn.return_values)]
        what_letter = single_valued(
                what for tgt, what, index in insn.return_values)

        nelements, nlege_nodes = self.curve.points.shape[1:]

        density = evaluate(insn.density).reshape(nelements, nlege_nodes)
        density_with_weights = (density
                * self.curve.expansion.weights[np.newaxis, :]
                * self.curve.speed/2).reshape(-1)

        hashable_ds_direction = insn.ds_direction
        if isinstance(hashable_ds_direction, np.ndarray):
            hashable_ds_direction = tuple(hashable_ds_direction)

        if insn.ds_direction is None:
            ds_direction = None
        elif insn.ds_direction == "n":
            ds_direction = self.curve.normals.reshape(2, -1)
        else:
            ds_direction = evaluate(insn.ds_direction)

        kernel = insn.kernel.evaluate(evaluate)

        from sumpy.p2p import P2P
        sumpy_kernels, _ = self.get_sumpy_kernels(kernel, what_letter,
                hashable_ds_direction)

        p2p = P2P(self.cl_context,
            sumpy_kernels, exclude_self=False, value_dtypes=np.complex128)

        p2p_kwargs = {}
        kwargs = {}
        kwargs.update(kernel.k_kwargs())

        if "zk" in kwargs:
            p2p_kwargs["k"] = kwargs["zk"]
        del kwargs

        if ds_direction is not None:
            p2p_kwargs["src_derivative_dir"] = ds_direction.T.copy()

        import pyopencl as cl
        queue = cl.CommandQueue(self.cl_context)
        evt, result = p2p(queue, target.points, self.nodes,
                [density_with_weights], **p2p_kwargs)

        return [
                (name, subresult.T[hellskitchen_map_idx(
                    self.curve.dimensions, what, idx)+(slice(None),)])
                for name, subresult, (target, what, idx) in
                zip(insn.names, result, insn.return_values)], []

    # }}}

# }}}

# vim: fdm=marker
