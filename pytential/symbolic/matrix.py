from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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
import pyopencl as cl  # noqa
import pyopencl.array  # noqa

import six
from six.moves import intern

from pytools import memoize_method

from pytential.symbolic.mappers import EvaluationMapperBase
import pytential.symbolic.primitives as sym
from pytential.symbolic.execution import bind


# {{{ helpers

def is_zero(x):
    return isinstance(x, (int, float, complex, np.number)) and x == 0


def _resample_arg(queue, source, x):
    if source is None:
        return x

    if not isinstance(x, np.ndarray):
        return x

    if len(x.shape) >= 2:
        raise RuntimeError("matrix variables in kernel arguments")

    def resample(y):
        return source.resampler(queue, cl.array.to_device(queue, y)).get(queue)

    from pytools.obj_array import with_object_array_or_scalar
    return with_object_array_or_scalar(resample, x)


def _get_layer_potential_args(mapper, expr, source):
    kernel_args = {}
    for arg_name, arg_expr in six.iteritems(expr.kernel_arguments):
        rec_arg = mapper.rec(arg_expr)
        kernel_args[arg_name] = _resample_arg(mapper.queue, source, rec_arg)

    return kernel_args


def _get_kernel_args(mapper, kernel, expr, source):
    # NOTE: copied from pytential.symbolic.primitives.IntG
    inner_kernel_args = kernel.get_args() + kernel.get_source_args()
    inner_kernel_args = set(arg.loopy_arg.name for arg in inner_kernel_args)

    kernel_args = {}
    for arg_name, arg_expr in six.iteritems(expr.kernel_arguments):
        if arg_name not in inner_kernel_args:
            continue

        rec_arg = mapper.rec(arg_expr)
        kernel_args[arg_name] = _resample_arg(mapper.queue, source, rec_arg)

    return kernel_args

# }}}


# {{{ QBX layer potential matrix builder

# FIXME: PyOpenCL doesn't do all the required matrix math yet.
# We'll cheat and build the matrix on the host.

class MatrixBuilder(EvaluationMapperBase):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source, dep_discr,
            places, context):
        super(MatrixBuilder, self).__init__(context=context)

        self.queue = queue
        self.dep_expr = dep_expr
        self.other_dep_exprs = other_dep_exprs
        self.dep_source = dep_source
        self.dep_discr = dep_discr
        self.places = places

    def map_variable(self, expr):
        if expr == self.dep_expr:
            return np.eye(self.dep_discr.nnodes, dtype=np.float64)
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super(MatrixBuilder, self).map_variable(expr)

    def map_subscript(self, expr):
        if expr == self.dep_expr:
            return np.eye(self.dep_discr.nnodes, dtype=np.float64)
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super(MatrixBuilder, self).map_subscript(expr)

    def map_sum(self, expr):
        sum_kind = None

        term_kind_matrix = intern("matrix")
        term_kind_vector = intern("vector")
        term_kind_scalar = intern("scalar")

        result = 0
        for child in expr.children:
            rec_child = self.rec(child)

            if is_zero(rec_child):
                continue

            if isinstance(rec_child, np.ndarray):
                if len(rec_child.shape) == 2:
                    term_kind = term_kind_matrix
                elif len(rec_child.shape) == 1:
                    term_kind = term_kind_vector
                else:
                    raise RuntimeError("unexpected array rank")

            else:
                term_kind = term_kind_scalar

            if sum_kind is None:
                sum_kind = term_kind

            if term_kind != sum_kind:
                raise RuntimeError("encountered %s in sum of kind %s"
                        % (term_kind, sum_kind))

            result = result + rec_child

        return result

    def map_product(self, expr):
        mat_result = None
        vecs_and_scalars = 1

        for term in expr.children:
            rec_term = self.rec(term)

            if is_zero(rec_term):
                return 0

            if isinstance(rec_term, (np.number, int, float, complex)):
                vecs_and_scalars = vecs_and_scalars * rec_term
            elif isinstance(rec_term, np.ndarray):
                if len(rec_term.shape) == 2:
                    if mat_result is not None:
                        raise RuntimeError("expression is nonlinear in %s"
                                % self.dep_expr)
                    else:
                        mat_result = rec_term
                else:
                    vecs_and_scalars = vecs_and_scalars * rec_term

        if mat_result is not None:
            if (
                    isinstance(vecs_and_scalars, np.ndarray)
                    and len(vecs_and_scalars.shape) == 1):
                vecs_and_scalars = vecs_and_scalars[:, np.newaxis]

            return mat_result * vecs_and_scalars
        else:
            return vecs_and_scalars

    def map_int_g(self, expr):
        from pytential.symbolic.execution import _get_discretization
        source, source_discr = _get_discretization(self.places, expr.source,
                default_discr="quad_stage2_density_discr")
        _, target_discr = _get_discretization(self.places, expr.target)

        rec_density = self.rec(expr.density)
        if is_zero(rec_density):
            return 0

        assert isinstance(rec_density, np.ndarray)
        if len(rec_density.shape) != 2:
            raise NotImplementedError("layer potentials on non-variables")

        kernel = expr.kernel
        kernel_args = _get_layer_potential_args(self, expr, source)

        from sumpy.expansion.local import LineTaylorLocalExpansion
        local_expn = LineTaylorLocalExpansion(kernel, source.qbx_order)

        from sumpy.qbx import LayerPotentialMatrixGenerator
        mat_gen = LayerPotentialMatrixGenerator(
                self.queue.context, (local_expn,))

        from pytential.qbx.utils import get_centers_on_side
        assert abs(expr.qbx_forced_limit) > 0

        _, (mat,) = mat_gen(self.queue,
                target_discr.nodes(),
                source_discr.nodes(),
                get_centers_on_side(source, expr.qbx_forced_limit),
                expansion_radii=self.dep_source._expansion_radii("nsources"),
                **kernel_args)
        mat = mat.get()

        waa = source.weights_and_area_elements().get(queue=self.queue)
        mat[:, :] *= waa

        resampler = source.direct_resampler
        resample_mat = resampler.full_resample_matrix(self.queue).get(self.queue)

        mat = mat.dot(resample_mat)
        mat = mat.dot(rec_density)

        return mat

    # IntGdSource should have been removed by a preprocessor

    def map_num_reference_derivative(self, expr):
        rec_operand = self.rec(expr.operand)

        assert isinstance(rec_operand, np.ndarray)
        if len(rec_operand.shape) == 2:
            raise NotImplementedError("derivatives")

        where_discr = self.places[expr.where]
        op = sym.NumReferenceDerivative(expr.ref_axes, sym.var("u"))
        return bind(where_discr, op)(
                self.queue, u=cl.array.to_device(self.queue, rec_operand)).get()

    def map_node_coordinate_component(self, expr):
        where_discr = self.places[expr.where]
        op = sym.NodeCoordinateComponent(expr.ambient_axis)
        return bind(where_discr, op)(self.queue).get()

    def map_call(self, expr):
        arg, = expr.parameters
        rec_arg = self.rec(arg)

        if isinstance(rec_arg, np.ndarray) and len(rec_arg.shape) == 2:
            raise RuntimeError("expression is nonlinear in variable")

        if isinstance(rec_arg, np.ndarray):
            rec_arg = cl.array.to_device(self.queue, rec_arg)

        op = expr.function(sym.var("u"))
        result = bind(self.dep_source, op)(self.queue, u=rec_arg)

        if isinstance(result, cl.array.Array):
            result = result.get()

        return result

# }}}


# {{{ p2p matrix builder

class P2PMatrixBuilder(MatrixBuilder):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source, dep_discr,
            places, context, exclude_self=True):
        super(P2PMatrixBuilder, self).__init__(queue,
                dep_expr, other_dep_exprs, dep_source, dep_discr, places, context)

        self.exclude_self = exclude_self

    def map_int_g(self, expr):
        source = self.places[expr.source]
        target_discr = self.places[expr.target]

        if source.density_discr is not target_discr:
            raise NotImplementedError()

        rec_density = self.rec(expr.density)
        if is_zero(rec_density):
            return 0

        assert isinstance(rec_density, np.ndarray)
        if len(rec_density.shape) != 2:
            raise NotImplementedError("layer potentials on non-variables")

        try:
            kernel = expr.kernel.inner_kernel
        except AttributeError:
            kernel = expr.kernel
        kernel_args = _get_kernel_args(self, kernel, expr, source)
        if self.exclude_self:
            kernel_args["target_to_source"] = \
                cl.array.arange(self.queue, 0, target_discr.nnodes, dtype=np.int)

        from sumpy.p2p import P2PMatrixGenerator
        mat_gen = P2PMatrixGenerator(
                self.queue.context, (kernel,), exclude_self=self.exclude_self)

        _, (mat,) = mat_gen(self.queue,
                targets=target_discr.nodes(),
                sources=source.density_discr.nodes(),
                **kernel_args)

        mat = mat.get()
        mat = mat.dot(rec_density)

        return mat
# }}}


# {{{ block matrix builders

class MatrixBlockBuilderBase(EvaluationMapperBase):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source,
            places, context, index_set):
        super(MatrixBlockBuilderBase, self).__init__(context=context)

        self.queue = queue
        self.dep_expr = dep_expr
        self.other_dep_exprs = other_dep_exprs
        self.dep_source = dep_source
        self.dep_discr = dep_source.density_discr
        self.places = places

        self.index_set = index_set

    def _map_dep_variable(self):
        return np.eye(self.index_set.col.indices.shape[0])

    def map_variable(self, expr):
        if expr == self.dep_expr:
            return self._map_dep_variable()
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super(MatrixBlockBuilderBase, self).map_variable(expr)

    def map_subscript(self, expr):
        if expr == self.dep_expr:
            return self.variable_identity()
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super(MatrixBlockBuilderBase, self).map_subscript(expr)

    def map_num_reference_derivative(self, expr):
        rec_operand = self.rec(expr.operand)

        assert isinstance(rec_operand, np.ndarray)
        if len(rec_operand.shape) == 2:
            raise NotImplementedError("derivatives")

        where_discr = self.places[expr.where]
        op = sym.NumReferenceDerivative(expr.ref_axes, sym.var("u"))
        return bind(where_discr, op)(
                self.queue, u=cl.array.to_device(self.queue, rec_operand)).get()

    def map_node_coordinate_component(self, expr):
        where_discr = self.places[expr.where]
        op = sym.NodeCoordinateComponent(expr.ambient_axis)
        return bind(where_discr, op)(self.queue).get()

    def map_call(self, expr):
        arg, = expr.parameters
        rec_arg = self.rec(arg)

        if isinstance(rec_arg, np.ndarray) and len(rec_arg.shape) == 2:
            raise RuntimeError("expression is nonlinear in variable")

        if isinstance(rec_arg, np.ndarray):
            rec_arg = cl.array.to_device(self.queue, rec_arg)

        op = expr.function(sym.var("u"))
        result = bind(self.dep_source, op)(self.queue, u=rec_arg)

        if isinstance(result, cl.array.Array):
            result = result.get()

        return result


class NearFieldBlockBuilder(MatrixBlockBuilderBase):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source,
            places, context, index_set):
        super(NearFieldBlockBuilder, self).__init__(queue,
            dep_expr, other_dep_exprs, dep_source, places, context, index_set)

        self.dummy = MatrixBlockBuilderBase(queue,
            dep_expr, other_dep_exprs, dep_source, places, context, index_set)

    @memoize_method
    def _oversampled_index_set(self, resampler):
        from pytential.linalg.proxy import partition_from_coarse
        srcindices = partition_from_coarse(resampler, self.index_set)

        from sumpy.tools import MatrixBlockIndexRanges
        ovsm_index_set = MatrixBlockIndexRanges(self.queue.context,
                self.index_set.row, srcindices)

        return ovsm_index_set

    def _map_dep_variable(self):
        tgtindices = self.index_set.row.indices.get(self.queue).reshape(-1, 1)
        srcindices = self.index_set.col.indices.get(self.queue).reshape(1, -1)

        return np.equal(tgtindices, srcindices).astype(np.float64)

    def map_int_g(self, expr):
        source = self.places[expr.source]
        target_discr = self.places[expr.target]

        if source.density_discr is not target_discr:
            raise NotImplementedError()

        rec_density = self.dummy.rec(expr.density)
        if is_zero(rec_density):
            return 0

        assert isinstance(rec_density, np.ndarray)
        if len(rec_density.shape) != 2:
            raise NotImplementedError("layer potentials on non-variables")

        resampler = source.direct_resampler
        ovsm_index_set = self._oversampled_index_set(resampler)

        kernel = expr.kernel
        kernel_args = _get_layer_potential_args(self, expr, source)

        from sumpy.expansion.local import LineTaylorLocalExpansion
        local_expn = LineTaylorLocalExpansion(kernel, source.qbx_order)

        from sumpy.qbx import LayerPotentialMatrixBlockGenerator
        mat_gen = LayerPotentialMatrixBlockGenerator(
                self.queue.context, (local_expn,))

        from pytential.qbx.utils import get_centers_on_side
        assert abs(expr.qbx_forced_limit) > 0

        _, (mat,) = mat_gen(self.queue,
                target_discr.nodes(),
                source.quad_stage2_density_discr.nodes(),
                get_centers_on_side(source, expr.qbx_forced_limit),
                expansion_radii=self.dep_source._expansion_radii("nsources"),
                index_set=ovsm_index_set,
                **kernel_args)

        waa = source.weights_and_area_elements().with_queue(self.queue)
        mat *= waa[ovsm_index_set.linear_col_indices]

        from sumpy.tools import MatrixBlockIndexRanges
        ovsm_col_index = ovsm_index_set.get(self.queue)
        ovsm_row_index = MatrixBlockIndexRanges(ovsm_col_index.cl_context,
                ovsm_col_index.col, ovsm_col_index.row)

        mat = mat.get(self.queue)
        resample_mat = resampler.full_resample_matrix(self.queue).get(self.queue)

        rmat = np.empty(ovsm_index_set.nblocks, dtype=np.object)
        for i in range(ovsm_index_set.nblocks):
            mat_blk = ovsm_col_index.block_take(mat, i)
            resample_blk = ovsm_row_index.take(resample_mat, i)

            rmat[i] = mat_blk.dot(resample_blk).reshape(-1)
        mat = np.hstack(rmat)

        # TODO:: multiply with rec_density

        return mat


class FarFieldBlockBuilder(MatrixBlockBuilderBase):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source,
            places, context, index_set, exclude_self=True):
        super(FarFieldBlockBuilder, self).__init__(queue,
            dep_expr, other_dep_exprs, dep_source, places, context, index_set)

        self.dummy = MatrixBlockBuilderBase(queue,
            dep_expr, other_dep_exprs, dep_source, places, context, index_set)
        self.exclude_self = exclude_self

    def _map_dep_variable(self):
        tgtindices = self.index_set.row.indices.get(self.queue).reshape(-1, 1)
        srcindices = self.index_set.col.indices.get(self.queue).reshape(1, -1)

        return np.equal(tgtindices, srcindices).astype(np.float64)

    def map_int_g(self, expr):
        source = self.places[expr.source]
        target_discr = self.places[expr.target]

        if source.density_discr is not target_discr:
            raise NotImplementedError()

        rec_density = self.dummy.rec(expr.density)
        if is_zero(rec_density):
            return 0

        assert isinstance(rec_density, np.ndarray)
        if len(rec_density.shape) != 2:
            raise NotImplementedError("layer potentials on non-variables")

        try:
            kernel = expr.kernel.inner_kernel
        except AttributeError:
            kernel = expr.kernel
        kernel_args = _get_kernel_args(self, kernel, expr, source)
        if self.exclude_self:
            kernel_args["target_to_source"] = \
                cl.array.arange(self.queue, 0, target_discr.nnodes, dtype=np.int)

        from sumpy.p2p import P2PMatrixBlockGenerator
        mat_gen = P2PMatrixBlockGenerator(
                self.queue.context, (kernel,), exclude_self=self.exclude_self)

        _, (mat,) = mat_gen(self.queue,
                targets=target_discr.nodes(),
                sources=source.density_discr.nodes(),
                index_set=self.index_set,
                **kernel_args)
        mat = mat.get()

        # TODO: need to multiply by rec_density

        return mat

# }}}

# vim: foldmethod=marker
