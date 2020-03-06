from __future__ import division, absolute_import

__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
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

import numpy as np
import pyopencl as cl  # noqa
import pyopencl.array  # noqa

import six
from six.moves import intern

from pytools import memoize_method
from pytential.symbolic.mappers import EvaluationMapperBase


# {{{ helpers

def is_zero(x):
    return isinstance(x, (int, float, complex, np.number)) and x == 0


def _get_layer_potential_args(mapper, expr, include_args=None):
    """
    :arg mapper: a :class:`~pytential.symbolic.matrix.MatrixBuilderBase`.
    :arg expr: symbolic layer potential expression.

    :return: a mapping of kernel arguments evaluated by the *mapper*.
    """

    kernel_args = {}
    for arg_name, arg_expr in six.iteritems(expr.kernel_arguments):
        if (include_args is not None
                and arg_name not in include_args):
            continue

        kernel_args[arg_name] = mapper.rec(arg_expr)

    return kernel_args

# }}}


# {{{ base classes for matrix builders

class MatrixBuilderBase(EvaluationMapperBase):
    def __init__(self, queue, dep_expr, other_dep_exprs,
            dep_source, dep_discr, places, context):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`.
        :arg dep_expr: symbolic expression for the input block column
            that the builder is evaluating.
        :arg other_dep_exprs: symbolic expressions for the remaining input
            block columns.
        :arg dep_source: a :class:`~pytential.source.LayerPotentialSourceBase`
            for the given *dep_expr*.
        :arg dep_discr: a concerete :class:`~meshmode.discretization.Discretization`
            for the given *dep_expr*.
        :arg places: a :class:`~pytential.symbolic.execution.GeometryCollection`
            for all the sources and targets the builder is expected to
            encounter.
        """
        super(MatrixBuilderBase, self).__init__(context=context)

        self.queue = queue
        self.dep_expr = dep_expr
        self.other_dep_exprs = other_dep_exprs
        self.dep_source = dep_source
        self.dep_discr = dep_discr
        self.places = places

    # {{{

    def get_dep_variable(self):
        return np.eye(self.dep_discr.nnodes, dtype=np.float64)

    def is_kind_vector(self, x):
        return len(x.shape) == 1

    def is_kind_matrix(self, x):
        return len(x.shape) == 2

    # }}}

    # {{{ map_xxx implementation

    def map_variable(self, expr):
        if expr == self.dep_expr:
            return self.get_dep_variable()
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super(MatrixBuilderBase, self).map_variable(expr)

    def map_subscript(self, expr):
        if expr == self.dep_expr:
            return self.get_dep_variable()
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super(MatrixBuilderBase, self).map_subscript(expr)

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
                if self.is_kind_matrix(rec_child):
                    term_kind = term_kind_matrix
                elif self.is_kind_vector(rec_child):
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

        for child in expr.children:
            rec_child = self.rec(child)

            if is_zero(rec_child):
                return 0

            if isinstance(rec_child, (np.number, int, float, complex)):
                vecs_and_scalars = vecs_and_scalars * rec_child
            elif isinstance(rec_child, np.ndarray):
                if self.is_kind_matrix(rec_child):
                    if mat_result is not None:
                        raise RuntimeError("expression is nonlinear in %s"
                                % self.dep_expr)
                    else:
                        mat_result = rec_child
                else:
                    vecs_and_scalars = vecs_and_scalars * rec_child

        if mat_result is not None:
            if (isinstance(vecs_and_scalars, np.ndarray)
                    and self.is_kind_vector(vecs_and_scalars)):
                vecs_and_scalars = vecs_and_scalars[:, np.newaxis]

            return mat_result * vecs_and_scalars
        else:
            return vecs_and_scalars

    def map_num_reference_derivative(self, expr):
        from pytential import bind, sym
        rec_operand = self.rec(expr.operand)

        assert isinstance(rec_operand, np.ndarray)
        if self.is_kind_matrix(rec_operand):
            raise NotImplementedError("derivatives")

        rec_operand = cl.array.to_device(self.queue, rec_operand)
        op = sym.NumReferenceDerivative(
                ref_axes=expr.ref_axes,
                operand=sym.var("u"),
                dofdesc=expr.dofdesc)
        return bind(self.places, op)(self.queue, u=rec_operand).get()

    def map_node_coordinate_component(self, expr):
        from pytential import bind, sym
        op = sym.NodeCoordinateComponent(expr.ambient_axis, dofdesc=expr.dofdesc)
        return bind(self.places, op)(self.queue).get()

    def map_call(self, expr):
        arg, = expr.parameters
        rec_arg = self.rec(arg)

        if isinstance(rec_arg, np.ndarray) and self.is_kind_matrix(rec_arg):
            raise RuntimeError("expression is nonlinear in variable")

        if isinstance(rec_arg, np.ndarray):
            rec_arg = cl.array.to_device(self.queue, rec_arg)

        from pytential import bind, sym
        op = expr.function(sym.var("u"))
        result = bind(self.places, op)(self.queue, u=rec_arg)

        if isinstance(result, cl.array.Array):
            result = result.get()

        return result

    # }}}


class MatrixBlockBuilderBase(MatrixBuilderBase):
    """Evaluate individual blocks of a matrix operator.

    Unlike, e.g. :class:`MatrixBuilder`, matrix block builders are
    significantly reduced in scope. They are basically just meant
    to evaluate linear combinations of layer potential operators.
    For example, they do not support composition of operators because we
    assume that each operator acts directly on the density.
    """

    def __init__(self, queue, dep_expr, other_dep_exprs,
            dep_source, dep_discr, places, index_set, context):
        """
        :arg index_set: a :class:`sumpy.tools.MatrixBlockIndexRanges` class
            describing which blocks are going to be evaluated.
        """

        super(MatrixBlockBuilderBase, self).__init__(queue,
                dep_expr, other_dep_exprs, dep_source, dep_discr,
                places, context)
        self.index_set = index_set

    @property
    @memoize_method
    def _mat_mapper(self):
        # mat_mapper is used to compute any kernel arguments that needs to
        # be computed on the full discretization, ignoring our index_set,
        # e.g the normal in a double layer potential

        return MatrixBuilderBase(self.queue,
                self.dep_expr,
                self.other_dep_exprs,
                self.dep_source,
                self.dep_discr,
                self.places, self.context)

    @property
    @memoize_method
    def _blk_mapper(self):
        # blk_mapper is used to recursively compute the density to
        # a layer potential operator to ensure there is no composition

        return MatrixBlockBuilderBase(self.queue,
                self.dep_expr,
                self.other_dep_exprs,
                self.dep_source,
                self.dep_discr,
                self.places,
                self.index_set, self.context)

    def get_dep_variable(self):
        return 1.0

    def is_kind_vector(self, x):
        # NOTE: since matrices are flattened, the only way to differentiate
        # them from a vector is by size
        return x.size == self.index_set.row.indices.size

    def is_kind_matrix(self, x):
        # NOTE: since matrices are flattened, we recognize them by checking
        # if they have the right size
        return x.size == self.index_set.linear_row_indices.size

# }}}


# {{{ QBX layer potential matrix builder

# FIXME: PyOpenCL doesn't do all the required matrix math yet.
# We'll cheat and build the matrix on the host.

class MatrixBuilder(MatrixBuilderBase):
    def __init__(self, queue, dep_expr, other_dep_exprs,
            dep_source, dep_discr, places, context):
        super(MatrixBuilder, self).__init__(queue, dep_expr, other_dep_exprs,
                dep_source, dep_discr, places, context)

    def map_interpolation(self, expr):
        from pytential import sym

        if expr.to_dd.discr_stage != sym.QBX_SOURCE_QUAD_STAGE2:
            raise RuntimeError("can only interpolate to QBX_SOURCE_QUAD_STAGE2")
        operand = self.rec(expr.operand)

        if isinstance(operand, (int, float, complex, np.number)):
            return operand
        elif isinstance(operand, np.ndarray) and operand.ndim == 1:
            conn = self.places.get_connection(expr.from_dd, expr.to_dd)
            return conn(self.queue,
                    cl.array.to_device(self.queue, operand)).get(self.queue)
        elif isinstance(operand, np.ndarray) and operand.ndim == 2:
            cache = self.places._get_cache("direct_resampler")
            key = (expr.from_dd.geometry,
                    expr.from_dd.discr_stage,
                    expr.to_dd.discr_stage)

            try:
                mat = cache[key]
            except KeyError:
                from meshmode.discretization.connection import \
                    flatten_chained_connection

                conn = self.places.get_connection(expr.from_dd, expr.to_dd)
                conn = flatten_chained_connection(self.queue, conn)
                mat = conn.full_resample_matrix(self.queue).get(self.queue)

                # FIXME: the resample matrix is slow to compute and very big
                # to store, so caching it may not be the best idea
                cache[key] = mat

            return mat.dot(operand)
        else:
            raise RuntimeError("unknown operand type: {}".format(type(operand)))

    def map_int_g(self, expr):
        lpot_source = self.places.get_geometry(expr.source.geometry)
        source_discr = self.places.get_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_discretization(
                expr.target.geometry, expr.target.discr_stage)

        rec_density = self.rec(expr.density)
        if is_zero(rec_density):
            return 0

        assert isinstance(rec_density, np.ndarray)
        if not self.is_kind_matrix(rec_density):
            raise NotImplementedError("layer potentials on non-variables")

        kernel = expr.kernel
        kernel_args = _get_layer_potential_args(self, expr)

        from sumpy.expansion.local import LineTaylorLocalExpansion
        local_expn = LineTaylorLocalExpansion(kernel, lpot_source.qbx_order)

        from sumpy.qbx import LayerPotentialMatrixGenerator
        mat_gen = LayerPotentialMatrixGenerator(
                self.queue.context, (local_expn,))

        assert abs(expr.qbx_forced_limit) > 0
        from pytential import bind, sym
        radii = bind(self.places, sym.expansion_radii(
            source_discr.ambient_dim,
            dofdesc=expr.target))(self.queue)
        centers = bind(self.places, sym.expansion_centers(
            source_discr.ambient_dim,
            expr.qbx_forced_limit,
            dofdesc=expr.target))(self.queue)

        _, (mat,) = mat_gen(self.queue,
                targets=target_discr.nodes(),
                sources=source_discr.nodes(),
                centers=centers,
                expansion_radii=radii,
                **kernel_args)
        mat = mat.get()

        waa = bind(self.places, sym.weights_and_area_elements(
            source_discr.ambient_dim,
            dofdesc=expr.source))(self.queue)
        mat[:, :] *= waa.get(self.queue)
        mat = mat.dot(rec_density)

        return mat

# }}}


# {{{ p2p matrix builder

class P2PMatrixBuilder(MatrixBuilderBase):
    def __init__(self, queue, dep_expr, other_dep_exprs,
            dep_source, dep_discr, places, context, exclude_self=True):
        super(P2PMatrixBuilder, self).__init__(queue,
                dep_expr, other_dep_exprs, dep_source, dep_discr,
                places, context)

        self.exclude_self = exclude_self

    def map_int_g(self, expr):
        source_discr = self.places.get_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_discretization(
                expr.target.geometry, expr.target.discr_stage)

        rec_density = self.rec(expr.density)
        if is_zero(rec_density):
            return 0

        assert isinstance(rec_density, np.ndarray)
        if not self.is_kind_matrix(rec_density):
            raise NotImplementedError("layer potentials on non-variables")

        # NOTE: copied from pytential.symbolic.primitives.IntG
        # NOTE: P2P evaluation only uses the inner kernel, so it should not
        # get other kernel_args, e.g. normal vectors in a double layer
        kernel = expr.kernel.get_base_kernel()
        kernel_args = kernel.get_args() + kernel.get_source_args()
        kernel_args = set(arg.loopy_arg.name for arg in kernel_args)

        kernel_args = _get_layer_potential_args(self,
                expr, include_args=kernel_args)
        if self.exclude_self:
            kernel_args["target_to_source"] = \
                cl.array.arange(self.queue, 0, target_discr.nnodes, dtype=np.int)

        from sumpy.p2p import P2PMatrixGenerator
        mat_gen = P2PMatrixGenerator(
                self.queue.context, (kernel,), exclude_self=self.exclude_self)

        _, (mat,) = mat_gen(self.queue,
                targets=target_discr.nodes(),
                sources=source_discr.nodes(),
                **kernel_args)

        mat = mat.get()
        mat = mat.dot(rec_density)

        return mat
# }}}


# {{{ block matrix builders

class NearFieldBlockBuilder(MatrixBlockBuilderBase):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source, dep_discr,
            places, index_set, context):
        super(NearFieldBlockBuilder, self).__init__(queue,
                dep_expr, other_dep_exprs, dep_source, dep_discr,
                places, index_set, context)

    def get_dep_variable(self):
        tgtindices = self.index_set.linear_row_indices.get(self.queue)
        srcindices = self.index_set.linear_col_indices.get(self.queue)

        return np.equal(tgtindices, srcindices).astype(np.float64)

    def map_int_g(self, expr):
        lpot_source = self.places.get_geometry(expr.source.geometry)
        source_discr = self.places.get_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_discretization(
                expr.target.geometry, expr.target.discr_stage)

        if source_discr is not target_discr:
            raise NotImplementedError

        rec_density = self._blk_mapper.rec(expr.density)
        if is_zero(rec_density):
            return 0

        if not np.isscalar(rec_density):
            raise NotImplementedError

        kernel = expr.kernel
        kernel_args = _get_layer_potential_args(self._mat_mapper, expr)

        from sumpy.expansion.local import LineTaylorLocalExpansion
        local_expn = LineTaylorLocalExpansion(kernel, lpot_source.qbx_order)

        from sumpy.qbx import LayerPotentialMatrixBlockGenerator
        mat_gen = LayerPotentialMatrixBlockGenerator(
                self.queue.context, (local_expn,))

        assert abs(expr.qbx_forced_limit) > 0
        from pytential import bind, sym
        radii = bind(self.places, sym.expansion_radii(
            source_discr.ambient_dim,
            dofdesc=expr.target))(self.queue)
        centers = bind(self.places, sym.expansion_centers(
            source_discr.ambient_dim,
            expr.qbx_forced_limit,
            dofdesc=expr.target))(self.queue)

        _, (mat,) = mat_gen(self.queue,
                targets=target_discr.nodes(),
                sources=source_discr.nodes(),
                centers=centers,
                expansion_radii=radii,
                index_set=self.index_set,
                **kernel_args)

        waa = bind(self.places, sym.weights_and_area_elements(
            source_discr.ambient_dim,
            dofdesc=expr.source))(self.queue)
        mat *= waa[self.index_set.linear_col_indices]
        mat = rec_density * mat.get(self.queue)

        return mat


class FarFieldBlockBuilder(MatrixBlockBuilderBase):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source, dep_discr,
            places, index_set, context, exclude_self=False):
        super(FarFieldBlockBuilder, self).__init__(queue,
                dep_expr, other_dep_exprs, dep_source, dep_discr,
                places, index_set, context)
        self.exclude_self = exclude_self

    def get_dep_variable(self):
        tgtindices = self.index_set.linear_row_indices.get(self.queue)
        srcindices = self.index_set.linear_col_indices.get(self.queue)

        return np.equal(tgtindices, srcindices).astype(np.float64)

    def map_int_g(self, expr):
        source_discr = self.places.get_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_discretization(
                expr.target.geometry, expr.target.discr_stage)

        if source_discr is not target_discr:
            raise NotImplementedError

        rec_density = self._blk_mapper.rec(expr.density)
        if is_zero(rec_density):
            return 0

        if not np.isscalar(rec_density):
            raise NotImplementedError

        # NOTE: copied from pytential.symbolic.primitives.IntG
        # NOTE: P2P evaluation only uses the inner kernel, so it should not
        # get other kernel_args, e.g. normal vectors in a double layer
        kernel = expr.kernel.get_base_kernel()
        kernel_args = kernel.get_args() + kernel.get_source_args()
        kernel_args = set(arg.loopy_arg.name for arg in kernel_args)

        kernel_args = _get_layer_potential_args(self._mat_mapper,
                expr, include_args=kernel_args)
        if self.exclude_self:
            kernel_args["target_to_source"] = \
                cl.array.arange(self.queue, 0, target_discr.nnodes, dtype=np.int)

        from sumpy.p2p import P2PMatrixBlockGenerator
        mat_gen = P2PMatrixBlockGenerator(
                self.queue.context, (kernel,), exclude_self=self.exclude_self)

        _, (mat,) = mat_gen(self.queue,
                targets=target_discr.nodes(),
                sources=source_discr.nodes(),
                index_set=self.index_set,
                **kernel_args)
        mat = rec_density * mat.get(self.queue)

        return mat

# }}}

# vim: foldmethod=marker
