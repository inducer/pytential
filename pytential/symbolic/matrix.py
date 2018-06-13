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

from pytential.symbolic.mappers import EvaluationMapperBase
import pytential.symbolic.primitives as sym
from pytential.symbolic.execution import bind


def is_zero(x):
    return isinstance(x, (int, float, complex, np.number)) and x == 0


# FIXME: PyOpenCL doesn't do all the required matrix math yet.
# We'll cheat and build the matrix on the host.

class MatrixBuilder(EvaluationMapperBase):
    def __init__(self, queue, dep_expr, other_dep_exprs, dep_source, places,
            context):
        self.queue = queue
        self.dep_expr = dep_expr
        self.other_dep_exprs = other_dep_exprs
        self.dep_source = dep_source
        self.dep_discr = dep_source.density_discr
        self.places = places
        self.context = context

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

        kernel = expr.kernel

        kernel_args = {}
        for arg_name, arg_expr in six.iteritems(expr.kernel_arguments):
            rec_arg = self.rec(arg_expr)

            if isinstance(rec_arg, np.ndarray):
                if len(rec_arg.shape) == 2:
                    raise RuntimeError("matrix variables in kernel arguments")
                if len(rec_arg.shape) == 1:
                    from pytools.obj_array import with_object_array_or_scalar

                    def resample(x):
                        return (
                                source.resampler(
                                    self.queue,
                                    cl.array.to_device(self.queue, x))
                                .get(queue=self.queue))

                    rec_arg = with_object_array_or_scalar(resample, rec_arg)

            kernel_args[arg_name] = rec_arg

        from sumpy.expansion.local import LineTaylorLocalExpansion
        local_expn = LineTaylorLocalExpansion(kernel, source.qbx_order)

        from sumpy.qbx import LayerPotentialMatrixGenerator
        mat_gen = LayerPotentialMatrixGenerator(
                self.queue.context, (local_expn,))

        assert target_discr is source.density_discr

        from pytential.qbx.utils import get_centers_on_side

        assert abs(expr.qbx_forced_limit) > 0
        _, (mat,) = mat_gen(self.queue,
                target_discr.nodes(),
                source.quad_stage2_density_discr.nodes(),
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

        if (
                isinstance(rec_arg, np.ndarray)
                and len(rec_arg.shape) == 2):
            raise RuntimeError("expression is nonlinear in variable")

        if isinstance(rec_arg, np.ndarray):
            rec_arg = cl.array.to_device(self.queue, rec_arg)

        op = expr.function(sym.var("u"))
        result = bind(self.dep_source, op)(self.queue, u=rec_arg)

        if isinstance(result, cl.array.Array):
            result = result.get()

        return result
