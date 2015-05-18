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

import six

from pytential.symbolic.mappers import EvaluationMapperBase


def is_zero(x):
    return isinstance(x, int) and x == 0


# FIXME: PyOpenCL doesn't do all the required matrix math yet.
# We'll cheat and build the matrix on the host.

class MatrixBuilder(EvaluationMapperBase):
    def __init__(self, queue, dep_expr, dep_source, places, context):
        self.queue = queue
        self.dep_expr = dep_expr
        self.dep_source = dep_source
        self.dep_discr = dep_source.density_discr
        self.places = places
        self.context = context

    def map_variable(self, expr):
        if expr == self.dep_expr:
            return np.eye(self.dep_discr.nnodes, np.float64)
        elif expr.name in self.context:
            return self.context[expr.name]
        else:
            return 0

    def map_subscript(self, expr):
        if expr == self.dep_expr:
            return np.eye(self.dep_discr.nnodes, np.float64)
        else:
            return super(MatrixBuilder, self).map_subscript(expr)

    def map_sum(self, expr):
        result = 0
        for child in expr.children:
            rec_child = self.rec(child)

            if is_zero(rec_child):
                continue

            if (
                    not isinstance(rec_child, np.ndarray)
                    or len(rec_child.shape) != 2):
                raise RuntimeError("non-matrix encountered in sum, "
                        "expression may be affine")

            result = result + rec_child

        return result

    def map_product(self, expr):
        mat_result = None
        vecs_and_scalars = 1

        for term in expr.children:
            rec_term = self.rec(term)

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

        return mat_result * vecs_and_scalars

    def map_int_g(self, expr):
        source = self.places[expr.source]
        target_discr = self.places[expr.target]

        if source.density_discr is not target_discr:
            raise NotImplementedError()

        kernel = expr.kernel

        kernel_args = {}
        for arg_name, arg_expr in six.iteritems(expr.kernel_arguments):
            kernel_args[arg_name] = self.rec(arg_expr)

        from sumpy.expansion.local import LineTaylorLocalExpansion
        local_expn = LineTaylorLocalExpansion(kernel, source.qbx_order)

        from sumpy.qbx import LayerPotentialMatrixGenerator
        mat_gen = LayerPotentialMatrixGenerator(
                self.queue.context, (local_expn,))

        assert abs(expr.qbx_forced_limit) > 0
        _, (mat,) = mat_gen(self.queue,
                target_discr.nodes(),
                source.fine_density_discr.nodes(),
                source.centers(target_discr, expr.qbx_forced_limit),
                **kernel_args)

        mat = mat.get()

        waa = source.weights_and_area_elements().get(queue=self.queue)
        mat[:, :] *= waa

        resample_mat = (
                source.resampler.full_resample_matrix(self.queue).get(self.queue))
        mat = mat.dot(resample_mat)

        return mat

    def map_int_g_ds(self, expr):
        return expr.copy(
                density=self.rec(expr.density),
                dsource=self.rec(expr.dsource),
                kernel_arguments=dict(
                    (name, self.rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))
