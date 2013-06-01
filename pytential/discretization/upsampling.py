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


from pytential.discretization import Discretization
from pytools import memoize_method, memoize_method_nested


class UpsampleToSourceDiscretization(Discretization):
    """A wrapper around a (coarse) *target_discr* and a (fine) *source_discr*
    that performs all operations except potential evaluations using
    the coarse *target_discr*. For potential evaluations, it upsamples
    the input density to the fine *source_discr* and defers to that
    to perform the evaluation.

    Requires that *target_discr* and *source_discr* are based on the
    same :class:`pytential.mesh.Mesh`
    """
    def __init__(self, target_discr, source_discr):
        if target_discr.mesh is not source_discr.mesh:
            raise ValueError("source_discr and target_discr "
                    "must be based on the same mesh.")

        self.target_discr = target_discr
        self.source_discr = source_discr

        if target_discr.cl_context != source_discr.cl_context:
            raise ValueError("source_discr and target_discr "
                    "must be based on the same OpenCL context.")

    @property
    def mesh(self):
        return self.target_discr.mesh

    @property
    def dim(self):
        return self.target_discr.dim

    @property
    def ambient_dim(self):
        return self.target_discr.ambient_dim

    @property
    def nnodes(self):
        return self.target_discr.nnodes

    def nodes(self, queue):
        return self.target_discr.nodes(queue)

    def parametrization_derivative_component(self,
            queue, ambient_axis, ref_axis):
        return self.target_discr.parametrization_derivative_component(
                queue, ambient_axis, ref_axis)

    def quad_weights(self, queue):
        return self.target_discr.quad_weights(queue)

    # {{{ related to layer potential evaluation

    def op_group_features(self, expr):
        return self.source_discr.op_group_features(expr)

    def gen_instruction_for_layer_pot_from_src(
            self, compiler, tgt_discr, expr, field_var):
        return self.source_discr.gen_instruction_for_layer_pot_from_src(
                compiler, tgt_discr, expr, field_var)

    @memoize_method
    def _upsample_matrix(self, elgroup_index):
        import modepy as mp
        tgrp = self.target_discr.groups[elgroup_index]
        sgrp = self.source_discr.groups[elgroup_index]

        return mp.resampling_matrix(
                mp.simplex_onb(self.dim, tgrp.order),
                sgrp.unit_nodes, tgrp.unit_nodes)

    def _upsample(self, queue, vec):
        @memoize_method_nested
        def knl():
            import loopy as lp
            knl = lp.make_kernel(self.cl_context.devices[0],
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<ndiscr_nodes and
                    0<=j<nmesh_nodes}""",
                "result[k,i] = sum(j, upsample_mat[i, j] * vec[k, j])")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = self.target_discr.empty(vec.dtype)

        for i_grp, (sgrp, tgrp) in enumerate(
                zip(self.source_discr.groups, self.target_discr.groups)):
            knl()(queue,
                    upsample_mat=self._upsample_matrix(i_grp),
                    result=sgrp.view(result), vec=tgrp.view(vec))

        return result

    def exec_layer_potential_insn(self, queue, insn, executor, evaluate):
        from pytools.obj_array import with_object_array_or_scalar
        from functools import partial
        upsample = partial(self._upsample, queue)

        def evaluate_wrapper(expr):
            return with_object_array_or_scalar(upsample, evaluate(expr))

        self.source_discr.exec_layer_potential_insn(
                queue, insn, executor, evaluate_wrapper)

    # }}}

# vim: fdm=marker
