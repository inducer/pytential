import numpy as np
import numpy.linalg as la

from pytools.obj_array import make_obj_array
from sumpy.tools import BlockIndexRanges, MatrixBlockIndexRanges

from pytential import sym

import extra_int_eq_data as extra


# {{{ helpers

def max_block_error(mat, blk, index_set, p=None):
    error = -np.inf
    for i in range(index_set.nblocks):
        mat_i = index_set.take(mat, i)
        blk_i = index_set.block_take(blk, i)

        error = max(error, la.norm(mat_i - blk_i, ord=p) / la.norm(mat_i, ord=p))

    return error

# }}}


# {{{ MatrixTestCase

class MatrixTestCaseMixin:
    # partitioning
    approx_block_count = 10
    max_particles_in_box = None
    tree_kind = "adaptive-level-restricted"
    index_sparsity_factor = 1.0

    # operators
    op_type = "scalar"

    # disable fmm for matrix tests
    fmm_backend = None

    def get_block_indices(self, actx, discr, matrix_indices=True):
        max_particles_in_box = self.max_particles_in_box
        if max_particles_in_box is None:
            max_particles_in_box = discr.ndofs // self.approx_block_count

        from pytential.linalg.proxy import partition_by_nodes
        indices = partition_by_nodes(actx, discr,
                tree_kind=self.tree_kind,
                max_nodes_in_box=max_particles_in_box)

        if abs(self.index_sparsity_factor - 1.0) < 1.0e-14:
            if not matrix_indices:
                return indices
            return MatrixBlockIndexRanges(actx.context, indices, indices)

        # randomly pick a subset of points
        indices = indices.get(actx.queue)

        subset = np.empty(indices.nblocks, dtype=object)
        for i in range(indices.nblocks):
            iidx = indices.block_indices(i)
            isize = int(self.index_sparsity_factor * len(iidx))
            isize = max(1, min(isize, len(iidx)))

            subset[i] = np.sort(np.random.choice(iidx, size=isize, replace=False))

        ranges = actx.from_numpy(np.cumsum([0] + [r.shape[0] for r in subset]))
        indices = actx.from_numpy(np.hstack(subset))

        indices = BlockIndexRanges(actx.context,
                actx.freeze(indices), actx.freeze(ranges))

        if not matrix_indices:
            return indices
        return MatrixBlockIndexRanges(actx.context, indices, indices)

    def get_operator(self, ambient_dim, qbx_forced_limit="avg"):
        knl = self.knl_class(ambient_dim)
        kwargs = self.knl_sym_kwargs.copy()
        kwargs["qbx_forced_limit"] = qbx_forced_limit

        if self.op_type in ("scalar", "single"):
            sym_u = sym.var("u")
            sym_op = sym.S(knl, sym_u, **kwargs)
        elif self.op_type == "double":
            sym_u = sym.var("u")
            sym_op = sym.D(knl, sym_u, **kwargs)
        elif self.op_type == "scalar_mixed":
            sym_u = sym.var("u")
            sym_op = sym.S(knl, 0.3 * sym_u, **kwargs) \
                    + sym.D(knl, 0.5 * sym_u, **kwargs)
        elif self.op_type == "vector":
            sym_u = sym.make_sym_vector("u", ambient_dim)

            sym_op = make_obj_array([
                sym.Sp(knl, sym_u[0], **kwargs)
                + sym.D(knl, sym_u[1], **kwargs),
                sym.S(knl, 0.4 * sym_u[0], **kwargs)
                + 0.3 * sym.D(knl, sym_u[0], **kwargs)
                ])
        else:
            raise ValueError(f"unknown operator type: '{self.op_type}'")

        if self.side is not None:
            sym_op = 0.5 * self.side * sym_u + sym_op

        return sym_u, sym_op


class CurveTestCase(MatrixTestCaseMixin, extra.CurveTestCase):
    pass


class TorusTestCase(MatrixTestCaseMixin, extra.TorusTestCase):
    pass

# }}}
