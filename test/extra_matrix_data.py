import numpy as np

from pytools.obj_array import make_obj_array

from pytential import sym
from pytential.linalg import MatrixBlockIndexRanges

import extra_int_eq_data as extra


# {{{ MatrixTestCase

class MatrixTestCaseMixin:
    # partitioning
    approx_block_count = 10
    max_particles_in_box = None
    tree_kind = "adaptive-level-restricted"
    index_sparsity_factor = 1.0

    # proxy
    proxy_radius_factor = 1.1
    proxy_approx_count = 32

    # operators
    op_type = "scalar"

    # disable fmm for matrix tests
    fmm_backend = None

    def get_block_indices(self, actx, places, dofdesc=None):
        if dofdesc is None:
            dofdesc = places.auto_source

        discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)

        max_particles_in_box = self.max_particles_in_box
        if max_particles_in_box is None:
            max_particles_in_box = discr.ndofs // self.approx_block_count

        from pytential.linalg import partition_by_nodes
        indices = partition_by_nodes(actx, places,
                dofdesc=dofdesc,
                tree_kind=self.tree_kind,
                max_particles_in_box=max_particles_in_box)

        # randomly pick a subset of points from each block
        if abs(self.index_sparsity_factor - 1.0) > 1.0e-14:
            subset = np.empty(indices.nblocks, dtype=object)
            for i in range(indices.nblocks):
                iidx = indices.block_indices(i)
                isize = int(self.index_sparsity_factor * len(iidx))
                isize = max(1, min(isize, len(iidx)))

                subset[i] = np.sort(
                        np.random.choice(iidx, size=isize, replace=False)
                        )

            from pytential.linalg import make_block_index_from_array
            indices = make_block_index_from_array(subset)

        return indices

    def get_matrix_block_indices(self, actx, places, dofdesc=None):
        indices = self.get_block_indices(actx, places, dofdesc=dofdesc)
        return MatrixBlockIndexRanges(indices, indices)

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
