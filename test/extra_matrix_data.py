import numpy as np

from pytools.obj_array import make_obj_array

from pytential import sym

import extra_int_eq_data as extra


# {{{ MatrixTestCase

class MatrixTestCaseMixin:
    # partitioning
    approx_cluster_count = 10
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

    def get_cluster_index(self, actx, places, dofdesc=None):
        if dofdesc is None:
            dofdesc = places.auto_source
        discr = places.get_discretization(dofdesc.geometry)

        max_particles_in_box = self.max_particles_in_box
        if max_particles_in_box is None:
            max_particles_in_box = discr.ndofs // self.approx_cluster_count

        from pytential.linalg import partition_by_nodes
        cindex = partition_by_nodes(actx, places,
                dofdesc=dofdesc,
                tree_kind=self.tree_kind,
                max_particles_in_box=max_particles_in_box)

        # randomly pick a subset of points from each cluster
        if abs(self.index_sparsity_factor - 1.0) > 1.0e-14:
            subset = np.empty(cindex.nclusters, dtype=object)

            for i in range(cindex.nclusters):
                iidx = cindex.cluster_indices(i)
                isize = int(self.index_sparsity_factor * len(iidx))
                isize = max(1, min(isize, len(iidx)))

                subset[i] = np.sort(
                        np.random.choice(iidx, size=isize, replace=False)
                        )

            from pytential.linalg import make_index_list
            cindex = make_index_list(subset)

        return cindex

    def get_tgt_src_cluster_index(self, actx, places, dofdesc=None):
        from pytential.linalg import TargetAndSourceClusterList
        cindex = self.get_cluster_index(actx, places, dofdesc=dofdesc)
        return TargetAndSourceClusterList(cindex, cindex)

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
