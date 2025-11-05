from __future__ import annotations


__copyright__ = "Copyright (C) 2021 Alexandru Fikl"

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

import logging

import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts
from meshmode import _acf  # noqa: F401
from meshmode.array_context import PytestPyOpenCLArrayContextFactory


logger = logging.getLogger(__name__)

from pytential.utils import pytest_teardown_function as teardown_function  # noqa: F401


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_matrix_cluster_index

def test_matrix_cluster_index(actx_factory: ArrayContextFactory):
    actx = actx_factory()

    # {{{ setup

    npoints = 256
    nclusters = 12

    indices = np.arange(0, npoints)
    starts = np.linspace(0, npoints, nclusters + 1, dtype=np.int64)

    from pytential.linalg import IndexList, TargetAndSourceClusterList
    index = IndexList(indices, starts)
    tgt_src_index = TargetAndSourceClusterList(index, index)

    # }}}

    # {{{ check the cartesian product

    from pytential.linalg import make_index_cluster_cartesian_product
    tgtindices, srcindices = (
            make_index_cluster_cartesian_product(actx, tgt_src_index))
    tgtindices = actx.to_numpy(tgtindices)
    srcindices = actx.to_numpy(srcindices)

    rng = np.random.default_rng()
    mat = rng.random(size=(npoints, npoints))
    flat_mat = mat[tgtindices, srcindices]

    starts = tgt_src_index._flat_cluster_starts
    for i in range(tgt_src_index.nclusters):
        shape = tgt_src_index.cluster_shape(i, i)
        istart, iend = starts[i:i + 2]

        cmat_ref = tgt_src_index.cluster_take(mat, i, i)
        cmat_flat = tgt_src_index.flat_cluster_take(flat_mat, i).reshape(shape)
        cmat = mat[tgtindices[istart:iend], srcindices[istart:iend]].reshape(shape)

        assert la.norm(cmat - cmat_ref) < 1.0e-15 * la.norm(cmat_ref)
        assert la.norm(cmat_flat - cmat_ref) < 1.0e-15 * la.norm(cmat_ref)

    # }}}

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
