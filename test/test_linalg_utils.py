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

import pytest

import numpy as np
import numpy.linalg as la

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_matrix_block_index

def test_matrix_block_index(actx_factory):
    actx = actx_factory()

    # {{{ setup

    npoints = 256
    nblocks = 12

    indices = np.arange(0, npoints)
    ranges = np.linspace(0, npoints, nblocks + 1, dtype=np.int64)

    rng = np.random.default_rng()
    mat = rng.random(size=(npoints, npoints))

    from pytential.linalg import BlockIndexRanges, MatrixBlockIndexRanges
    row = BlockIndexRanges(indices, ranges)
    col = BlockIndexRanges(indices, ranges)
    idx = MatrixBlockIndexRanges(row, col)

    # }}}

    # {{{ check the cartesian product

    from pytential.linalg import make_index_blockwise_product
    rowindices, colindices = make_index_blockwise_product(actx, idx)
    rowindices = actx.to_numpy(rowindices)
    colindices = actx.to_numpy(colindices)

    ranges = idx._block_ranges
    for i in range(idx.nblocks):
        istart, iend = ranges[i:i + 2]

        blk_ref = idx.block_take(mat, i, i)
        blk = mat[rowindices[istart:iend], colindices[istart:iend]].reshape(
                idx.block_shape(i, i))

        assert la.norm(blk - blk_ref) < 1.0e-15 * la.norm(blk_ref)

    # }}}

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
