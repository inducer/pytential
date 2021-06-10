__copyright__ = "Copyright (C) 2018-2021 Alexandru Fikl"

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

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from arraycontext import PyOpenCLArrayContext
from pytools import memoize_in


# {{{ block index handling

@dataclass(frozen=True)
class BlockIndexRanges:
    """Convenience class for working with subsets (blocks) of an array.

    .. attribute:: nblocks
    .. attribute:: indices

        An :class:`~numpy.ndarray` of not necessarily continuous or increasing
        integers representing the indices of a global array. The individual
        blocks are delimited using :attr:`ranges`.

    .. attribute:: ranges

        An :class:`~numpy.ndarray` of size ``(nblocks,)`` consisting of
        nondecreasing integers used to index into :attr:`indices`. A block
        :math:`i` can be retrieved using ``indices[ranges[i]:ranges[i + 1]]``.

    .. automethod:: block_size
    .. automethod:: block_indices
    .. automethod:: block_take
    """

    indices: np.ndarray
    ranges: np.ndarray

    @property
    def nblocks(self) -> int:
        return self.ranges.size - 1

    def block_size(self, i: int) -> int:
        """
        :returns: the number of indices in the block *i*.
        """
        if not (0 <= i < self.nblocks):
            raise IndexError(f"block {i} is out of bounds for {self.nblocks} blocks")

        return self.ranges[i + 1] - self.ranges[i]

    def block_indices(self, i: int) -> np.ndarray:
        """
        :returns: the actual indices in block *i*.
        """
        if not (0 <= i < self.nblocks):
            raise IndexError(f"block {i} is out of bounds for {self.nblocks} blocks")

        return self.indices[self.ranges[i]:self.ranges[i + 1]]

    def block_take(self, x: np.ndarray, i: int) -> np.ndarray:
        """
        :returns: a subset of *x* corresponding to the indices in block *i*.
            The returned array is a copy (not a view) of the elements of *x*.
        """
        if not (0 <= i < self.nblocks):
            raise IndexError(f"block {i} is out of bounds for {self.nblocks} blocks")

        return x[self.block_indices(i)]


@dataclass(frozen=True)
class MatrixBlockIndexRanges:
    """Convenience class for working with subsets (blocks) of a matrix.

    .. attribute:: nblocks
    .. attribute:: row

        A :class:`BlockIndexRanges` encapsulating row block indices.

    .. attribute:: col

        A :class:`BlockIndexRanges` encapsulating column block indices.

    .. automethod:: block_shape
    .. automethod:: block_indices
    .. automethod:: block_take
    """

    row: BlockIndexRanges
    col: BlockIndexRanges

    def __post_init__(self):
        if self.row.nblocks != self.col.nblocks:
            raise ValueError("row and column must have the same number of blocks: "
                    f"got {self.row.nblocks} row blocks "
                    f"and {self.col.nblocks} column blocks")

    @property
    def nblocks(self):
        return self.row.nblocks

    def block_shape(self, i: int, j: int) -> Tuple[int, int]:
        r"""
        :returns: the shape of the block ``(i, j)``, where *i* indexes into
            the :attr:`row`\ s and *j* into the :attr:`col`\ s.
        """
        return (self.row.block_size(i), self.col.block_size(j))

    def block_indices(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        :returns: the indices that make up the block ``(i, j)`` in the matrix.
        """
        return (self.row.block_indices(i), self.col.block_indices(j))

    def block_take(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        :returns: a subset of the matrix *x* corresponding to the indices in
            the block ``(i, j)``. The returned array is a copy of the elements
            of *x*.
        """
        irow, icol = self.block_indices(i, j)
        return x[np.ix_(irow, icol)]


def make_index_blockwise_product(
        actx: PyOpenCLArrayContext,
        idx: MatrixBlockIndexRanges) -> Tuple[Any, Any]:
    """Constructs a Cartesian product of all the indices in *idx* block by block.

    The indices in the resulting arrays are laid out in *C* order. Retrieving
    """
    @memoize_in(actx, (make_index_blockwise_product, "index_set_product_knl"))
    def prg():
        import loopy as lp
        from loopy.version import MOST_RECENT_LANGUAGE_VERSION

        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[i, j]: 0 <= i < nrows and 0 <= j < ncols}"
            ],
            """
            for irange
                <> nrows = rowranges[irange + 1] - rowranges[irange]
                <> ncols = colranges[irange + 1] - colranges[irange]
                <> offset = rowranges[irange] * colranges[irange]

                for i, j
                    rowindices[offset + ncols * i + j] = \
                            rowindices[rowranges[irange] + i] \
                            {id_prefix=write_index}
                    colindices[offset + ncols * i + j] = \
                            colindices[colranges[irange] + j] \
                            {id_prefix=write_index}
                end
            end
            """, [
                lp.GlobalArg("rowindices", None, shape="nresults"),
                lp.GlobalArg("colindices", None, shape="nresults"),
                lp.ValueArg("nresults", None),
                ...
                ],
            name="index_set_product_knl",
            default_offset=lp.auto,
            assumptions="nranges>=1",
            silenced_warnings="write_race(write_index*)",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.split_iname(knl, "irange", 128, outer_tag="g.0")
        return knl

    @memoize_in(idx, (make_index_blockwise_product, "index_set_product"))
    def _make_product():
        _, (rowindices, colindices) = prg()(actx.queue,
                rowindices=actx.from_numpy(idx.row.indices),
                rowranges=actx.from_numpy(idx.row.ranges),
                colindices=actx.from_numpy(idx.col.indices),
                colranges=actx.from_numpy(idx.col.ranges),
                nresults=idx.row.indices.size * idx.col.indices.size,
                )

        return actx.freeze(rowindices), actx.freeze(colindices)

    return _make_product()

# }}}
