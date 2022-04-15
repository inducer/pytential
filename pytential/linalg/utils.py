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
from typing import Optional, Tuple

import numpy as np

from arraycontext import PyOpenCLArrayContext, Array
from pytools import memoize_in, memoize_method

__doc__ = """
Misc
~~~~

.. currentmodule:: pytential.linalg

.. autoclass:: IndexList
.. autoclass:: TargetAndSourceClusterList

.. autofunction:: make_index_list
.. autofunction:: make_index_cluster_cartesian_product
"""


# {{{ cluster index handling

@dataclass(frozen=True)
class IndexList:
    """Convenience class for working with clusters (subsets) of an array.

    .. attribute:: nclusters
    .. attribute:: indices

        An :class:`~numpy.ndarray` of not necessarily continuous or increasing
        integers representing the indices of a global array. The individual
        cluster slices are delimited using :attr:`starts`.

    .. attribute:: starts

        An :class:`~numpy.ndarray` of size ``(nclusters + 1,)`` consisting of
        nondecreasing integers used to index into :attr:`indices`. A cluster
        :math:`i` can be retrieved using ``indices[starts[i]:starts[i + 1]]``.

    .. automethod:: cluster_size
    .. automethod:: cluster_indices
    .. automethod:: cluster_take
    """

    indices: np.ndarray
    starts: np.ndarray

    @property
    def nclusters(self) -> int:
        return self.starts.size - 1

    def cluster_size(self, i: int) -> int:
        if not (0 <= i < self.nclusters):
            raise IndexError(
                    f"cluster {i} is out of bounds for {self.nclusters} clusters")

        return self.starts[i + 1] - self.starts[i]

    def cluster_indices(self, i: int) -> np.ndarray:
        """
        :returns: a view into the :attr:`indices` array for the range
            corresponding to cluster *i*.
        """
        if not (0 <= i < self.nclusters):
            raise IndexError(
                    f"cluster {i} is out of bounds for {self.nclusters} clusters")

        return self.indices[self.starts[i]:self.starts[i + 1]]

    def cluster_take(self, x: np.ndarray, i: int) -> np.ndarray:
        """
        :returns: a subset of *x* corresponding to the indices in cluster *i*.
            The returned array is a copy (not a view) of the elements of *x*.
        """
        if not (0 <= i < self.nclusters):
            raise IndexError(
                    f"cluster {i} is out of bounds for {self.nclusters} clusters")

        return x[self.cluster_indices(i)]


@dataclass(frozen=True)
class TargetAndSourceClusterList:
    """Convenience class for working with clusters (subsets) of a matrix.

    .. attribute:: nclusters
    .. attribute:: targets

        An :class:`IndexList` encapsulating target cluster indices.

    .. attribute:: sources

        An :class:`IndexList` encapsulating source cluster indices.

    .. automethod:: cluster_shape
    .. automethod:: cluster_indices
    .. automethod:: cluster_take
    .. automethod:: flat_cluster_take
    """

    targets: IndexList
    sources: IndexList

    def __post_init__(self):
        if self.targets.nclusters != self.sources.nclusters:
            raise ValueError(
                    "targets and sources must have the same number of clusters: "
                    f"got {self.targets.nclusters} target clusters "
                    f"and {self.sources.nclusters} source clusters")

    @property
    def nclusters(self):
        return self.targets.nclusters

    @property
    @memoize_method
    def _flat_cluster_starts(self):
        return np.cumsum([0] + [
            self.targets.cluster_size(i) * self.sources.cluster_size(i)
            for i in range(self.nclusters)
            ])

    @property
    def _flat_total_size(self):
        return self._flat_cluster_starts[-1]

    def cluster_shape(self, i: int, j: int) -> Tuple[int, int]:
        r"""
        :returns: the shape of the cluster ``(i, j)``, where *i* indexes into
            the :attr:`targets` and *j* into the :attr:`sources`.
        """
        return (self.targets.cluster_size(i), self.sources.cluster_size(j))

    def cluster_indices(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        :returns: a view into the indices that make up the cluster ``(i, j)``.
        """
        return (self.targets.cluster_indices(i), self.sources.cluster_indices(j))

    def cluster_take(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        :returns: a subset of the matrix *x* corresponding to the indices in
            the cluster ``(i, j)``. The returned array is a copy of the elements
            of *x*.
        """
        assert x.ndim == 2

        itargets, isources = self.cluster_indices(i, j)
        return x[np.ix_(itargets, isources)]

    def flat_cluster_take(self, x: np.ndarray, i: int) -> np.ndarray:
        """
        :returns: a subset of an array *x* corresponding to the indices in
            the cluster *i*. Unlike :meth:`cluster_take`, this method indexes
            into flattened cluster arrays (see also
            :func:`make_index_cluster_cartesian_product`).
        """
        assert x.ndim == 1
        istart, iend = self._flat_cluster_starts[i:i + 2]

        return x[istart:iend]


def make_index_list(
        indices: np.ndarray,
        starts: Optional[np.ndarray] = None) -> IndexList:
    """Wrap a ``(indices, starts)`` tuple into an :class:`IndexList`.

    :param starts: if *None*, then *indices* is expected to be an object
        array of indices, so that the *starts* can be reconstructed.
    """
    if starts is None:
        assert indices.dtype.char == "O"

        starts = np.cumsum([0] + [r.size for r in indices])
        indices = np.hstack(indices)
    else:
        if starts[-1] != indices.size:
            raise ValueError("size of 'indices' does not match 'starts' endpoint; "
                    f"expected {indices.size}, but got {starts[-1]}")

    return IndexList(indices=indices, starts=starts)


def make_index_cluster_cartesian_product(
        actx: PyOpenCLArrayContext,
        mindex: TargetAndSourceClusterList) -> Tuple[Array, Array]:
    """Constructs a cluster by cluster Cartesian product of all the
    indices in *mindex*.

    The indices in the resulting arrays are laid out in *C* order. Retrieving
    two-dimensional data for a cluster :math:`i` using the resulting
    index arrays can be done as follows

    .. code:: python

        offsets = np.cumsum([0] + [
            mindex.targets.cluster_size(i) * mindex.sources.cluster_size(i)
            for i in range(mindex.nclusters)
            ])
        istart = offsets[i]
        iend = offsets[i + 1]

        cluster_1d = x[tgtindices[istart:iend], srcindices[istart:iend]]
        cluster_2d = cluster_1d.reshape(*mindex.cluster_shape(i, i))

        assert np.allclose(cluster_2d, mindex.cluster_take(x, i, i))

    The result is equivalent to :meth:`~TargetAndSourceClusterList.cluster_take`,
    which takes the Cartesian product as well.

    :returns: a :class:`tuple` containing ``(tgtindices, srcindices)``, where
        the type of the arrays is the base array type of *actx*.
    """
    @memoize_in(actx, (
        make_index_cluster_cartesian_product, "index_product_knl"))
    def prg():
        import loopy as lp
        from loopy.version import MOST_RECENT_LANGUAGE_VERSION

        knl = lp.make_kernel([
            "{[icluster]: 0 <= icluster < nclusters}",
            "{[i, j]: 0 <= i < ntargets and 0 <= j < nsources}"
            ],
            """
            for icluster
                <> offset = offsets[icluster]
                <> tgtstart = tgtstarts[icluster]
                <> srcstart = srcstarts[icluster]
                <> ntargets = tgtstarts[icluster + 1] - tgtstart
                <> nsources = srcstarts[icluster + 1] - srcstart

                for i, j
                    tgtproduct[offset + nsources * i + j] = \
                            tgtindices[tgtstart + i] \
                            {id_prefix=write_index}
                    srcproduct[offset + nsources * i + j] = \
                            srcindices[srcstart + j] \
                            {id_prefix=write_index}
                end
            end
            """, [
                lp.GlobalArg("offsets", None, shape="nclusters + 1"),
                lp.GlobalArg("tgtproduct", None, shape="nresults"),
                lp.GlobalArg("srcproduct", None, shape="nresults"),
                lp.ValueArg("nresults", np.int64),
                ...
                ],
            name="index_product_knl",
            default_offset=lp.auto,
            assumptions="nclusters>=1",
            silenced_warnings="write_race(write_index*)",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.split_iname(knl, "icluster", 128, outer_tag="g.0")
        return knl

    @memoize_in(mindex, (make_index_cluster_cartesian_product, "index_product"))
    def _product():
        _, (tgtindices, srcindices) = prg()(actx.queue,
                tgtindices=actx.from_numpy(mindex.targets.indices),
                tgtstarts=actx.from_numpy(mindex.targets.starts),
                srcindices=actx.from_numpy(mindex.sources.indices),
                srcstarts=actx.from_numpy(mindex.sources.starts),
                offsets=actx.from_numpy(mindex._flat_cluster_starts),
                nresults=mindex._flat_total_size,
                )

        return actx.freeze(tgtindices), actx.freeze(srcindices)

    return _product()


def make_flat_cluster_diag(
        mat: np.ndarray,
        mindex: TargetAndSourceClusterList) -> np.ndarray:
    """
    :param mat: a one-dimensional :class:`~numpy.ndarray` that has a one-to-one
        correspondence to the index sets constructed by
        :func:`make_index_cluster_cartesian_product` for *mindex*.

    :returns: a block diagonal object :class:`~numpy.ndarray`, where each
        diagonal element :math:`(i, i)` is the reshaped slice of *mat* that
        corresponds to the cluster :math:`i`.
    """
    block_mat = np.full((mindex.nclusters, mindex.nclusters), 0, dtype=object)
    for i in range(mindex.nclusters):
        shape = mindex.cluster_shape(i, i)
        block_mat[i, i] = mindex.flat_cluster_take(mat, i).reshape(*shape)

    return block_mat

# }}}
