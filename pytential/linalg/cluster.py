__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

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

from dataclasses import dataclass, replace
from functools import singledispatch
from typing import Any, Iterator, Optional, Tuple

import numpy as np

from pytools import memoize_method
from pytools.obj_array import make_obj_array

from arraycontext import PyOpenCLArrayContext
from boxtree.tree import Tree
from meshmode.discretization import Discretization
from pytential import sym, GeometryCollection
from pytential.linalg.utils import IndexList, TargetAndSourceClusterList
from pytential.qbx import QBXLayerPotentialSource

__doc__ = """
Clustering
~~~~~~~~~~

.. autoclass:: ClusterLevel
.. autoclass:: ClusterTree

.. autofunction:: split_array
.. autofunction:: cluster
.. autofunction:: uncluster

.. autofunction:: partition_by_nodes
"""

# FIXME: this is just an arbitrary value
_DEFAULT_MAX_PARTICLES_IN_BOX = 32


# {{{ cluster tree


def make_cluster_parent_map(parent_ids: np.ndarray) -> np.ndarray:
    """Construct a parent map for :attr:`ClusterLevel.parent_map`."""
    # NOTE: np.unique returns a sorted array
    unique_parent_ids = np.unique(parent_ids)
    ids = np.arange(parent_ids.size)

    return make_obj_array([
        ids[parent_ids == unique_parent_ids[i]]
        for i in range(unique_parent_ids.size)
        ])


@dataclass(frozen=True)
class ClusterLevel:
    """A level in a :class:`ClusterTree`.

    .. attribute:: level

        Current level that is represented.

    .. attribute:: nclusters

        Number of clusters on the current level (same as number of boxes
        in :attr:`box_ids`).

    .. attribute:: box_ids

        Box IDs on the current level.

    .. attribute:: parent_map

        An object :class:`~numpy.ndarray` containing buckets of child indices,
        i.e. ``parent_map[i]`` contains all the child indices that will cluster
        into the same parent. Note that this indexing is local to this level
        and is not related to the tree indexing stored by the :class:`ClusterTree`.
    """

    level: int
    box_ids: np.ndarray
    parent_map: np.ndarray

    @property
    def nclusters(self):
        return self.box_ids.size


@dataclass(frozen=True)
class ClusterTree:
    r"""Hierarchical cluster representation.

    .. attribute:: nlevels

        Total number of levels in the tree.

    .. attribute:: leaf_cluster_box_ids

        Box IDs for each cluster on the leaf level of the tree.

    .. attribute:: tree_cluster_parent_ids

        Parent box IDs for :attr:`leaf_cluster_box_ids`.

    .. attribute:: levels

        An :class:`~numpy.ndarray` of :class:`ClusterLevel`\ s.
    """

    nlevels: int
    leaf_cluster_box_ids: np.ndarray
    tree_cluster_parent_ids: np.ndarray

    # NOTE: only here to allow easier debugging + testing
    _tree: Optional[Tree]

    @property
    def nclusters(self):
        return self.leaf_cluster_box_ids.size

    @property
    @memoize_method
    def levels(self) -> np.ndarray:
        return make_obj_array(list(self.iter_levels()))

    def iter_levels(self) -> Iterator[ClusterLevel]:
        """
        :returns: an iterator over all the :class:`ClusterLevel` levels.
        """

        box_ids = self.leaf_cluster_box_ids
        parent_ids = self.tree_cluster_parent_ids[box_ids]
        clevel = ClusterLevel(
            level=self.nlevels - 1,
            box_ids=box_ids,
            parent_map=make_cluster_parent_map(parent_ids),
            )

        for _ in range(self.nlevels - 1, -1, -1):
            yield clevel

            box_ids = np.unique(self.tree_cluster_parent_ids[clevel.box_ids])
            parent_ids = self.tree_cluster_parent_ids[box_ids]
            clevel = ClusterLevel(
                level=clevel.level - 1,
                box_ids=box_ids,
                parent_map=make_cluster_parent_map(parent_ids)
                )

        assert clevel.nclusters == 1

# }}}


# {{{ cluster

def split_array(x: np.ndarray, index: IndexList) -> np.ndarray:
    """
    :returns: an object :class:`~numpy.ndarray` where each entry contains the
        elements of the :math:`i`-th cluster in *index*.
    """
    assert x.size == index.nindices

    from pytools.obj_array import make_obj_array
    return make_obj_array([
        index.cluster_take(x, i) for i in range(index.nclusters)
        ])


@singledispatch
def cluster(obj: Any, clevel: ClusterLevel) -> Any:
    """Merge together elements of *obj* into their parent object, as described
    by :attr:`ClusterLevel.parent_map`.
    """
    raise NotImplementedError(type(obj).__name__)


@cluster.register(IndexList)
def _cluster_index_list(obj: IndexList, clevel: ClusterLevel) -> IndexList:
    assert obj.nclusters == clevel.nclusters

    if clevel.nclusters == 1:
        return obj

    from pytential.linalg.utils import make_index_list
    indices = make_obj_array([
        np.concatenate([obj.cluster_indices(i) for i in ppm])
        for ppm in clevel.parent_map
        ])

    return make_index_list(indices)


@cluster.register(TargetAndSourceClusterList)
def _cluster_target_and_source_cluster_list(
        obj: TargetAndSourceClusterList, clevel: ClusterLevel,
        ) -> TargetAndSourceClusterList:
    assert obj.nclusters == clevel.nclusters

    if clevel.nclusters == 1:
        return obj

    return replace(obj,
        targets=cluster(obj.targets, clevel),
        sources=cluster(obj.sources, clevel))


@cluster.register(np.ndarray)
def _cluster_ndarray(obj: np.ndarray, clevel: ClusterLevel) -> np.ndarray:
    assert obj.shape == (clevel.nclusters,)
    if clevel.nclusters == 1:
        return obj

    def make_block(i: int, j: int):
        if i == j:
            return obj[i]

        return np.zeros((obj[i].shape[0], obj[j].shape[1]), dtype=obj[i].dtype)

    from pytools import single_valued
    ndim = single_valued(block.ndim for block in obj)

    if ndim == 1:
        return make_obj_array([
            np.concatenate([obj[i] for i in ppm]) for ppm in clevel.parent_map
            ])
    elif ndim == 2:
        return make_obj_array([
            np.block([[make_block(i, j) for j in ppm] for i in ppm])
            for ppm in clevel.parent_map
            ])
    else:
        raise ValueError(f"unsupported ndarray dimension: '{ndim}'")

# }}}


# {{{ uncluster

def uncluster(ary: np.ndarray, index: IndexList, clevel: ClusterLevel) -> np.ndarray:
    """Performs the reverse of :func:`cluster` on object arrays.

    :arg ary: an object :class:`~numpy.ndarray` with a shape that matches
        :attr:`ClusterLevel.parent_map`.
    :arg index: an :class:`~pytential.linalg.utils.IndexList` for the
        current level, as given by :attr:`ClusterLevel.box_ids`.
    :returns: an object :class:`~numpy.ndarray` with a shape that matches
        :attr:`ClusterLevel.box_ids` of all the elements of *ary* that belong
        to each child cluster.
    """
    assert ary.dtype.char == "O"
    assert ary.shape == (clevel.parent_map.size,)

    if index.nclusters == 1:
        return ary

    result: np.ndarray = np.empty(index.nclusters, dtype=object)
    for ifrom, ppm in enumerate(clevel.parent_map):
        offset = 0
        for ito in ppm:
            cluster_size = index.cluster_size(ito)
            result[ito] = ary[ifrom][offset:offset + cluster_size]
            offset += cluster_size

        assert ary[ifrom].shape == (offset,)

    return result

# }}}


# {{{ cluster generation

def _build_binary_ish_tree_from_starts(starts: np.ndarray) -> ClusterTree:
    partition_box_ids = np.arange(starts.size - 1)

    box_ids = partition_box_ids

    box_parent_ids = []
    offset = box_ids.size
    while box_ids.size > 1:
        # NOTE: this is probably not the most efficient way to do it, but this
        # code is mostly meant for debugging using a simple tree
        clusters = np.array_split(box_ids, box_ids.size // 2)
        parent_ids = offset + np.arange(len(clusters))
        box_parent_ids.append(np.repeat(parent_ids, [len(c) for c in clusters]))

        box_ids = parent_ids
        offset += box_ids.size

    # NOTE: make the root point to itself
    box_parent_ids.append(np.array([offset - 1]))
    nlevels = len(box_parent_ids)

    return ClusterTree(
        nlevels=nlevels,
        leaf_cluster_box_ids=partition_box_ids,
        tree_cluster_parent_ids=np.concatenate(box_parent_ids),
        _tree=None)


def partition_by_nodes(
        actx: PyOpenCLArrayContext, places: GeometryCollection, *,
        dofdesc: Optional[sym.DOFDescriptorLike] = None,
        tree_kind: Optional[str] = "adaptive-level-restricted",
        max_particles_in_box: Optional[int] = None) -> Tuple[IndexList, ClusterTree]:
    """Generate equally sized ranges of nodes. The partition is created at the
    lowest level of granularity, i.e. nodes. This results in balanced ranges
    of points, but will split elements across different ranges.

    :arg dofdesc: a :class:`~pytential.symbolic.dof_desc.DOFDescriptor` for
        the geometry in *places* which should be partitioned.
    :arg tree_kind: if not *None*, it is passed to :class:`boxtree.TreeBuilder`.
    :arg max_particles_in_box: value used to control the number of points
        in each partition (and thus the number of partitions). See the documentation
        in :class:`boxtree.TreeBuilder`.
    """
    if dofdesc is None:
        dofdesc = places.auto_source
    dofdesc = sym.as_dofdesc(dofdesc)

    if max_particles_in_box is None:
        max_particles_in_box = _DEFAULT_MAX_PARTICLES_IN_BOX

    lpot_source = places.get_geometry(dofdesc.geometry)
    assert isinstance(lpot_source, (Discretization, QBXLayerPotentialSource))

    discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    assert isinstance(discr, Discretization)

    if tree_kind is not None:
        setup_actx = lpot_source._setup_actx
        assert isinstance(setup_actx, PyOpenCLArrayContext)

        from pytential.qbx.utils import tree_code_container
        tcc = tree_code_container(setup_actx)

        from arraycontext import flatten
        from meshmode.dof_array import DOFArray
        tree, _ = tcc.build_tree()(actx.queue,
                particles=flatten(
                    actx.thaw(discr.nodes()), actx, leaf_class=DOFArray
                    ),
                max_particles_in_box=max_particles_in_box,
                kind=tree_kind)

        from boxtree import box_flags_enum
        tree = tree.get(actx.queue)
        # FIXME maybe this should use IS_LEAF once available?
        leaf_boxes, = (
                tree.box_flags & box_flags_enum.HAS_SOURCE_OR_TARGET_CHILD_BOXES == 0
                ).nonzero()

        # FIXME: this annotation is not needed with numpy 2.0
        indices: np.ndarray = np.empty(len(leaf_boxes), dtype=object)
        starts = None

        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]
            indices[i] = tree.user_source_ids[box_start:box_end]

        ctree = ClusterTree(
            nlevels=tree.nlevels,
            leaf_cluster_box_ids=leaf_boxes,
            tree_cluster_parent_ids=tree.box_parent_ids,
            _tree=tree)
    else:
        if discr.ambient_dim != 2 and discr.dim == 1:
            raise ValueError("only curves are supported for 'tree_kind=None'")

        nclusters = max(discr.ndofs // max_particles_in_box, 2)
        indices = np.arange(0, discr.ndofs, dtype=np.int64)
        starts = np.linspace(0, discr.ndofs, nclusters + 1, dtype=np.int64)

        # FIXME: mypy seems to be able to figure this out with numpy 2.0
        assert starts is not None
        assert starts[-1] == discr.ndofs

        ctree = _build_binary_ish_tree_from_starts(starts)

    from pytential.linalg import make_index_list
    return make_index_list(indices, starts=starts), ctree

# }}}

# vim: foldmethod=marker
