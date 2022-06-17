from __future__ import annotations


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

import logging
import pathlib
from dataclasses import dataclass, replace
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import numpy as np

from arraycontext import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from pytools import log_process, memoize_method, obj_array

from pytential import GeometryCollection, sym
from pytential.linalg.utils import IndexList, TargetAndSourceClusterList
from pytential.qbx import QBXLayerPotentialSource


if TYPE_CHECKING:
    from collections.abc import Iterator

    import optype.numpy as onp

    from boxtree.tree import Tree
    from boxtree.tree_build import TreeKind

    from pytential.linalg.proxy import ProxyGenerator


logger = logging.getLogger(__name__)

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


def make_cluster_parent_map(
        parent_ids: onp.Array1D[np.integer],
    ) -> obj_array.ObjectArray1D[onp.Array1D[np.integer]]:
    """Construct a parent map for :attr:`ClusterLevel.parent_map`."""
    # NOTE: np.unique returns a sorted array
    unique_parent_ids = np.unique(parent_ids)
    ids = np.arange(parent_ids.size)

    return obj_array.new_1d([
        ids[parent_ids == unique_parent_ids[i]]
        for i in range(unique_parent_ids.size)
        ])


@dataclass(frozen=True)
class ClusterLevel:
    """A level in a :class:`ClusterTree`.

    .. autoattribute:: level
    .. autoattribute:: box_ids
    .. autoattribute:: parent_map
    .. autoproperty:: nclusters
    """

    level: int
    """Current level that is represented."""
    box_ids: onp.Array1D[np.integer]
    """Box IDs on the current level."""
    parent_map: obj_array.ObjectArray1D[onp.Array1D[np.integer]]
    """An object :class:`~numpy.ndarray` containing buckets of child indices,
    i.e. ``parent_map[i]`` contains all the child indices that will cluster
    into the same parent. Note that this indexing is local to this level
    and is not related to the tree indexing stored by the :class:`ClusterTree`.
    """

    @property
    def nclusters(self) -> int:
        """Number of clusters on the current level (same as number of boxes
        in :attr:`box_ids`).
        """
        return self.box_ids.size


@dataclass(frozen=True)
class ClusterTree:
    r"""Hierarchical cluster representation.

    .. autoattribute:: nlevels
    .. autoattribute:: leaf_cluster_box_ids
    .. autoattribute:: tree_cluster_parent_ids

    .. autoproperty:: nclusters
    .. autoproperty:: levels
    .. automethod:: iter_levels
    """

    nlevels: int
    """Total number of levels in the tree."""
    leaf_cluster_box_ids: onp.Array1D[np.integer]
    """Box IDs for each cluster on the leaf level of the tree."""
    tree_cluster_parent_ids: onp.Array1D[np.integer]
    """Parent box IDs for :attr:`leaf_cluster_box_ids`."""

    # NOTE: only here to allow easier debugging + testing
    _tree: Tree | None

    @property
    def nclusters(self) -> int:
        """Number of clusters in the leaf level of the tree."""
        return self.leaf_cluster_box_ids.size

    @property
    @memoize_method
    def levels(self) -> obj_array.ObjectArray1D[ClusterLevel]:
        r"""An :class:`~numpy.ndarray` of :class:`ClusterLevel`\ s."""
        return obj_array.new_1d(list(self.iter_levels()))

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

def split_array(x: onp.Array1D[Any],
                index: IndexList) -> obj_array.ObjectArray1D[onp.Array1D[Any]]:
    """
    :returns: an object :class:`~numpy.ndarray` where each entry contains the
        elements of the :math:`i`-th cluster in *index*.
    """
    assert x.size == index.nindices

    return obj_array.new_1d([
        index.cluster_take(x, i) for i in range(index.nclusters)
        ])


@singledispatch
def cluster(obj: object, clevel: ClusterLevel) -> Any:
    """Merge together elements of *obj* into their parent object, as described
    by :attr:`ClusterLevel.parent_map`.
    """
    raise NotImplementedError(type(obj).__name__)


@cluster.register(IndexList)
def cluster_index_list(obj: IndexList, clevel: ClusterLevel) -> IndexList:
    assert obj.nclusters == clevel.nclusters

    if clevel.nclusters == 1:
        return obj

    from pytential.linalg.utils import make_index_list
    indices = obj_array.new_1d([
        np.concatenate([obj.cluster_indices(i) for i in ppm])
        for ppm in clevel.parent_map
        ])

    return make_index_list(indices)


@cluster.register(TargetAndSourceClusterList)
def cluster_target_and_source_cluster_list(
        obj: TargetAndSourceClusterList, clevel: ClusterLevel,
        ) -> TargetAndSourceClusterList:
    assert obj.nclusters == clevel.nclusters

    if clevel.nclusters == 1:
        return obj

    return replace(obj,
        targets=cluster(obj.targets, clevel),
        sources=cluster(obj.sources, clevel))


@cluster.register(np.ndarray)
def cluster_ndarray(obj: obj_array.ObjectArray1D[onp.ArrayND[Any]],
                    clevel: ClusterLevel) -> obj_array.ObjectArray1D[onp.ArrayND[Any]]:
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
        return obj_array.new_1d([
            np.concatenate([obj[i] for i in ppm]) for ppm in clevel.parent_map
            ])
    elif ndim == 2:
        return obj_array.new_1d([
            np.block([[make_block(i, j) for j in ppm] for i in ppm])
            for ppm in clevel.parent_map
            ])
    else:
        raise ValueError(f"unsupported ndarray dimension: '{ndim}'")

# }}}


# {{{ uncluster

def uncluster(ary: obj_array.ObjectArray1D[onp.Array1D[Any]],
              index: IndexList,
              clevel: ClusterLevel) -> obj_array.ObjectArray1D[onp.Array1D[Any]]:
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

def _build_binary_ish_tree_from_starts(starts: onp.Array1D[np.integer]) -> ClusterTree:
    partition_box_ids = np.arange(starts.size - 1)
    box_ids = partition_box_ids

    box_parent_ids: list[onp.Array1D[np.integer]] = []
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


@log_process(logger)
def partition_by_nodes(
        actx: PyOpenCLArrayContext, places: GeometryCollection, *,
        dofdesc: sym.DOFDescriptorLike | None = None,
        tree_kind: TreeKind | None = "adaptive-level-restricted",
        max_particles_in_box: int | None = None) -> tuple[IndexList, ClusterTree]:
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
    assert isinstance(lpot_source, Discretization | QBXLayerPotentialSource)

    discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    assert isinstance(discr, Discretization)

    if tree_kind is not None:
        setup_actx = lpot_source._setup_actx
        assert isinstance(setup_actx, PyOpenCLArrayContext)

        from pytential.qbx.utils import tree_code_container
        tcc = tree_code_container(setup_actx)

        from arraycontext import flatten
        from meshmode.dof_array import DOFArray
        tree, _ = tcc.build_tree()(actx,
                particles=flatten(
                    actx.thaw(discr.nodes()), actx, leaf_class=DOFArray
                    ),
                max_particles_in_box=max_particles_in_box,
                kind=tree_kind)
        tree = actx.to_numpy(tree)

        # FIXME maybe this should use IS_LEAF once available?
        from boxtree import box_flags_enum
        assert tree.box_flags is not None
        leaf_boxes, = (
                tree.box_flags & box_flags_enum.HAS_SOURCE_OR_TARGET_CHILD_BOXES == 0
                ).nonzero()

        # FIXME: this annotation is not needed with numpy 2.0
        indices = np.empty(len(leaf_boxes), dtype=object)
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


# {{{ visualize clusters

def visualize_clusters(actx: PyOpenCLArrayContext,
                       generator: ProxyGenerator,
                       srcindex: IndexList,
                       tree: ClusterTree,
                       filename: str | pathlib.Path, *,
                       dofdesc: sym.DOFDescriptorLike = None,
                       overwrite: bool = False) -> None:
    filename = pathlib.Path(filename)

    places = generator.places
    if dofdesc is None:
        dofdesc = places.auto_source
    dofdesc = sym.as_dofdesc(dofdesc)

    discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    assert isinstance(discr, Discretization)

    if discr.ambient_dim == 2:
        _visualize_clusters_2d(actx, generator, discr, srcindex, tree, filename,
                               dofdesc=dofdesc, overwrite=overwrite)
    elif discr.ambient_dim == 2:
        _visualize_clusters_3d(actx, generator, discr, srcindex, tree, filename,
                               dofdesc=dofdesc, overwrite=overwrite)
    else:
        raise NotImplementedError(f"Unsupported dimension: {discr.ambient_dim}")


def _visualize_clusters_2d(actx: PyOpenCLArrayContext,
                           generator: ProxyGenerator,
                           discr: Discretization,
                           srcindex: IndexList,
                           tree: ClusterTree,
                           filename: pathlib.Path, *,
                           dofdesc: sym.DOFDescriptor,
                           overwrite: bool = False) -> None:
    import matplotlib.pyplot as pt

    from arraycontext import flatten
    from boxtree.visualization import TreePlotter
    from meshmode.dof_array import DOFArray

    assert discr.ambient_dim == 2
    x, y = actx.to_numpy(flatten(discr.nodes(), actx, leaf_class=DOFArray))
    for clevel in tree.levels:
        outfile = filename.with_stem(f"{filename.stem}-{clevel.level:03d}")
        if not overwrite and outfile.exists():
            raise FileExistsError(f"Output file '{outfile}' already exists")

        pxy = generator(actx, dofdesc, srcindex).to_numpy(actx)
        pxycenters = pxy.centers
        pxyradii = pxy.radii
        clsradii = pxy.cluster_radii

        fig = pt.figure()
        ax = fig.gca()

        plotter = TreePlotter(tree._tree)
        plotter.set_bounding_box()
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)

        ax.plot(x, y, "ko", ms=2.0)
        for i in range(srcindex.nclusters):
            isrc = srcindex.cluster_indices(i)
            ax.plot(x[isrc], y[isrc], "o", ms=2.0)

        from itertools import cycle
        colors = cycle(pt.rcParams["axes.prop_cycle"].by_key()["color"])

        for ppm in clevel.parent_map:
            color = next(colors)
            for j in ppm:
                center = (pxycenters[0, j], pxycenters[1, j])
                c = pt.Circle(center, pxyradii[j], color=color, alpha=0.1)
                ax.add_artist(c)
                c = pt.Circle(center, clsradii[j], color=color, alpha=0.1)
                ax.add_artist(c)
                ax.text(*center, f"{j}", fontsize=18)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.relim()
        ax.autoscale()
        ax.set_aspect("equal")

        fig.savefig(outfile)
        pt.close(fig)

        srcindex = cluster(srcindex, clevel)


def _visualize_clusters_3d(actx: PyOpenCLArrayContext,
                           generator: ProxyGenerator,
                           discr: Discretization,
                           srcindex: IndexList,
                           tree: ClusterTree,
                           filename: pathlib.Path, *,
                           dofdesc: sym.DOFDescriptor,
                           overwrite: bool = False) -> None:
    from arraycontext import unflatten
    from meshmode.discretization.visualization import make_visualizer

    # NOTE: This writes out one vtu file for each level that contains
    #   * a mesh that's the union of `discr` and a sphere for each proxy ball
    #   * marker: a marker on `discr` (NaN on the proxy balls) for each of the
    #     clusters at the current level
    #   * proxies: a marker on the proxy balls (NaN on `discr`)
    #
    # Not quite sure how to best visualize the whole geometry here, so the
    # proposed workflow is to load the vtu file twice, set opacity to 0 for
    # NaNs and set opacity to something small for the proxy balls.

    # TODO:
    #   * color proxy balls based on their parent so we can easily see how they
    #    will cluster

    assert discr.ambient_dim == 3
    for clevel in tree.levels:
        outfile = filename.with_stem(f"{filename.stem}-lvl{clevel.level:03d}")
        outfile = outfile.with_suffix(".vtu")
        if not overwrite and outfile.exists():
            raise FileExistsError(f"Output file '{outfile}' already exists")

        # construct proxy balls
        pxy = generator(actx, dofdesc, srcindex).to_numpy(actx)
        pxycenters = pxy.centers
        pxyradii = pxy.radii
        nclusters = srcindex.nclusters

        # construct meshes for each proxy ball
        from meshmode.mesh.generation import generate_sphere
        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes

        ref_mesh = generate_sphere(1, 4, uniform_refinement_rounds=1)
        pxymeshes = [
            affine_map(ref_mesh, A=pxyradii[i], b=pxycenters[:, i].squeeze())
            for i in range(nclusters)
        ]

        # merge meshes into a single discretization
        from meshmode.discretization.poly_element import (
            InterpolatoryEdgeClusteredGroupFactory,
        )
        pxymesh = merge_disjoint_meshes([discr.mesh, *pxymeshes])
        pxydiscr = Discretization(actx, pxymesh,
                                  InterpolatoryEdgeClusteredGroupFactory(4))

        # add a marker field for all clusters
        marker = np.full((pxydiscr.ndofs,), np.nan, dtype=np.float64)
        template_ary = actx.thaw(pxydiscr.nodes()[0])

        for i in range(srcindex.nclusters):
            isrc = srcindex.cluster_indices(i)
            marker[isrc] = 10.0 * (i + 1.0)
        marker_dev = unflatten(template_ary, actx.from_numpy(marker), actx)

        # add a marker field for all proxies
        pxymarker = np.full((pxydiscr.ndofs,), np.nan, dtype=np.float64)
        pxymarker[discr.ndofs:] = 1.0
        pxymarker_dev = unflatten(template_ary, actx.from_numpy(pxymarker), actx)

        # write it all out
        vis = make_visualizer(actx, pxydiscr)
        vis.write_vtk_file(str(outfile), [
            ("marker", marker_dev),
            ("proxies", pxymarker_dev),
            ], overwrite=overwrite)

        srcindex = cluster(srcindex, clevel)


# }}}


# vim: foldmethod=marker
