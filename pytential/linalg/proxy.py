from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018 Alexandru Fikl"

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


import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array # noqa
from pyopencl.array import to_device

from pytools.obj_array import make_obj_array
from pytools import memoize_method, memoize_in

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


__doc__ = """
Proxy Point Generation
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: partition_by_nodes

.. autofunction:: partition_by_elements

.. autofunction:: partition_from_coarse

.. autoclass:: ProxyGenerator
"""


# {{{ point index partitioning

def _element_node_range(group, ielement):
    istart = group.node_nr_base + group.nunit_nodes * ielement
    iend = group.node_nr_base + group.nunit_nodes * (ielement + 1)

    return np.arange(istart, iend)


def partition_by_nodes(queue, discr,
                       use_tree=True,
                       max_particles_in_box=30):
    """Generate clusters / ranges of nodes. The partition is created at the
    lowest level of granularity, i.e. nodes. This results in balanced ranges
    of points, but will split elements across different ranges.

    :arg queue: a :class:`pyopencl.CommandQueue`.
    :arg discr: a :class:`~meshmode.discretization.Discretization`.
    :arg use_tree: if `True`, node partitions are generated using a
        :class:`boxtree.TreeBuilder`, which leads to geometrically close
        points to belong to the same partition. If `False`, a simple linear
        partition is constructed.
    :arg max_particles_in_box: passed to :class:`boxtree.TreeBuilder`.

    :return: a tuple `(indices, ranges)` of :class:`pyopencl.array.Array`
        integer arrays. The indices in a range can be retrieved using
        `indices[ranges[i]:ranges[i + 1]]`.
    """

    if use_tree:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        builder = TreeBuilder(discr.cl_context)

        tree, _ = builder(queue, discr.nodes(),
            max_particles_in_box=max_particles_in_box)

        tree = tree.get(queue)
        leaf_boxes, = (tree.box_flags &
                       box_flags_enum.HAS_CHILDREN == 0).nonzero()

        indices = np.empty(len(leaf_boxes), dtype=np.object)
        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]
            indices[i] = tree.user_source_ids[box_start:box_end]

        ranges = to_device(queue,
            np.cumsum([0] + [box.shape[0] for box in indices]))
        indices = to_device(queue, np.hstack(indices))
    else:
        indices = cl.array.arange(queue, 0, discr.nnodes,
                                  dtype=np.int)
        ranges = cl.array.arange(queue, 0, discr.nnodes + 1,
                                 discr.nnodes // max_particles_in_box,
                                 dtype=np.int)

    assert ranges[-1] == discr.nnodes
    return indices, ranges


def partition_by_elements(queue, discr,
                          use_tree=True,
                          max_particles_in_box=10):
    """Generate clusters / ranges of points. The partition is created at the
    element level, so that all the nodes belonging to an element belong to
    the same range. This can result in slightly larger differences in size
    between the ranges, but can be very useful when the individual partitions
    need to be resampled, integrated, etc.

    :arg queue: a :class:`pyopencl.CommandQueue`.
    :arg discr: a :class:`~meshmode.discretization.Discretization`.
    :arg use_tree: if True, node partitions are generated using a
        :class:`boxtree.TreeBuilder`, which leads to geometrically close
        points to belong to the same partition. If False, a simple linear
        partition is constructed.
    :arg max_particles_in_box: passed to :class:`boxtree.TreeBuilder`.

    :return: a tuple `(indices, ranges)` of :class:`pyopencl.array.Array`
        integer arrays. The indices in a range can be retrieved using
        `indices[ranges[i]:ranges[i + 1]]`.
    """
    if use_tree:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        builder = TreeBuilder(discr.cl_context)

        from pytential.qbx.utils import element_centers_of_mass
        elranges = np.cumsum([group.nelements for group in discr.mesh.groups])
        elcenters = element_centers_of_mass(discr)

        tree, _ = builder(queue, elcenters,
            max_particles_in_box=max_particles_in_box)

        groups = discr.groups
        tree = tree.get(queue)
        leaf_boxes, = (tree.box_flags &
                       box_flags_enum.HAS_CHILDREN == 0).nonzero()

        indices = np.empty(len(leaf_boxes), dtype=np.object)
        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]

            ielement = tree.user_source_ids[box_start:box_end]
            igroup = np.digitize(ielement, elranges)

            indices[i] = np.hstack([_element_node_range(groups[j], k)
                                    for j, k in zip(igroup, ielement)])
    else:
        nelements = discr.mesh.nelements
        elements = np.array_split(np.arange(0, nelements),
                                  nelements // max_particles_in_box)

        elranges = np.cumsum([g.nelements for g in discr.groups])
        elgroups = [np.digitize(elements[i], elranges)
                    for i in range(len(elements))]

        indices = np.empty(len(elements), dtype=np.object)
        for i in range(indices.shape[0]):
            indices[i] = np.hstack([_element_node_range(discr.groups[j], k)
                                    for j, k in zip(elgroups[i], elements[i])])

    ranges = to_device(queue,
            np.cumsum([0] + [box.shape[0] for box in indices]))
    indices = to_device(queue, np.hstack(indices))

    assert ranges[-1] == discr.nnodes
    return indices, ranges


def partition_from_coarse(queue, resampler, from_indices, from_ranges):
    """Generate a partition of nodes from an existing partition on a
    coarser discretization. The new partition is generated based on element
    refinement relationships in :attr:`resampler`, so the existing partition
    needs to be created using :func:`partition_by_element`, since we assume
    that each range contains all the nodes in an element.

    The new partition will have the same number of ranges as the old partition.
    The nodes inside each range in the new partition are all the nodes in
    :attr:`resampler.to_discr` that belong to the same region as the nodes
    in the same range from :attr:`resampler.from_discr`. These nodes are
    obtained using :attr:`mesmode.discretization.connection.InterpolationBatch`.

    :arg queue: a :class:`pyopencl.CommandQueue`.
    :arg resampler: a
        :class:`~meshmode.discretization.connection.DirectDiscretizationConnection`.
    :arg from_indices: a set of indices into the nodes in
        :attr:`resampler.from_discr`.
    :arg from_ranges: array used to index into :attr:`from_indices`.

    :return: a tuple `(indices, ranges)` of :class:`pyopencl.array.Array`
        integer arrays. The indices in a range can be retrieved using
        `indices[ranges[i]:ranges[i + 1]]`.
    """

    if not hasattr(resampler, "groups"):
        raise ValueError("resampler must be a DirectDiscretizationConnection.")

    if isinstance(from_ranges, cl.array.Array):
        from_indices = from_indices.get(queue)
        from_ranges = from_ranges.get(queue)

    # construct ranges
    from_discr = resampler.from_discr
    from_grp_ranges = np.cumsum([0] +
            [grp.nelements for grp in from_discr.mesh.groups])
    from_el_ranges = np.hstack([
        np.arange(grp.node_nr_base, grp.nnodes + 1, grp.nunit_nodes)
        for grp in from_discr.groups])

    # construct coarse element arrays in each from_range
    el_indices = np.empty(from_ranges.shape[0] - 1, dtype=np.object)
    el_ranges = np.full(from_grp_ranges[-1], -1, dtype=np.int)
    for i in range(from_ranges.shape[0] - 1):
        irange = np.s_[from_ranges[i]:from_ranges[i + 1]]
        el_indices[i] = \
            np.unique(np.digitize(from_indices[irange], from_el_ranges)) - 1
        el_ranges[el_indices[i]] = i
    el_indices = np.hstack(el_indices)

    # construct lookup table
    to_el_table = [np.full(g.nelements, -1, dtype=np.int)
                   for g in resampler.to_discr.groups]

    for igrp, grp in enumerate(resampler.groups):
        for batch in grp.batches:
            to_el_table[igrp][batch.to_element_indices.get(queue)] = \
                from_grp_ranges[igrp] + batch.from_element_indices.get(queue)

    # construct fine node index list
    indices = [np.empty(0, dtype=np.int)
               for _ in range(from_ranges.shape[0] - 1)]
    for igrp in range(len(resampler.groups)):
        to_element_indices = \
                np.where(np.isin(to_el_table[igrp], el_indices))[0]

        for i, j in zip(el_ranges[to_el_table[igrp][to_element_indices]],
                        to_element_indices):
            indices[i] = np.hstack([indices[i],
                _element_node_range(resampler.to_discr.groups[igrp], j)])

    ranges = to_device(queue,
            np.cumsum([0] + [box.shape[0] for box in indices]))
    indices = to_device(queue, np.hstack(indices))

    return indices, ranges

# }}}


# {{{ proxy point generator

class ProxyGenerator(object):
    def __init__(self, queue, source, ratio=1.5, nproxy=31):
        r"""
        :arg queue: a :class:`pyopencl.CommandQueue`.
        :arg source: a :class:`pytential.qbx.LayerPotentialSourceBase`.
        :arg ratio: a ratio used to compute the proxy point radius. The radius
            is computed in the :math:`L_2` norm, resulting in a circle or
            sphere of proxy points. For QBX, we have two radii of interest
            for a set of points: the radius :math:`r_{block}` of the
            smallest ball containing all the points and the radius
            :math:`r_{qbx}` of the smallest ball containing all the QBX
            expansion balls in the block. If the ratio :math:`\theta \in
            [0, 1]`, then the radius of the proxy ball is

            .. math::

                r = (1 - \theta) r_{block} + \theta r_{qbx}.

            If the ratio :math:`\theta > 1`, the the radius is simply

            .. math::

                r = \theta r_{qbx}.

        :arg nproxy: number of proxy points.
        """

        self.queue = queue
        self.source = source
        self.context = self.queue.context
        self.ratio = abs(ratio)
        self.nproxy = int(abs(nproxy))
        self.dim = self.source.ambient_dim

        if self.dim == 2:
            from meshmode.mesh.generation import ellipse, make_curve_mesh

            self.mesh = make_curve_mesh(lambda t: ellipse(1.0, t),
                                   np.linspace(0.0, 1.0, self.nproxy + 1),
                                   self.nproxy)
        elif self.dim == 3:
            from meshmode.mesh.generation import generate_icosphere

            self.mesh = generate_icosphere(1, self.nproxy)
        else:
            raise ValueError("unsupported ambient dimension")

    @memoize_method
    def get_kernel(self):
        if self.ratio < 1.0:
            radius_expr = "(1.0 - ratio) * rblk + ratio * rqbx"
        else:
            radius_expr = "ratio * rqbx"

        # NOTE: centers of mass are computed using a second-order approximation
        # that currently matches what's in `element_centers_of_mass`.
        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[i]: 0 <= i < npoints}",
            "{[idim]: 0 <= idim < dim}"
            ],
            ["""
            for irange
                <> npoints = ranges[irange + 1] - ranges[irange]
                proxy_center[idim, irange] = 1.0 / npoints * \
                    reduce(sum, i, nodes[idim, indices[i + ranges[irange]]]) \
                        {{dup=idim:i}}

                <> rblk = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         nodes[idim, indices[i + ranges[irange]]]) ** 2)))

                <> rqbx_int = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_int[idim, indices[i + ranges[irange]]]) ** 2)) + \
                         expansion_radii[indices[i + ranges[irange]]])
                <> rqbx_ext = simul_reduce(max, i, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_ext[idim, indices[i + ranges[irange]]]) ** 2)) + \
                         expansion_radii[indices[i + ranges[irange]]])
                <> rqbx = if(rqbx_ext < rqbx_int, rqbx_int, rqbx_ext)

                proxy_radius[irange] = {radius_expr}
            end
            """.format(radius_expr=radius_expr)],
            [
                lp.GlobalArg("nodes", None,
                    shape=(self.dim, "nnodes")),
                lp.GlobalArg("center_int", None,
                    shape=(self.dim, "nnodes"), dim_tags="sep,C"),
                lp.GlobalArg("center_ext", None,
                    shape=(self.dim, "nnodes"), dim_tags="sep,C"),
                lp.GlobalArg("expansion_radii", None,
                    shape="nnodes"),
                lp.GlobalArg("ranges", None,
                    shape="nranges + 1"),
                lp.GlobalArg("indices", None,
                    shape="nindices"),
                lp.GlobalArg("proxy_center", None,
                    shape=(self.dim, "nranges")),
                lp.GlobalArg("proxy_radius", None,
                    shape="nranges"),
                lp.ValueArg("nnodes", np.int64),
                lp.ValueArg("nranges", None),
                lp.ValueArg("nindices", np.int64)
            ],
            name="proxy_generator_knl",
            assumptions="dim>=1 and nranges>=1",
            fixed_parameters=dict(
                dim=self.dim,
                ratio=self.ratio),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.tag_inames(knl, "idim*:unr")

        return knl

    @memoize_method
    def get_optimized_kernel(self):
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "irange", 128, outer_tag="g.0")

        return knl

    def get_proxies(self, indices, ranges, **kwargs):
        """Generate proxy points for each given range of source points in
        the discretization :attr:`source`.

        :arg indices: a :class:`pyopencl.array.Array` of indices around
            which to construct proxy balls.
        :arg ranges: an :class:`pyopencl.array.Array` of size `(nranges + 1,)`
            used to index into :attr:`indices`. Each one of the `nranges`
            ranges will get a proxy ball.

        :return: a tuple of `(centers, radii, proxies, pryranges)`, where
            each value is a :class:`pyopencl.array.Array`. The
            sizes of the arrays are as follows: `centers` is of size
            `(2, nranges)`, `radii` is of size `(nranges,)`, `pryranges` is
            of size `(nranges + 1,)` and `proxies` is of size
            `(2, nranges * nproxies)`. The proxy points in a range :math:`i`
            can be obtained by a slice `proxies[pryranges[i]:pryranges[i + 1]]`
            and are all at a distance `radii[i]` from the range center
            `centers[i]`.
        """

        from pytential.qbx.utils import get_centers_on_side

        knl = self.get_kernel()
        _, (centers_dev, radii_dev,) = knl(self.queue,
            nodes=self.source.density_discr.nodes(),
            center_int=get_centers_on_side(self.source, -1),
            center_ext=get_centers_on_side(self.source, +1),
            expansion_radii=self.source._expansion_radii("nsources"),
            indices=indices, ranges=ranges, **kwargs)
        centers = centers_dev.get()
        radii = radii_dev.get()

        from meshmode.mesh.processing import affine_map
        proxies = np.empty(ranges.shape[0] - 1, dtype=np.object)

        for i in range(ranges.shape[0] - 1):
            mesh = affine_map(self.mesh,
                A=(radii[i] * np.eye(self.dim)),
                b=centers[:, i].reshape(-1))
            proxies[i] = mesh.vertices

        pxyranges = np.cumsum([0] + [p.shape[-1] for p in proxies])
        pxyranges = cl.array.to_device(self.queue, pxyranges)
        proxies = make_obj_array([
            cl.array.to_device(self.queue, np.hstack([p[idim] for p in proxies]))
            for idim in range(self.dim)])
        centers_dev = make_obj_array([
            centers_dev[idim].with_queue(self.queue).copy()
            for idim in range(self.dim)])

        return centers_dev, radii_dev, proxies, pxyranges

    def get_neighbors(self, indices, ranges, centers, radii):
        """Generate a set of neighboring points for each range of source
        points in :attr:`source`. Neighboring points are defined as all
        the points inside a proxy ball :math:`i` that do not also belong to
        the set of source points in the same range :math:`i`.

        :arg indices: a :class:`pyopencl.array.Array` of indices for a subset
            of the source points.
        :arg ranges: a :class:`pyopencl.array.Array` used to index into
            the :attr:`indices` array.
        :arg centers: a :class:`pyopencl.array.Array` containing the center
            of each proxy ball.
        :arg radii: a :class:`pyopencl.array.Array` containing the radius
            of each proxy ball.

        :return: a tuple `(neighbours, nbrranges)` where each value is a
            :class:`pyopencl.array.Array` of integers. For a range :math:`i`
            in `nbrranges`, the corresponding slice of the `neighbours` array
            is a subset of :attr:`indices` such that all points are inside
            the proxy ball centered at `centers[i]` of radius `radii[i]`
            that is not included in `indices[ranges[i]:ranges[i + 1]]`.
        """

        if isinstance(indices, cl.array.Array):
            indices = indices.get(self.queue)
            ranges = ranges.get(self.queue)

        nranges = radii.shape[0] + 1
        sources = self.source.density_discr.nodes().get(self.queue)
        sources = make_obj_array([
            cl.array.to_device(self.queue, sources[idim, indices])
            for idim in range(self.dim)])

        # construct tree
        from boxtree import TreeBuilder
        builder = TreeBuilder(self.context)
        tree, _ = builder(self.queue, sources, max_particles_in_box=30)

        from boxtree.area_query import AreaQueryBuilder
        builder = AreaQueryBuilder(self.context)
        query, _ = builder(self.queue, tree, centers, radii)

        # find nodes inside each proxy ball
        tree = tree.get(self.queue)
        query = query.get(self.queue)
        centers_ = np.vstack([centers[idim].get(self.queue)
                              for idim in range(self.dim)])
        radii_ = radii.get(self.queue)

        neighbors = np.empty(nranges - 1, dtype=np.object)
        for iproxy in range(nranges - 1):
            # get list of boxes intersecting the current ball
            istart = query.leaves_near_ball_starts[iproxy]
            iend = query.leaves_near_ball_starts[iproxy + 1]
            iboxes = query.leaves_near_ball_lists[istart:iend]

            # get nodes inside the boxes
            istart = tree.box_source_starts[iboxes]
            iend = istart + tree.box_source_counts_cumul[iboxes]
            isources = np.hstack([np.arange(s, e)
                                  for s, e in zip(istart, iend)])
            nodes = np.vstack([tree.sources[idim][isources]
                               for idim in range(self.dim)])
            isources = tree.user_source_ids[isources]

            # get nodes inside the ball but outside the current range
            center = centers_[:, iproxy].reshape(-1, 1)
            radius = radii_[iproxy]
            mask = (la.norm(nodes - center, axis=0) < radius) & \
                   ((isources < ranges[iproxy]) | (ranges[iproxy + 1] <= isources))

            neighbors[iproxy] = indices[isources[mask]]

        nbrranges = np.cumsum([0] + [n.shape[0] for n in neighbors])
        neighbors = np.hstack(neighbors)

        return (to_device(self.queue, neighbors),
                to_device(self.queue, nbrranges))

    def __call__(self, indices, ranges, **kwargs):
        """
        :arg indices: a :class:`pyopencl.array.Array` of indices for points
            in :attr:`source`.
        :arg ranges: a :class:`pyopencl.array.Array` describing the ranges
            from :attr:`indices` around which to build proxy points. For each
            range, this builds a ball of proxy points centered
            at the center of mass of the points in the range with a radius
            defined by :attr:`ratio`.

        :returns: a tuple `(skeletons, sklranges)` where each value is a
            :class:`pyopencl.array.Array`. For a range :math:`i`, we can
            get the slice using `skeletons[sklranges[i]:sklranges[i + 1]]`.
            The skeleton points in a range represents the union of a set
            of generated proxy points and all the source points inside the
            proxy ball that do not also belong to the current range in
            :attr:`indices`.
        """

        @memoize_in(self, "concat_skl_knl")
        def knl():
            loopy_knl = lp.make_kernel([
                "{[irange, idim]: 0 <= irange < nranges and \
                                  0 <= idim < dim}",
                "{[ipxy, ingb]: 0 <= ipxy < npxyblock and \
                                0 <= ingb < nngbblock}"
                ],
                """
                for irange
                    <> pxystart = pxyranges[irange]
                    <> pxyend = pxyranges[irange + 1]
                    <> npxyblock = pxyend - pxystart

                    <> ngbstart = nbrranges[irange]
                    <> ngbend = nbrranges[irange + 1]
                    <> nngbblock = ngbend - ngbstart

                    <> sklstart = pxyranges[irange] + nbrranges[irange]
                    skeletons[idim, sklstart + ipxy] = \
                        proxies[idim, pxystart + ipxy] \
                        {id_prefix=write_pxy,nosync=write_ngb}
                    skeletons[idim, sklstart + npxyblock + ingb] = \
                        sources[idim, neighbors[ngbstart + ingb]] \
                        {id_prefix=write_ngb,nosync=write_pxy}
                    sklranges[irange + 1] = sklranges[irange] + \
                            npxyblock + nngbblock
                end
                """,
                [
                    lp.GlobalArg("sources", None,
                        shape=(self.dim, "nsources")),
                    lp.GlobalArg("proxies", None,
                        shape=(self.dim, "nproxies"), dim_tags="sep,C"),
                    lp.GlobalArg("neighbors", None,
                        shape="nneighbors"),
                    lp.GlobalArg("pxyranges", None,
                        shape="nranges + 1"),
                    lp.GlobalArg("nbrranges", None,
                        shape="nranges + 1"),
                    lp.GlobalArg("skeletons", None,
                        shape=(self.dim, "nproxies + nneighbors")),
                    lp.GlobalArg("sklranges", None,
                        shape="nranges + 1"),
                    lp.ValueArg("nsources", np.int32),
                    lp.ValueArg("nproxies", np.int32),
                    lp.ValueArg("nneighbors", np.int32),
                    "..."
                ],
                name="concat_skl",
                default_offset=lp.auto,
                silenced_warnings="write_race(write_*)",
                fixed_parameters=dict(dim=self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

            loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
            loopy_knl = lp.split_iname(loopy_knl, "irange", 128, outer_tag="g.0")

            return loopy_knl

        # construct point arrays
        sources = self.source.density_discr.nodes()
        centers, radii, proxies, pxyranges = \
            self.get_proxies(indices, ranges, **kwargs)
        neighbors, nbrranges = \
            self.get_neighbors(indices, ranges, centers, radii)

        # construct joint array
        sklranges = cl.array.zeros(self.queue, ranges.shape, dtype=np.int)
        _, (skeletons, sklranges) = knl()(self.queue,
                sources=sources, proxies=proxies, neighbors=neighbors,
                pxyranges=pxyranges, nbrranges=nbrranges,
                sklranges=sklranges)

        return skeletons, sklranges

# }}}

# vim: foldmethod=marker
