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

import pyopencl as cl
import pyopencl.array # noqa

from pytools import memoize_method, memoize_in

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


# {{{ point index partitioning

def _element_node_range(groups, igroup, ielement):
    istart = groups[igroup].node_nr_base + \
             groups[igroup].nunit_nodes * ielement
    iend = groups[igroup].node_nr_base + \
           groups[igroup].nunit_nodes * (ielement + 1)
    return np.arange(istart, iend)


def partition_by_nodes(discr, use_tree=True, max_particles_in_box=30):
    if use_tree:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        with cl.CommandQueue(discr.cl_context) as queue:
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

            ranges = np.cumsum([0] + [box.shape[0] for box in indices])
            indices = np.hstack(indices)
    else:
        indices = np.arange(0, discr.nnodes)
        ranges = np.arange(0, discr.nnodes + 1,
                           discr.nnodes // max_particles_in_box)

    assert ranges[-1] == discr.nnodes
    return indices, ranges


def partition_by_elements(discr, use_tree=True, max_particles_in_box=10):
    if use_tree:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        with cl.CommandQueue(discr.cl_context) as queue:
            builder = TreeBuilder(discr.cl_context)

            from pytential.qbx.utils import element_centers_of_mass
            elranges = np.cumsum([0] +
                [group.nelements for group in discr.mesh.groups])
            elcenters = element_centers_of_mass(discr)

            tree, _ = builder(queue, elcenters,
                max_particles_in_box=max_particles_in_box)

            groups = discr.groups
            tree = tree.get(queue)
            leaf_boxes, = (tree.box_flags & 
                           box_flags_enum.HAS_CHILDREN == 0).nonzero()

            indices = np.empty(len(leaf_boxes), dtype=np.object)
            elements = np.empty(len(leaf_boxes), dtype=np.object)
            for i, ibox in enumerate(leaf_boxes):
                box_start = tree.box_source_starts[ibox]
                box_end = box_start + tree.box_source_counts_cumul[ibox]

                ielement = tree.user_source_ids[box_start:box_end]
                igroup = np.digitize(ielement, elranges) - 1

                indices[i] = np.hstack([_element_node_range(groups, j, k)
                                        for j, k in zip(igroup, ielement)])
    else:
        groups = discr.groups
        elranges = np.cumsum([0] +
            [group.nelements for group in discr.mesh.groups])

        nelements = discr.mesh.nelements
        indices = np.array_split(np.arange(0, nelements),
                                 nelements // max_particles_in_box)
        for i in range(len(indices)):
            ielement = indices[i]
            igroup = np.digitize(ielement, elranges) - 1

            indices[i] = np.hstack([_element_node_range(groups, j, k)
                                    for j, k in zip(igroup, ielement)])

    ranges = np.cumsum([0] + [box.shape[0] for box in indices])
    indices = np.hstack(indices)

    return indices, ranges


def partition_from_coarse(queue, resampler, from_indices, from_ranges):
    if not hasattr(resampler, "groups"):
        raise ValueError("resampler must be a DirectDiscretizationConnection.")

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
                _element_node_range(resampler.to_discr.groups, igrp, j)])

    ranges = np.cumsum([0] + [box.shape[0] for box in indices])
    indices = np.hstack(indices)

    return indices, ranges

# }}}


# {{{ proxy point generator

class ProxyGenerator(object):
    def __init__(self, queue, places,
                 ratio=1.5, nproxy=31, target_order=0):
        r"""
        :arg queue: a :class:`pyopencl.CommandQueue` used to synchronize the
            calculations.
        :arg places: commonly a :class:`pytential.qbx.LayerPotentialSourceBase`.
        :arg ratio: a ratio used to compute the proxy point radius. For proxy
            points, we have two radii of interest for a set of points: the
            radius :math:`r_{block}` of the smallest ball containing all the
            points in a block and the radius :math:`r_{qbx}` of the smallest
            ball containing all the QBX expansion balls in the block. If the 
            ratio :math:`\theta \in [0, 1]`, then the radius of the proxy
            ball is computed as:

            .. math::

                r = (1 - \theta) r_{block} + \theta r_{qbx}.

            If the ratio :math:`\theta > 1`, the the radius is simply:

            .. math::

                r = \theta r_{qbx}.

        :arg nproxy: number of proxy points for each block.
        :arg target_order: order of the discretization of proxy points. Can
            be quite small, since proxy points are evaluated point-to-point
            at the moment.
        """

        self.queue = queue
        self.places = places
        self.context = self.queue.context
        self.ratio = abs(ratio)
        self.nproxy = int(abs(nproxy))
        self.target_order = target_order
        self.dim = self.places.ambient_dim

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

        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[iblk]: 0 <= iblk < nblock}",
            "{[idim]: 0 <= idim < dim}"
            ],
            ["""
            for irange
                <> blk_start = ranges[irange]
                <> blk_end = ranges[irange + 1]
                <> nblock = blk_end - blk_start

                proxy_center[idim, irange] = 1.0 / blk_length * \
                    reduce(sum, iblk, nodes[idim, indices[blk_start + iblk]]) \
                        {{dup=idim:iblk}}

                <> rblk = simul_reduce(max, iblk, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         nodes[idim, indices[blk_start + iblk]]) ** 2)))
                <> rqbx_int = simul_reduce(max, iblk, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_int[idim, indices[blk_start + iblk]]) ** 2)) + \
                         expansion_radii[indices[blk_start + iblk]])
                <> rqbx_ext = simul_reduce(max, iblk, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_ext[idim, indices[blk_start + iblk]]) ** 2)) + \
                        expansion_radii[indices[blk_start + iblk]])

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
            name="proxy_kernel",
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
        """
        :arg indices: a set of indices around which to construct proxy balls.
        :arg ranges: an array of size `nranges + 1` used to index into the
        set of indices. Each one of the `nranges` ranges will get a proxy
        ball of a predefined size.

        :returns: `centers` is a list of centers for each proxy ball.
        :returns: `radii` is a list of radii for each proxy ball.
        :returns: `proxies` is a set of proxy points.
        :returns: `pxyranges` is an array used to index into the list of
        proxy points.
        """

        from pytential.qbx.utils import get_centers_on_side

        knl = self.get_kernel()
        _, (centers_dev, radii_dev,) = knl(self.queue,
            nodes=self.places.density_discr.nodes(),
            center_int=get_centers_on_side(self.places, -1),
            center_ext=get_centers_on_side(self.places, +1),
            expansion_radii=self.places._expansion_radii("nsources"),
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
        proxies = make_obj_array([
            cl.array.to_device(self.queue, np.hstack([p[idim] for p in proxies]))
            for idim in range(self.dim)])
        centers_dev = make_obj_array([
            centers_dev[idim].with_queue(self.queue).copy()
            for idim in range(self.dim)])

        return centers_dev, radii_dev, proxies, pxyranges

    def get_neighbors(self, indices, ranges, centers, radii):
        """
        :arg indices: indices for a subset of discretization nodes.
        :arg ranges: array used to index into the :attr:`indices` array.
        :arg centers: centers for each proxy ball.
        :arg radii: radii for each proxy ball.

        :returns: `neighbors` is a set of indices into the full set of
        discretization nodes (already mapped through the given set `indices`).
        It contains all nodes that are inside a given proxy ball :math:`i`,
        but not contained in the range :math:`i`, given by
        `indices[ranges[i]:ranges[i + 1]]`.
        :returns: `nbrranges` is an array used to index into the set of
        neighbors and return the subset for a given proxy ball.
        """

        nranges = radii.shape[0] + 1
        sources = self.places.density_discr.nodes().get(self.queue)
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

        return (cl.array.to_device(self.queue, neighbors),
                cl.array.to_device(self.queue, nbrranges))

    def __call__(self, indices, ranges, **kwargs):
        """
        :arg indices: Set of indices for points in :attr:`places`.
        :arg ranges: Set of ranges around which to build a set of proxy
        points. For each range, this builds a ball of proxy points centered
        at the center of mass of the points in the range with a radius
        defined by :attr:`ratio`.

        :returns: `skeletons` A set of skeleton points for each range,
        which are supposed to contain all the necessary information about
        the farfield interactions. This set of points contains a set of
        proxy points constructed around each range and the closest neighbors
        that are inside the proxy ball.
        :returns: `sklranges` An array used to index into the skeleton
        points.
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
                    lp.ValueArg("nsources", np.int32),
                    lp.ValueArg("nproxies", np.int32),
                    lp.ValueArg("nneighbors", np.int32),
                    "..."
                ],
                name="concat_skl",
                default_offset=lp.auto,
                fixed_parameters=dict(dim=self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

            loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
            # loopy_knl = lp.split_iname(loopy_knl, "irange", 16, inner_tag="l.0")
            return loopy_knl

        # construct point arrays
        sources = self.places.density_discr.nodes()
        centers, radii, proxies, pxyranges = \
            self.get_proxies(indices, ranges, **kwargs)
        neighbors, nbrranges = \
            self.get_neighbors(indices, ranges, centers, radii)

        # construct joint array
        _, (skeletons,) = knl()(self.queue,
                sources=sources, proxies=proxies, neighbors=neighbors,
                pxyranges=pxyranges, nbrranges=nbrranges)
        sklranges = np.array([p + n for p, n in zip(pxyranges, nbrranges)])

        return skeletons, sklranges


# }}}

# vim: foldmethod=marker
