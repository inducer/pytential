from __future__ import division, absolute_import, print_function

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

import os
import time

import numpy as np
import numpy.linalg as la

import pyopencl as cl
from pyopencl.array import to_device

import loopy as lp
from pytential import sym
from meshmode.mesh.generation import ( # noqa
        ellipse, NArmedStarfish, generate_torus, make_curve_mesh)

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)


def _build_qbx_discr(queue,
        ndim=2,
        nelements=30,
        target_order=7,
        qbx_order=4,
        curve_f=None):

    if curve_f is None:
        curve_f = NArmedStarfish(5, 0.25)

    if ndim == 2:
        mesh = make_curve_mesh(curve_f,
                np.linspace(0, 1, nelements + 1),
                target_order)
    elif ndim == 3:
        mesh = generate_torus(10.0, 2.0, order=target_order)
    else:
        raise ValueError("unsupported ambient dimension")

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx, _ = QBXLayerPotentialSource(density_discr,
            fine_order=4 * target_order,
            qbx_order=qbx_order,
            fmm_order=False).with_refinement()

    return qbx


def _build_block_index(queue, discr,
        nblks=10,
        factor=1.0,
        method='elements',
        use_tree=True):

    from pytential.linalg.proxy import (
            partition_by_nodes, partition_by_elements)

    if method == 'elements':
        factor = 1.0

    if method == 'nodes':
        nnodes = discr.nnodes
    else:
        nnodes = discr.mesh.nelements
    max_particles_in_box = nnodes // nblks

    # create index ranges
    if method == "nodes":
        indices, ranges = partition_by_nodes(queue, discr,
            use_tree=use_tree, max_particles_in_box=max_particles_in_box)
    elif method == "elements":
        indices, ranges = partition_by_elements(queue, discr,
            use_tree=use_tree, max_particles_in_box=max_particles_in_box)
    else:
        raise ValueError("unknown method: {}".format(method))

    # randomly pick a subset of points
    if abs(factor - 1.0) > 1.0e-14:
        indices = indices.get(queue)
        ranges = ranges.get(queue)

        indices_ = np.empty(ranges.shape[0] - 1, dtype=np.object)
        for i in range(ranges.shape[0] - 1):
            iidx = indices[np.s_[ranges[i]:ranges[i + 1]]]
            isize = int(factor * len(iidx))
            isize = max(1, min(isize, len(iidx)))

            indices_[i] = np.sort(
                    np.random.choice(iidx, size=isize, replace=False))

        ranges = to_device(queue,
                np.cumsum([0] + [r.shape[0] for r in indices_]))
        indices = to_device(queue, np.hstack(indices_))

    return indices, ranges


def _plot_partition_indices(queue, discr, indices, ranges, **kwargs):
    import matplotlib.pyplot as pt
    args = [
        kwargs.get("method", "unknown"),
        "tree" if kwargs.get("use_tree", False) else "linear",
        kwargs.get("pid", "stage1"),
        discr.ambient_dim
        ]

    if isinstance(indices, cl.array.Array):
        indices = indices.get()
        ranges = ranges.get()

    pt.figure(figsize=(10, 8), dpi=300)
    pt.plot(np.diff(ranges))
    pt.savefig("test_partition_{0}_{1}_{3}d_ranges_{2}.png".format(*args))
    pt.clf()

    if discr.ambient_dim == 2:
        sources = discr.nodes().get(queue)

        pt.figure(figsize=(10, 8), dpi=300)

        if indices.shape[0] != discr.nnodes:
            pt.plot(sources[0], sources[1], 'ko', alpha=0.5)
        for i in range(ranges.shape[0] - 1):
            isrc = indices[np.s_[ranges[i]:ranges[i + 1]]]
            pt.plot(sources[0][isrc], sources[1][isrc], 'o')

        pt.xlim([-1.5, 1.5])
        pt.ylim([-1.5, 1.5])
        pt.savefig("test_partition_{0}_{1}_{3}d_{2}.png".format(*args))
        pt.clf()
    elif discr.ambient_dim == 3:
        from meshmode.discretization import NoninterpolatoryElementGroupError
        try:
            discr.groups[0].basis()
        except NoninterpolatoryElementGroupError:
            return

        from meshmode.discretization.visualization import make_visualizer
        marker = -42.0 * np.ones(discr.nnodes)

        for i in range(ranges.shape[0] - 1):
            isrc = indices[np.s_[ranges[i]:ranges[i + 1]]]
            marker[isrc] = 10.0 * (i + 1.0)

        vis = make_visualizer(queue, discr, 10)

        filename = "test_partition_{0}_{1}_{3}d_{2}.png".format(*args)
        if os.path.isfile(filename):
            os.remove(filename)

        vis.write_vtk_file(filename, [
            ("marker", cl.array.to_device(queue, marker))
            ])


@pytest.mark.parametrize("method", ["nodes", "elements"])
@pytest.mark.parametrize("use_tree", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
def test_partition_points(ctx_factory, method, use_tree, ndim, visualize=True):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    qbx = _build_qbx_discr(queue, ndim=ndim)
    srcindices, srcranges = _build_block_index(queue, qbx.density_discr,
            method=method, use_tree=use_tree, factor=0.6)


@pytest.mark.parametrize("use_tree", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
def test_partition_coarse(ctx_factory, use_tree, ndim, visualize=True):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    qbx = _build_qbx_discr(queue, ndim=ndim)
    srcindices, srcranges = _build_block_index(queue, qbx.density_discr, 
            method="elements", use_tree=use_tree)

    if visualize:
        discr = qbx.resampler.from_discr
        _plot_partition_indices(queue, discr, srcindices, srcranges,
                method="elements", use_tree=use_tree, pid="stage1")

    from pytential.linalg.proxy import partition_from_coarse
    resampler = qbx.direct_resampler

    t_start = time.time()
    srcindices_, srcranges_ = \
        partition_from_coarse(queue, resampler, srcindices, srcranges)
    t_end = time.time()
    if visualize:
        print('Time: {:.5f}s'.format(t_end - t_start))

    srcindices = srcindices.get()
    srcranges = srcranges.get()
    srcindices_ = srcindices_.get()
    srcranges_ = srcranges_.get()

    sources = resampler.from_discr.nodes().get(queue)
    sources_ = resampler.to_discr.nodes().get(queue)

    for i in range(srcranges.shape[0] - 1):
        isrc = srcindices[np.s_[srcranges[i]:srcranges[i + 1]]]
        isrc_ = srcindices_[np.s_[srcranges_[i]:srcranges_[i + 1]]]

        for j in range(ndim):
            assert np.min(sources_[j][isrc_]) <= np.min(sources[j][isrc])
            assert np.max(sources_[j][isrc_]) >= np.max(sources[j][isrc])

    if visualize:
        discr = resampler.to_discr
        _plot_partition_indices(queue, discr, srcindices_, srcranges_,
                method="elements", use_tree=use_tree, pid="stage2")


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("factor", [1.0, 0.6])
def test_proxy_generator(ctx_factory, ndim, factor, visualize=True):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    qbx = _build_qbx_discr(queue, ndim=ndim)
    srcindices, srcranges = _build_block_index(queue, qbx.density_discr,
            method='nodes', factor=factor)

    from pytential.linalg.proxy import ProxyGenerator
    generator = ProxyGenerator(qbx, ratio=1.1)
    proxies, pxyranges, pxycenters, pxyradii = \
        generator(queue, srcindices, srcranges)

    proxies = np.vstack([p.get() for p in proxies])
    pxyranges = pxyranges.get()
    pxycenters = np.vstack([c.get() for c in pxycenters])
    pxyradii = pxyradii.get()

    for i in range(srcranges.shape[0] - 1):
        ipxy = np.s_[pxyranges[i]:pxyranges[i + 1]]

        r = la.norm(proxies[:, ipxy] - pxycenters[:, i].reshape(-1, 1), axis=0)
        assert np.allclose(r - pxyradii[i], 0.0, atol=1.0e-14)

    if visualize:
        srcindices = srcindices.get()
        srcranges = srcranges.get()

    if visualize:
        if qbx.ambient_dim == 2:
            import matplotlib.pyplot as pt
            from pytential.qbx.utils import get_centers_on_side

            density_nodes = qbx.density_discr.nodes().get(queue)
            ci = get_centers_on_side(qbx, -1)
            ci = np.vstack([c.get(queue) for c in ci])
            ce = get_centers_on_side(qbx, +1)
            ce = np.vstack([c.get(queue) for c in ce])
            r = qbx._expansion_radii("nsources").get(queue)

            for i in range(srcranges.shape[0] - 1):
                isrc = srcindices[np.s_[srcranges[i]:srcranges[i + 1]]]
                ipxy = np.s_[pxyranges[i]:pxyranges[i + 1]]

                pt.figure(figsize=(10, 8))
                axis = pt.gca()
                for j in isrc:
                    c = pt.Circle(ci[:, j], r[j], color='k', alpha=0.1)
                    axis.add_artist(c)
                    c = pt.Circle(ce[:, j], r[j], color='k', alpha=0.1)
                    axis.add_artist(c)

                pt.plot(density_nodes[0], density_nodes[1],
                        'ko', ms=2.0, alpha=0.5)
                pt.plot(density_nodes[0, srcindices], density_nodes[1, srcindices],
                        'o', ms=2.0)
                pt.plot(density_nodes[0, isrc], density_nodes[1, isrc],
                        'o', ms=2.0)
                pt.plot(proxies[0, ipxy], proxies[1, ipxy],
                        'o', ms=2.0)
                pt.xlim([-1.5, 1.5])
                pt.ylim([-1.5, 1.5])

                filename = "test_proxy_generator_{}d_{:04}.png".format(ndim, i)
                pt.savefig(filename, dpi=300)
                pt.clf()
        else:
            from meshmode.discretization.visualization import make_visualizer
            from meshmode.mesh.processing import ( # noqa
                    affine_map, merge_disjoint_meshes)
            from meshmode.discretization import Discretization
            from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory

            for i in range(srcranges.shape[0] - 1):
                mesh = affine_map(generator.ref_mesh,
                    A=(pxyradii[i] * np.eye(ndim)),
                    b=pxycenters[:, i].reshape(-1))

                mesh = merge_disjoint_meshes([mesh, qbx.density_discr.mesh])
                discr = Discretization(ctx, mesh,
                    InterpolatoryQuadratureSimplexGroupFactory(10))

                vis = make_visualizer(queue, discr, 10)
                filename = "test_proxy_generator_{}d_{:04}.vtu".format(ndim, i)
                if os.path.isfile(filename):
                    os.remove(filename)
                vis.write_vtk_file(filename, [])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("factor", [1.0, 0.6])
def test_area_query(ctx_factory, ndim, factor, visualize=True):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    qbx = _build_qbx_discr(queue, ndim=ndim)
    srcindices, srcranges = _build_block_index(queue, qbx.density_discr,
            method='nodes', factor=factor)

    # generate proxy points
    from pytential.linalg.proxy import ProxyGenerator
    generator = ProxyGenerator(qbx, ratio=1.1)
    _, _, pxycenters, pxyradii = generator(queue, srcindices, srcranges)

    from pytential.linalg.proxy import build_neighbor_list, build_skeleton_list
    neighbors, nbrranges = build_neighbor_list(qbx.density_discr,
            srcindices, srcranges, pxycenters, pxyradii)
    skeletons, sklranges = build_skeleton_list(qbx, srcindices, srcranges,
                                               ratio=1.1)

    srcindices = srcindices.get()
    srcranges = srcranges.get()
    neighbors = neighbors.get()
    nbrranges = nbrranges.get()

    for i in range(srcranges.shape[0] - 1):
        isrc = np.s_[srcranges[i]:srcranges[i + 1]]
        inbr = np.s_[nbrranges[i]:nbrranges[i + 1]]

        assert not np.any(np.isin(neighbors[inbr], srcindices[isrc]))

    if visualize:
        if ndim == 2:
            import matplotlib.pyplot as pt
            density_nodes = qbx.density_discr.nodes().get(queue)
            skeletons = skeletons.get(queue)
            sklranges = sklranges.get(queue)

            for i in range(srcranges.shape[0] - 1):
                isrc = srcindices[np.s_[srcranges[i]:srcranges[i + 1]]]
                ingb = neighbors[nbrranges[i]:nbrranges[i + 1]]
                iskl = np.s_[sklranges[i]:sklranges[i + 1]]

                pt.figure(figsize=(10, 8))
                pt.plot(density_nodes[0], density_nodes[1],
                        'ko', ms=2.0, alpha=0.5)
                pt.plot(density_nodes[0, srcindices], density_nodes[1, srcindices],
                        'o', ms=2.0)
                pt.plot(density_nodes[0, isrc], density_nodes[1, isrc],
                        'o', ms=2.0)
                pt.plot(density_nodes[0, ingb], density_nodes[1, ingb],
                        'o', ms=2.0)
                pt.plot(skeletons[0, iskl], skeletons[1, iskl],
                        'x', ms=2.0)
                pt.xlim([-1.5, 1.5])
                pt.ylim([-1.5, 1.5])

                filename = "test_area_query_{}d_{:04}.png".format(ndim, i)
                pt.savefig(filename, dpi=300)
                pt.clf()
        elif ndim == 3:
            from meshmode.discretization.visualization import make_visualizer
            marker = np.empty(qbx.density_discr.nnodes)

            for i in range(srcranges.shape[0] - 1):
                isrc = srcindices[np.s_[srcranges[i]:srcranges[i + 1]]]
                ingb = neighbors[nbrranges[i]:nbrranges[i + 1]]

                # TODO: some way to turn off some of the interpolations
                # would help visualize this better.
                marker.fill(0.0)
                marker[srcindices] = 0.0
                marker[isrc] = -42.0
                marker[ingb] = +42.0
                marker_dev = cl.array.to_device(queue, marker)

                vis = make_visualizer(queue, qbx.density_discr, 10)
                filename = "test_area_query_{}d_{:04}.vtu".format(ndim, i)
                if os.path.isfile(filename):
                    os.remove(filename)

                vis.write_vtk_file(filename, [
                    ("marker", marker_dev),
                    ])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
