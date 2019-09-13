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

import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array   # noqa

from pytential import bind, sym
from meshmode.mesh.generation import ( # noqa
        ellipse, NArmedStarfish, generate_torus, make_curve_mesh)

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)


from test_matrix import _build_geometry, _build_block_index


def _plot_partition_indices(queue, discr, indices, **kwargs):
    import matplotlib.pyplot as pt
    indices = indices.get(queue)

    args = [
        kwargs.get("method", "unknown"),
        "tree" if kwargs.get("use_tree", False) else "linear",
        kwargs.get("pid", "stage1"),
        discr.ambient_dim
        ]

    pt.figure(figsize=(10, 8), dpi=300)
    pt.plot(np.diff(indices.ranges))
    pt.savefig("test_partition_{0}_{1}_{3}d_ranges_{2}.png".format(*args))
    pt.clf()

    if discr.ambient_dim == 2:
        sources = discr.nodes().get(queue)

        pt.figure(figsize=(10, 8), dpi=300)

        if indices.indices.shape[0] != discr.nnodes:
            pt.plot(sources[0], sources[1], 'ko', alpha=0.5)
        for i in range(indices.nblocks):
            isrc = indices.block_indices(i)
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

        for i in range(indices.nblocks):
            isrc = indices.block_indices(i)
            marker[isrc] = 10.0 * (i + 1.0)

        vis = make_visualizer(queue, discr, 10)

        filename = "test_partition_{0}_{1}_{3}d_{2}.png".format(*args)
        vis.write_vtk_file(filename, [
            ("marker", cl.array.to_device(queue, marker))
            ])


@pytest.mark.parametrize("use_tree", [True, False])
@pytest.mark.parametrize("ambient_dim", [2, 3])
def test_partition_points(ctx_factory, use_tree, ambient_dim, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    places, dofdesc = _build_geometry(queue, ambient_dim=ambient_dim)
    _build_block_index(queue,
            places.get_discretization(dofdesc),
            use_tree=use_tree,
            factor=0.6)


@pytest.mark.parametrize("ambient_dim", [2, 3])
@pytest.mark.parametrize("factor", [1.0, 0.6])
def test_proxy_generator(ctx_factory, ambient_dim, factor, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    places, dofdesc = _build_geometry(queue, ambient_dim=ambient_dim)
    dofdesc = dofdesc.to_stage1()

    density_discr = places.get_discretization(dofdesc)
    srcindices = _build_block_index(queue,
            density_discr,
            factor=factor)

    from pytential.linalg.proxy import ProxyGenerator
    generator = ProxyGenerator(places, dofdesc=dofdesc)
    proxies, pxyranges, pxycenters, pxyradii = generator(queue, srcindices)

    proxies = np.vstack([p.get() for p in proxies])
    pxyranges = pxyranges.get()
    pxycenters = np.vstack([c.get() for c in pxycenters])
    pxyradii = pxyradii.get()

    for i in range(srcindices.nblocks):
        ipxy = np.s_[pxyranges[i]:pxyranges[i + 1]]

        r = la.norm(proxies[:, ipxy] - pxycenters[:, i].reshape(-1, 1), axis=0)
        assert np.allclose(r - pxyradii[i], 0.0, atol=1.0e-14)

    srcindices = srcindices.get(queue)
    if visualize:
        if ambient_dim == 2:
            import matplotlib.pyplot as pt

            density_nodes = density_discr.nodes().get(queue)
            ci = bind(places, sym.expansion_centers(ambient_dim, -1))(queue)
            ci = np.vstack([c.get(queue) for c in ci])
            ce = bind(places, sym.expansion_centers(ambient_dim, +1))(queue)
            ce = np.vstack([c.get(queue) for c in ce])
            r = bind(places, sym.expansion_radii(ambient_dim))(queue).get()

            for i in range(srcindices.nblocks):
                isrc = srcindices.block_indices(i)
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
                pt.plot(density_nodes[0, srcindices.indices],
                        density_nodes[1, srcindices.indices],
                        'o', ms=2.0)
                pt.plot(density_nodes[0, isrc], density_nodes[1, isrc],
                        'o', ms=2.0)
                pt.plot(proxies[0, ipxy], proxies[1, ipxy],
                        'o', ms=2.0)
                pt.xlim([-1.5, 1.5])
                pt.ylim([-1.5, 1.5])

                filename = "test_proxy_generator_{}d_{:04}.png".format(
                        ambient_dim, i)
                pt.savefig(filename, dpi=300)
                pt.clf()
        else:
            from meshmode.discretization.visualization import make_visualizer
            from meshmode.mesh.processing import ( # noqa
                    affine_map, merge_disjoint_meshes)
            from meshmode.discretization import Discretization
            from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory

            from meshmode.mesh.generation import generate_icosphere
            ref_mesh = generate_icosphere(1, generator.nproxy)

            # NOTE: this does not plot the actual proxy points
            for i in range(srcindices.nblocks):
                mesh = affine_map(ref_mesh,
                    A=(pxyradii[i] * np.eye(ambient_dim)),
                    b=pxycenters[:, i].reshape(-1))

                mesh = merge_disjoint_meshes([mesh, density_discr.mesh])
                discr = Discretization(ctx, mesh,
                    InterpolatoryQuadratureSimplexGroupFactory(10))

                vis = make_visualizer(queue, discr, 10)
                filename = "test_proxy_generator_{}d_{:04}.vtu".format(
                        ambient_dim, i)
                vis.write_vtk_file(filename, [])


@pytest.mark.parametrize("ambient_dim", [2, 3])
@pytest.mark.parametrize("factor", [1.0, 0.6])
def test_interaction_points(ctx_factory, ambient_dim, factor, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    places, dofdesc = _build_geometry(queue, ambient_dim=ambient_dim)
    dofdesc = dofdesc.to_stage1()

    density_discr = places.get_discretization(dofdesc)
    srcindices = _build_block_index(queue,
            density_discr,
            factor=factor)

    # generate proxy points
    from pytential.linalg.proxy import ProxyGenerator
    generator = ProxyGenerator(places, dofdesc=dofdesc)
    _, _, pxycenters, pxyradii = generator(queue, srcindices)

    from pytential.linalg.proxy import (  # noqa
            gather_block_neighbor_points,
            gather_block_interaction_points)
    nbrindices = gather_block_neighbor_points(density_discr,
            srcindices, pxycenters, pxyradii)
    nodes, ranges = gather_block_interaction_points(
            places, dofdesc, srcindices)

    srcindices = srcindices.get(queue)
    nbrindices = nbrindices.get(queue)

    for i in range(srcindices.nblocks):
        isrc = srcindices.block_indices(i)
        inbr = nbrindices.block_indices(i)

        assert not np.any(np.isin(inbr, isrc))

    if visualize:
        if ambient_dim == 2:
            import matplotlib.pyplot as pt
            density_nodes = density_discr.nodes().get(queue)
            nodes = nodes.get(queue)
            ranges = ranges.get(queue)

            for i in range(srcindices.nblocks):
                isrc = srcindices.block_indices(i)
                inbr = nbrindices.block_indices(i)
                iall = np.s_[ranges[i]:ranges[i + 1]]

                pt.figure(figsize=(10, 8))
                pt.plot(density_nodes[0], density_nodes[1],
                        'ko', ms=2.0, alpha=0.5)
                pt.plot(density_nodes[0, srcindices.indices],
                        density_nodes[1, srcindices.indices],
                        'o', ms=2.0)
                pt.plot(density_nodes[0, isrc], density_nodes[1, isrc],
                        'o', ms=2.0)
                pt.plot(density_nodes[0, inbr], density_nodes[1, inbr],
                        'o', ms=2.0)
                pt.plot(nodes[0, iall], nodes[1, iall],
                        'x', ms=2.0)
                pt.xlim([-1.5, 1.5])
                pt.ylim([-1.5, 1.5])

                filename = "test_area_query_{}d_{:04}.png".format(ambient_dim, i)
                pt.savefig(filename, dpi=300)
                pt.clf()
        elif ambient_dim == 3:
            from meshmode.discretization.visualization import make_visualizer
            marker = np.empty(density_discr.nnodes)

            for i in range(srcindices.nblocks):
                isrc = srcindices.block_indices(i)
                inbr = nbrindices.block_indices(i)

                marker.fill(0.0)
                marker[srcindices.indices] = 0.0
                marker[isrc] = -42.0
                marker[inbr] = +42.0
                marker_dev = cl.array.to_device(queue, marker)

                vis = make_visualizer(queue, density_discr, 10)
                filename = "test_area_query_{}d_{:04}.vtu".format(ambient_dim, i)
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
