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

from functools import partial

import numpy as np
import numpy.linalg as la

import pyopencl as cl

from pytential import bind, sym
from pytential import GeometryCollection

from arraycontext import PyOpenCLArrayContext
from meshmode.mesh.generation import ellipse

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import extra_matrix_data as extra
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_partition_indices(actx, discr, indices, **kwargs):
    try:
        import matplotlib.pyplot as pt
    except ImportError:
        return

    indices = indices.get(actx.queue)
    args = [
        kwargs.get("tree_kind", "linear").replace("-", "_"),
        kwargs.get("discr_stage", "stage1"),
        discr.ambient_dim
        ]

    pt.figure(figsize=(10, 8), dpi=300)
    pt.plot(np.diff(indices.ranges))
    pt.savefig("test_partition_{1}_{3}d_ranges_{2}.png".format(*args))
    pt.clf()

    from pytential.utils import flatten_to_numpy
    if discr.ambient_dim == 2:
        sources = flatten_to_numpy(actx, discr.nodes())

        pt.figure(figsize=(10, 8), dpi=300)
        if indices.indices.shape[0] != discr.ndofs:
            pt.plot(sources[0], sources[1], "ko", alpha=0.5)

        for i in range(indices.nblocks):
            isrc = indices.block_indices(i)
            pt.plot(sources[0][isrc], sources[1][isrc], "o")

        pt.xlim([-1.5, 1.5])
        pt.ylim([-1.5, 1.5])
        pt.savefig("test_partition_{1}_{3}d_{2}.png".format(*args))
        pt.clf()
    elif discr.ambient_dim == 3:
        from meshmode.discretization.visualization import make_visualizer
        marker = -42.0 * np.ones(discr.ndofs)

        for i in range(indices.nblocks):
            isrc = indices.block_indices(i)
            marker[isrc] = 10.0 * (i + 1.0)

        from meshmode.dof_array import unflatten
        marker = unflatten(actx, discr, actx.from_numpy(marker))

        vis = make_visualizer(actx, discr, 10)

        filename = "test_partition_{0}_{1}_{3}d_{2}.vtu".format(*args)
        vis.write_vtk_file(filename, [
            ("marker", marker)
            ])


PROXY_TEST_CASES = [
        extra.CurveTestCase(
            name="ellipse",
            target_order=7,
            curve_fn=partial(ellipse, 3.0)),
        extra.TorusTestCase(
            target_order=2,
            resolutions=[0])
        ]


@pytest.mark.skip(reason="only useful with visualize=True")
@pytest.mark.parametrize("tree_kind", ["adaptive", None])
@pytest.mark.parametrize("case", PROXY_TEST_CASES)
def test_partition_points(ctx_factory, tree_kind, case, visualize=False):
    """Tests that the points are correctly partitioned (by visualization)."""

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = case.copy(tree_kind=tree_kind, index_sparsity_factor=0.6)
    logger.info("\n%s", case)

    # {{{

    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    density_discr = places.get_discretization(case.name)
    indices = case.get_block_indices(actx, density_discr)

    if visualize:
        plot_partition_indices(actx, density_discr, indices, tree_kind=tree_kind)

    # }}}


@pytest.mark.parametrize("case", PROXY_TEST_CASES)
@pytest.mark.parametrize("index_sparsity_factor", [1.0, 0.6])
def test_proxy_generator(ctx_factory, case, index_sparsity_factor, visualize=False):
    """Tests that the proxies generated are all at the correct radius from the
    points in the cluster, etc.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = case.copy(index_sparsity_factor=index_sparsity_factor)
    logger.info("\n%s", case)

    # {{{ generate proxies

    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    density_discr = places.get_discretization(case.name)
    srcindices = case.get_block_indices(actx, density_discr, matrix_indices=False)

    from pytential.linalg.proxy import ProxyGenerator
    generator = ProxyGenerator(places)
    proxies, pxyranges, pxycenters, pxyradii = \
            generator(actx, places.auto_source, srcindices)

    proxies = np.vstack([actx.to_numpy(p) for p in proxies])
    pxyranges = actx.to_numpy(pxyranges)
    pxycenters = np.vstack([actx.to_numpy(c) for c in pxycenters])
    pxyradii = actx.to_numpy(pxyradii)

    for i in range(srcindices.nblocks):
        ipxy = np.s_[pxyranges[i]:pxyranges[i + 1]]

        r = la.norm(proxies[:, ipxy] - pxycenters[:, i].reshape(-1, 1), axis=0)
        assert np.allclose(r - pxyradii[i], 0.0, atol=1.0e-14)

    # }}}

    # {{{ visualization

    srcindices = srcindices.get(queue)
    if visualize:
        ambient_dim = places.ambient_dim
        if ambient_dim == 2:
            import matplotlib.pyplot as pt

            from pytential.utils import flatten_to_numpy
            density_nodes = np.vstack(flatten_to_numpy(actx, density_discr.nodes()))
            ci = bind(places, sym.expansion_centers(ambient_dim, -1))(actx)
            ci = np.vstack(flatten_to_numpy(actx, ci))
            ce = bind(places, sym.expansion_centers(ambient_dim, +1))(actx)
            ce = np.vstack(flatten_to_numpy(actx, ce))
            r = bind(places, sym.expansion_radii(ambient_dim))(actx)
            r = flatten_to_numpy(actx, r)

            for i in range(srcindices.nblocks):
                isrc = srcindices.block_indices(i)
                ipxy = np.s_[pxyranges[i]:pxyranges[i + 1]]

                pt.figure(figsize=(10, 8))
                axis = pt.gca()
                for j in isrc:
                    c = pt.Circle(ci[:, j], r[j], color="k", alpha=0.1)
                    axis.add_artist(c)
                    c = pt.Circle(ce[:, j], r[j], color="k", alpha=0.1)
                    axis.add_artist(c)

                pt.plot(density_nodes[0], density_nodes[1],
                        "ko", ms=2.0, alpha=0.5)
                pt.plot(density_nodes[0, srcindices.indices],
                        density_nodes[1, srcindices.indices],
                        "o", ms=2.0)
                pt.plot(density_nodes[0, isrc], density_nodes[1, isrc],
                        "o", ms=2.0)
                pt.plot(proxies[0, ipxy], proxies[1, ipxy],
                        "o", ms=2.0)
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
                discr = Discretization(actx, mesh,
                    InterpolatoryQuadratureSimplexGroupFactory(10))

                vis = make_visualizer(actx, discr, 10)
                filename = "test_proxy_generator_{}d_{:04}.vtu".format(
                        ambient_dim, i)
                vis.write_vtk_file(filename, [])

    # }}}


@pytest.mark.parametrize("case", PROXY_TEST_CASES)
@pytest.mark.parametrize("index_sparsity_factor", [1.0, 0.6])
def test_interaction_points(ctx_factory,
        case, index_sparsity_factor, visualize=False):
    """Test that neighboring points (inside the proxy balls, but outside the
    current block/cluster) are actually inside.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = case.copy(index_sparsity_factor=index_sparsity_factor)
    logger.info("\n%s", case)

    # {{{ check neighboring points

    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    density_discr = places.get_discretization(case.name)
    srcindices = case.get_block_indices(actx, density_discr, matrix_indices=False)

    # generate proxy points
    from pytential.linalg.proxy import ProxyGenerator
    generator = ProxyGenerator(places)
    _, _, pxycenters, pxyradii = generator(actx, places.auto_source, srcindices)

    # get neighboring points
    from pytential.linalg.proxy import (  # noqa
            gather_block_neighbor_points,
            gather_block_interaction_points)
    nbrindices = gather_block_neighbor_points(actx, density_discr,
            srcindices, pxycenters, pxyradii)
    nodes, ranges = gather_block_interaction_points(actx,
            places, places.auto_source, srcindices)

    srcindices = srcindices.get(queue)
    nbrindices = nbrindices.get(queue)

    for i in range(srcindices.nblocks):
        isrc = srcindices.block_indices(i)
        inbr = nbrindices.block_indices(i)

        assert not np.any(np.isin(inbr, isrc))

    # }}}

    # {{{ visualize

    from pytential.utils import flatten_to_numpy
    if visualize:
        ambient_dim = places.ambient_dim
        if ambient_dim == 2:
            import matplotlib.pyplot as pt
            density_nodes = flatten_to_numpy(actx, density_discr.nodes())
            nodes = flatten_to_numpy(actx, nodes)
            ranges = actx.to_numpy(ranges)

            for i in range(srcindices.nblocks):
                isrc = srcindices.block_indices(i)
                inbr = nbrindices.block_indices(i)
                iall = np.s_[ranges[i]:ranges[i + 1]]

                pt.figure(figsize=(10, 8))
                pt.plot(density_nodes[0], density_nodes[1],
                        "ko", ms=2.0, alpha=0.5)
                pt.plot(density_nodes[0][srcindices.indices],
                        density_nodes[1][srcindices.indices],
                        "o", ms=2.0)
                pt.plot(density_nodes[0][isrc], density_nodes[1][isrc],
                        "o", ms=2.0)
                pt.plot(density_nodes[0][inbr], density_nodes[1][inbr],
                        "o", ms=2.0)
                pt.plot(nodes[0][iall], nodes[1][iall],
                        "x", ms=2.0)
                pt.xlim([-1.5, 1.5])
                pt.ylim([-1.5, 1.5])

                filename = f"test_area_query_{ambient_dim}d_{i:04}.png"
                pt.savefig(filename, dpi=300)
                pt.clf()
        elif ambient_dim == 3:
            from meshmode.discretization.visualization import make_visualizer
            marker = np.empty(density_discr.ndofs)

            for i in range(srcindices.nblocks):
                isrc = srcindices.block_indices(i)
                inbr = nbrindices.block_indices(i)

                marker.fill(0.0)
                marker[srcindices.indices] = 0.0
                marker[isrc] = -42.0
                marker[inbr] = +42.0

                from meshmode.dof_array import unflatten
                marker_dev = unflatten(actx, density_discr, actx.from_numpy(marker))

                vis = make_visualizer(actx, density_discr, 10)
                filename = f"test_area_query_{ambient_dim}d_{i:04}.vtu"
                vis.write_vtk_file(filename, [
                    ("marker", marker_dev),
                    ])
    # }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
