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
import pyopencl as cl

import loopy as lp
from pytential import sym
from meshmode.mesh.generation import ( # noqa
        ellipse, NArmedStarfish, generate_torus, make_curve_mesh)

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)


def create_data(queue,
        target_order=7,
        qbx_order=4,
        nelements=30,
        curve_f=None,
        ndim=2, k=0,
        lpot_idx=2):

    if curve_f is None:
        curve_f = NArmedStarfish(5, 0.25)
        # curve_f = partial(ellipse, 3)

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    if k:
        knl = HelmholtzKernel(ndim)
        knl_kwargs = {"k": k}
    else:
        knl = LaplaceKernel(ndim)
        knl_kwargs = {}

    if lpot_idx == 1:
        u_sym = sym.var("u")
        op = sym.D(knl, u_sym, **knl_kwargs)
    else:
        u_sym = sym.var("u")
        op = sym.S(knl, u_sym, **knl_kwargs)

    if ndim == 2:
        mesh = make_curve_mesh(curve_f,
                np.linspace(0, 1, nelements + 1),
                target_order)
    else:
        mesh = generate_torus(10.0, 2.0, order=target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx, _ = QBXLayerPotentialSource(pre_density_discr, 4 * target_order,
            qbx_order,
            # Don't use FMM for now
            fmm_order=False).with_refinement()

    return qbx, op, u_sym


def create_indices(qbx, nblks,
                   factor=1.0,
                   random=True,
                   method='elements',
                   use_tree=True,
                   use_stage2=False):
    from pytential.direct_solver import (
            partition_by_nodes, partition_by_elements)

    if use_stage2:
        density_discr = qbx.quad_stage2_density_discr
    else:
        density_discr = qbx.density_discr

    if method == 'nodes':
        nnodes = density_discr.nnodes
    else:
        nnodes = density_discr.mesh.nelements
    max_particles_in_box = nnodes // nblks

    if method == "nodes":
        indices, ranges = partition_by_nodes(density_discr,
            use_tree=use_tree, max_particles_in_box=max_particles_in_box)
    elif method == "elements":
        indices, ranges = partition_by_elements(density_discr,
            use_tree=use_tree, max_particles_in_box=max_particles_in_box)
    else:
        raise ValueError("unknown method: {}".format(method))

    # take a subset of the points
    if abs(factor - 1.0) > 1.0e-14:
        indices_ = np.empty(ranges.shape[0] - 1, dtype=np.object)
        for i in range(ranges.shape[0] - 1):
            iidx = indices[np.s_[ranges[i]:ranges[i + 1]]]
            isize = int(factor * len(iidx))

            if random:
                indices_[i] = np.sort(np.random.choice(iidx,
                    size=isize, replace=False))
            else:
                indices_[i] = iidx[:isize]

        ranges = np.cumsum([0] + [r.shape[0] for r in indices_])
        indices = np.hstack(indices_)

    return indices, ranges


@pytest.mark.parametrize("method", ["nodes", "elements"])
@pytest.mark.parametrize("use_tree", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
def test_partition_points(ctx_factory, method, use_tree, ndim, visualize=False):
    def plot_indices(pid, discr, indices, ranges):
        import matplotlib.pyplot as pt

        pt.figure(figsize=(10, 8))
        pt.plot(np.diff(ranges))

        rangefile = "test_partition_{}_{}_ranges_{}.png".format(method,
                "tree" if use_tree else "linear", pid)
        pt.savefig(rangefile, dpi=300)
        pt.clf()

        if ndim == 2:
            sources = discr.nodes().get(queue)

            pt.figure(figsize=(10, 8))
            pt.plot(sources[0], sources[1], 'ko', alpha=0.5)
            for i in range(nblks):
                isrc = indices[np.s_[ranges[i]:ranges[i + 1]]]
                pt.plot(sources[0][isrc], sources[1][isrc], 'o')
            pt.xlim([-1.5, 1.5])
            pt.ylim([-1.5, 1.5])

            pointfile = "test_partition_{}_{}_{}d_{}.png".format(method,
                "tree" if use_tree else "linear", ndim, pid)
            pt.savefig(pointfile, dpi=300)
            pt.clf()
        elif ndim == 3:
            from meshmode.discretization import NoninterpolatoryElementGroupError
            try:
                discr.groups[0].basis()
            except NoninterpolatoryElementGroupError:
                return

            from meshmode.discretization.visualization import make_visualizer
            marker = -42.0 * np.ones(discr.nnodes)

            for i in range(nblks):
                isrc = indices[np.s_[ranges[i]:ranges[i + 1]]]
                marker[isrc] = 10.0 * (i + 1.0)

            vis = make_visualizer(queue, discr, 10)

            pointfile = "test_partition_{}_{}_{}d_{}.vtu".format(method,
                "tree" if use_tree else "linear", ndim, pid)
            if os.path.isfile(pointfile):
                os.remove(pointfile)

            vis.write_vtk_file(pointfile, [
                ("marker", cl.array.to_device(queue, marker))
                ])

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from sympy.core.cache import clear_cache
    clear_cache()
    qbx = create_data(queue, ndim=ndim)[0]

    nblks = 10
    t_start = time.time()
    srcindices, srcranges = create_indices(qbx, nblks,
            method=method, use_tree=use_tree, factor=1.0)
    nblks = srcranges.shape[0] - 1
    t_end = time.time()
    if visualize:
        print('Time: {:.5f}s'.format(t_end - t_start))

    if visualize:
        discr = qbx.resampler.from_discr
        plot_indices(0, discr, srcindices, srcranges)

    if method == "elements":
        from meshmode.discretization.connection import flatten_chained_connection
        resampler = flatten_chained_connection(queue, qbx.resampler)

        from pytential.direct_solver import partition_from_coarse
        t_start = time.time()
        srcindices_, srcranges_ = \
            partition_from_coarse(queue, resampler, srcindices, srcranges)
        t_end = time.time()
        if visualize:
            print('Time: {:.5f}s'.format(t_end - t_start))

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
            plot_indices(1, discr, srcindices_, srcranges_)


@pytest.mark.parametrize("ndim", [2, 3])
def test_proxy_generator(ctx_factory, ndim, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # prevent cache explosion
    from sympy.core.cache import clear_cache
    clear_cache()
    qbx = create_data(queue, ndim=ndim)[0]

    nblks = 10
    srcindices, srcranges = create_indices(qbx, nblks,
                                           method='nodes', factor=0.6)
    nblks = srcranges.shape[0] - 1

    from pytential.direct_solver import ProxyGenerator
    gen = ProxyGenerator(queue, qbx, ratio=1.1)
    centers, radii, proxies, pxyranges = \
        gen.get_proxies(srcindices, srcranges)

    if visualize:
        knl = gen.get_kernel()
        knl = lp.add_dtypes(knl, {
            "nodes": np.float64,
            "center_int": np.float64,
            "center_ext": np.float64,
            "expansion_radii": np.float64,
            "ranges": np.int64,
            "indices": np.int64,
            "nnodes": np.int64,
            "nranges": np.int64,
            "nindices": np.int64})
        print(knl)
        print(lp.generate_code_v2(knl).device_code())

    proxies = np.vstack([p.get(queue) for p in proxies])
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
            pxyranges = pxyranges.get(queue)

            for i in range(nblks):
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

            centers = np.vstack([c.get(queue) for c in centers])
            radii = radii.get(queue)

            for i in range(nblks):
                mesh = affine_map(gen.mesh,
                    A=(radii[i] * np.eye(ndim)),
                    b=centers[:, i].reshape(-1))

                mesh = merge_disjoint_meshes([mesh, qbx.density_discr.mesh])
                discr = Discretization(ctx, mesh,
                    InterpolatoryQuadratureSimplexGroupFactory(10))

                vis = make_visualizer(queue, discr, 10)
                filename = "test_proxy_generator_{}d_{:04}.vtu".format(ndim, i)
                if os.path.isfile(filename):
                    os.remove(filename)
                vis.write_vtk_file(filename, [])


@pytest.mark.parametrize("ndim", [2, 3])
def test_area_query(ctx_factory, ndim, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from sympy.core.cache import clear_cache
    clear_cache()
    qbx = create_data(queue, ndim=ndim)[0]

    nblks = 10
    srcindices, srcranges = create_indices(qbx, nblks,
                                           method='nodes', factor=0.6)
    nblks = srcranges.shape[0] - 1

    # generate proxy points
    from pytential.direct_solver import ProxyGenerator
    gen = ProxyGenerator(queue, qbx, ratio=1.1)
    centers, radii, _, _ = gen.get_proxies(srcindices, srcranges)
    neighbors, ngbranges = gen.get_neighbors(srcindices, srcranges, centers, radii)
    skeleton_nodes, sklranges = gen(srcindices, srcranges)

    neighbors = neighbors.get(queue)
    ngbranges = ngbranges.get(queue)
    if visualize:
        if ndim == 2:
            import matplotlib.pyplot as pt
            density_nodes = qbx.density_discr.nodes().get(queue)
            skeleton_nodes = skeleton_nodes.get(queue)

            for i in range(nblks):
                isrc = srcindices[np.s_[srcranges[i]:srcranges[i + 1]]]
                ingb = neighbors[ngbranges[i]:ngbranges[i + 1]]
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
                pt.plot(skeleton_nodes[0, iskl], skeleton_nodes[1, iskl],
                        'x', ms=2.0)
                pt.xlim([-1.5, 1.5])
                pt.ylim([-1.5, 1.5])

                filename = "test_area_query_{}d_{:04}.png".format(ndim, i)
                pt.savefig(filename, dpi=300)
                pt.clf()
        elif ndim == 3:
            from meshmode.discretization.visualization import make_visualizer
            marker = np.empty(qbx.density_discr.nnodes)

            for i in range(nblks):
                isrc = srcindices[np.s_[srcranges[i]:srcranges[i + 1]]]
                ingb = neighbors[ngbranges[i]:ngbranges[i + 1]]

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
