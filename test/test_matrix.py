from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
"""

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
from pytools.obj_array import make_obj_array, is_obj_array

from sumpy.symbolic import USE_SYMENGINE
from meshmode.mesh.generation import \
        ellipse, NArmedStarfish, make_curve_mesh

from pytential import bind, sym
from pytential.symbolic.primitives import DEFAULT_SOURCE, DEFAULT_TARGET
from pytential.symbolic.primitives import QBXSourceStage1, QBXSourceQuadStage2

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)


def _build_op(lpot_id,
              k=0,
              ndim=2,
              qbx_forced_limit=-1):

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    if k:
        knl = HelmholtzKernel(ndim)
        knl_kwargs = {"k": k}
    else:
        knl = LaplaceKernel(ndim)
        knl_kwargs = {}

    lpot_kwargs = {"qbx_forced_limit": qbx_forced_limit}
    lpot_kwargs.update(knl_kwargs)
    if lpot_id == 1:
        # scalar single-layer potential
        u_sym = sym.var("u")
        op = sym.S(knl, u_sym, **lpot_kwargs)
    elif lpot_id == 2:
        # scalar double-layer potential
        u_sym = sym.var("u")
        op = sym.D(knl, u_sym, **lpot_kwargs)
    elif lpot_id == 3:
        # vector potential
        u_sym = sym.make_sym_vector("u", 2)
        u0_sym, u1_sym = u_sym

        op = make_obj_array([
            sym.Sp(knl, u0_sym, **lpot_kwargs) +
            sym.D(knl, u1_sym, **lpot_kwargs),

            sym.S(knl, 0.4 * u0_sym, **lpot_kwargs) +
            0.3 * sym.D(knl, u0_sym, **lpot_kwargs)
            ])
    else:
        raise ValueError("Unknown lpot_id: {}".format(lpot_id))

    return op, u_sym, knl_kwargs


@pytest.mark.skipif(USE_SYMENGINE,
        reason="https://gitlab.tiker.net/inducer/sumpy/issues/25")
@pytest.mark.parametrize("k", [0, 42])
@pytest.mark.parametrize("curve_f", [
    partial(ellipse, 3),
    NArmedStarfish(5, 0.25)])
@pytest.mark.parametrize("lpot_id", [1, 3])
def test_matrix_build(ctx_factory, k, curve_f, lpot_id, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    target_order = 7
    qbx_order = 4
    nelements = 30
    mesh = make_curve_mesh(curve_f,
            np.linspace(0, 1, nelements + 1),
            target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx, _ = QBXLayerPotentialSource(pre_density_discr, 4 * target_order,
            qbx_order,
            # Don't use FMM for now
            fmm_order=False).with_refinement()
    density_discr = qbx.density_discr

    op, u_sym, knl_kwargs = _build_op(lpot_id, k=k)
    bound_op = bind(qbx, op)

    from pytential.symbolic.execution import build_matrix
    mat = build_matrix(queue, qbx, op, u_sym).get()

    if visualize:
        from sumpy.tools import build_matrix as build_matrix_via_matvec
        mat2 = bound_op.scipy_op(queue, "u", dtype=mat.dtype, **knl_kwargs)
        mat2 = build_matrix_via_matvec(mat2)
        print(la.norm((mat - mat2).real, "fro") / la.norm(mat2.real, "fro"),
              la.norm((mat - mat2).imag, "fro") / la.norm(mat2.imag, "fro"))

        import matplotlib.pyplot as pt
        pt.subplot(121)
        pt.imshow(np.log10(np.abs(1.0e-20 + (mat - mat2).real)))
        pt.colorbar()
        pt.subplot(122)
        pt.imshow(np.log10(np.abs(1.0e-20 + (mat - mat2).imag)))
        pt.colorbar()
        pt.show()

    if visualize:
        import matplotlib.pyplot as pt
        pt.subplot(121)
        pt.imshow(mat.real)
        pt.colorbar()
        pt.subplot(122)
        pt.imshow(mat.imag)
        pt.colorbar()
        pt.show()

    from sumpy.tools import vector_to_device, vector_from_device
    np.random.seed(12)
    for i in range(5):
        if is_obj_array(u_sym):
            u = make_obj_array([
                np.random.randn(density_discr.nnodes)
                for _ in range(len(u_sym))
                ])
        else:
            u = np.random.randn(density_discr.nnodes)

        u_dev = vector_to_device(queue, u)
        res_matvec = np.hstack(
                list(vector_from_device(
                    queue, bound_op(queue, u=u_dev))))

        res_mat = mat.dot(np.hstack(list(u)))

        abs_err = la.norm(res_mat - res_matvec, np.inf)
        rel_err = abs_err / la.norm(res_matvec, np.inf)

        print("AbsErr {:.5e} RelErr {:.5e}".format(abs_err, rel_err))
        assert rel_err < 1e-13


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("factor", [1.0, 0.6])
@pytest.mark.parametrize("lpot_id", [1, 2])
def test_p2p_block_builder(ctx_factory, factor, ndim, lpot_id,
                           visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # prevent cache explosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from test_linalg_proxy import _build_qbx_discr, _build_block_index
    target_order = 2 if ndim == 3 else 7
    qbx = _build_qbx_discr(queue, target_order=target_order, ndim=ndim)
    op, u_sym, _ = _build_op(lpot_id, ndim=ndim)

    srcindices = _build_block_index(qbx.density_discr,
            method='nodes', factor=factor)
    tgtindices = _build_block_index(qbx.density_discr,
            method='nodes', factor=factor)

    from pytential.symbolic.execution import GeometryCollection
    from pytential.symbolic.execution import _prepare_expr, _prepare_domains
    places = GeometryCollection(qbx)
    expr = _prepare_expr(places, op)
    domains = _prepare_domains(1, places, None, places.source)

    from sumpy.tools import MatrixBlockIndexRanges
    index_set = MatrixBlockIndexRanges(ctx, tgtindices, srcindices)

    from pytential.symbolic.matrix import FarFieldBlockBuilder
    mbuilder = FarFieldBlockBuilder(queue,
            dep_expr=u_sym,
            other_dep_exprs=[],
            dep_source=places[domains[0]],
            dep_discr=places.get_discretization(domains[0]),
            places=places,
            index_set=index_set,
            context={})
    blk = mbuilder(expr)

    from pytential.symbolic.matrix import P2PMatrixBuilder
    mbuilder = P2PMatrixBuilder(queue,
            dep_expr=u_sym,
            other_dep_exprs=[],
            dep_source=places[domains[0]],
            dep_discr=places.get_discretization(domains[0]),
            places=places,
            context={})
    mat = mbuilder(expr)

    index_set = index_set.get(queue)
    if visualize and ndim == 2:
        blk_full = np.zeros_like(mat)
        mat_full = np.zeros_like(mat)

        for i in range(index_set.nblocks):
            itgt, isrc = index_set.block_indices(i)

            blk_full[np.ix_(itgt, isrc)] = index_set.block_take(blk, i)
            mat_full[np.ix_(itgt, isrc)] = index_set.take(mat, i)

        import matplotlib.pyplot as mp
        _, (ax1, ax2) = mp.subplots(1, 2,
                figsize=(10, 8), dpi=300, constrained_layout=True)
        ax1.imshow(blk_full)
        ax1.set_title('FarFieldBlockBuilder')
        ax2.imshow(mat_full)
        ax2.set_title('P2PMatrixBuilder')
        mp.savefig("test_p2p_block_{}d_{:.1f}.png".format(ndim, factor))

    for i in range(index_set.nblocks):
        eps = 1.0e-14 * la.norm(index_set.take(mat, i))
        error = la.norm(index_set.block_take(blk, i) -
                        index_set.take(mat, i))

        if visualize:
            print('block[{:04}]: {:.5e}'.format(i, error))
        assert error < eps


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("lpot_id", [1])
def test_qbx_block_builder(ctx_factory, ndim, lpot_id, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # prevent cache explosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from test_linalg_proxy import _build_qbx_discr, _build_block_index
    target_order = 2 if ndim == 3 else 7
    qbx = _build_qbx_discr(queue, target_order=target_order, ndim=ndim)
    op, u_sym, _ = _build_op(lpot_id, ndim=ndim)

    tgtindices = _build_block_index(qbx.density_discr)
    srcindices = _build_block_index(qbx.density_discr)

    # NOTE: NearFieldBlockBuilder only does stage1/stage1 or stage2/stage2,
    # so we need to hardcode the discr for MatrixBuilder too, since the
    # defaults are different
    where = (QBXSourceStage1(DEFAULT_SOURCE), QBXSourceStage1(DEFAULT_TARGET))

    from pytential.symbolic.execution import GeometryCollection, _prepare_expr
    places = GeometryCollection(qbx, auto_where=where)
    expr = _prepare_expr(places, op)

    from sumpy.tools import MatrixBlockIndexRanges
    index_set = MatrixBlockIndexRanges(ctx, tgtindices, srcindices)

    from pytential.symbolic.matrix import NearFieldBlockBuilder
    mbuilder = NearFieldBlockBuilder(queue,
            dep_expr=u_sym,
            other_dep_exprs=[],
            dep_source=places[where[0]],
            dep_discr=places.get_discretization(where[0]),
            places=places,
            index_set=index_set,
            context={})
    blk = mbuilder(expr)

    from pytential.symbolic.matrix import MatrixBuilder
    mbuilder = MatrixBuilder(queue,
            dep_expr=u_sym,
            other_dep_exprs=[],
            dep_source=places[where[0]],
            dep_discr=places.get_discretization(where[0]),
            places=places,
            context={})
    mat = mbuilder(expr)

    index_set = index_set.get(queue)
    if visualize:
        blk_full = np.zeros_like(mat)
        mat_full = np.zeros_like(mat)

        for i in range(index_set.nblocks):
            itgt, isrc = index_set.block_indices(i)

            blk_full[np.ix_(itgt, isrc)] = index_set.block_take(blk, i)
            mat_full[np.ix_(itgt, isrc)] = index_set.take(mat, i)

        import matplotlib.pyplot as mp
        _, (ax1, ax2) = mp.subplots(1, 2,
                figsize=(10, 8), constrained_layout=True)
        ax1.imshow(mat_full)
        ax1.set_title('MatrixBuilder')
        ax2.imshow(blk_full)
        ax2.set_title('NearFieldBlockBuilder')
        mp.savefig("test_qbx_block_builder.png", dpi=300)

    for i in range(index_set.nblocks):
        eps = 1.0e-14 * la.norm(index_set.take(mat, i))
        error = la.norm(index_set.block_take(blk, i) -
                        index_set.take(mat, i))

        if visualize:
            print('block[{:04}]: {:.5e}'.format(i, error))
        assert error < eps


@pytest.mark.parametrize('where',
        [(DEFAULT_SOURCE, DEFAULT_TARGET),
         (QBXSourceStage1(DEFAULT_SOURCE),
          QBXSourceStage1(DEFAULT_TARGET)),
         (QBXSourceQuadStage2(DEFAULT_SOURCE),
          QBXSourceQuadStage2(DEFAULT_TARGET))])
def test_build_matrix_where(ctx_factory, where, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # prevent cache explosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from test_linalg_proxy import _build_qbx_discr
    qbx = _build_qbx_discr(queue, nelements=8, target_order=2, ndim=2,
                           curve_f=partial(ellipse, 1.0))

    qbx_forced_limit = -1
    op, u_sym, _ = _build_op(lpot_id=1, ndim=2,
            qbx_forced_limit=qbx_forced_limit)

    # associate quad_stage2 targets to centers
    from pytential.qbx.target_assoc import associate_targets_to_qbx_centers

    code_container = qbx.target_association_code_container
    target_discr = qbx.quad_stage2_density_discr
    target_assoc = associate_targets_to_qbx_centers(
            qbx,
            code_container.get_wrangler(queue),
            [(target_discr, qbx_forced_limit)],
            target_association_tolerance=1.0e-1).get(queue)
    target_assoc = target_assoc.target_to_center

    if qbx.ambient_dim == 2 and visualize:
        import matplotlib.pyplot as pt
        from pytential.qbx.utils import get_interleaved_centers
        sources = qbx.density_discr.nodes().get(queue)
        targets = target_discr.nodes().get(queue)
        centers = get_interleaved_centers(queue, qbx)
        centers = np.vstack(c.get(queue) for c in centers)
        radii = qbx._expansion_radii('nsources').get(queue)

        for i in range(centers[0].size):
            itgt = np.where(target_assoc == i)[0]
            if not len(itgt):
                continue

            pt.figure(figsize=(10, 8), dpi=300)
            pt.plot(sources[0], sources[1], 'k')
            pt.plot(targets[0], targets[1], 'ko')

            line = pt.plot(targets[0, itgt], targets[1, itgt], 'o')
            c = pt.Circle([centers[0][i], centers[1][i]], radii[i // 2],
                           color=line[0].get_color(), alpha=0.5)
            pt.gca().add_artist(c)

            pt.savefig('test_assoc_quad_targets_to_centers_{:05}.png'.format(i // 2))
            pt.close()

    from pytential.symbolic.execution import build_matrix
    mat = build_matrix(queue, qbx, op, u_sym, auto_where=where)

    from pytential.symbolic.execution import GeometryCollection
    places = GeometryCollection(qbx, auto_where=where)
    source_discr = places.get_discretization(where[0])
    target_discr = places.get_discretization(where[1])

    assert mat.shape == (target_discr.nnodes, source_discr.nnodes)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
