from __future__ import division, print_function

__copyright__ = "Copyright (C) 2017 Matt Wala"

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

import pytest

import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from pytential import bind, sym

import logging
logger = logging.getLogger(__name__)

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh)

from functools import partial
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory


# {{{ discretization getters

def get_ellipse_with_ref_mean_curvature(cl_ctx, nelements, aspect=1):
    order = 4
    mesh = make_curve_mesh(
            partial(ellipse, aspect),
            np.linspace(0, 1, nelements+1),
            order)

    discr = Discretization(cl_ctx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(order))

    with cl.CommandQueue(cl_ctx) as queue:
        nodes = discr.nodes().get(queue=queue)

    a = 1
    b = 1/aspect
    t = np.arctan2(nodes[1] * aspect, nodes[0])

    return discr, a*b / ((a*np.sin(t))**2 + (b*np.cos(t))**2)**(3/2)


def get_torus_with_ref_mean_curvature(cl_ctx, h):
    order = 4
    r_inner = 1.0
    r_outer = 3.0

    from meshmode.mesh.generation import generate_torus
    mesh = generate_torus(r_outer, r_inner,
            n_outer=h, n_inner=h, order=order)
    discr = Discretization(cl_ctx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(order))

    with cl.CommandQueue(cl_ctx) as queue:
        nodes = discr.nodes().get(queue=queue)

    # copied from meshmode.mesh.generation.generate_torus
    a = r_outer
    b = r_inner

    u = np.arctan2(nodes[1], nodes[0])
    rvec = np.array([np.cos(u), np.sin(u), np.zeros_like(u)])
    rvec = np.sum(nodes * rvec, axis=0) - a
    cosv = np.cos(np.arctan2(nodes[2], rvec))

    return discr, (a + 2.0 * b * cosv) / (2 * b * (a + b * cosv))

# }}}


# {{{ test_mean_curvature

@pytest.mark.parametrize(("discr_name",
        "resolutions",
        "discr_and_ref_mean_curvature_getter"), [
    ("unit_circle", [16, 32, 64],
        get_ellipse_with_ref_mean_curvature),
    ("2-to-1 ellipse", [16, 32, 64],
        partial(get_ellipse_with_ref_mean_curvature, aspect=2)),
    ("torus", [8, 10, 12, 16],
        get_torus_with_ref_mean_curvature),
    ])
def test_mean_curvature(ctx_factory, discr_name, resolutions,
        discr_and_ref_mean_curvature_getter, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for r in resolutions:
        discr, ref_mean_curvature = \
                discr_and_ref_mean_curvature_getter(ctx, r)
        mean_curvature = bind(
            discr,
            sym.mean_curvature(discr.ambient_dim))(queue).get(queue)

        h = 1.0 / r
        h_error = la.norm(mean_curvature - ref_mean_curvature, np.inf)
        eoc.add_data_point(h, h_error)
    print(eoc)

    order = min([g.order for g in discr.groups])
    assert eoc.order_estimate() > order - 1.1

# }}}


# {{{ test_tangential_onb

def test_tangential_onb(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_torus
    mesh = generate_torus(5, 2, order=3)

    discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(3))

    tob = sym.tangential_onb(mesh.ambient_dim)
    nvecs = tob.shape[1]

    # make sure tangential_onb is mutually orthogonal and normalized
    orth_check = bind(discr, sym.make_obj_array([
        np.dot(tob[:, i], tob[:, j]) - (1 if i == j else 0)
        for i in range(nvecs) for j in range(nvecs)])
        )(queue)

    for i, orth_i in enumerate(orth_check):
        assert (cl.clmath.fabs(orth_i) < 1e-13).get().all()

    # make sure tangential_onb is orthogonal to normal
    orth_check = bind(discr, sym.make_obj_array([
        np.dot(tob[:, i], sym.normal(mesh.ambient_dim).as_vector())
        for i in range(nvecs)])
        )(queue)

    for i, orth_i in enumerate(orth_check):
        assert (cl.clmath.fabs(orth_i) < 1e-13).get().all()

# }}}


# {{{ test_expr_pickling

def test_expr_pickling():
    import pickle
    from sumpy.kernel import LaplaceKernel, AxisTargetDerivative

    ops_for_testing = [
        sym.d_dx(
            2,
            sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=-2)
        ),
        sym.D(
            AxisTargetDerivative(0, LaplaceKernel(2)),
            sym.var("sigma"),
            qbx_forced_limit=-2
        )
    ]

    for op in ops_for_testing:
        pickled_op = pickle.dumps(op)
        after_pickle_op = pickle.loads(pickled_op)

        assert op == after_pickle_op

# }}}


@pytest.mark.parametrize(("name", "source_discr_stage", "target_granularity"), [
    ("default", None, None),
    ("default-explicit", sym.QBX_SOURCE_STAGE1, sym.GRANULARITY_NODE),
    ("stage2", sym.QBX_SOURCE_STAGE2, sym.GRANULARITY_NODE),
    ("stage2-center", sym.QBX_SOURCE_STAGE2, sym.GRANULARITY_CENTER),
    ("quad", sym.QBX_SOURCE_QUAD_STAGE2, sym.GRANULARITY_NODE)
    ])
def test_interpolation(ctx_factory, name, source_discr_stage, target_granularity):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nelements = 32
    target_order = 7
    qbx_order = 4

    mesh = make_curve_mesh(starfish,
            np.linspace(0.0, 1.0, nelements + 1),
            target_order)
    discr = Discretization(ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx, _ = QBXLayerPotentialSource(discr,
            fine_order=4 * target_order,
            qbx_order=qbx_order,
            fmm_order=False).with_refinement()

    where = 'test-interpolation'
    from_dd = sym.DOFDescriptor(
            geometry=where,
            discr_stage=source_discr_stage,
            granularity=sym.GRANULARITY_NODE)
    to_dd = sym.DOFDescriptor(
            geometry=where,
            discr_stage=sym.QBX_SOURCE_QUAD_STAGE2,
            granularity=target_granularity)

    sigma_sym = sym.var("sigma")
    op_sym = sym.sin(sym.interp(from_dd, to_dd, sigma_sym))
    bound_op = bind(qbx, op_sym, auto_where=where)

    target_nodes = qbx.quad_stage2_density_discr.nodes().get(queue)
    if source_discr_stage == sym.QBX_SOURCE_STAGE2:
        source_nodes = qbx.stage2_density_discr.nodes().get(queue)
    elif source_discr_stage == sym.QBX_SOURCE_QUAD_STAGE2:
        source_nodes = target_nodes
    else:
        source_nodes = qbx.density_discr.nodes().get(queue)

    sigma_dev = cl.array.to_device(queue, la.norm(source_nodes, axis=0))
    sigma_target = np.sin(la.norm(target_nodes, axis=0))
    sigma_target_interp = bound_op(queue, sigma=sigma_dev).get(queue)

    if name in ('default', 'default-explicit', 'stage2', 'quad'):
        error = la.norm(sigma_target_interp - sigma_target) / la.norm(sigma_target)
        assert error < 1.0e-10
    elif name in ('stage2-center',):
        assert len(sigma_target_interp) == 2 * len(sigma_target)
    else:
        raise ValueError('unknown test case name: {}'.format(name))


# You can test individual routines by typing
# $ python test_symbolic.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
