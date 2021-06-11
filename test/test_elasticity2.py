__copyright__ = "Copyright (C) 2017 Natalie Beams"

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

from arraycontext import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytools.obj_array import make_obj_array

from pytential import bind, sym
from pytential import GeometryCollection

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)


# {{{ test_exterior_stokes

def run_exterior_stokes(actx, *,
        ambient_dim, target_order, qbx_order, resolution,
        fmm_order=None,    # FIXME: FMM is slower than direct evaluation
        source_ovsmp=None,
        radius=1.5,
        mu=1.0,
        nu=0.4,
        verbose=False,
        method="naive",

        _target_association_tolerance=0.05,
        _expansions_in_tree_have_extent=True):
    # {{{ geometry

    if source_ovsmp is None:
        source_ovsmp = 4 if ambient_dim == 2 else 8

    places = {}

    if ambient_dim == 3:
        from meshmode.mesh.generation import generate_surface_of_revolution
        nheight = 10*(resolution + 1)
        nangle = 10*(resolution + 1)
        mesh = generate_surface_of_revolution(
            lambda x, y: np.ones(x.shape),
            np.linspace(-1.5, -0.5, nheight),
            np.linspace(0, 2*np.pi, nangle, endpoint=False), target_order)
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    mesh_vertices_image = mesh.vertices.copy()
    mesh_vertices_image[2, :] *= -1
    mesh_grp, = mesh.groups
    mesh_grp_nodes_image = mesh_grp.nodes.copy()
    mesh_grp_nodes_image[2, :] *= -1
    mesh_grp_image = mesh_grp.copy(nodes=mesh_grp_nodes_image)
    mesh_image = mesh.copy(vertices=mesh_vertices_image, groups=[mesh_grp_image])

    pre_density_discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))
    pre_density_discr_image = pre_density_discr.copy(mesh=mesh_image)

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(pre_density_discr,
            fine_order=source_ovsmp * target_order,
            qbx_order=qbx_order,
            fmm_order=fmm_order,
            _max_leaf_refine_weight=64,
            target_association_tolerance=_target_association_tolerance,
            _expansions_in_tree_have_extent=_expansions_in_tree_have_extent)
    places["source"] = qbx
    places["target"] = qbx
    places["target_image"] = qbx.copy(density_discr=pre_density_discr_image)

    from extra_int_eq_data import make_source_and_target_points
    point_source, point_target = make_source_and_target_points(
            actx,
            side=+1,
            inner_radius=0.5 * radius,
            outer_radius=2.0 * radius,
            ambient_dim=ambient_dim,
            )
    places["point_source"] = point_source
    places["point_target"] = point_target

    places = GeometryCollection(places, auto_where="source")

    density_discr = places.get_discretization("source")
    logger.info("ndofs:     %d", density_discr.ndofs)
    logger.info("nelements: %d", density_discr.mesh.nelements)

    # }}}

    # {{{ symbolic

    sym_normal = sym.make_sym_vector("normal", ambient_dim)
    sym_mu = sym.var("mu")
    sym_nu = sym.var("nu")

    if ambient_dim == 3:
        from pytential.symbolic.elasticity import MindlinOperator
        op = MindlinOperator(method=method, mu_sym=sym_mu, nu_sym=sym_nu)
    else:
        raise AssertionError()

    sym_sigma = op.get_density_var("sigma")
    sym_bc = op.get_density_var("bc")
    sym_rhs = sym_bc
    sym_source_pot = op.free_space_op.stokeslet.apply(sym_sigma, qbx_forced_limit=None)

    # }}}

    # {{{ boundary conditions

    normal = bind(places, sym.normal(ambient_dim).as_vector())(actx)

    np.random.seed(42)
    charges = make_obj_array([
        actx.from_numpy(np.random.randn(point_source.ndofs))
        for _ in range(ambient_dim)
        ])

    bc_context = {"nu": nu}
    op_context = {"mu": mu, "normal": normal, "nu": nu}
    direct_context = {"mu": mu, "nu": nu}

    bc_op = bind(places, sym_source_pot,
            auto_where=("point_source", "source"))
    bc = bc_op(actx, sigma=charges, **direct_context)

    rhs = bind(places, sym_rhs)(actx, bc=bc, **bc_context)

    sym_op = op.free_space_operator(sym_sigma, normal=sym_normal, qbx_forced_limit=1)
    sym_op_image = op.operator(sym_sigma, normal=sym_normal, qbx_forced_limit=2)
    bound_op = bind(places, sym_op, auto_where=("source", "target"))
    bound_op_image = bind(places, sym_op_image, auto_where=("source", "target_image"))
    # }}}

    def print_timing_data(timings, name):
        result = {k: 0 for k in list(timings.values())[0].keys()}
        total = 0
        for k, timing in timings.items():
            for k, v in timing.items():
                result[k] += v['wall_elapsed']
                total += v['wall_elapsed']
        result['total'] = total
        print(f"{name}={result}")

    fmm_timing_data = {}
    v1 = bound_op.eval({"sigma": rhs, **op_context}, array_context=actx,
            timing_data=fmm_timing_data)
    print_timing_data(fmm_timing_data, method)
    
    fmm_timing_data = {}
    v2 = bound_op_image.eval({"sigma": rhs, **op_context}, array_context=actx,
            timing_data=fmm_timing_data)
    print_timing_data(fmm_timing_data, method)
    
    h_max = bind(places, sym.h_max(ambient_dim))(actx)

    return h_max, v1 + v2


@pytest.mark.parametrize("method, nu", [
    ("biharmonic", 0.4),
    ("laplace", 0.4),
    ])
def test_exterior_stokes(ctx_factory, method, nu, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    def rnorm2(x, y):
        y_norm = actx.np.linalg.norm(y.dot(y), ord=2)
        if y_norm < 1.0e-14:
            y_norm = 1.0

        d = x - y
        return actx.np.linalg.norm(d.dot(d), ord=2) / y_norm

    target_order = 3
    qbx_order = 3
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    actx = PyOpenCLArrayContext(queue)
    fmm_order = 6
    resolution = 0
    ambient_dim = 3

    kwargs = dict(
        actx=actx,
        ambient_dim=ambient_dim,
        target_order=target_order,
        qbx_order=qbx_order,
        resolution=resolution,
        verbose=verbose,
        nu=nu,
        method=method,
    )

    h_max, fmm_result = run_exterior_stokes(fmm_order=fmm_order, **kwargs)
    _, direct_result = run_exterior_stokes(fmm_order=False, **kwargs)

    v_error = rnorm2(fmm_result, direct_result)
    print(v_error)

    assert v_error < 1e-5


# }}}


# You can test individual routines by typing
# $ python test_stokes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
