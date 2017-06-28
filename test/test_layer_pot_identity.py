from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2013-2017 Andreas Kloeckner"

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
import pyopencl.clmath  # noqa
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh)
# from sumpy.visualization import FieldPlotter
from pytential import bind, sym, norm

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


# {{{ integral identity tester

d1 = sym.Derivative()
d2 = sym.Derivative()


def get_starfish_mesh(refinement_increment, target_order):
    nelements = [30, 50, 70][refinement_increment]
    return make_curve_mesh(starfish,
                np.linspace(0, 1, nelements+1),
                target_order)


def get_wobbly_circle_mesh(refinement_increment, target_order):
    nelements = [3000, 5000, 7000][refinement_increment]
    return make_curve_mesh(WobblyCircle.random(30, seed=30),
                np.linspace(0, 1, nelements+1),
                target_order)


def get_sphere_mesh(refinement_increment, target_order):
    from meshmode.mesh.generation import generate_icosphere
    mesh = generate_icosphere(1, target_order)
    from meshmode.mesh.refinement import Refiner

    refiner = Refiner(mesh)
    for i in range(refinement_increment):
        flags = np.ones(mesh.nelements, dtype=bool)
        refiner.refine(flags)
        mesh = refiner.get_current_mesh()

    return mesh


@pytest.mark.parametrize(("mesh_name", "mesh_getter", "qbx_order"), [
    #("circle", partial(ellipse, 1)),
    #("3-to-1 ellipse", partial(ellipse, 3)),
    ("starfish", get_starfish_mesh, 5),
    ("sphere", get_sphere_mesh, 3),
    ])
@pytest.mark.parametrize(("zero_op_name", "k"), [
    ("green", 0),
    ("green", 1.2),
    ("green_grad", 0),
    ("green_grad", 1.2),
    ("zero_calderon", 0),
    ])
# sample invocation to copy and paste:
# 'test_identities(cl._csc, "green", "starfish", get_starfish_mesh, 4, 0)'
def test_identities(ctx_getter, zero_op_name, mesh_name, mesh_getter, qbx_order, k):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    if mesh_name == "sphere" and k != 0:
        pytest.skip("both direct eval and generating the FMM kernels are too slow")

    if mesh_name == "sphere" and zero_op_name == "green_grad":
        pytest.skip("does not achieve sufficient precision")

    target_order = 8

    order_table = {
            "green": qbx_order,
            "green_grad": qbx_order-1,
            "zero_calderon": qbx_order-1,
            }

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for refinement_increment in [0, 1, 2]:
        mesh = mesh_getter(refinement_increment, target_order)
        if mesh is None:
            break

        d = mesh.ambient_dim

        u_sym = sym.var("u")
        grad_u_sym = sym.make_sym_mv("grad_u",  d)
        dn_u_sym = sym.var("dn_u")

        from sumpy.kernel import LaplaceKernel, HelmholtzKernel
        lap_k_sym = LaplaceKernel(d)
        if k == 0:
            k_sym = lap_k_sym
            knl_kwargs = {}
        else:
            k_sym = HelmholtzKernel(d)
            knl_kwargs = {"k": sym.var("k")}

        zero_op_table = {
                "green":
                sym.S(k_sym, dn_u_sym, qbx_forced_limit=-1, **knl_kwargs)
                - sym.D(k_sym, u_sym, qbx_forced_limit="avg", **knl_kwargs)
                - 0.5*u_sym,

                "green_grad":
                d1.resolve(d1.dnabla(d) * d1(sym.S(k_sym, dn_u_sym,
                    qbx_forced_limit="avg", **knl_kwargs)))
                - d2.resolve(d2.dnabla(d) * d2(sym.D(k_sym, u_sym,
                    qbx_forced_limit="avg", **knl_kwargs)))
                - 0.5*grad_u_sym,

                # only for k==0:
                "zero_calderon":
                -sym.Dp(lap_k_sym, sym.S(lap_k_sym, u_sym))
                - 0.25*u_sym + sym.Sp(lap_k_sym, sym.Sp(lap_k_sym, u_sym))
                }

        zero_op = zero_op_table[zero_op_name]

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        from pytential.qbx import QBXLayerPotentialSource
        pre_density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))

        if d == 2:
            order_bump = 15
        elif d == 3:
            order_bump = 8

        refiner_extra_kwargs = {}

        if k != 0:
            refiner_extra_kwargs["kernel_length_scale"] = 5/k

        qbx, _ = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                qbx_order, fmm_order=qbx_order + order_bump
                ).with_refinement(**refiner_extra_kwargs)

        density_discr = qbx.density_discr

        # {{{ compute values of a solution to the PDE

        nodes_host = density_discr.nodes().get(queue)
        normal = bind(density_discr, sym.normal(d))(queue).as_vector(np.object)
        normal_host = [normal[j].get() for j in range(d)]

        if k != 0:
            if d == 2:
                angle = 0.3
                wave_vec = np.array([np.cos(angle), np.sin(angle)])
                u = np.exp(1j*k*np.tensordot(wave_vec, nodes_host, axes=1))
                grad_u = 1j*k*wave_vec[:, np.newaxis]*u
            else:
                center = np.array([3, 1, 2])
                diff = nodes_host - center[:, np.newaxis]
                r = la.norm(diff, axis=0)
                u = np.exp(1j*k*r) / r
                grad_u = diff * (1j*k*u/r - u/r**2)
        else:
            center = np.array([3, 1, 2])[:d]
            diff = nodes_host - center[:, np.newaxis]
            dist_squared = np.sum(diff**2, axis=0)
            dist = np.sqrt(dist_squared)
            if d == 2:
                u = np.log(dist)
                grad_u = diff/dist_squared
            elif d == 3:
                u = 1/dist
                grad_u = -diff/dist**3
            else:
                assert False

        dn_u = 0
        for i in range(d):
            dn_u = dn_u + normal_host[i]*grad_u[i]

        # }}}

        u_dev = cl.array.to_device(queue, u)
        dn_u_dev = cl.array.to_device(queue, dn_u)
        grad_u_dev = cl.array.to_device(queue, grad_u)

        key = (qbx_order, mesh_name, refinement_increment, zero_op_name)

        bound_op = bind(qbx, zero_op)
        error = bound_op(
                queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=k)
        if 0:
            pt.plot(error)
            pt.show()

        l2_error_norm = norm(density_discr, queue, error)
        print(key, l2_error_norm)

        eoc_rec.add_data_point(qbx.h_max, l2_error_norm)

    print(eoc_rec)
    tgt_order = order_table[zero_op_name]
    assert eoc_rec.order_estimate() > tgt_order - 1.3

# }}}


# You can test individual routines by typing
# $ python test_layer_pot_identity.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
