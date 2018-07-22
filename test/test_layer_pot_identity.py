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
        NArmedStarfish,
        make_curve_mesh)
# from sumpy.visualization import FieldPlotter
from pytential import bind, sym, norm
from sumpy.kernel import LaplaceKernel, HelmholtzKernel

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


d1 = sym.Derivative()
d2 = sym.Derivative()


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


class StarfishGeometry(object):
    def __init__(self, n_arms=5, amplitude=0.25):
        self.n_arms = n_arms
        self.amplitude = amplitude

    @property
    def mesh_name(self):
        return "%d-starfish-%s" % (
                self.n_arms,
                self.amplitude)

    dim = 2

    resolutions = [30, 50, 70, 90]

    def get_mesh(self, nelements, target_order):
        return make_curve_mesh(
                NArmedStarfish(self.n_arms, self.amplitude),
                np.linspace(0, 1, nelements+1),
                target_order)


class WobblyCircleGeometry(object):
    dim = 2
    mesh_name = "wobbly-circle"

    resolutions = [2000, 3000, 4000]

    def get_mesh(self, resolution, target_order):
        return make_curve_mesh(
                WobblyCircle.random(30, seed=30),
                np.linspace(0, 1, resolution+1),
                target_order)


class SphereGeometry(object):
    mesh_name = "sphere"
    dim = 3

    resolutions = [0, 1]

    def get_mesh(self, resolution, tgt_order):
        return get_sphere_mesh(resolution, tgt_order)


class GreenExpr(object):
    zero_op_name = "green"

    def get_zero_op(self, kernel, **knl_kwargs):

        u_sym = sym.var("u")
        dn_u_sym = sym.var("dn_u")

        return (
            sym.S(kernel, dn_u_sym, qbx_forced_limit=-1, **knl_kwargs)
            - sym.D(kernel, u_sym, qbx_forced_limit="avg", **knl_kwargs)
            - 0.5*u_sym)

    order_drop = 0


class GradGreenExpr(object):
    zero_op_name = "grad_green"

    def get_zero_op(self, kernel, **knl_kwargs):
        d = kernel.dim
        u_sym = sym.var("u")
        grad_u_sym = sym.make_sym_mv("grad_u",  d)
        dn_u_sym = sym.var("dn_u")

        return (
                d1.resolve(d1.dnabla(d) * d1(sym.S(kernel, dn_u_sym,
                    qbx_forced_limit="avg", **knl_kwargs)))
                - d2.resolve(d2.dnabla(d) * d2(sym.D(kernel, u_sym,
                    qbx_forced_limit="avg", **knl_kwargs)))
                - 0.5*grad_u_sym
                )

    order_drop = 1


class ZeroCalderonExpr(object):
    zero_op_name = "calderon"

    def get_zero_op(self, kernel, **knl_kwargs):
        assert isinstance(kernel, LaplaceKernel)
        assert not knl_kwargs

        u_sym = sym.var("u")

        return (
                    -sym.Dp(kernel, sym.S(kernel, u_sym))
                    - 0.25*u_sym + sym.Sp(kernel, sym.Sp(kernel, u_sym))
                    )

    order_drop = 1


class StaticTestCase(object):
    def check(self):
        pass


class StarfishGreenTest(StaticTestCase):
    expr = GreenExpr()
    geometry = StarfishGeometry()
    k = 0
    qbx_order = 5
    fmm_order = 15

    resolutions = [30, 50]

    _expansion_stick_out_factor = 0.5

    fmm_backend = "fmmlib"


class WobblyCircleGreenTest(StaticTestCase):
    expr = GreenExpr()
    geometry = WobblyCircleGeometry()
    k = 0
    qbx_order = 3
    fmm_order = 10

    _expansion_stick_out_factor = 0.5

    fmm_backend = "sumpy"


class SphereGreenTest(StaticTestCase):
    expr = GreenExpr()
    geometry = SphereGeometry()
    k = 0
    qbx_order = 3
    fmm_order = 10

    resolutions = [0, 1]

    _expansion_stick_out_factor = 0.5

    fmm_backend = "fmmlib"


class DynamicTestCase(object):
    fmm_backend = "sumpy"

    def __init__(self, geometry, expr, k, fmm_backend="sumpy"):
        self.geometry = geometry
        self.expr = expr
        self.k = k
        self.qbx_order = 5 if geometry.dim == 2 else 3
        self.fmm_backend = fmm_backend

        if geometry.dim == 2:
            order_bump = 15
        elif geometry.dim == 3:
            order_bump = 8

        self.fmm_order = self.qbx_order + order_bump

    def check(self):
        if (self.geometry.mesh_name == "sphere"
                and self.k != 0
                and self.fmm_backend == "sumpy"):
            raise ValueError("both direct eval and generating the FMM kernels "
                    "are too slow")

        if (self.geometry.mesh_name == "sphere"
                and self.expr.zero_op_name == "green_grad"):
            raise ValueError("does not achieve sufficient precision")


# {{{ integral identity tester


@pytest.mark.slowtest
@pytest.mark.parametrize("case", [
        DynamicTestCase(SphereGeometry(), GreenExpr(), 0),
])
def test_identity_convergence_slow(ctx_getter, case):
    test_identity_convergence(ctx_getter, case)


@pytest.mark.parametrize("case", [
        # 2d
        DynamicTestCase(StarfishGeometry(), GreenExpr(), 0),
        DynamicTestCase(StarfishGeometry(), GreenExpr(), 1.2),
        DynamicTestCase(StarfishGeometry(), GradGreenExpr(), 0),
        DynamicTestCase(StarfishGeometry(), GradGreenExpr(), 1.2),
        DynamicTestCase(StarfishGeometry(), ZeroCalderonExpr(), 0),
        DynamicTestCase(StarfishGeometry(), GreenExpr(), 0, fmm_backend="fmmlib"),
        DynamicTestCase(StarfishGeometry(), GreenExpr(), 1.2, fmm_backend="fmmlib"),
        # 3d
        DynamicTestCase(SphereGeometry(), GreenExpr(), 0, fmm_backend="fmmlib"),
        DynamicTestCase(SphereGeometry(), GreenExpr(), 1.2, fmm_backend="fmmlib")
])
def test_identity_convergence(ctx_getter,  case, visualize=False):
    logging.basicConfig(level=logging.INFO)

    case.check()

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    target_order = 8

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for resolution in (
            getattr(case, "resolutions", None)
            or case.geometry.resolutions
            ):
        mesh = case.geometry.get_mesh(resolution, target_order)
        if mesh is None:
            break

        d = mesh.ambient_dim
        k = case.k

        lap_k_sym = LaplaceKernel(d)
        if k == 0:
            k_sym = lap_k_sym
            knl_kwargs = {}
        else:
            k_sym = HelmholtzKernel(d)
            knl_kwargs = {"k": sym.var("k")}

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        from pytential.qbx import QBXLayerPotentialSource
        pre_density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))

        refiner_extra_kwargs = {}

        if case.k != 0:
            refiner_extra_kwargs["kernel_length_scale"] = 5/case.k

        qbx, _ = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                case.qbx_order,
                fmm_order=case.fmm_order,
                fmm_backend=case.fmm_backend,
                _expansions_in_tree_have_extent=True,
                _expansion_stick_out_factor=getattr(
                    case, "_expansion_stick_out_factor", 0),
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
            elif d == 3:
                center = np.array([3, 1, 2])
                diff = nodes_host - center[:, np.newaxis]
                r = la.norm(diff, axis=0)
                u = np.exp(1j*k*r) / r
                grad_u = diff * (1j*k*u/r - u/r**2)
            else:
                raise ValueError("invalid dim")
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

        key = (case.qbx_order, case.geometry.mesh_name, resolution,
                case.expr.zero_op_name)

        bound_op = bind(qbx, case.expr.get_zero_op(k_sym, **knl_kwargs))
        error = bound_op(
                queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=case.k)
        if 0:
            pt.plot(error)
            pt.show()

        linf_error_norm = norm(density_discr, queue, error, p=np.inf)
        print("--->", key, linf_error_norm)

        eoc_rec.add_data_point(qbx.h_max, linf_error_norm)

        if visualize:
            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(queue, density_discr, target_order)

            bdry_normals = bind(density_discr, sym.normal(mesh.ambient_dim))(queue)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
                ("u", u_dev),
                ("bdry_normals", bdry_normals),
                ("error", error),
                ])

    print(eoc_rec)
    tgt_order = case.qbx_order - case.expr.order_drop
    assert eoc_rec.order_estimate() > tgt_order - 1.6

# }}}


# You can test individual routines by typing
# $ python test_layer_pot_identity.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
