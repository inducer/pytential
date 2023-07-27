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

import pytest

import numpy as np
import numpy.linalg as la

from arraycontext import flatten, unflatten
from pytential import bind, sym, norm
from pytential import GeometryCollection
from sumpy.kernel import LaplaceKernel, HelmholtzKernel
# from sumpy.visualization import FieldPlotter

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import extra_int_eq_data as ied
import logging
logger = logging.getLogger(__name__)

from pytential.utils import (  # noqa: F401
        pytest_teardown_function as teardown_function)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])

d1 = sym.Derivative()
d2 = sym.Derivative()


class SphereTestCase(ied.SphereTestCase):
    resolutions = [0, 1]


class QuadSphereTestCase(ied.QuadSpheroidTestCase):
    name = "sphere-quad"
    aspect_ratio = 1
    resolutions = [0, 1]


class GreenExpr:
    zero_op_name = "green"
    order_drop = 0

    def get_zero_op(self, kernel, **knl_kwargs):
        u_sym = sym.var("u")
        dn_u_sym = sym.var("dn_u")

        return (
            sym.S(kernel, dn_u_sym, qbx_forced_limit=-1, **knl_kwargs)
            - sym.D(kernel, u_sym, qbx_forced_limit="avg", **knl_kwargs)
            - 0.5*u_sym)


class GradGreenExpr:
    zero_op_name = "grad_green"
    order_drop = 1

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
                ).as_vector()


class ZeroCalderonExpr:
    zero_op_name = "calderon"
    order_drop = 1

    def get_zero_op(self, kernel, **knl_kwargs):
        assert isinstance(kernel, LaplaceKernel)
        assert not knl_kwargs

        u_sym = sym.var("u")

        from functools import partial
        S = partial(sym.S, qbx_forced_limit=+1)
        Dp = partial(sym.Dp, qbx_forced_limit="avg")
        Sp = partial(sym.Sp, qbx_forced_limit="avg")

        return (
                -Dp(kernel, S(kernel, u_sym))
                - 0.25*u_sym + Sp(kernel, Sp(kernel, u_sym))
                )


class StaticTestCase:
    def check(self):
        pass


class StarfishGreenTest(StaticTestCase):
    expr = GreenExpr()
    geometry = ied.StarfishTestCase()
    k = 0
    qbx_order = 5
    fmm_order = 15

    resolutions = [30, 50]

    _expansion_stick_out_factor = 0.5

    fmm_backend = "fmmlib"


class WobblyCircleGreenTest(StaticTestCase):
    expr = GreenExpr()
    geometry = ied.WobbleCircleTestCase()
    k = 0
    qbx_order = 3
    fmm_order = 10

    _expansion_stick_out_factor = 0.5

    fmm_backend = "sumpy"


class SphereGreenTest(StaticTestCase):
    expr = GreenExpr()
    geometry = SphereTestCase()
    k = 0
    qbx_order = 3
    fmm_order = 10

    resolutions = [0, 1]

    _expansion_stick_out_factor = 0.5

    fmm_backend = "fmmlib"


class DynamicTestCase:
    fmm_backend = "sumpy"

    def __init__(self, geometry, expr, k, fmm_backend="sumpy", fmm_order=None):
        self.geometry = geometry
        self.expr = expr
        self.k = k
        self.qbx_order = 5 if geometry.ambient_dim == 2 else 3
        self.fmm_backend = fmm_backend

        if geometry.ambient_dim == 2:
            order_bump = 15
        elif geometry.ambient_dim == 3:
            order_bump = 8
        else:
            raise ValueError(f"unsupported dimension: {geometry.ambient_dim}")

        if fmm_order is None:
            self.fmm_order = self.qbx_order + order_bump
        else:
            self.fmm_order = fmm_order

    @property
    def resolutions(self):
        return self.geometry.resolutions

    def check(self):
        from warnings import warn
        if (self.geometry.name == "sphere"
                and self.k != 0
                and self.fmm_backend == "sumpy"):
            warn("both direct eval and generating the FMM kernels are too slow")

        if (self.geometry.name == "sphere"
                and self.expr.zero_op_name == "grad_green"):
            warn("does not achieve sufficient precision")


# {{{ integral identity tester


@pytest.mark.slowtest
@pytest.mark.parametrize("case", [
        DynamicTestCase(SphereTestCase(), GreenExpr(), 0),
])
def test_identity_convergence_slow(actx_factory, case):
    test_identity_convergence(actx_factory, case)


@pytest.mark.parametrize("case", [
        # 2d
        DynamicTestCase(ied.StarfishTestCase(), GreenExpr(), 0),
        DynamicTestCase(ied.StarfishTestCase(), GreenExpr(), 1.2),
        DynamicTestCase(ied.StarfishTestCase(), GradGreenExpr(), 0),
        DynamicTestCase(ied.StarfishTestCase(), GradGreenExpr(), 1.2),
        DynamicTestCase(ied.StarfishTestCase(), ZeroCalderonExpr(), 0),
        # test target derivatives with direct evaluation
        DynamicTestCase(
            ied.StarfishTestCase(), ZeroCalderonExpr(), 0, fmm_order=False),
        DynamicTestCase(
            ied.StarfishTestCase(), GreenExpr(), 0, fmm_backend="fmmlib"),
        DynamicTestCase(
            ied.StarfishTestCase(), GreenExpr(), 1.2, fmm_backend="fmmlib"),
        # 3d
        DynamicTestCase(SphereTestCase(), GreenExpr(), 0, fmm_backend="fmmlib"),
        DynamicTestCase(SphereTestCase(), GreenExpr(), 1.2, fmm_backend="fmmlib"),
        DynamicTestCase(QuadSphereTestCase(), GreenExpr(), 0, fmm_backend="fmmlib"),
])
def test_identity_convergence(actx_factory,  case, visualize=False):
    if case.fmm_backend == "fmmlib":
        pytest.importorskip("pyfmmlib")

    logging.basicConfig(level=logging.INFO)

    case.check()

    actx = actx_factory()

    target_order = 8

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for resolution in case.resolutions:
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
                InterpolatoryQuadratureGroupFactory
        pre_density_discr = Discretization(
                actx, mesh, InterpolatoryQuadratureGroupFactory(target_order))

        from pytential.qbx import QBXLayerPotentialSource
        qbx = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                case.qbx_order,
                fmm_order=case.fmm_order,
                fmm_backend=case.fmm_backend,
                target_association_tolerance=1.0e-1,
                _expansions_in_tree_have_extent=True,
                _expansion_stick_out_factor=getattr(
                    case, "_expansion_stick_out_factor", 0),
                )
        places = GeometryCollection(qbx)

        from pytential.qbx.refinement import refine_geometry_collection
        kernel_length_scale = 5 / case.k if case.k else None
        places = refine_geometry_collection(places,
                kernel_length_scale=kernel_length_scale)

        # {{{ compute values of a solution to the PDE

        density_discr = places.get_discretization(places.auto_source.geometry)
        ambient_dim = places.ambient_dim

        nodes_host = actx.to_numpy(
                flatten(density_discr.nodes(), actx)
                ).reshape(ambient_dim, -1)
        normal = bind(places, sym.normal(d))(actx).as_vector(object)
        normal_host = actx.to_numpy(flatten(normal, actx)).reshape(ambient_dim, -1)

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
                raise AssertionError()

        dn_u = 0
        for i in range(d):
            dn_u = dn_u + normal_host[i]*grad_u[i]

        # }}}

        u_dev = unflatten(
                normal[0], actx.from_numpy(u), actx, strict=False)
        dn_u_dev = unflatten(
                normal[0], actx.from_numpy(dn_u), actx, strict=False)
        grad_u_dev = unflatten(
                normal, actx.from_numpy(grad_u.ravel()), actx, strict=False)

        key = (case.qbx_order, case.geometry.name, resolution,
                case.expr.zero_op_name)

        bound_op = bind(places, case.expr.get_zero_op(k_sym, **knl_kwargs))
        error = bound_op(
                actx, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=case.k)

        linf_error_norm = actx.to_numpy(norm(density_discr, error, p=np.inf))
        logger.info("---> key %s error %.5e", key, linf_error_norm)

        h_max = actx.to_numpy(
                bind(places, sym.h_max(qbx.ambient_dim))(actx)
                )
        eoc_rec.add_data_point(h_max, linf_error_norm)

        if visualize:
            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(actx, density_discr, target_order)

            bdry_normals = bind(places, sym.normal(mesh.ambient_dim))(actx)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
                ("u", u_dev),
                ("bdry_normals", bdry_normals),
                ("error", error),
                ])

    logger.info("\n%s", eoc_rec)
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
