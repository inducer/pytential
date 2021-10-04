__copyright__ = "Copyright (C) 2021 Alexandru Fikl"

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

from dataclasses import dataclass
import pytest

import numpy as np

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

from pytential import bind, sym
from pytential.symbolic.pde.beltrami import (
        LaplaceBeltramiOperator, YukawaBeltramiOperator)

import extra_int_eq_data as eid
import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ solutions

def evaluate_spharm(actx, discr, m: int, n: int) -> np.ndarray:
    assert discr.ambient_dim == 3

    # {{{ get spherical coordinates

    from arraycontext import thaw
    x, y, z = thaw(discr.nodes(), actx)

    theta = actx.np.arctan2(actx.np.sqrt(x**2 + y**2), z)
    phi = actx.np.arctan2(y, x)

    # }}}

    # {{{ evaluate Y^m_n

    from scipy.special import sph_harm      # pylint: disable=no-name-in-module
    y_mn = []
    for gtheta, gphi in zip(theta, phi):
        result = sph_harm(m, n, actx.to_numpy(gphi), actx.to_numpy(gtheta))

        y_mn.append(actx.from_numpy(result.real.copy()))

    # }}}

    return type(x)(actx, tuple(y_mn))


@dataclass(frozen=True)
class LaplaceBeltramiSolution:
    name: str = "laplace"
    radius: float = 1.0
    m: int = 1
    n: int = 3

    @property
    def eig(self):
        # NOTE: eigenvalue of the -Laplacian on the sphere
        return self.n * (self.n + 1) / self.radius ** 2

    @property
    def context(self):
        return {}

    def source(self, actx, discr):
        return self.eig * evaluate_spharm(actx, discr, self.m, self.n)

    def exact(self, actx, discr):
        return evaluate_spharm(actx, discr, self.m, self.n)


class YukawaBeltramiSolution(LaplaceBeltramiSolution):
    name: str = "yukawa"
    k: float = 1.0

    @property
    def context(self):
        return {"k": self.k}

    def source(self, actx, discr):
        return (self.eig + self.k**2) * evaluate_spharm(actx, discr, self.m, self.n)

# }}}


# {{{ test_beltrami_convergence

@pytest.mark.slowtest
@pytest.mark.parametrize(("operator", "solution"), [
    (LaplaceBeltramiOperator(3, precond="left"), LaplaceBeltramiSolution()),
    (LaplaceBeltramiOperator(3, precond="right"), LaplaceBeltramiSolution()),
    (YukawaBeltramiOperator(3, precond="left"), YukawaBeltramiSolution()),
    ])
def test_beltrami_convergence(actx_factory, operator, solution, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    radius = 1
    case = eid.SphereTestCase(
            target_order=5,
            qbx_order=5,
            source_ovsmp=8,
            fmm_order=False, fmm_tol=None, fmm_backend=None,
            radius=radius,
            resolutions=[0, 1, 2]
            )
    logger.info("\n%s", case)

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for resolution in case.resolutions:
        # {{{ geometry

        from pytential import GeometryCollection
        qbx = case.get_layer_potential(actx, resolution, case.target_order)
        places = GeometryCollection(qbx, auto_where=case.name)

        density_discr = places.get_discretization(case.name)
        logger.info("ndofs:     %d", density_discr.ndofs)
        logger.info("nelements: %d", density_discr.mesh.nelements)

        # }}}

        # {{{ symbolic

        sym_sigma = operator.get_density_var("sigma")
        sym_b = operator.get_density_var("b")

        sym_rhs = operator.prepare_rhs(sym_b)
        sym_result = operator.prepare_solution(sym_sigma)

        sym_op = operator.operator(sym_sigma, mean_curvature=1/radius)

        # }}}

        # {{{ solve

        scipy_op = bind(places, sym_op).scipy_op(
                actx, "sigma", operator.dtype, **solution.context)
        rhs = bind(places, sym_rhs)(
                actx, b=solution.source(actx, density_discr),
                **solution.context)

        from pytential.solve import gmres
        result = gmres(
                scipy_op, rhs,
                x0=rhs,
                tol=1.0e-7,
                progress=visualize,
                stall_iterations=0,
                hard_failure=True)

        result = bind(places, sym_result)(
                actx, sigma=actx.np.real(result.solution),
                **solution.context)
        ref_result = solution.exact(actx, density_discr)

        # }}}

        from pytential import norm
        h_max = actx.to_numpy(
                bind(places, sym.h_max(places.ambient_dim))(actx)
                )
        error = actx.to_numpy(
                norm(density_discr, result - ref_result, p=2)
                / norm(density_discr, ref_result, p=2)
                )

        eoc.add_data_point(h_max, error)
        logger.info("resolution %3d h_max %.5e rel_error %.5e",
                resolution, h_max, error)

        if not visualize:
            continue

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, density_discr, case.target_order)

        filename = f"beltrami_{case.name}_{solution.name}_{resolution}.vtu"
        vis.write_vtk_file(filename, [
            ("result", result),
            ("ref_result", ref_result),
            ("error", result - ref_result)
            ], overwrite=True)

    logger.info("\n%s", eoc.pretty_print(
        abscissa_format="%.8e",
        error_format="%.8e",
        eoc_format="%.2f"))

    # NOTE: expected order is `order - 2` because the formulation has two
    # additional target derivatives on the kernel
    order = min(case.target_order, case.qbx_order)
    assert eoc.order_estimate() > order - 2.5

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
