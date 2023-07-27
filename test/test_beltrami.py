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
from meshmode.dof_array import DOFArray

from pytential import bind, sym
from pytential.symbolic.pde.beltrami import (
        LaplaceBeltramiOperator, YukawaBeltramiOperator)

import extra_int_eq_data as eid
import logging
logger = logging.getLogger(__name__)

from pytential.utils import (  # noqa: F401
        pytest_teardown_function as teardown_function)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ solutions

def evaluate_circle_eigf(actx, discr, k: int) -> np.ndarray:
    assert discr.ambient_dim == 2

    # {{{ get polar coordinates

    x, y = actx.thaw(discr.nodes())
    theta = actx.np.arctan2(y, x)

    # }}}

    return actx.np.exp(-1j * k * theta)


def evaluate_sphere_eigf(actx, discr, m: int, n: int) -> DOFArray:
    assert discr.ambient_dim == 3

    # {{{ get spherical coordinates

    x, y, z = actx.thaw(discr.nodes())

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

    return DOFArray(actx, tuple(y_mn))


@dataclass(frozen=True)
class LaplaceBeltramiSolution:
    name: str = "laplace"
    dim: int = 3
    radius: float = 1.0
    m: int = 1
    n: int = 3

    @property
    def eigenvalue(self):
        # NOTE: eigenvalue of the -Laplacian on a ball of radius *radius*
        if self.dim == 2:
            return self.n**2 / self.radius ** 2
        elif self.dim == 3:
            return self.n * (self.n + 1) / self.radius ** 2
        else:
            raise ValueError(f"unsupported dimension: {self.dim}")

    def eigenfunction(self, actx, discr):
        if self.dim == 2:
            f = evaluate_circle_eigf(actx, discr, k=self.n)
        elif self.dim == 3:
            f = evaluate_sphere_eigf(actx, discr, self.m, self.n)
        else:
            raise ValueError(f"unsupported dimension: {self.dim}")

        return actx.np.real(f)

    @property
    def context(self):
        return {}

    def source(self, actx, discr):
        return self.eigenvalue * self.eigenfunction(actx, discr)

    def exact(self, actx, discr):
        return self.eigenfunction(actx, discr)


@dataclass(frozen=True)
class YukawaBeltramiSolution(LaplaceBeltramiSolution):
    name: str = "yukawa"
    k: float = 2.0

    @property
    def context(self):
        return {"k": self.k}

    def source(self, actx, discr):
        return (self.eigenvalue + self.k**2) * self.eigenfunction(actx, discr)

# }}}


# {{{ test_beltrami_convergence

@pytest.mark.parametrize(("operator", "solution"), [
    (LaplaceBeltramiOperator(2, precond="left"), LaplaceBeltramiSolution(dim=2)),
    (LaplaceBeltramiOperator(2, precond="right"), LaplaceBeltramiSolution(dim=2)),
    (YukawaBeltramiOperator(2, precond="left"), YukawaBeltramiSolution(dim=2)),
    (YukawaBeltramiOperator(2, precond="right"), YukawaBeltramiSolution(dim=2)),
    pytest.param(
        LaplaceBeltramiOperator(3, precond="left"), LaplaceBeltramiSolution(),
        marks=pytest.mark.slowtest),
    pytest.param(
        LaplaceBeltramiOperator(3, precond="right"), LaplaceBeltramiSolution(),
        marks=pytest.mark.slowtest),
    pytest.param(
        YukawaBeltramiOperator(3, precond="left"), YukawaBeltramiSolution(),
        marks=pytest.mark.slowtest),
    pytest.param(
        YukawaBeltramiOperator(3, precond="right"), YukawaBeltramiSolution(),
        marks=pytest.mark.slowtest),
    ])
def test_beltrami_convergence(actx_factory, operator, solution, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    from dataclasses import replace
    radius = 1.5
    solution = replace(solution, radius=radius)

    if operator.ambient_dim == 2:
        case = eid.CircleTestCase(
                target_order=5,
                qbx_order=5,
                source_ovsmp=4,
                resolutions=[32, 64, 96, 128],
                # FIXME: FMM should not be slower!
                fmm_order=False, fmm_backend=None,
                radius=radius
                )
    elif operator.ambient_dim == 3:
        case = eid.SphereTestCase(
                target_order=5,
                qbx_order=4,
                source_ovsmp=8,
                # FIXME: FMM should not be slower!
                fmm_order=False, fmm_tol=None, fmm_backend=None,
                radius=radius
                )
    else:
        raise ValueError(f"unsupported dimension: {operator.ambient_dim}")

    logger.info("\n%s", case)
    logger.info("\n%s", solution)

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

        from pytential.linalg.gmres import gmres
        result = gmres(
                scipy_op, rhs,
                x0=rhs,
                tol=1.0e-7,
                progress=visualize,
                stall_iterations=0,
                hard_failure=True)

        result = bind(places, sym.real(sym_result))(
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
