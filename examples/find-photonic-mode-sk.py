from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2015 Shidong Jiang, Andreas Kloeckner
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

import numpy as np
import pyopencl as cl
from pytential import sym, bind
from functools import partial

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import logging


def find_mode():
    logging.basicConfig(level=logging.INFO)

    import warnings
    warnings.simplefilter("error", np.ComplexWarning)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    k0 = 1.4447
    k1 = k0*1.02

    from pytential.symbolic.pde.maxwell.fiber import \
            SecondKindInfZMuellerOperator

    pde_op = SecondKindInfZMuellerOperator(
            k_vacuum="k_v",
            interfaces=((0, 1, sym.DEFAULT_SOURCE),),
            domain_k_exprs=("k0", "k1"),
            beta="beta",
            use_l2_weighting=True)

    base_context = {
            "k0": k0,
            "k1": k1,
            "k_v": 1,
            }

    u_sym = pde_op.make_unknown("u")
    op = pde_op.operator(u_sym)

    # {{{ discretization setup

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    curve_f = partial(ellipse, 1)

    target_order = 7
    qbx_order = 3
    nelements = 30

    from meshmode.mesh.processing import affine_map
    mesh = make_curve_mesh(curve_f,
            np.linspace(0, 1, nelements+1),
            target_order)
    lambda_ = 1.55
    circle_radius = 3.4*2*np.pi/lambda_
    mesh = affine_map(mesh, A=circle_radius*np.eye(2))

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx, _ = QBXLayerPotentialSource(pre_density_discr, 4*target_order,
            qbx_order,
            # Don't use FMM for now
            fmm_order=qbx_order+5).with_refinement()
    density_discr = qbx.density_discr

    # }}}

    x_vec = np.random.randn(len(u_sym)*density_discr.nnodes)
    y_vec = np.random.randn(len(u_sym)*density_discr.nnodes)

    bound_op = bind(qbx, op)

    def muller_solve_func(beta):
        from pytential.solve import gmres
        gmres_result = gmres(
                bound_op.scipy_op(queue, "u",
                    np.complex128, beta=beta, **base_context),
                y_vec, tol=1e-12, progress=True,
                stall_iterations=0,
                hard_failure=False)
        minv_y = gmres_result.solution
        print("gmres state:", gmres_result.state)

        z = 1/x_vec.dot(minv_y)
        print("muller func value:", repr(z))
        return z

    starting_guesses = (1+0j)*(
            k0
            + (k1-k0) * np.random.rand(3))

    from pytential.muller import muller
    beta, niter = muller(muller_solve_func, z_start=starting_guesses)
    print("beta")


if __name__ == "__main__":
    import sys
    if sys.argv[1:]:

        exec(sys.argv[1])
    else:
        find_mode()

# vim: fdm=marker
