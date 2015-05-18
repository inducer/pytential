from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut,
        make_curve_mesh)
from pytential import bind, sym
from functools import partial

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)


def test_matrix_build(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    target_order = 7
    qbx_order = 4
    nelements = 30
    curve_f = partial(ellipse, 1)

    k = 0

    u_sym = sym.var("u")

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    if k:
        knl = HelmholtzKernel(2)
        knl_kwargs = {"k": k}
    else:
        knl = LaplaceKernel(2)
        knl_kwargs = {}

    op = sym.S(knl, u_sym, **knl_kwargs)

    mesh = make_curve_mesh(curve_f,
            np.linspace(0, 1, nelements+1),
            target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx = QBXLayerPotentialSource(density_discr, 4*target_order,
            qbx_order,
            # Don't use FMM for now
            fmm_order=False)

    bound_op = bind(qbx, op)

    from pytential.symbolic.execution import build_matrix
    mat = build_matrix(queue, qbx, op, (u_sym,)).get()

    if 0:
        from sumpy.tools import build_matrix as build_matrix_via_matvec
        mat2 = build_matrix_via_matvec(bound_op.scipy_op(queue, "u"))

        print(
                la.norm((mat-mat2).real, "fro")/la.norm(mat2.real, "fro"),
                la.norm((mat-mat2).imag, "fro")/la.norm(mat2.imag, "fro"))
        import matplotlib.pyplot as pt
        pt.subplot(121)
        pt.imshow(np.log10(np.abs(1e-20+(mat-mat2).real)))
        pt.colorbar()
        pt.subplot(122)
        pt.imshow(np.log10(np.abs(1e-20+(mat-mat2).imag)))
        pt.colorbar()
        pt.show()

    np.random.seed(12)
    for i in range(5):
        u = np.random.randn(density_discr.nnodes)
        res_mat = mat.dot(u)

        u_dev = cl.array.to_device(queue, u)
        res_matvec = bound_op(queue, u=u_dev).get()

        abs_err = la.norm(res_mat - res_matvec, np.inf)
        rel_err = abs_err / la.norm(res_matvec, np.inf)

        print(abs_err, rel_err)
        assert rel_err < 1e-13


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
