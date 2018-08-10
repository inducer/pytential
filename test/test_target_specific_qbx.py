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
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clmath as clmath
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial  # noqa
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        NArmedStarfish,
        make_curve_mesh)

from pytential import bind, sym, norm  # noqa
from sumpy.kernel import LaplaceKernel, HelmholtzKernel

import logging
logger = logging.getLogger(__name__)


def test_spherical_bessel_functions():
    import pytential.qbx.target_specific as ts
    
    nterms = 10
    z = 3j
    scale = 1
    j = np.zeros(nterms, dtype=np.complex)
    jder = np.zeros(nterms, dtype=np.complex)
    ts.jfuns3d_wrapper(nterms, z, scale, j, jder)

    # Reference solution from scipy.special.spherical_jn
    
    j_expected = np.array([
            +3.33929164246994992e+00 + +0.00000000000000000e+00j,
            +0.00000000000000000e+00 + +2.24279011776926884e+00j,
            -1.09650152470070195e+00 + +0.00000000000000000e+00j,
            +2.77555756156289135e-17 + -4.15287576601431119e-01j,
            +1.27497179297362317e-01 + +0.00000000000000000e+00j,
            -3.46944695195361419e-18 + +3.27960387093445271e-02j,
            -7.24503736309898075e-03 + -4.33680868994201774e-19j,
            +0.00000000000000000e+00 + -1.40087680258226812e-03j,
            +2.40653350187633002e-04 + +0.00000000000000000e+00j,
            +0.00000000000000000e+00 + +3.71744848523478122e-05j,
        ])
    
    assert np.allclose(j, j_expected, rtol=1e-13, atol=0)

    jder_expected = np.array([
            -0.00000000000000000e+00 + -2.24279011776926884e+00j,
            +1.84409823062377076e+00 + +0.00000000000000000e+00j,
            +0.00000000000000000e+00 + +1.14628859306856690e+00j,
            -5.42784755898793825e-01 + +3.70074341541718826e-17j,
            +2.77555756156289135e-17 + -2.02792277772493951e-01j,
            +6.19051018786732632e-02 + -6.93889390390722838e-18j,
            -2.45752492430047685e-18 + +1.58909515287802387e-02j,
            -3.50936588954626604e-03 + -4.33680868994201774e-19j,
            +0.00000000000000000e+00 + -6.78916752019369197e-04j,
            +1.16738400679806980e-04 + +0.00000000000000000e+00j,
        ])

    assert np.allclose(jder, jder_expected, rtol=1e-13, atol=0)
    

@pytest.mark.parametrize("op", ["S", "D"])
@pytest.mark.parametrize("helmholtz_k", [0, 1.2])
def test_target_specific_qbx(ctx_getter, op, helmholtz_k):
    logging.basicConfig(level=logging.INFO)

    if helmholtz_k != 0:
        pytest.xfail("not implemented yet")

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 8
    fmm_tol = 1e-5

    from meshmode.mesh.generation import generate_icosphere
    mesh = generate_icosphere(1, target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

    refiner_extra_kwargs = {}

    if helmholtz_k != 0:
        refiner_extra_kwargs["kernel_length_scale"] = 5 / helmholtz_k

    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order,
            qbx_order=5,
            fmm_level_to_order=SimpleExpansionOrderFinder(fmm_tol),
            fmm_backend="fmmlib",
            _expansions_in_tree_have_extent=True,
            _expansion_stick_out_factor=0.9,
            ).with_refinement(**refiner_extra_kwargs)

    density_discr = qbx.density_discr

    nodes = density_discr.nodes().with_queue(queue)
    u_dev = clmath.sin(nodes[0])

    if helmholtz_k == 0:
        kernel = LaplaceKernel(3)
        kernel_kwargs = {}
    else:
        kernel = HelmholtzKernel(3)
        kernel_kwargs = {"k": sym.var("k")}

    u_sym = sym.var("u")

    if op == "S":
        op = sym.S
    elif op == "D":
        op = sym.D

    expr = op(kernel, u_sym, qbx_forced_limit=-1, **kernel_kwargs)

    bound_op = bind(qbx, expr)
    pot_ref = bound_op(queue, u=u_dev, k=helmholtz_k)

    qbx = qbx.copy(_use_tsqbx=True)
    bound_op = bind(qbx, expr)
    pot_tsqbx = bound_op(queue, u=u_dev, k=helmholtz_k)

    assert (np.max(np.abs(pot_ref.get() - pot_tsqbx.get()))) < 1e-13


# You can test individual routines by typing
# $ python test_target_specific_qbx.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
