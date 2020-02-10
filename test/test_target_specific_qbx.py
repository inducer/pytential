from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2018 Matt Wala"

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

    nterms = 9
    z = 3j
    scale = 1
    j = np.zeros(1 + nterms, dtype=np.complex)
    jder = np.zeros(1 + nterms, dtype=np.complex)
    ts.jfuns3d_wrapper(nterms, z, scale, j, jder)

    # Reference solution computed using scipy.special.spherical_jn

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

    assert np.allclose(j, j_expected)

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

    assert np.allclose(jder, jder_expected)


def test_spherical_hankel_functions():
    import pytential.qbx.target_specific as ts

    nterms = 9
    z = 2 + 3j
    scale = 1
    h = np.zeros(1 + nterms, dtype=np.complex)
    hder = np.zeros(1 + nterms, dtype=np.complex)
    ts.h3dall_wrapper(nterms, z, scale, h, hder)

    # Reference solution computed using
    # scipy.special.spherical_jn + 1j * scipy.special.spherical_yn
    h_expected = np.array([
            +1.17460537937623677e-02 + -7.25971518952217565e-03j,
            -7.12794888037171503e-03 + -1.55735608522498126e-02j,
            -2.58175723285687941e-02 + +5.00665171335734627e-03j,
            -6.95481631849959037e-03 + +4.92143379339500253e-02j,
            +9.78278544942576822e-02 + +5.92281078069348405e-02j,
            +2.65420992601874961e-01 + -1.70387117227806167e-01j,
            -8.11750107462848453e-02 + -1.02133651818182791e+00j,
            -3.49178056863992792e+00 + -1.62876088689699405e+00j,
            -1.36147986022969878e+01 + +9.34959028601928743e+00j,
            +4.56300765393887087e+00 + +7.94934376901125432e+01j,
    ])

    assert np.allclose(h, h_expected)

    hder_expected = np.array([
            +7.12794888037171503e-03 + +1.55735608522498126e-02j,
            +2.11270661502996893e-02 + -5.75767287207851197e-03j,
            +1.32171023895111261e-03 + -3.57580271012700734e-02j,
            -6.69663049946767064e-02 + -3.16989251553807527e-02j,
            -1.50547136475930293e-01 + +1.16532548652759055e-01j,
            +8.87444851771816839e-02 + +5.84014513465967444e-01j,
            +2.00269153354544205e+00 + +7.98384884993240895e-01j,
            +7.22334424954346144e+00 + -5.46307186102847187e+00j,
            -4.05890079026877615e+00 + -4.28512368415405192e+01j,
            -2.04081205047078043e+02 + -1.02417988497371837e+02j,
    ])

    assert np.allclose(hder, hder_expected)


@pytest.mark.parametrize("op", ["S", "D", "Sp"])
@pytest.mark.parametrize("helmholtz_k", [0, 1.2, 12 + 1.2j])
@pytest.mark.parametrize("qbx_order", [0, 1, 5])
def test_target_specific_qbx(ctx_factory, op, helmholtz_k, qbx_order):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 4
    fmm_tol = 1e-3

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
        refiner_extra_kwargs["kernel_length_scale"] = 5 / abs(helmholtz_k)

    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order,
            qbx_order=qbx_order,
            fmm_level_to_order=SimpleExpansionOrderFinder(fmm_tol),
            fmm_backend="fmmlib",
            _expansions_in_tree_have_extent=True,
            _expansion_stick_out_factor=0.9,
            _use_target_specific_qbx=False,
            ).with_refinement(**refiner_extra_kwargs)

    density_discr = qbx.density_discr

    nodes = density_discr.nodes().with_queue(queue)
    u_dev = clmath.sin(nodes[0])

    if helmholtz_k == 0:
        kernel = LaplaceKernel(3)
        kernel_kwargs = {}
    else:
        kernel = HelmholtzKernel(3, allow_evanescent=True)
        kernel_kwargs = {"k": sym.var("k")}

    u_sym = sym.var("u")

    if op == "S":
        op = sym.S
    elif op == "D":
        op = sym.D
    elif op == "Sp":
        op = sym.Sp
    else:
        raise ValueError("unknown operator: '%s'" % op)

    expr = op(kernel, u_sym, qbx_forced_limit=-1, **kernel_kwargs)

    bound_op = bind(qbx, expr)
    pot_ref = bound_op(queue, u=u_dev, k=helmholtz_k).get()

    qbx = qbx.copy(_use_target_specific_qbx=True)
    bound_op = bind(qbx, expr)
    pot_tsqbx = bound_op(queue, u=u_dev, k=helmholtz_k).get()

    assert np.allclose(pot_tsqbx, pot_ref, atol=1e-13, rtol=1e-13)


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
