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

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


@pytest.mark.parametrize("op", ["S", "D"])
def test_target_specific_qbx(ctx_getter, op):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 4

    from meshmode.mesh.generation import generate_icosphere
    mesh = generate_icosphere(1, target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    pre_density_discr = Discretization(
        cl_ctx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order,
            qbx_order=5,
            fmm_order=10,
            fmm_backend="fmmlib",
            _expansions_in_tree_have_extent=True,
            _expansion_stick_out_factor=0.9,
            ).with_refinement()

    density_discr = qbx.density_discr

    nodes_host = density_discr.nodes().get(queue)
    center = np.array([3, 1, 2])
    diff = nodes_host - center[:, np.newaxis]
    
    dist_squared = np.sum(diff**2, axis=0)
    dist = np.sqrt(dist_squared)
    u = 1/dist
    
    u_dev = cl.array.to_device(queue, u)

    kernel = LaplaceKernel(3)
    u_sym = sym.var("u")
    
    if op == "S":
        op = sym.S
    elif op == "D":
        op = sym.D
    expr = op(kernel, u_sym, qbx_forced_limit=-1)

    bound_op = bind(qbx, expr)
    slp_ref = bound_op(queue, u=u_dev)

    qbx = qbx.copy(_use_tsqbx_list1=True)
    bound_op = bind(qbx, expr)
    slp_tsqbx = bound_op(queue, u=u_dev)

    assert (np.max(np.abs(slp_ref.get() - slp_tsqbx.get()))) < 1e-13


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
