from __future__ import division, print_function

__copyright__ = "Copyright (C) 2017 Matt Wala"

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
import pyopencl as cl

import logging
logger = logging.getLogger(__name__)

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh)

from functools import partial
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory


# {{{ discretization getters

def get_ellipse_with_ref_mean_curvature(cl_ctx, aspect=1):
    nelements = 20
    order = 16

    mesh = make_curve_mesh(
            partial(ellipse, aspect),
            np.linspace(0, 1, nelements+1),
            order)

    a = 1
    b = 1/aspect

    discr = Discretization(cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    with cl.CommandQueue(cl_ctx) as queue:
        nodes = discr.nodes().get(queue=queue)

    t = np.arctan2(nodes[1] * aspect, nodes[0])

    return discr, a*b / ((a*np.sin(t))**2 + (b*np.cos(t))**2)**(3/2)


def get_square_with_ref_mean_curvature(cl_ctx):
    nelements = 8
    order = 8

    from extra_curve_data import unit_square

    mesh = make_curve_mesh(
            unit_square,
            np.linspace(0, 1, nelements+1),
            order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr = Discretization(cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    return discr, 0


def get_unit_sphere_with_ref_mean_curvature(cl_ctx):
    order = 8

    from meshmode.mesh.generation import generate_icosphere
    mesh = generate_icosphere(1, order)

    discr = Discretization(cl_ctx, mesh,
           InterpolatoryQuadratureSimplexGroupFactory(order))

    return discr, 1

# }}}


@pytest.mark.parametrize(("discr_name", "discr_and_ref_mean_curvature_getter"), [
    ("unit_circle", get_ellipse_with_ref_mean_curvature),
    ("2-to-1 ellipse", partial(get_ellipse_with_ref_mean_curvature, aspect=2)),
    ("square", get_square_with_ref_mean_curvature),
    ("unit sphere", get_unit_sphere_with_ref_mean_curvature),
    ])
def test_mean_curvature(ctx_getter, discr_name, discr_and_ref_mean_curvature_getter):
    if discr_name == "unit sphere":
        pytest.skip("not implemented in 3D yet")

    import pytential.symbolic.primitives as prim
    ctx = ctx_getter()

    discr, ref_mean_curvature = discr_and_ref_mean_curvature_getter(ctx)

    with cl.CommandQueue(ctx) as queue:
        from pytential import bind
        mean_curvature = bind(
            discr,
            prim.mean_curvature(discr.ambient_dim))(queue).as_vector(np.object)

    assert np.allclose(mean_curvature[0].get(), ref_mean_curvature)


# You can test individual routines by typing
# $ python test_tools.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
