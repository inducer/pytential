from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
#import numpy.linalg as la
import pyopencl as cl
import pytest
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

from functools import partial
from pytential.mesh.generation import (
        ellipse, cloverleaf, starfish, drop, n_gon,
        make_curve_mesh)

circle = partial(ellipse, 1)

__all__ = [
        "pytest_generate_tests",

        # difficult curves not currently used for testing
        "drop", "n_gon", "cloverleaf"
        ]

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass




@pytest.mark.parametrize(("curve_name", "curve_f"), [
    ("circle", partial(ellipse, 1)),
    ("5-to-1 ellipse", partial(ellipse, 5)),
    ("starfish", starfish),
    ])
@pytest.mark.parametrize("kernel", [0])
@pytest.mark.parametrize("bc_type", ["dirichlet", "neumann"])
@pytest.mark.parametrize("loc_sign", [+1, -1])
@pytest.mark.parametrize("order", [+1, -1])
def test_integral_equation(
        ctx_getter, curve_f, nelements,
        #kernel, bc_dtype, loc_sign, order, curve_name=None
        ):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    mesh = make_curve_mesh(curve_f,
            np.linspace(0, 1, nelements),
            8)

    if 0:
        from pytential.visualization import show_mesh
        show_mesh(mesh)

        pt.gca().set_aspect("equal")
        pt.show()







# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
