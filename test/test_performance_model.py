from __future__ import division, print_function

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
import pyopencl.clmath  # noqa
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from pytential import bind, sym, norm  # noqa


# {{{ global params

TARGET_ORDER = 8
OVSMP_FACTOR = 5
TCF = 0.9
QBX_ORDER = 5
FMM_ORDER = 10

DEFAULT_LPOT_KWARGS = {
        "_box_extent_norm": "l2",
        "_from_sep_smaller_crit": "static_l2",
        }

# }}}


# {{{ test_timing_data_gathering

def test_timing_data_gathering(ctx_getter):
    pytest.importorskip("pyfmmlib")

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    from meshmode.mesh.generation import starfish, make_curve_mesh
    mesh = make_curve_mesh(starfish, np.linspace(0, 1, 1000), order=target_order)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()
    lpot_kwargs.update(
            _expansion_stick_out_factor=TCF,
            fmm_order=FMM_ORDER, qbx_order=QBX_ORDER,
            fmm_backend="fmmlib",
            )

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            **lpot_kwargs)

    lpot_source, _ = lpot_source.with_refinement()

    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)

    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
    op_S = bind(lpot_source, sym_op_S)

    timing_data = {}
    op_S.eval(queue, dict(sigma=sigma), timing_data=timing_data)
    assert timing_data
    print(timing_data)

# }}}


# {{{ test_performance_model

@pytest.mark.parametrize("dim", (2, 3))
def test_performance_model(ctx_getter, dim):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ get lpot source

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    if dim == 2:
        from meshmode.mesh.generation import starfish, make_curve_mesh
        mesh = make_curve_mesh(starfish, np.linspace(0, 1, 50), order=target_order)
    elif dim == 3:
        from meshmode.mesh.generation import generate_icosphere
        mesh = generate_icosphere(r=1, order=target_order)
    else:
        raise ValueError("unknown dimension: %d" % dim)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()
    lpot_kwargs.update(
            _expansion_stick_out_factor=TCF,
            fmm_order=FMM_ORDER, qbx_order=QBX_ORDER
            )

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            **lpot_kwargs)

    lpot_source, _ = lpot_source.with_refinement()

    # }}}

    # {{{ run performance model

    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var("sigma")
    k_sym = LaplaceKernel(lpot_source.ambient_dim)

    sym_op_S = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
    op_S = bind(lpot_source, sym_op_S)
    perf_S = op_S.get_modeled_performance(queue, sigma=sigma)
    assert len(perf_S) == 1

    sym_op_S_plus_D = (
            sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)
            + sym.D(k_sym, sigma_sym))
    op_S_plus_D = bind(lpot_source, sym_op_S_plus_D)
    perf_S_plus_D = op_S_plus_D.get_modeled_performance(queue, sigma=sigma)
    assert len(perf_S_plus_D) == 2

    # }}}

# }}}


# You can test individual routines by typing
# $ python test_performance_model.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])


# vim: foldmethod=marker
