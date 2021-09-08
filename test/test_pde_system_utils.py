__copyright__ = "Copyright (C) 2021 Isuru Fernando"

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

from sumpy.kernel import (LaplaceKernel, AxisSourceDerivative,
    AxisTargetDerivative, TargetPointMultiplier, BiharmonicKernel)
from pytential.symbolic.primitives import int_g_vec, IntG, NodeCoordinateComponent
from pytential.symbolic.pde.system_utils import merge_int_g_exprs
from pymbolic.primitives import make_sym_vector, Variable


def test_reduce_number_of_fmms():
    dim = 3
    knl = LaplaceKernel(dim)
    densities = make_sym_vector("sigma", 2)
    mu = Variable("mu")

    int_g1 = \
        int_g_vec(AxisSourceDerivative(1, AxisSourceDerivative(0, knl)),
             densities[0] * mu, qbx_forced_limit=1) + \
        int_g_vec(AxisSourceDerivative(1, AxisSourceDerivative(1, knl)),
             densities[1] * mu, qbx_forced_limit=1)

    int_g2 = \
        int_g_vec(AxisSourceDerivative(2, AxisSourceDerivative(0, knl)),
             densities[0], qbx_forced_limit=1) + \
        int_g_vec(AxisSourceDerivative(2, AxisSourceDerivative(1, knl)),
             densities[1], qbx_forced_limit=1)

    # Merging reduces 4 FMMs to 2 FMMs and then further reduced to 1 FMM
    result = merge_int_g_exprs([int_g1, int_g2], source_dependent_variables=[])

    int_g3 = \
        IntG(target_kernel=AxisTargetDerivative(1, knl),
             source_kernels=[AxisSourceDerivative(0, knl),
                 AxisSourceDerivative(1, knl)],
             densities=[-mu * densities[0], -mu * densities[1]],
             qbx_forced_limit=1)

    int_g4 = \
        IntG(target_kernel=AxisTargetDerivative(2, knl),
             source_kernels=[AxisSourceDerivative(0, knl),
                 AxisSourceDerivative(1, knl)],
             densities=[-mu * densities[0], -mu * densities[1]],
             qbx_forced_limit=1)

    assert result[0] == int_g3
    assert result[1] == int_g4 * mu**(-1)


def test_source_dependent_variable():
    # Same example as test_reduce_number_of_fmms, but with
    # mu marked as a source dependent variable
    dim = 3
    knl = LaplaceKernel(dim)
    densities = make_sym_vector("sigma", 2)
    mu = Variable("mu")

    int_g1 = \
        int_g_vec(AxisSourceDerivative(1, AxisSourceDerivative(0, knl)),
             mu * densities[0], qbx_forced_limit=1) + \
        int_g_vec(AxisSourceDerivative(1, AxisSourceDerivative(1, knl)),
             mu * densities[1], qbx_forced_limit=1)

    int_g2 = \
        int_g_vec(AxisSourceDerivative(2, AxisSourceDerivative(0, knl)),
             densities[0], qbx_forced_limit=1) + \
        int_g_vec(AxisSourceDerivative(2, AxisSourceDerivative(1, knl)),
             densities[1], qbx_forced_limit=1)

    result = merge_int_g_exprs([int_g1, int_g2],
            source_dependent_variables=[mu])

    # Merging reduces 4 FMMs to 2 FMMs. No further reduction of FMMs.
    int_g3 = \
        IntG(target_kernel=knl,
             source_kernels=[AxisSourceDerivative(1, AxisSourceDerivative(0, knl)),
                 AxisSourceDerivative(1, AxisSourceDerivative(1, knl))],
             densities=[mu * densities[0], mu * densities[1]],
             qbx_forced_limit=1)

    int_g4 = \
        IntG(target_kernel=knl,
             source_kernels=[AxisSourceDerivative(2, AxisSourceDerivative(0, knl)),
                 AxisSourceDerivative(2, AxisSourceDerivative(1, knl))],
             densities=[densities[0], densities[1]],
             qbx_forced_limit=1)

    assert result[0] == int_g3
    assert result[1] == int_g4


def test_base_kernel_merge():
    # Same example as test_reduce_number_of_fmms, but with
    # mu marked as a source dependent variable
    dim = 3
    knl = LaplaceKernel(dim)
    biharm_knl = BiharmonicKernel(dim)
    density = make_sym_vector("sigma", 1)[0]

    int_g1 = \
        int_g_vec(TargetPointMultiplier(0, knl),
             density, qbx_forced_limit=1)

    int_g2 = \
        int_g_vec(TargetPointMultiplier(1, knl),
             density, qbx_forced_limit=1)

    result = merge_int_g_exprs([int_g1, int_g2],
            source_dependent_variables=[],
            base_kernel=biharm_knl)

    sources = [NodeCoordinateComponent(i) for i in range(dim)]

    source_kernels = [
        AxisSourceDerivative(i, AxisSourceDerivative(i, biharm_knl))
        for i in range(dim)]

    int_g3 = IntG(target_kernel=biharm_knl,
            source_kernels=source_kernels + [AxisSourceDerivative(0, biharm_knl)],
            densities=[density*sources[0]*(-1.0) for i in range(dim)]
                    + [2*density],
            qbx_forced_limit=1)
    int_g4 = IntG(target_kernel=biharm_knl,
            source_kernels=source_kernels + [AxisSourceDerivative(1, biharm_knl)],
            densities=[density*sources[1]*(-1.0) for i in range(dim)]
                    + [2*density],
            qbx_forced_limit=1)

    assert int_g3 == result[0]
    assert int_g4 == result[1]


# You can test individual routines by typing
# $ python test_pde_system_tools.py 'test_reduce_number_of_fmms()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
