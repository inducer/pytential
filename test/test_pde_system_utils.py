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
    AxisTargetDerivative, TargetPointMultiplier, BiharmonicKernel, HelmholtzKernel)
from pytential.symbolic.primitives import (int_g_vec, D, IntG,
    NodeCoordinateComponent)
from pytential.symbolic.pde.system_utils import (merge_int_g_exprs,
    rewrite_using_base_kernel)
from pymbolic.primitives import make_sym_vector, Variable


def test_reduce_number_of_fmms():
    dim = 3
    knl = LaplaceKernel(dim)
    densities = make_sym_vector("sigma", 2)
    mu = Variable("mu")

    int_g1 = \
        mu * int_g_vec(AxisSourceDerivative(1, AxisSourceDerivative(0, knl)),
             densities[0], qbx_forced_limit=1) + \
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
    nu = Variable("nu")

    int_g1 = \
        int_g_vec(AxisSourceDerivative(1, AxisSourceDerivative(0, knl)),
             mu * nu * densities[0], qbx_forced_limit=1) + \
        int_g_vec(AxisSourceDerivative(1, AxisSourceDerivative(1, knl)),
             mu * nu * densities[1], qbx_forced_limit=1)

    int_g2 = \
        int_g_vec(AxisSourceDerivative(2, AxisSourceDerivative(0, knl)),
             densities[0], qbx_forced_limit=1) + \
        int_g_vec(AxisSourceDerivative(2, AxisSourceDerivative(1, knl)),
             densities[1], qbx_forced_limit=1)

    result = merge_int_g_exprs([int_g1, int_g2],
            source_dependent_variables=[mu, nu])

    # Merging reduces 4 FMMs to 2 FMMs. No further reduction of FMMs.
    int_g3 = \
        IntG(target_kernel=knl,
             source_kernels=[AxisSourceDerivative(1, AxisSourceDerivative(1, knl)),
                 AxisSourceDerivative(1, AxisSourceDerivative(0, knl))],
             densities=[mu * nu * densities[1], mu * nu * densities[0]],
             qbx_forced_limit=1)

    int_g4 = \
        IntG(target_kernel=knl,
             source_kernels=[AxisSourceDerivative(2, AxisSourceDerivative(1, knl)),
                 AxisSourceDerivative(2, AxisSourceDerivative(0, knl))],
             densities=[densities[1], densities[0]],
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

    exprs_rewritten = rewrite_using_base_kernel([int_g1, int_g2],
            base_kernel=biharm_knl)
    result = merge_int_g_exprs(exprs_rewritten, source_dependent_variables=[])

    sources = [NodeCoordinateComponent(i) for i in range(dim)]

    source_kernels = list(reversed([
        AxisSourceDerivative(i, AxisSourceDerivative(i, biharm_knl))
        for i in range(dim)]))

    int_g3 = IntG(target_kernel=biharm_knl,
            source_kernels=[AxisSourceDerivative(0, biharm_knl)] + source_kernels,
            densities=[2*density] + [density*sources[0]*(-1.0) for _ in range(dim)],
            qbx_forced_limit=1)
    int_g4 = IntG(target_kernel=biharm_knl,
            source_kernels=[AxisSourceDerivative(1, biharm_knl)] + source_kernels,
            densities=[2*density] + [density*sources[1]*(-1.0) for _ in range(dim)],
            qbx_forced_limit=1)

    assert result[0] == int_g3
    assert result[1] == int_g4


def test_merge_different_kernels():
    # Test different kernels Laplace, Helmholtz(k=1), Helmholtz(k=2)
    dim = 3
    laplace_knl = LaplaceKernel(dim)
    helmholtz_knl = HelmholtzKernel(dim)
    density = make_sym_vector("sigma", 1)[0]

    int_g1 = int_g_vec(laplace_knl, density, qbx_forced_limit=1) \
        + int_g_vec(helmholtz_knl, density, qbx_forced_limit=1, k=1) \
        + int_g_vec(AxisTargetDerivative(0, helmholtz_knl),
                density, qbx_forced_limit=1, k=1) \
        + int_g_vec(helmholtz_knl, density, qbx_forced_limit=1, k=2)

    int_g2 = int_g_vec(AxisTargetDerivative(0, laplace_knl),
            density, qbx_forced_limit=1)

    result = merge_int_g_exprs([int_g1, int_g2],
            source_dependent_variables=[])

    int_g3 = int_g_vec(laplace_knl, density, qbx_forced_limit=1) \
        + IntG(target_kernel=helmholtz_knl,
            source_kernels=[AxisSourceDerivative(0, helmholtz_knl), helmholtz_knl],
            densities=[-density, density],
            qbx_forced_limit=1, k=1) \
        + int_g_vec(helmholtz_knl, density, qbx_forced_limit=1, k=2)

    assert result[0] == int_g3
    assert result[1] == int_g2


def test_merge_different_qbx_forced_limit():
    dim = 3
    laplace_knl = LaplaceKernel(dim)
    density = make_sym_vector("sigma", 1)[0]

    int_g1 = int_g_vec(laplace_knl, density, qbx_forced_limit=1)
    int_g2 = int_g1.copy(target_kernel=AxisTargetDerivative(0, laplace_knl))

    int_g3, = merge_int_g_exprs([int_g2 + int_g1])
    int_g4 = int_g1.copy(qbx_forced_limit=2) + int_g2.copy(qbx_forced_limit=-2)
    int_g5 = int_g1.copy(qbx_forced_limit=-2) + int_g2.copy(qbx_forced_limit=2)

    result = merge_int_g_exprs([int_g3, int_g4, int_g5],
            source_dependent_variables=[])

    int_g6 = int_g_vec(laplace_knl, -density, qbx_forced_limit=1)
    int_g7 = int_g6.copy(target_kernel=AxisTargetDerivative(0, laplace_knl))
    int_g8 = int_g7 * (-1) + int_g6 * (-1)
    int_g9 = int_g6.copy(qbx_forced_limit=2) * (-1) \
                + int_g7.copy(qbx_forced_limit=-2) * (-1)
    int_g10 = int_g6.copy(qbx_forced_limit=-2) * (-1) \
                + int_g7.copy(qbx_forced_limit=2) * (-1)

    assert result[0] == int_g8
    assert result[1] == int_g9
    assert result[2] == int_g10


def test_merge_directional_source():
    from pymbolic.primitives import Variable
    from pytential.symbolic.primitives import cse

    dim = 3
    laplace_knl = LaplaceKernel(dim)
    density = Variable("density")

    int_g1 = int_g_vec(laplace_knl, density, qbx_forced_limit=1)
    int_g2 = D(laplace_knl, density, qbx_forced_limit=1)

    source_kernels = [AxisSourceDerivative(d, laplace_knl)
            for d in range(dim)] + [laplace_knl]
    dsource = int_g2.kernel_arguments["dsource_vec"]
    densities = [dsource[d]*cse(density) for d in range(dim)] + [density]
    int_g3 = int_g2.copy(source_kernels=source_kernels, densities=densities,
                         kernel_arguments={})

    result = merge_int_g_exprs([int_g1 + int_g2],
            source_dependent_variables=[density])
    assert result[0] == int_g3

    result = merge_int_g_exprs([int_g1 + int_g2])
    assert result[0] == int_g3


def test_restoring_target_attributes():
    from pymbolic.primitives import Variable
    dim = 3
    laplace_knl = LaplaceKernel(dim)
    density = Variable("density")

    int_g1 = int_g_vec(TargetPointMultiplier(0, AxisTargetDerivative(0,
        laplace_knl)), density, qbx_forced_limit=1)
    int_g2 = int_g_vec(AxisTargetDerivative(1, laplace_knl),
            density, qbx_forced_limit=1)

    result = merge_int_g_exprs([int_g1, int_g2],
            source_dependent_variables=[])

    assert result[0] == int_g1
    assert result[1] == int_g2


def test_int_gs_in_densities():
    from pymbolic.primitives import Variable, Quotient
    dim = 3
    laplace_knl = LaplaceKernel(dim)
    density = Variable("density")

    int_g1 = \
        int_g_vec(laplace_knl,
            int_g_vec(AxisSourceDerivative(2, laplace_knl), density,
                qbx_forced_limit=1), qbx_forced_limit=1) + \
        int_g_vec(AxisTargetDerivative(0, laplace_knl),
            int_g_vec(AxisSourceDerivative(1, laplace_knl), 2*density,
                qbx_forced_limit=1), qbx_forced_limit=1)

    # In the above example the two inner source derivatives should
    # be converted to target derivatives and the two outermost
    # IntGs should be merged into one by converting the target
    # derivative in the last term to a source derivative
    result = merge_int_g_exprs([int_g1])

    source_kernels = [AxisSourceDerivative(0, laplace_knl), laplace_knl]
    densities = [
        (-1)*int_g_vec(AxisTargetDerivative(1, laplace_knl),
            (-2)*density, qbx_forced_limit=1),
        int_g_vec(AxisTargetDerivative(2, laplace_knl),
            (-2)*density, qbx_forced_limit=1) * Quotient(1, 2)
    ]
    int_g3 = IntG(target_kernel=laplace_knl,
                  source_kernels=tuple(source_kernels),
                  densities=tuple(densities),
                  qbx_forced_limit=1)

    assert result[0] == int_g3


# You can test individual routines by typing
# $ python test_pde_system_tools.py 'test_reduce_number_of_fmms()'

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
