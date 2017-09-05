from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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
import pyopencl.clrandom  # noqa
import pytest

from pytential import bind, sym

from sumpy.visualization import FieldPlotter  # noqa
from sumpy.point_calculus import CalculusPatch, frequency_domain_maxwell
from pytential.target import PointsTarget

import logging
logger = logging.getLogger(__name__)

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


class TestCase:
    def __init__(self, k, is_interior, resolutions, qbx_order,
            fmm_order):
        self.k = k
        self.is_interior = is_interior
        self.resolutions = resolutions
        self.qbx_order = qbx_order
        self.fmm_order = fmm_order


class SphereTestCase(TestCase):
    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.generation import generate_icosphere
        from meshmode.mesh.refinement import refine_uniformly
        return refine_uniformly(
                generate_icosphere(2, target_order),
                resolution)

    def get_test_mesh(self, target_order):
        from meshmode.mesh.generation import generate_icosphere

        if self.is_interior:

            return generate_icosphere(5, target_order)
        else:
            return generate_icosphere(0.5, target_order)

    def get_source(self, queue):
        if self.is_interior:
            source_ctr = np.array([[0.35, 0.1, 0.15]]).T
        else:
            source_ctr = np.array([[5, 0.1, 0.15]]).T

        sources = source_ctr + np.random.rand(3, 10)
        from pytential.source import PointPotentialSource
        return PointPotentialSource(
                queue.context,
                cl.array.to_device(queue, sources))


tc_int = SphereTestCase(k=10.2, is_interior=True, resolutions=[0, 1],
        qbx_order=3, fmm_order=10)
tc_ext = SphereTestCase(k=10.2, is_interior=False, resolutions=[0, 1],
        qbx_order=3, fmm_order=10)


def get_sym_maxwell_source(kernel, jxyz, k):
    # This ensures div A = 0, which is simply a consequence of div curl S=0.
    # This means we use the Coulomb gauge to generate this field.

    A = sym.curl(sym.S(kernel, jxyz, k=k, qbx_forced_limit=None))

    # https://en.wikipedia.org/w/index.php?title=Maxwell%27s_equations&oldid=798940325#Alternative_formulations
    return sym.join_fields(
        - 1j*k*A,
        sym.curl(A))


@pytest.mark.parametrize("case", [
    tc_int,
    tc_ext
    ])
def test_mfie_from_source(ctx_getter, case, visualize=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    pytest.importorskip("pyfmmlib")
    pytest.importorskip("scipy")

    np.random.seed(12)

    # {{{ come up with a solution to Maxwell's equations

    j_sym = sym.make_sym_vector("j", 3)

    from pytential.symbolic.pde.maxwell import PECAugmentedMFIEOperator
    mfie = PECAugmentedMFIEOperator()

    test_source = case.get_source(queue)

    calc_patch = CalculusPatch(np.array([-2, 0, 0]), h=0.01)
    calc_patch_tgt = PointsTarget(calc_patch.points)

    rng = cl.clrandom.PhiloxGenerator(cl_ctx, seed=12)
    src_j = rng.normal(queue, (3, test_source.nnodes), dtype=np.float64)

    pde_test_field = bind(
            (test_source, calc_patch_tgt),
            get_sym_maxwell_source(mfie.kernel, j_sym, mfie.k)
            )(queue, j=src_j, k=case.k)

    from sumpy.tools import vector_from_device
    pde_test_e = vector_from_device(queue, pde_test_field[:3])
    pde_test_h = vector_from_device(queue, pde_test_field[3:])

    print([
            calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_e, np.inf)
            for x in frequency_domain_maxwell(
                calc_patch, pde_test_e, pde_test_h, case.k)])


    1/0





    mfie_ext_op = xyz_to_tangential(mfie.boundary_field(+1, Jxyz))
    mfie_int_op = xyz_to_tangential(mfie.boundary_field(-1, Jxyz))
    xyz_mfie_bdry_vol_op = mfie.boundary_field(0, Jxyz)
    xyz_mfie_vol_op = mfie.volume_field(Jxyz)

    n_cross_op = n_cross(make_vector_field("vec", 3))

    from pytools.convergence import EOCRecorder

    eoc_rec_nxH = EOCRecorder()
    eoc_rec_E = EOCRecorder()
    eoc_rec_H = EOCRecorder()


    for resolution in case.resolutions:
        scat_mesh = generate_icosphere(2, i)
        if is_interior:
            tgt_mesh = generate_icosphere(5, 0)
        else:
            tgt_mesh = generate_icosphere(0.5, 1)

        scat_discr = FlatConstantDiscretization(scat_mesh)
        tgt_discr = FlatConstantDiscretization(tgt_mesh)

        def sbind(op): return bind(op, scat_discr, iprec=iprec)

        source_func = gen_field_multipole

        E_scat, H_scat = source_func(k, is_interior, scat_mesh.centroids)
        E_tgt_true, H_tgt_true = source_func(k, is_interior, tgt_mesh.centroids)

        nxHinc_xyz_mid = sbind(n_cross_op)(vec=H_scat)
        nxH_tgt_true = bind(n_cross_op, tgt_discr, iprec=iprec)(
                vec=H_tgt_true)

        # {{{ system solve

        nxHinc_t_mid = sbind(
                xyz_to_tangential(make_vector_field("vec", 3))
                )(vec=nxHinc_xyz_mid)

        if is_interior:
            mfie_solve_op = mfie_int_op
        else:
            mfie_solve_op = mfie_ext_op

        Jt_mid = solve_lin_op(
                sbind(mfie_solve_op).scipy_op("Jt"),
                nxHinc_t_mid)

        # }}}

        # {{{ error in nxH

        nxH_tgt = bind(xyz_mfie_bdry_vol_op, (scat_discr, tgt_discr), iprec=iprec) \
                (Jt=Jt_mid)
        rel_err_nxH = (tgt_discr.norm(nxH_tgt_true+nxH_tgt)
                / tgt_discr.norm(nxH_tgt_true))

        # }}}

        # {{{ error in E, H

        EH_tgt = bind(xyz_mfie_vol_op, (scat_discr, tgt_discr), iprec=iprec) \
                (Jt=Jt_mid)
        E_tgt = EH_tgt[:3]
        H_tgt = EH_tgt[3:]
        rel_err_H = (tgt_discr.norm(H_tgt_true-H_tgt)
                / tgt_discr.norm(H_tgt_true))
        rel_err_E = (tgt_discr.norm(E_tgt_true-E_tgt)
                / tgt_discr.norm(E_tgt_true))

        # }}}

        print("ERR", h, rel_err_nxH, rel_err_H, rel_err_E)

        eoc_rec_nxH.add_data_point(h, rel_err_nxH)
        eoc_rec_H.add_data_point(h, rel_err_H)
        eoc_rec_E.add_data_point(h, rel_err_E)

        # {{{ visualization
        if write_vis:
            from pyvisfile.silo import SiloFile
            from hellskitchen.visualization import add_to_silo_file
            from pytools.obj_array import oarray_real_copy

            Jxyz_mid = bind(
                    tangential_to_xyz(make_vector_field("vec", 2)))(vec=Jt_mid)

            silo = SiloFile("mfie-int%s-%d.silo" % (is_interior, i))
            add_to_silo_file(silo, scat_geo, cell_data=[
                ("nxHinc_mid", oarray_real_copy(nxHinc_xyz_mid)),
                ("J_mid", oarray_real_copy(Jxyz_mid)),
                ])
            add_to_silo_file(silo, tgt_geo, cell_data=[
                ("nxHinc_tgt", oarray_real_copy(nxH_tgt)),
                ("nxHinc_tgt_true", oarray_real_copy(nxH_tgt_true)),
                ("Hinc_tgt", oarray_real_copy(H_tgt)),
                ("Hinc_tgt_true", oarray_real_copy(H_tgt_true)),
                ("Einc_tgt", oarray_real_copy(E_tgt)),
                ("Einc_tgt_true", oarray_real_copy(E_tgt_true)),
                ], mesh_name="tgt_mesh")
            silo.close()

        # }}}

    print("--------------------------------------------------------")
    print("is_interior=%s" % case.is_interior)
    print("--------------------------------------------------------")
    for which_eoc, eoc_rec in [
            ("nxH", eoc_rec_nxH),
            ("H", eoc_rec_H),
            ("E", eoc_rec_E)]:
        print(which_eoc)
        print(eoc_rec.pretty_print())

        if len(eoc_rec.history) > 1:
            assert eoc_rec.order_estimate() > 0.8






# You can test individual routines by typing
# $ python test_maxwell.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
