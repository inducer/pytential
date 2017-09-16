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

from pytential import bind, sym, norm

from sumpy.visualization import FieldPlotter  # noqa
from sumpy.point_calculus import CalculusPatch, frequency_domain_maxwell
from sumpy.tools import vector_from_device
from pytential.target import PointsTarget

import logging
logger = logging.getLogger(__name__)

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


class TestCase:
    fmm_backend = "fmmlib"

    def __init__(self, k, is_interior, resolutions, qbx_order,
            fmm_order):
        self.k = k
        self.is_interior = is_interior
        self.resolutions = resolutions
        self.qbx_order = qbx_order
        self.fmm_order = fmm_order


class SphereTestCase(TestCase):
    target_order = 8
    gmres_tol = 1e-10

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.generation import generate_icosphere
        from meshmode.mesh.refinement import refine_uniformly
        return refine_uniformly(
                generate_icosphere(2, target_order),
                resolution)

    def get_observation_mesh(self, target_order):
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

        source_rad = 0.3

        sources = source_ctr + source_rad*2*(np.random.rand(3, 10)-0.5)
        from pytential.source import PointPotentialSource
        return PointPotentialSource(
                queue.context,
                cl.array.to_device(queue, sources))


tc_int = SphereTestCase(k=1.2, is_interior=True, resolutions=[0, 1],
        qbx_order=3, fmm_order=10)
tc_ext = SphereTestCase(k=1.2, is_interior=False, resolutions=[0, 1],
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

    np.random.seed(12)

    knl_kwargs = {"k": case.k}

    # {{{ come up with a solution to Maxwell's equations

    j_sym = sym.make_sym_vector("j", 3)
    jt_sym = sym.make_sym_vector("jt", 2)
    rho_sym = sym.var("rho")

    from pytential.symbolic.pde.maxwell import PECAugmentedMFIEOperator
    mfie = PECAugmentedMFIEOperator()

    test_source = case.get_source(queue)

    calc_patch = CalculusPatch(np.array([-3, 0, 0]), h=0.01)
    calc_patch_tgt = PointsTarget(calc_patch.points)

    rng = cl.clrandom.PhiloxGenerator(cl_ctx, seed=12)
    src_j = rng.normal(queue, (3, test_source.nnodes), dtype=np.float64)

    def eval_inc_field_at(tgt):
        return bind(
                (test_source, tgt),
                get_sym_maxwell_source(mfie.kernel, j_sym, mfie.k)
                )(queue, j=src_j, k=case.k)

    pde_test_inc = eval_inc_field_at(calc_patch_tgt)

    pde_test_e = vector_from_device(queue, pde_test_inc[:3])
    pde_test_h = vector_from_device(queue, pde_test_inc[3:])

    assert max(
            calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_e, np.inf)
            for x in frequency_domain_maxwell(
                calc_patch, pde_test_e, pde_test_h, case.k)) < 1e-6

    # }}}

    loc_sign = -1 if case.is_interior else +1
    # xyz_mfie_bdry_vol_op = mfie.boundary_field(0, Jxyz)
    # xyz_mfie_vol_op = mfie.volume_field(Jxyz)

    # n_cross_op = n_cross(make_vector_field("vec", 3))

    from pytools.convergence import EOCRecorder

    eoc_rec_repr_maxwell = EOCRecorder()
    eoc_rec_e = EOCRecorder()
    eoc_rec_h = EOCRecorder()

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    for resolution in case.resolutions:
        scat_mesh = case.get_mesh(resolution, case.target_order)
        observation_mesh = case.get_observation_mesh(case.target_order)

        pre_scat_discr = Discretization(
                cl_ctx, scat_mesh,
                InterpolatoryQuadratureSimplexGroupFactory(case.target_order))
        qbx, _ = QBXLayerPotentialSource(
                pre_scat_discr, fine_order=4*case.target_order,
                qbx_order=case.qbx_order,
                fmm_order=case.fmm_order, fmm_backend=case.fmm_backend
                ).with_refinement()
        h_max = qbx.h_max

        scat_discr = qbx.density_discr
        obs_discr = Discretization(
                cl_ctx, observation_mesh,
                InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

        inc_field_scat = eval_inc_field_at(scat_discr)
        inc_field_obs = eval_inc_field_at(obs_discr)

        # nxHinc_xyz_mid = sbind(n_cross_op)(vec=H_scat)
        # nxH_tgt_true = bind(n_cross_op, tgt_discr, iprec=iprec)(
        #         vec=H_tgt_true)

        # {{{ system solve

        # nxHinc_t_mid = sbind(
        #         xyz_to_tangential(make_vector_field("vec", 3))
        #         )(vec=nxHinc_xyz_mid)

        inc_xyz_sym = sym.make_sym_vector("inc_fld", 6)
        e_inc_xyz_sym = inc_xyz_sym[:3]
        h_inc_xyz_sym = inc_xyz_sym[3:]

        bound_j_op = bind(qbx, mfie.j_operator(loc_sign, jt_sym))
        j_rhs = bind(qbx, mfie.j_rhs(h_inc_xyz_sym))(
                queue, inc_fld=inc_field_scat, **knl_kwargs)

        from pytential.solve import gmres
        gmres_result = gmres(
                bound_j_op.scipy_op(queue, "jt", np.complex128, **knl_kwargs),
                j_rhs,
                tol=case.gmres_tol,
                progress=True,
                hard_failure=True)

        jt = gmres_result.solution

        bound_rho_op = bind(qbx, mfie.rho_operator(loc_sign, rho_sym))
        rho_rhs = bind(qbx, mfie.rho_rhs(jt_sym, e_inc_xyz_sym))(
                queue, jt=jt, inc_fld=inc_field_scat, **knl_kwargs)

        gmres_result = gmres(
                bound_rho_op.scipy_op(queue, "rho", np.complex128, **knl_kwargs),
                rho_rhs,
                tol=case.gmres_tol,
                progress=True,
                hard_failure=True)

        rho = gmres_result.solution

        # }}}

        jxyz = bind(qbx, sym.tangential_to_xyz(jt_sym))(queue, jt=jt)

        # {{{ volume eval

        sym_repr = mfie.scattered_volume_field(jt_sym, rho_sym)

        def eval_repr_at(tgt, source=None):
            if source is None:
                source = qbx

            return bind((source, tgt), sym_repr)(queue, jt=jt, rho=rho, **knl_kwargs)

        pde_test_repr = eval_repr_at(calc_patch_tgt)

        pde_test_e = vector_from_device(queue, pde_test_repr[:3])
        pde_test_h = vector_from_device(queue, pde_test_repr[3:])

        maxwell_residuals = [
                calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_e, np.inf)
                for x in frequency_domain_maxwell(
                    calc_patch, pde_test_e, pde_test_h, case.k)]
        print("Maxwell residuals:", maxwell_residuals)

        eoc_rec_repr_maxwell.add_data_point(h_max, max(maxwell_residuals))

        # }}}

        # {{{ visualization

        if visualize:
            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(queue, scat_discr, case.target_order+3)

            bdry_normals = bind(scat_discr, sym.normal(3))(queue)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
                ("j", jxyz),
                ("rho", rho),
                ("Einc", inc_field_scat[:3]),
                ("Hinc", inc_field_scat[3:]),
                ("bdry_normals", bdry_normals),
                ])

            bbox_center = np.zeros(3)
            bbox_size = 6

            fplot = FieldPlotter(
                    bbox_center, extent=bbox_size, npoints=(150, 150, 5))

            from pytential.qbx import QBXTargetAssociationFailedException

            qbx_tgt_tol = qbx.copy(target_association_tolerance=0.2)

            fplot_tgt = PointsTarget(fplot.points)
            try:
                fplot_repr = eval_repr_at(fplot_tgt, source=qbx_tgt_tol)
            except QBXTargetAssociationFailedException as e:
                fplot.write_vtk_file(
                        "failed-targets.vts",
                        [
                            ("failed_targets", e.failed_target_flags.get(queue))
                            ])
                raise

            fplot_inc = eval_inc_field_at(fplot_tgt)

            fplot.write_vtk_file(
                    "potential.vts",
                    [
                        ("E", vector_from_device(queue, fplot_repr[:3])),
                        ("H", vector_from_device(queue, fplot_repr[3:])),
                        ("Einc", vector_from_device(queue, fplot_inc[:3])),
                        ("Hinc", vector_from_device(queue, fplot_inc[3:])),
                        ]
                    )

        # }}}

        # {{{ error in E, H

        obs_repr = eval_repr_at(obs_discr)

        obs_e = obs_repr[:3]
        obs_h = obs_repr[3:]

        inc_obs_e = inc_field_obs[:3]
        inc_obs_h = inc_field_obs[3:]

        def obs_norm(f):
            return norm(obs_discr, queue, f, p=np.inf)

        # FIXME: Why "+"?
        rel_err_e = (obs_norm(inc_obs_e + obs_e)
                / obs_norm(inc_obs_e))
        rel_err_h = (obs_norm(inc_obs_h + obs_h)
                / obs_norm(inc_obs_h))

        # }}}

        print("ERR", h_max, rel_err_h, rel_err_e)

        eoc_rec_h.add_data_point(h_max, rel_err_h)
        eoc_rec_e.add_data_point(h_max, rel_err_e)

    # TODO: Check that total field verifies BC
    # TODO: Check for extinction on the interior

    print("--------------------------------------------------------")
    print("is_interior=%s" % case.is_interior)
    print("--------------------------------------------------------")

    good = True
    for which_eoc, eoc_rec, order_tol in [
            ("maxwell", eoc_rec_repr_maxwell, 1.5),
            ("H", eoc_rec_h, 1),
            ("E", eoc_rec_e, 1)]:
        print(which_eoc)
        print(eoc_rec.pretty_print())

        if len(eoc_rec.history) > 1:
            if eoc_rec.order_estimate() < case.qbx_order - order_tol:
                good = False

    assert good


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
