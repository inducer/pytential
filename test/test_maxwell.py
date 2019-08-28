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

from sumpy.visualization import make_field_plotter_from_bbox  # noqa
from sumpy.point_calculus import CalculusPatch, frequency_domain_maxwell
from sumpy.tools import vector_from_device
from pytential.target import PointsTarget
from meshmode.mesh.processing import find_bounding_box

import logging
logger = logging.getLogger(__name__)

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


# {{{ test cases

class MaxwellTestCase:
    fmm_backend = "fmmlib"

    def __init__(self, k, is_interior, resolutions, qbx_order,
            fmm_tolerance):
        self.k = k
        self.is_interior = is_interior
        self.resolutions = resolutions
        self.qbx_order = qbx_order
        self.fmm_tolerance = fmm_tolerance


class SphereTestCase(MaxwellTestCase):
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


class RoundedCubeTestCase(MaxwellTestCase):
    target_order = 8
    gmres_tol = 1e-10

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("rounded-cube.step"), 2, order=3,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        from meshmode.mesh.processing import perform_flips, affine_map
        mesh = affine_map(mesh, b=np.array([-0.5, -0.5, -0.5]))
        mesh = affine_map(mesh, A=np.eye(3)*2)

        # now centered at origin and extends to -1,1

        # Flip elements--gmsh generates inside-out geometry.
        return perform_flips(mesh, np.ones(mesh.nelements))

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


class ElliptiPlaneTestCase(MaxwellTestCase):
    target_order = 4
    gmres_tol = 1e-7

    def get_mesh(self, resolution, target_order):
        from pytools import download_from_web_if_not_present

        download_from_web_if_not_present(
                "https://raw.githubusercontent.com/inducer/geometries/master/"
                "surface-3d/elliptiplane.brep")

        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("elliptiplane.brep"), 2, order=2,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        # now centered at origin and extends to -1,1

        # Flip elements--gmsh generates inside-out geometry.
        from meshmode.mesh.processing import perform_flips
        return perform_flips(mesh, np.ones(mesh.nelements))

    def get_observation_mesh(self, target_order):
        from meshmode.mesh.generation import generate_icosphere

        if self.is_interior:
            return generate_icosphere(12, target_order)
        else:
            return generate_icosphere(0.5, target_order)

    def get_source(self, queue):
        if self.is_interior:
            source_ctr = np.array([[0.35, 0.1, 0.15]]).T
        else:
            source_ctr = np.array([[3, 1, 10]]).T

        source_rad = 0.3

        sources = source_ctr + source_rad*2*(np.random.rand(3, 10)-0.5)
        from pytential.source import PointPotentialSource
        return PointPotentialSource(
                queue.context,
                cl.array.to_device(queue, sources))

# }}}


tc_int = SphereTestCase(k=1.2, is_interior=True, resolutions=[0, 1],
        qbx_order=3, fmm_tolerance=1e-4)
tc_ext = SphereTestCase(k=1.2, is_interior=False, resolutions=[0, 1],
        qbx_order=3, fmm_tolerance=1e-4)

tc_rc_ext = RoundedCubeTestCase(k=6.4, is_interior=False, resolutions=[0.1],
        qbx_order=3, fmm_tolerance=1e-4)

tc_plane_ext = ElliptiPlaneTestCase(k=2, is_interior=False, resolutions=[0.15],
        qbx_order=3, fmm_tolerance=1e-4)


class EHField(object):
    def __init__(self, eh_field):
        assert len(eh_field) == 6
        self.field = eh_field

    @property
    def e(self):
        return self.field[:3]

    @property
    def h(self):
        return self.field[3:]


# {{{ driver

@pytest.mark.slowtest
@pytest.mark.parametrize("case", [
    #tc_int,
    tc_ext,
    ])
def test_pec_mfie_extinction(ctx_factory, case, visualize=False):
    """For (say) is_interior=False (the 'exterior' MFIE), this test verifies
    extinction of the combined (incoming + scattered) field on the interior
    of the scatterer.
    """
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    np.random.seed(12)

    knl_kwargs = {"k": case.k}

    # {{{ come up with a solution to Maxwell's equations

    j_sym = sym.make_sym_vector("j", 3)
    jt_sym = sym.make_sym_vector("jt", 2)
    rho_sym = sym.var("rho")

    from pytential.symbolic.pde.maxwell import (
            PECChargeCurrentMFIEOperator,
            get_sym_maxwell_point_source,
            get_sym_maxwell_plane_wave)
    mfie = PECChargeCurrentMFIEOperator()

    test_source = case.get_source(queue)

    calc_patch = CalculusPatch(np.array([-3, 0, 0]), h=0.01)
    calc_patch_tgt = PointsTarget(cl.array.to_device(queue, calc_patch.points))

    rng = cl.clrandom.PhiloxGenerator(cl_ctx, seed=12)
    src_j = rng.normal(queue, (3, test_source.nnodes), dtype=np.float64)

    def eval_inc_field_at(tgt):
        if 0:
            # plane wave
            return bind(
                    tgt,
                    get_sym_maxwell_plane_wave(
                        amplitude_vec=np.array([1, 1, 1]),
                        v=np.array([1, 0, 0]),
                        omega=case.k)
                    )(queue)
        else:
            # point source
            return bind(
                    (test_source, tgt),
                    get_sym_maxwell_point_source(mfie.kernel, j_sym, mfie.k)
                    )(queue, j=src_j, k=case.k)

    pde_test_inc = EHField(
            vector_from_device(queue, eval_inc_field_at(calc_patch_tgt)))

    source_maxwell_resids = [
            calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_inc.e, np.inf)
            for x in frequency_domain_maxwell(
                calc_patch, pde_test_inc.e, pde_test_inc.h, case.k)]
    print("Source Maxwell residuals:", source_maxwell_resids)
    assert max(source_maxwell_resids) < 1e-6

    # }}}

    loc_sign = -1 if case.is_interior else +1

    from pytools.convergence import EOCRecorder

    eoc_rec_repr_maxwell = EOCRecorder()
    eoc_pec_bc = EOCRecorder()
    eoc_rec_e = EOCRecorder()
    eoc_rec_h = EOCRecorder()

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

    for resolution in case.resolutions:
        scat_mesh = case.get_mesh(resolution, case.target_order)
        observation_mesh = case.get_observation_mesh(case.target_order)

        pre_scat_discr = Discretization(
                cl_ctx, scat_mesh,
                InterpolatoryQuadratureSimplexGroupFactory(case.target_order))
        qbx, _ = QBXLayerPotentialSource(
                pre_scat_discr, fine_order=4*case.target_order,
                qbx_order=case.qbx_order,
                fmm_level_to_order=SimpleExpansionOrderFinder(
                    case.fmm_tolerance),
                fmm_backend=case.fmm_backend
                ).with_refinement(_expansion_disturbance_tolerance=0.05)
        h_max = bind(qbx, sym.h_max(qbx.ambient_dim))(queue)

        scat_discr = qbx.density_discr
        obs_discr = Discretization(
                cl_ctx, observation_mesh,
                InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

        inc_field_scat = EHField(eval_inc_field_at(scat_discr))
        inc_field_obs = EHField(eval_inc_field_at(obs_discr))

        # {{{ system solve

        inc_xyz_sym = EHField(sym.make_sym_vector("inc_fld", 6))

        bound_j_op = bind(qbx, mfie.j_operator(loc_sign, jt_sym))
        j_rhs = bind(qbx, mfie.j_rhs(inc_xyz_sym.h))(
                queue, inc_fld=inc_field_scat.field, **knl_kwargs)

        gmres_settings = dict(
                tol=case.gmres_tol,
                progress=True,
                hard_failure=True,
                stall_iterations=50, no_progress_factor=1.05)
        from pytential.solve import gmres
        gmres_result = gmres(
                bound_j_op.scipy_op(queue, "jt", np.complex128, **knl_kwargs),
                j_rhs, **gmres_settings)

        jt = gmres_result.solution

        bound_rho_op = bind(qbx, mfie.rho_operator(loc_sign, rho_sym))
        rho_rhs = bind(qbx, mfie.rho_rhs(jt_sym, inc_xyz_sym.e))(
                queue, jt=jt, inc_fld=inc_field_scat.field, **knl_kwargs)

        gmres_result = gmres(
                bound_rho_op.scipy_op(queue, "rho", np.complex128, **knl_kwargs),
                rho_rhs, **gmres_settings)

        rho = gmres_result.solution

        # }}}

        jxyz = bind(qbx, sym.tangential_to_xyz(jt_sym))(queue, jt=jt)

        # {{{ volume eval

        sym_repr = mfie.scattered_volume_field(jt_sym, rho_sym)

        def eval_repr_at(tgt, source=None):
            if source is None:
                source = qbx

            return bind((source, tgt), sym_repr)(queue, jt=jt, rho=rho, **knl_kwargs)

        pde_test_repr = EHField(
                vector_from_device(queue, eval_repr_at(calc_patch_tgt)))

        maxwell_residuals = [
                calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_repr.e, np.inf)
                for x in frequency_domain_maxwell(
                    calc_patch, pde_test_repr.e, pde_test_repr.h, case.k)]
        print("Maxwell residuals:", maxwell_residuals)

        eoc_rec_repr_maxwell.add_data_point(h_max, max(maxwell_residuals))

        # }}}

        # {{{ check PEC BC on total field

        bc_repr = EHField(mfie.scattered_volume_field(
            jt_sym, rho_sym, qbx_forced_limit=loc_sign))
        pec_bc_e = sym.n_cross(bc_repr.e + inc_xyz_sym.e)
        pec_bc_h = sym.normal(3).as_vector().dot(bc_repr.h + inc_xyz_sym.h)

        eh_bc_values = bind(qbx, sym.join_fields(pec_bc_e, pec_bc_h))(
                    queue, jt=jt, rho=rho, inc_fld=inc_field_scat.field,
                    **knl_kwargs)

        def scat_norm(f):
            return norm(qbx, queue, f, p=np.inf)

        e_bc_residual = scat_norm(eh_bc_values[:3]) / scat_norm(inc_field_scat.e)
        h_bc_residual = scat_norm(eh_bc_values[3]) / scat_norm(inc_field_scat.h)

        print("E/H PEC BC residuals:", h_max, e_bc_residual, h_bc_residual)

        eoc_pec_bc.add_data_point(h_max, max(e_bc_residual, h_bc_residual))

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
                ("Einc", inc_field_scat.e),
                ("Hinc", inc_field_scat.h),
                ("bdry_normals", bdry_normals),
                ("e_bc_residual", eh_bc_values[:3]),
                ("h_bc_residual", eh_bc_values[3]),
                ])

            fplot = make_field_plotter_from_bbox(
                    find_bounding_box(scat_discr.mesh), h=(0.05, 0.05, 0.3),
                    extend_factor=0.3)

            from pytential.qbx import QBXTargetAssociationFailedException

            qbx_tgt_tol = qbx.copy(target_association_tolerance=0.2)

            fplot_tgt = PointsTarget(cl.array.to_device(queue, fplot.points))
            try:
                fplot_repr = eval_repr_at(fplot_tgt, source=qbx_tgt_tol)
            except QBXTargetAssociationFailedException as e:
                fplot.write_vtk_file(
                        "failed-targets.vts",
                        [
                            ("failed_targets", e.failed_target_flags.get(queue))
                            ])
                raise

            fplot_repr = EHField(vector_from_device(queue, fplot_repr))

            fplot_inc = EHField(
                    vector_from_device(queue, eval_inc_field_at(fplot_tgt)))

            fplot.write_vtk_file(
                    "potential-%s.vts" % resolution,
                    [
                        ("E", fplot_repr.e),
                        ("H", fplot_repr.h),
                        ("Einc", fplot_inc.e),
                        ("Hinc", fplot_inc.h),
                        ]
                    )

        # }}}

        # {{{ error in E, H

        obs_repr = EHField(eval_repr_at(obs_discr))

        def obs_norm(f):
            return norm(obs_discr, queue, f, p=np.inf)

        rel_err_e = (obs_norm(inc_field_obs.e + obs_repr.e)
                / obs_norm(inc_field_obs.e))
        rel_err_h = (obs_norm(inc_field_obs.h + obs_repr.h)
                / obs_norm(inc_field_obs.h))

        # }}}

        print("ERR", h_max, rel_err_h, rel_err_e)

        eoc_rec_h.add_data_point(h_max, rel_err_h)
        eoc_rec_e.add_data_point(h_max, rel_err_e)

    print("--------------------------------------------------------")
    print("is_interior=%s" % case.is_interior)
    print("--------------------------------------------------------")

    good = True
    for which_eoc, eoc_rec, order_tol in [
            ("maxwell", eoc_rec_repr_maxwell, 1.5),
            ("PEC BC", eoc_pec_bc, 1.5),
            ("H", eoc_rec_h, 1.5),
            ("E", eoc_rec_e, 1.5)]:
        print(which_eoc)
        print(eoc_rec.pretty_print())

        if len(eoc_rec.history) > 1:
            if eoc_rec.order_estimate() < case.qbx_order - order_tol:
                good = False

    assert good

# }}}


# You can test individual routines by typing
# $ python test_maxwell.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
