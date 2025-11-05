from __future__ import annotations


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

import logging
from functools import partial

import numpy as np
import pytest

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts
from meshmode import _acf  # noqa: F401
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from meshmode.mesh.processing import find_bounding_box
from sumpy.point_calculus import CalculusPatch, frequency_domain_maxwell
from sumpy.tools import vector_from_device
from sumpy.visualization import make_field_plotter_from_bbox

from pytential import bind, norm, sym
from pytential.target import PointsTarget
from pytential.utils import pytest_teardown_function as teardown_function  # noqa: F401


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


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
        from meshmode.mesh.generation import generate_sphere
        return generate_sphere(2, target_order,
                uniform_refinement_rounds=resolution)

    def get_observation_mesh(self, target_order):
        from meshmode.mesh.generation import generate_sphere

        if self.is_interior:
            return generate_sphere(5, target_order)
        else:
            return generate_sphere(0.5, target_order)

    def get_source(self, actx):
        if self.is_interior:
            source_ctr = np.array([[0.35, 0.1, 0.15]]).T
        else:
            source_ctr = np.array([[5, 0.1, 0.15]]).T

        source_rad = 0.3

        rng = np.random.default_rng(seed=42)
        sources = source_ctr + source_rad*2*(rng.random(size=(3, 10))-0.5)
        from pytential.source import PointPotentialSource
        return PointPotentialSource(actx.from_numpy(sources))


class RoundedCubeTestCase(MaxwellTestCase):
    target_order = 8
    gmres_tol = 1e-10

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.io import FileSource, generate_gmsh
        mesh = generate_gmsh(
                FileSource("rounded-cube.step"), 2, order=3,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        from meshmode.mesh.processing import affine_map, perform_flips
        mesh = affine_map(mesh, b=np.array([-0.5, -0.5, -0.5]))
        mesh = affine_map(mesh, A=np.eye(3)*2)

        # now centered at origin and extends to -1,1

        # Flip elements--gmsh generates inside-out geometry.
        return perform_flips(mesh, np.ones(mesh.nelements, dtype=np.bool))

    def get_observation_mesh(self, target_order):
        from meshmode.mesh.generation import generate_sphere

        if self.is_interior:
            return generate_sphere(5, target_order)
        else:
            return generate_sphere(0.5, target_order)

    def get_source(self, actx):
        if self.is_interior:
            source_ctr = np.array([[0.35, 0.1, 0.15]]).T
        else:
            source_ctr = np.array([[5, 0.1, 0.15]]).T

        source_rad = 0.3

        rng = np.random.default_rng(seed=42)
        sources = source_ctr + source_rad*2*(rng.random(size=(3, 10))-0.5)
        from pytential.source import PointPotentialSource
        return PointPotentialSource(actx.from_numpy(sources))


class ElliptiPlaneTestCase(MaxwellTestCase):
    target_order = 4
    gmres_tol = 1e-7

    def get_mesh(self, resolution, target_order):
        from pytools import download_from_web_if_not_present

        download_from_web_if_not_present(
                "https://raw.githubusercontent.com/inducer/geometries/master/"
                "surface-3d/elliptiplane.brep")

        from meshmode.mesh.io import FileSource, generate_gmsh
        mesh = generate_gmsh(
                FileSource("elliptiplane.brep"), 2, order=2,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        # now centered at origin and extends to -1,1

        # Flip elements--gmsh generates inside-out geometry.
        from meshmode.mesh.processing import perform_flips
        return perform_flips(mesh, np.ones(mesh.nelements, dtype=np.bool))

    def get_observation_mesh(self, target_order):
        from meshmode.mesh.generation import generate_sphere

        if self.is_interior:
            return generate_sphere(12, target_order)
        else:
            return generate_sphere(0.5, target_order)

    def get_source(self, actx):
        if self.is_interior:
            source_ctr = np.array([[0.35, 0.1, 0.15]]).T
        else:
            source_ctr = np.array([[3, 1, 10]]).T

        source_rad = 0.3

        rng = np.random.default_rng(seed=42)
        sources = source_ctr + source_rad*2*(rng.random(size=(3, 10))-0.5)
        from pytential.source import PointPotentialSource
        return PointPotentialSource(actx.from_numpy(sources))

# }}}


tc_int = SphereTestCase(k=1.2, is_interior=True, resolutions=[0, 1],
        qbx_order=3, fmm_tolerance=1e-4)
tc_ext = SphereTestCase(k=1.2, is_interior=False, resolutions=[0, 1],
        qbx_order=3, fmm_tolerance=1e-4)

tc_rc_ext = RoundedCubeTestCase(k=6.4, is_interior=False, resolutions=[0.1],
        qbx_order=3, fmm_tolerance=1e-4)

tc_plane_ext = ElliptiPlaneTestCase(k=2, is_interior=False, resolutions=[0.15],
        qbx_order=3, fmm_tolerance=1e-4)


class EHField:
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
    # tc_int,
    tc_ext,
    ])
def test_pec_mfie_extinction(actx_factory: ArrayContextFactory, case,
        use_plane_wave=False, visualize=False):
    """For (say) is_interior=False (the 'exterior' MFIE), this test verifies
    extinction of the combined (incoming + scattered) field on the interior
    of the scatterer.
    """
    actx = actx_factory()
    knl_kwargs = {"k": case.k}

    # {{{ come up with a solution to Maxwell's equations

    j_sym = sym.make_sym_vector("j", 3)
    jt_sym = sym.make_sym_vector("jt", 2)
    rho_sym = sym.var("rho")

    from pytential.symbolic.pde.maxwell import (
        PECChargeCurrentMFIEOperator,
        get_sym_maxwell_plane_wave,
        get_sym_maxwell_point_source,
    )
    mfie = PECChargeCurrentMFIEOperator()

    test_source = case.get_source(actx)

    calc_patch = CalculusPatch(np.array([-3, 0, 0]), h=0.01)
    calc_patch_tgt = PointsTarget(actx.from_numpy(calc_patch.points))

    import pyopencl.clrandom as clrandom
    rng = clrandom.PhiloxGenerator(actx.context, seed=12)

    from pytools import obj_array

    src_j = obj_array.new_1d([
            rng.normal(actx.queue, (test_source.ndofs), dtype=np.float64)
            for _ in range(3)])

    def eval_inc_field_at(places, source=None, target=None):
        if source is None:
            source = "test_source"

        if use_plane_wave:
            # plane wave
            return bind(places,
                    get_sym_maxwell_plane_wave(
                        amplitude_vec=np.array([1, 1, 1]),
                        v=np.array([1, 0, 0]),
                        omega=case.k),
                    auto_where=target)(actx)
        else:
            # point source
            return bind(places,
                    get_sym_maxwell_point_source(mfie.kernel, j_sym, mfie.k),
                    auto_where=(source, target))(actx, j=src_j, k=case.k)

    # }}}

    loc_sign = -1 if case.is_interior else +1

    from pytools.convergence import EOCRecorder

    eoc_rec_repr_maxwell = EOCRecorder()
    eoc_pec_bc = EOCRecorder()
    eoc_rec_e = EOCRecorder()
    eoc_rec_h = EOCRecorder()

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )
    from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

    from pytential.qbx import QBXLayerPotentialSource

    for resolution in case.resolutions:
        places = {}
        scat_mesh = case.get_mesh(resolution, case.target_order)
        observation_mesh = case.get_observation_mesh(case.target_order)

        pre_scat_discr = Discretization(
                actx, scat_mesh,
                InterpolatoryQuadratureSimplexGroupFactory(case.target_order))
        qbx = QBXLayerPotentialSource(
                pre_scat_discr, fine_order=4*case.target_order,
                qbx_order=case.qbx_order,
                fmm_level_to_order=SimpleExpansionOrderFinder(
                    case.fmm_tolerance),
                fmm_backend=case.fmm_backend,
                )

        scat_discr = qbx.density_discr
        obs_discr = Discretization(
                actx, observation_mesh,
                InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

        places.update({
            "source": qbx,
            "target": qbx.density_discr,
            "test_source": test_source,
            "scat_discr": scat_discr,
            "obs_discr": obs_discr,
            "patch_target": calc_patch_tgt,
            })

        if visualize:
            qbx_tgt_tol = qbx.copy(target_association_tolerance=0.2)

            fplot = make_field_plotter_from_bbox(
                    find_bounding_box(scat_discr.mesh), h=(0.05, 0.05, 0.3),
                    extend_factor=0.3)
            fplot_tgt = PointsTarget(actx.from_numpy(fplot.points))

            places.update({
                "qbx_target_tol": qbx_tgt_tol,
                "plot_targets": fplot_tgt,
                })

        from pytential import GeometryCollection
        places = GeometryCollection(places, auto_where=("source", "target"))
        density_discr = places.get_discretization(places.auto_source.geometry)

        # {{{ system solve

        h_max = actx.to_numpy(
                bind(places, sym.h_max(qbx.ambient_dim))(actx)
                )

        pde_test_inc = EHField(vector_from_device(actx.queue,
            eval_inc_field_at(places, target="patch_target")))

        source_maxwell_resids = [
                calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_inc.e, np.inf)
                for x in frequency_domain_maxwell(
                    calc_patch, pde_test_inc.e, pde_test_inc.h, case.k)]
        print("Source Maxwell residuals:", source_maxwell_resids)
        assert max(source_maxwell_resids) < 1e-6

        inc_field_scat = EHField(eval_inc_field_at(places, target="scat_discr"))
        inc_field_obs = EHField(eval_inc_field_at(places, target="obs_discr"))

        inc_xyz_sym = EHField(sym.make_sym_vector("inc_fld", 6))

        bound_j_op = bind(places, mfie.j_operator(loc_sign, jt_sym))
        j_rhs = bind(places, mfie.j_rhs(inc_xyz_sym.h))(
                actx, inc_fld=inc_field_scat.field, **knl_kwargs)

        gmres_settings = {
                "tol": case.gmres_tol,
                "progress": True,
                "hard_failure": True,
                "stall_iterations": 50,
                "no_progress_factor": 1.05}
        from pytential.linalg.gmres import gmres
        gmres_result = gmres(
                bound_j_op.scipy_op(actx, "jt", np.complex128, **knl_kwargs),
                j_rhs, **gmres_settings)

        jt = gmres_result.solution

        bound_rho_op = bind(places, mfie.rho_operator(loc_sign, rho_sym))
        rho_rhs = bind(places, mfie.rho_rhs(jt_sym, inc_xyz_sym.e))(
                actx, jt=jt, inc_fld=inc_field_scat.field, **knl_kwargs)

        gmres_result = gmres(
                bound_rho_op.scipy_op(actx, "rho", np.complex128, **knl_kwargs),
                rho_rhs, **gmres_settings)

        rho = gmres_result.solution

        # }}}

        jxyz = bind(places, sym.tangential_to_xyz(jt_sym))(actx, jt=jt)

        # {{{ volume eval

        sym_repr = mfie.scattered_volume_field(jt_sym, rho_sym)

        def eval_repr_at(tgt, source=None, target=None):
            if source is None:
                source = "source"

            return bind(
                places, sym_repr, auto_where=(source, target)       # noqa: B023
                )(actx, jt=jt, rho=rho, **knl_kwargs)               # noqa: B023

        pde_test_repr = EHField(vector_from_device(actx.queue,
            eval_repr_at(places, target="patch_target")))

        maxwell_residuals = [
                actx.to_numpy(
                    calc_patch.norm(x, np.inf)
                    / calc_patch.norm(pde_test_repr.e, np.inf))
                for x in frequency_domain_maxwell(
                    calc_patch, pde_test_repr.e, pde_test_repr.h, case.k)]
        print("Maxwell residuals:", maxwell_residuals)

        eoc_rec_repr_maxwell.add_data_point(h_max, max(maxwell_residuals))

        # }}}

        # {{{ check PEC BC on total field

        from pytools import obj_array

        bc_repr = EHField(mfie.scattered_volume_field(
            jt_sym, rho_sym, qbx_forced_limit=loc_sign))
        pec_bc_e = sym.n_cross(bc_repr.e + inc_xyz_sym.e)
        pec_bc_h = sym.normal(3).as_vector().dot(bc_repr.h + inc_xyz_sym.h)

        eh_bc_values = bind(places, obj_array.flat(pec_bc_e, pec_bc_h))(
                    actx, jt=jt, rho=rho, inc_fld=inc_field_scat.field,
                    **knl_kwargs)

        scat_norm = partial(norm, density_discr, p=np.inf)
        e_bc_residual = actx.to_numpy(
                scat_norm(eh_bc_values[:3]) / scat_norm(inc_field_scat.e)
                )
        h_bc_residual = actx.to_numpy(
                scat_norm(eh_bc_values[3]) / scat_norm(inc_field_scat.h)
                )

        print("E/H PEC BC residuals:", h_max, e_bc_residual, h_bc_residual)

        eoc_pec_bc.add_data_point(h_max, max(e_bc_residual, h_bc_residual))

        # }}}

        # {{{ visualization

        if visualize:
            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(actx, scat_discr, case.target_order+3)

            bdry_normals = bind(places,
                    sym.normal(3, dofdesc="scat_discr")
                    )(actx).as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
                ("j", jxyz),
                ("rho", rho),
                ("Einc", inc_field_scat.e),
                ("Hinc", inc_field_scat.h),
                ("bdry_normals", bdry_normals),
                ("e_bc_residual", eh_bc_values[:3]),
                ("h_bc_residual", eh_bc_values[3]),
                ])

            from pytential.qbx import QBXTargetAssociationFailedError
            try:
                fplot_repr = eval_repr_at(places,
                        target="plot_targets", source="qbx_target_tol")
            except QBXTargetAssociationFailedError as e:
                fplot.write_vtk_file(
                        "failed-targets.vts",
                        [
                            ("failed_targets", actx.to_numpy(
                                actx.thaw(e.failed_target_flags))),
                            ])
                raise

            fplot_repr = EHField(vector_from_device(actx.queue, fplot_repr))
            fplot_inc = EHField(vector_from_device(actx.queue,
                eval_inc_field_at(places, target="plot_targets")))

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

        obs_repr = EHField(eval_repr_at(places, target="obs_discr"))

        obs_norm = partial(norm, obs_discr, p=np.inf)
        rel_err_e = actx.to_numpy(
                obs_norm(inc_field_obs.e + obs_repr.e)
                / obs_norm(inc_field_obs.e))
        rel_err_h = actx.to_numpy(
                obs_norm(inc_field_obs.h + obs_repr.h)
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
