from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2017 - 2018 Andreas Kloeckner
Copyright (C) 2017 - 2018 Christian Howard
"""

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
    #fmm_backend = "sumpy"

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

@pytest.mark.parametrize("case", [
    #tc_int,
    tc_ext,
    ])
def test_pec_dpie_extinction(ctx_getter, case, visualize=False):
    """For (say) is_interior=False (the 'exterior' MFIE), this test verifies
    extinction of the combined (incoming + scattered) field on the interior
    of the scatterer.
    """

    # setup the basic config for logging
    logging.basicConfig(level=logging.INFO)

    # setup the OpenCL context and queue
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # import or skip pyfmmlib
    pytest.importorskip("pyfmmlib")

    # initialize the random seed
    np.random.seed(12)

    # specify a dictionary with some useful arguments
    knl_kwargs = {"k": case.k}

    # specify the list of geometry objects being used
    geom_list = ["obj0"]

    # {{{ come up with a solution to Maxwell's equations

    # define symbolic variable for current density
    # for use in defining incident field
    j_sym = sym.make_sym_vector("j", 3)

    # import some functionality from maxwell into this
    # local scope environment
    #from pytential.symbolic.pde.maxwell import (get_sym_maxwell_point_source,get_sym_maxwell_point_source_potentials,get_sym_maxwell_plane_wave,DPIEOperator)
    #from pytential.symbolic.pde.maxwell.dpie import (get_sym_maxwell_point_source,get_sym_maxwell_point_source_potentials,get_sym_maxwell_plane_wave,DPIEOperator)
    import pytential.symbolic.pde.maxwell       as mw
    import pytential.symbolic.pde.maxwell.dpie  as mw_dpie
    
    # initialize the DPIE operator based on the geometry list
    dpie = mw_dpie.DPIEOperator(geometry_list=geom_list)

    # specify some symbolic variables that will be used
    # in the process to solve integral equations for the DPIE
    phi_densities   = sym.make_sym_vector("phi_densities", dpie.num_scalar_potential_densities())
    A_densities     = sym.make_sym_vector("A_densities", dpie.num_vector_potential_densities())
    tau_densities = sym.make_sym_vector("tau_densities", dpie.num_distinct_objects())

    # get test source locations from the passed in case's queue
    test_source = case.get_source(queue)

    # create the calculus patch and get calculus patch for targets
    calc_patch = CalculusPatch(np.array([-3, 0, 0]), h=0.01)
    calc_patch_tgt = PointsTarget(cl.array.to_device(queue, calc_patch.points))

    # define a random number generator based on OpenCL
    rng = cl.clrandom.PhiloxGenerator(cl_ctx, seed=12)

    # define some parameters for the incident wave
    # direction for the wave
    u_dir = np.array([1, 0, 0],dtype=np.complex128)

    # polarization vector
    Ep = np.array([1, 1, 1],dtype=np.complex128)

    # define symbolic vectors for use
    uvar = sym.make_sym_vector("u", 3)
    Evar = sym.make_sym_vector("Ep",3)

    # define functions that can be used to generate incident fields for an input discretization
    # define potentials based on incident plane wave
    def get_incident_plane_wave_EHField(tgt):
        return bind((test_source,tgt),mw.get_sym_maxwell_plane_wave(amplitude_vec=Ep, v=u_dir, omega=dpie.k))(queue,k=case.k,u=u_dir,Ep=Ep)

    # get the gradphi_inc field evaluated at some source locations
    def get_incident_gradphi(objects, target=None):
        return bind(objects,mw.get_sym_maxwell_planewave_gradphi(u=u_dir, Ep=Ep, k=dpie.k,where=target))(queue,k=case.k,u=u_dir,Ep=Ep)

    # get the incident plane wave div(A)
    def get_incident_divA(objects, target=None):
        return bind(objects,mw.get_sym_maxwell_planewave_divA(u=uvar, Ep=Evar, k=dpie.k,where=target))(queue,k=case.k,u=u_dir,Ep=Ep)

    # method to get vector potential and scalar potential for incident 
    # E-M fields
    def get_incident_potentials(objects, target=None):
        return bind(objects,mw.get_sym_maxwell_planewave_potentials(u=uvar, Ep=Evar, k=dpie.k,where=target))(queue, k=case.k,u=u_dir,Ep=Ep)

    # define a smooth function to represent the density
    def dummy_density(omega = 1.0, where=None):
        x = sym.nodes(3, where).as_vector()
        return sym.sin(omega*sym.n_dot(x,where))

    # get the Electromagnetic field evaluated at the target calculus patch
    pde_test_inc = EHField(vector_from_device(queue, get_incident_plane_wave_EHField(calc_patch_tgt)))

    # compute residuals of incident field at source points
    source_maxwell_resids = [
            calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_inc.e, np.inf)
            for x in frequency_domain_maxwell(
                calc_patch, pde_test_inc.e, pde_test_inc.h, case.k)]

    # make sure Maxwell residuals are small so we know the incident field
    # properly satisfies the maxwell equations
    print("Source Maxwell residuals:", source_maxwell_resids)
    assert max(source_maxwell_resids) < 1e-6

    # }}}


    # # {{{ Test the representations
    test_representations = True

    if test_representations:

        # import a bunch of stuff that will be useful
        from pytools.convergence import EOCRecorder
        from pytential.qbx import QBXLayerPotentialSource
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
        from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder


        # # loop through the case's resolutions and compute the scattered field solution
        rep_error = []
        for resolution in case.resolutions:

            # get the scattered and observation mesh
            scat_mesh           = case.get_mesh(resolution, case.target_order)
            observation_mesh    = case.get_observation_mesh(case.target_order)

            # define the pre-scattered discretization
            pre_scat_discr = Discretization(
                    cl_ctx, scat_mesh,
                    InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

            # use OpenCL random number generator to create a set of random
            # source locations for various variables being solved for
            dpie0 = mw_dpie.DPIEOperator(geometry_list=geom_list)
            qbx0, _ = QBXLayerPotentialSource(
                        pre_scat_discr, fine_order=4*case.target_order,
                        #fmm_order=False,
                        qbx_order=case.qbx_order,
                        fmm_level_to_order=SimpleExpansionOrderFinder(case.fmm_tolerance),
                        fmm_backend=case.fmm_backend
                        ).with_refinement(_expansion_disturbance_tolerance=0.05)

            # define the geometry dictionary
            geom_map = {"obj0":qbx0, "obj0t":qbx0.density_discr, "scat":qbx0.density_discr}
            dummy_phi = np.array([None]*dpie0.num_scalar_potential_densities(),dtype=dpie0.stype)
            dummy_A = np.array([None]*dpie0.num_vector_potential_densities(),dtype=dpie0.stype)
            v = rng.normal(queue, (qbx0.density_discr.nnodes,), dtype=np.float64)
            s = rng.normal(queue, (), dtype=np.float64)
            for i in range(0,len(dummy_phi)):
                dummy_phi[i] = bind(geom_map,dummy_density(where='obj0'))(queue)
            for i in range(0,len(dummy_A)):
                dummy_A[i] = bind(geom_map,dummy_density(where='obj0'))(queue)
            test_tau_op= bind(geom_map,dpie0.subproblem_operator(tau_densities=tau_densities))
            test_tau_rhs= bind(geom_map,dpie0.subproblem_rhs(A_densities=A_densities))(queue,A_densities=dummy_A,**knl_kwargs)

            # set GMRES settings for solving
            gmres_settings = dict(
                    tol=case.gmres_tol,
                    progress=True,
                    hard_failure=True,
                    stall_iterations=50, no_progress_factor=1.05)

            # get the GMRES functionality
            from pytential.solve import gmres

            subprob_result = gmres(
                    test_tau_op.scipy_op(queue, "tau_densities", np.complex128, domains=dpie0.get_subproblem_domain_list(), **knl_kwargs),
                    test_tau_rhs, **gmres_settings)
            dummy_tau    = subprob_result.solution

            sym_repr0 = dpie.scattered_volume_field(phi_densities,A_densities,tau_densities,target='tgt')

            def eval_test_repr_at(tgt):
                map = geom_map
                map['tgt'] = tgt
                return bind(map, sym_repr0)(queue, phi_densities=dummy_phi, A_densities=dummy_A, tau_densities=dummy_tau, **knl_kwargs)

            pde_test_repr = EHField(vector_from_device(queue, eval_test_repr_at(calc_patch_tgt)))

            maxwell_residuals = [
                    calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_repr.e, np.inf)
                    for x in frequency_domain_maxwell(calc_patch, pde_test_repr.e, pde_test_repr.h, case.k)]
            print("Maxwell residuals:", maxwell_residuals)
            rep_error.append(maxwell_residuals)

        print("Representation Error Results:")
        for n in range(0,len(rep_error)):
            print("Case {0}: {1}".format(n+1,rep_error[n]))

    # #}}}


    # # {{{ Test the operators
    test_operators = False
    if test_operators:

        # define error array
        op_error = []

        # define method to get locations to evaluate representation
        def epsilon_off_boundary(where=None, epsilon=1e-4):
            x = sym.nodes(3, where).as_vector()
            return x + sym.normal(3,3,where).as_vector()*epsilon

        # import a bunch of stuff that will be useful
        from pytools.convergence import EOCRecorder
        from pytential.qbx import QBXLayerPotentialSource
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
        from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder


        # loop through the case's resolutions and compute the scattered field solution
        for resolution in case.resolutions:

            # get the scattered and observation mesh
            scat_mesh           = case.get_mesh(resolution, case.target_order)
            observation_mesh    = case.get_observation_mesh(case.target_order)

            # define the pre-scattered discretization
            pre_scat_discr = Discretization(
                    cl_ctx, scat_mesh,
                    InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

            # use OpenCL random number generator to create a set of random
            # source locations for various variables being solved for
            dpie0 = mw_dpie.DPIEOperator(geometry_list=geom_list)
            qbx0, _ = QBXLayerPotentialSource(
                        pre_scat_discr, fine_order=4*case.target_order,
                        #fmm_order=False,
                        qbx_order=case.qbx_order,
                        fmm_level_to_order=SimpleExpansionOrderFinder(case.fmm_tolerance),
                        fmm_backend=case.fmm_backend
                        ).with_refinement(_expansion_disturbance_tolerance=0.05)

            # define the geometry dictionary
            geom_map = {"obj0":qbx0, "obj0t":qbx0.density_discr, "scat":qbx0.density_discr}

            # init random dummy densities for the vector and scalar potentials
            dummy_phi = np.array([None]*dpie0.num_scalar_potential_densities(),dtype=dpie0.stype)
            dummy_A = np.array([None]*dpie0.num_vector_potential_densities(),dtype=dpie0.stype)
            dummy_tau = np.array([None]*dpie0.num_distinct_objects(),dtype=dpie0.stype)

            v = rng.normal(queue, (qbx0.density_discr.nnodes,), dtype=np.float64)
            s = rng.normal(queue, (1,), dtype=np.float64)[0]
            for i in range(0,len(dummy_phi)-1):
                dummy_phi[i] = bind(geom_map,dummy_density(where='obj0'))(queue)
            dummy_phi[1] = s
            for i in range(0,len(dummy_A)-1):
                dummy_A[i] = bind(geom_map,dummy_density(where='obj0'))(queue)
            dummy_A[3] = s
            for i in range(0,len(dummy_tau)):
                dummy_tau[i] = bind(geom_map,dummy_density(where='obj0'))(queue)

            # check that the scalar density operator and representation are similar
            scalar_op = dpie0.phi_operator(phi_densities=phi_densities)
            vector_op = dpie0.a_operator(A_densities=A_densities)
            tau_op = dpie0.subproblem_operator(tau_densities=tau_densities)

            # evaluate operators at the dummy densities
            scalar_op_eval = bind(geom_map, scalar_op)(queue, phi_densities=dummy_phi, **knl_kwargs)
            vector_op_eval = bind(geom_map, vector_op)(queue, A_densities=dummy_A, **knl_kwargs)
            tau_op_eval = bind(geom_map, tau_op)(queue, tau_densities=dummy_tau, **knl_kwargs)

            # compute off-boundary locations that the representation will need to be evaluated at
            tgt_n = bind(geom_map,epsilon_off_boundary(where='obj0'))
            geom_map['tgt'] = tgt_n
            scalar_rep_eval = bind(geom_map, dpie0.scalar_potential_rep(phi_densities=phi_densities, target='tgt'))(queue, phi_densities=dummy_phi, **knl_kwargs)
            vector_rep_eval = bind(geom_map, dpie0.vector_potential_rep(A_densities=A_densities, target='tgt'))(queue, A_densities=dummy_A, **knl_kwargs)
            tau_op_eval = bind(geom_map, dpie0.subproblem_rep(tau_densities=tau_densities,target='tgt'))(queue, tau_densities=dummy_tau, **knl_kwargs)

            # compute the error between the operator values and the representation values
            def error_diff(u,v):
                return norm(tgt_n, queue, u-v, p=np.inf)
            op_error.append([error_diff(scalar_op_eval,scalar_rep_eval), error_diff(vector_op_eval,vector_rep_eval), error_diff(sigma_op_eval,sigma_op_eval)])

        # print the resulting error results
        print("Operator Error Results:")
        for n in range(0,len(op_error)):
            print("Case {0}: {1}".format(n+1,op_error[n]))

    # #}}}


    # {{{ Solve for the scattered field
    solve_scattered_field = False
    if solve_scattered_field:
        loc_sign = -1 if case.is_interior else +1

        # import a bunch of stuff that will be useful
        from pytools.convergence import EOCRecorder
        from pytential.qbx import QBXLayerPotentialSource
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
        from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

        # setup an EOC Recorder
        eoc_rec_repr_maxwell    = EOCRecorder()
        eoc_pec_bc              = EOCRecorder()
        eoc_rec_e               = EOCRecorder()
        eoc_rec_h               = EOCRecorder()

        # loop through the case's resolutions and compute the scattered field solution
        for resolution in case.resolutions:

            # get the scattered and observation mesh
            scat_mesh           = case.get_mesh(resolution, case.target_order)
            observation_mesh    = case.get_observation_mesh(case.target_order)

            # define the pre-scattered discretization
            pre_scat_discr = Discretization(
                    cl_ctx, scat_mesh,
                    InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

            # obtain the QBX layer potential source object
            qbx, _ = QBXLayerPotentialSource(
                    pre_scat_discr, fine_order=4*case.target_order,
                    #fmm_order=False,
                    qbx_order=case.qbx_order,
                    fmm_level_to_order=SimpleExpansionOrderFinder(case.fmm_tolerance),
                    fmm_backend=case.fmm_backend
                    ).with_refinement(_expansion_disturbance_tolerance=0.05)

            # define the geometry dictionary
            #geom_map = {"g0": qbx}
            geom_map = {"obj0":qbx, "obj0t":qbx.density_discr, "scat":qbx.density_discr}

            # get the maximum mesh element edge length
            h_max = qbx.h_max

            # define the scattered and observation discretization
            scat_discr  = qbx.density_discr
            obs_discr   = Discretization(cl_ctx, observation_mesh,
                                         InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

            # get the incident field at the scatter and observation locations
            #inc_EM_field_scat   = EHField(eval_inc_field_at(scat_discr))
            #inc_EM_field_obs    = EHField(eval_inc_field_at(obs_discr))
            #inc_vec_field_scat  = get_inc_potentials(scat_discr)
            #inc_vec_field_obs   = get_inc_potentials(obs_discr)

            # get the incident fields used for boundary conditions
            (phi_inc, A_inc) = get_incident_potentials(geom_map,'scat')
            inc_divA_scat = get_incident_divA(geom_map,'scat')
            inc_gradPhi_scat = get_incident_gradphi(geom_map,'scat')

            # {{{ solve the system of integral equations
            inc_A = sym.make_sym_vector("inc_A", 3)
            inc_phi = sym.make_sym_vector("inc_phi",1)
            inc_divA = sym.make_sym_vector("inc_divA",1)
            inc_gradPhi = sym.make_sym_vector("inc_gradPhi", 3)

            # setup operators that will be solved
            phi_op  = bind(geom_map,dpie.phi_operator(phi_densities=phi_densities))
            A_op    = bind(geom_map,dpie.a_operator(A_densities=A_densities))

            # setup the RHS with provided data so we can solve for density values across the domain
            phi_rhs = bind(geom_map,dpie.phi_rhs(phi_inc=inc_phi,gradphi_inc=inc_gradPhi))(queue,inc_phi=phi_inc,inc_gradPhi=inc_gradPhi_scat,**knl_kwargs)
            A_rhs   = bind(geom_map,dpie.a_rhs(A_inc=inc_A,divA_inc=inc_divA))(queue,inc_A=A_inc,inc_divA=inc_divA_scat,**knl_kwargs)

            # set GMRES settings for solving
            gmres_settings = dict(
                    tol=case.gmres_tol,
                    progress=True,
                    hard_failure=True,
                    stall_iterations=50, no_progress_factor=1.05)

            # get the GMRES functionality
            from pytential.solve import gmres

            # solve for the scalar potential densities
            gmres_result = gmres(
                            phi_op.scipy_op(queue, "phi_densities", np.complex128, domains=dpie.get_scalar_domain_list(),**knl_kwargs),
                            phi_rhs, **gmres_settings)
            phi_dens    = gmres_result.solution

            # solve for the vector potential densities
            gmres_result = gmres(
                    A_op.scipy_op(queue, "A_densities", np.complex128, domains=dpie.get_vector_domain_list(), **knl_kwargs),
                    A_rhs, **gmres_settings)
            A_dens      = gmres_result.solution

            # solve sub problem for sigma densities
            tau_op= bind(geom_map,dpie.subproblem_operator(tau_densities=tau_densities))
            tau_rhs= bind(geom_map,dpie.subproblem_rhs(A_densities=A_densities))(queue,A_densities=A_dens,**knl_kwargs)
            gmres_result = gmres(
                    sigma_op.scipy_op(queue, "tau_densities", np.complex128, domains=dpie.get_subproblem_domain_list(), **knl_kwargs),
                    sigma_rhs, **gmres_settings)
            tau_dens      = gmres_result.solution

            # extract useful solutions
            #phi     = bind(geom_map, dpie.scalar_potential_rep(phi_densities=phi_densities))(queue, phi_densities=phi_dens)
            #Axyz    = bind(geom_map, dpie.vector_potential_rep(A_densities=A_densities))(queue, A_densities=A_dens)

            # }}}

            # {{{ volume eval

            sym_repr = dpie.scattered_volume_field(phi_densities,A_densities,tau_densities,target='tgt')

            def eval_repr_at(tgt):
                map = geom_map
                map['tgt'] = tgt
                return bind(map, sym_repr)(queue, phi_densities=phi_dens, A_densities=A_dens, tau_densities=tau_dens, **knl_kwargs)

            pde_test_repr = EHField(vector_from_device(queue, eval_repr_at(calc_patch_tgt)))

            maxwell_residuals = [
                    calc_patch.norm(x, np.inf) / calc_patch.norm(pde_test_repr.e, np.inf)
                    for x in frequency_domain_maxwell(calc_patch, pde_test_repr.e, pde_test_repr.h, case.k)]
            print("Maxwell residuals:", maxwell_residuals)

            eoc_rec_repr_maxwell.add_data_point(h_max, max(maxwell_residuals))

            # }}}

            # {{{ check PEC BC on total field

            #bc_repr = EHField(dpie.scattered_volume_field(
            #    phi_densities, A_densities, qbx_forced_limit=loc_sign))
            #pec_bc_e = sym.n_cross(bc_repr.e + inc_xyz_sym.e)
            #pec_bc_h = sym.normal(3).as_vector().dot(bc_repr.h + sym.curl(inc_xyz_vec_sym[1:]))

            #eh_bc_values = bind(qbx, sym.join_fields(pec_bc_e, pec_bc_h))(
            #            queue, phi_densities=phi_dens, A_densities=A_dens, inc_vec_fld=inc_field_vec_scat,
            #            **knl_kwargs)

            #def scat_norm(f):
            #    return norm(qbx, queue, f, p=np.inf)

            #e_bc_residual = scat_norm(eh_bc_values[:3]) / scat_norm(inc_EM_field_scat.e)
            #h_bc_residual = scat_norm(eh_bc_values[3]) / scat_norm(inc_EM_field_scat.h)

            #print("E/H PEC BC residuals:", h_max, e_bc_residual, h_bc_residual)

            #eoc_pec_bc.add_data_point(h_max, max(e_bc_residual, h_bc_residual))

            # }}}

            # {{{ visualization

            if visualize:
                from meshmode.discretization.visualization import make_visualizer
                bdry_vis = make_visualizer(queue, scat_discr, case.target_order+3)

                bdry_normals = bind(scat_discr, sym.normal(3))(queue)\
                        .as_vector(dtype=object)

                bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
                    ("phi", phi),
                    ("Axyz", Axyz),
                    ("Einc", inc_EM_field_scat.e),
                    ("Hinc", inc_EM_field_scat.h),
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

            #inc_field_scat = EHField(get_incident_plane_wave_EHField(scat_discr))
            #inc_field_obs = EHField(get_incident_plane_wave_EHField(obs_discr))

            #rel_err_e = (obs_norm(inc_field_obs.e + obs_repr.e)
            #        / obs_norm(inc_field_obs.e))
            #rel_err_h = (obs_norm(inc_field_obs.h + obs_repr.h)
            #        / obs_norm(inc_field_obs.h))

            # }}}

            #print("ERR", h_max, rel_err_h, rel_err_e)

            #eoc_rec_h.add_data_point(h_max, rel_err_h)
            #eoc_rec_e.add_data_point(h_max, rel_err_e)

        print("--------------------------------------------------------")
        print("is_interior=%s" % case.is_interior)
        print("--------------------------------------------------------")

        good = True
        for which_eoc, eoc_rec, order_tol in [
                ("maxwell", eoc_rec_repr_maxwell, 1.5)
                #("PEC BC", eoc_pec_bc, 1.5),
                #("H", eoc_rec_h, 1.5),
                #("E", eoc_rec_e, 1.5)
                ]:
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
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
