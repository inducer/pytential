from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pytools import Record
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh)
from meshmode.discretization.visualization import make_visualizer
from sumpy.symbolic import USE_SYMENGINE
from sumpy.tools import build_matrix
from sumpy.visualization import FieldPlotter
from pytential import bind, sym
from pytential.qbx import QBXTargetAssociationFailedException

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


def make_circular_point_group(ambient_dim, npoints, radius,
        center=np.array([0., 0.]), func=lambda x: x):
    t = func(np.linspace(0, 1, npoints, endpoint=False)) * (2 * np.pi)
    center = np.asarray(center)
    result = np.zeros((ambient_dim, npoints))
    result[:2, :] = center[:, np.newaxis] + radius*np.vstack((np.cos(t), np.sin(t)))
    return result


# {{{ test cases

class IntEqTestCase:

    @property
    def default_helmholtz_k(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def qbx_order(self):
        raise NotImplementedError

    @property
    def target_order(self):
        raise NotImplementedError

    def __init__(self, helmholtz_k, bc_type, prob_side):
        """
        :arg prob_side: may be -1, +1, or ``'scat'`` for a scattering problem
        """

        if helmholtz_k is None:
            helmholtz_k = self.default_helmholtz_k

        self.helmholtz_k = helmholtz_k
        self.bc_type = bc_type
        self.prob_side = prob_side

    @property
    def k(self):
        return self.helmholtz_k

    def __str__(self):
        return ("name: %s, bc_type: %s, prob_side: %s, "
                "helmholtz_k: %s, qbx_order: %d, target_order: %d"
            % (self.name, self.bc_type, self.prob_side, self.helmholtz_k,
                self.qbx_order, self.target_order))

    fmm_backend = "sumpy"
    gmres_tol = 1e-12


class CurveIntEqTestCase(IntEqTestCase):
    resolutions = [40, 50, 60]

    def curve_func(self, *args, **kwargs):
        raise NotImplementedError

    def get_mesh(self, resolution, target_order):
        return make_curve_mesh(
                self.curve_func,
                np.linspace(0, 1, resolution+1),
                target_order)

    use_refinement = True

    inner_radius = 0.1
    outer_radius = 2

    qbx_order = 5
    target_order = 5
    fmm_order = 15
    fmm_backend = "sumpy"



class EllipseIntEqTestCase(CurveIntEqTestCase):
    name = "3-to-1 ellipse"

    def __init__(self, ratio=3, *args, **kwargs):
        super(EllipseIntEqTestCase, self).__init__(*args, **kwargs)
        self.ratio = ratio

    def curve_func(self, x):
        return ellipse(self.ratio, x)

# }}}


# {{{ test backend
# }}}


ctx_factory = cl._csc
case = EllipseIntEqTestCase(helmholtz_k=0, bc_type="clamped_plate",
        prob_side=-1, ratio=3)
visualize=False


logging.basicConfig(level=logging.INFO)

cl_ctx = ctx_factory()
queue = cl.CommandQueue(cl_ctx)
resolution = 30
max_err = []
x = []
for resolution in [10]:
    x.append(resolution)

    if USE_SYMENGINE and case.fmm_backend is None:
        pytest.skip("https://gitlab.tiker.net/inducer/sumpy/issues/25")

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()


    mesh = case.get_mesh(resolution, case.target_order)
    print("%d elements" % mesh.nelements)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

    source_order = 4*case.target_order

    refiner_extra_kwargs = {}

    qbx_lpot_kwargs = {}
    if case.fmm_backend is None:
        qbx_lpot_kwargs["fmm_order"] = False
    else:
        if hasattr(case, "fmm_tol"):
            from sumpy.expansion.flevel_to_order import SimpleExpansionOrderFinder
            qbx_lpot_kwargs["fmm_level_to_order"] = SimpleExpansionOrderFinder(
                    case.fmm_tol)

        elif hasattr(case, "fmm_order"):
            qbx_lpot_kwargs["fmm_order"] = case.fmm_order
        else:
            qbx_lpot_kwargs["fmm_order"] = case.qbx_order + 5

    case.qbx_order = 5

    qbx = QBXLayerPotentialSource(
            pre_density_discr,
            fine_order=source_order,
            qbx_order=case.qbx_order,
            target_association_tolerance=0.05,
            _box_extent_norm=getattr(case, "box_extent_norm", None),
            _from_sep_smaller_crit=getattr(case, "from_sep_smaller_crit", None),
            _from_sep_smaller_min_nsources_cumul=30,
            fmm_backend=case.fmm_backend, **qbx_lpot_kwargs)

    if case.use_refinement:
        if case.k != 0 and getattr(case, "refine_on_helmholtz_k", True):
            refiner_extra_kwargs["kernel_length_scale"] = 5/case.k

        if hasattr(case, "scaled_max_curvature_threshold"):
            refiner_extra_kwargs["_scaled_max_curvature_threshold"] = \
                    case.scaled_max_curvature_threshold

        if hasattr(case, "expansion_disturbance_tolerance"):
            refiner_extra_kwargs["_expansion_disturbance_tolerance"] = \
                    case.expansion_disturbance_tolerance

        if hasattr(case, "refinement_maxiter"):
            refiner_extra_kwargs["maxiter"] = case.refinement_maxiter

        #refiner_extra_kwargs["visualize"] = True

        print("%d elements before refinement" % pre_density_discr.mesh.nelements)
        qbx, _ = qbx.with_refinement(**refiner_extra_kwargs)
        print("%d stage-1 elements after refinement"
                % qbx.density_discr.mesh.nelements)
        print("%d stage-2 elements after refinement"
                % qbx.stage2_density_discr.mesh.nelements)
        print("quad stage-2 elements have %d nodes"
                % qbx.quad_stage2_density_discr.groups[0].nunit_nodes)

    density_discr = qbx.density_discr

    if hasattr(case, "visualize_geometry") and case.visualize_geometry:
        bdry_normals = bind(
                density_discr, sym.normal(mesh.ambient_dim)
                )(queue).as_vector(dtype=object)

        bdry_vis = make_visualizer(queue, density_discr, case.target_order)
        bdry_vis.write_vtk_file("geometry.vtu", [
            ("normals", bdry_normals)
            ])

    # {{{ plot geometry

    if 0:
        if mesh.ambient_dim == 2:
            # show geometry, centers, normals
            nodes_h = density_discr.nodes().get(queue=queue)
            pt.plot(nodes_h[0], nodes_h[1], "x-")
            normal = bind(density_discr, sym.normal(2))(queue).as_vector(np.object)
            pt.quiver(nodes_h[0], nodes_h[1],
                    normal[0].get(queue), normal[1].get(queue))
            pt.gca().set_aspect("equal")
            pt.show()

        elif mesh.ambient_dim == 3:
            bdry_vis = make_visualizer(queue, density_discr, case.target_order+3)

            bdry_normals = bind(density_discr, sym.normal(3))(queue)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("pre-solve-source-%s.vtu" % resolution, [
                ("bdry_normals", bdry_normals),
                ])

        else:
            raise ValueError("invalid mesh dim")

    # }}}

    # {{{ set up operator

    from pytential.symbolic.pde.scalar import (
        DirichletOperator,
        NeumannOperator,
        BiharmonicClampedPlateOperator,
    )

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel, BiharmonicKernel

    knl = BiharmonicKernel(mesh.ambient_dim)
    knl_kwargs = {}
    concrete_knl_kwargs = {}

    sigma_sym = sym.make_sym_vector("sigma", 2)

    if knl.is_complex_valued:
        dtype = np.complex128
    else:
        dtype = np.float64

    loc_sign = +1 if case.prob_side in [+1, "scat"] else -1

    if case.bc_type == "dirichlet":
        op = DirichletOperator(knl, loc_sign, use_l2_weighting=True,
                kernel_arguments=knl_kwargs)
    elif case.bc_type == "neumann":
        op = NeumannOperator(knl, loc_sign, use_l2_weighting=True,
                 use_improved_operator=False, kernel_arguments=knl_kwargs)
    elif case.bc_type == "clamped_plate":
        op = BiharmonicClampedPlateOperator(knl)
    else:
        assert False

    # }}}

    # {{{ set up test data

    if case.prob_side == -1:
        test_src_geo_radius = case.outer_radius
        test_tgt_geo_radius = case.inner_radius
    elif case.prob_side == +1:
        test_src_geo_radius = case.inner_radius
        test_tgt_geo_radius = case.outer_radius
    elif case.prob_side == "scat":
        test_src_geo_radius = case.outer_radius
        test_tgt_geo_radius = case.outer_radius
    else:
        raise ValueError("unknown problem_side")

    point_sources = make_circular_point_group(
            mesh.ambient_dim, 10, test_src_geo_radius,
            func=lambda x: x**1.5)
    test_targets = make_circular_point_group(
            mesh.ambient_dim, 20, test_tgt_geo_radius)

    np.random.seed(22)
    source_charges = np.random.randn(point_sources.shape[1])
    source_charges[-1] = -np.sum(source_charges[:-1])
    source_charges = source_charges.astype(dtype)
    assert np.sum(source_charges) < 1e-15

    source_charges_dev = cl.array.to_device(queue, source_charges)

    # }}}

    # {{{ establish BCs

    from pytential.source import PointPotentialSource
    from pytential.target import PointsTarget

    point_source = PointPotentialSource(cl_ctx, point_sources)

    pot_src = sym.IntG(
        # FIXME: qbx_forced_limit--really?
        knl, sym.var("charges"), qbx_forced_limit=None, **knl_kwargs)

    if case.bc_type != "clamped_plate":
        raise RuntimeError("Not Implemented")
    bc1 = bind((point_source, density_discr), pot_src)(
            queue, charges=source_charges_dev, **concrete_knl_kwargs)
    bc2 = bind(
            (point_source, density_discr),
            sym.normal_derivative(
                qbx.ambient_dim, pot_src, where=sym.DEFAULT_TARGET)
            )(queue, charges=source_charges_dev, **concrete_knl_kwargs)


    dv = sym.normal(2).as_vector()
    dt = [-dv[1], dv[0]]

    from sumpy.kernel import DirectionalSourceDerivative, DirectionalTargetDerivative, AxisTargetDerivative
    def dv(knl):
        return DirectionalSourceDerivative(knl, "normal_dir")
    def dt(knl):
        return DirectionalSourceDerivative(knl, "tangent_dir")

    normal_dir = sym.normal(2).as_vector()
    tangent_dir = np.array([-normal_dir[1], normal_dir[0]])


    if 1:

        k1 = sym.S(dv(dv(dv(knl))), sigma_sym[0], kernel_arguments={"normal_dir": normal_dir}, qbx_forced_limit='avg') + \
                    3*sym.S(dv(dt(dt(knl))), sigma_sym[0], kernel_arguments={"normal_dir": normal_dir, "tangent_dir": tangent_dir}, qbx_forced_limit='avg')

        k2 = -sym.S(dv(dv(knl)), sigma_sym[1], kernel_arguments={"normal_dir": normal_dir}, qbx_forced_limit='avg') + \
                    sym.S(dt(dt(knl)), sigma_sym[1], kernel_arguments={"tangent_dir": tangent_dir}, qbx_forced_limit='avg')

        k3 = sym.normal_derivative(2, k1)
        k4 = sym.normal_derivative(2, k2)

        int1 = sigma_sym[0]/2 + k1 + k2
        int2 = -sym.mean_curvature(qbx.ambient_dim)*sigma_sym[0] + sigma_sym[1]/2 + k3 + k4

        bound_op = bind(qbx, np.array([int1, int2]))


        def dv2(knl):
            return DirectionalSourceDerivative(knl, "normal_dir_a")
        def dt2(knl):
            return DirectionalSourceDerivative(knl, "tangent_dir_a")


        sigma_sym2 = sym.make_sym_vector("sigma2", 2)
        _k1 = sym.S(dv2(dv2(dv2(knl))), sigma_sym2[0], kernel_arguments={"normal_dir_a": normal_dir}, qbx_forced_limit=None) + \
                    3*sym.S(dv2(dt2(dt2(knl))), sigma_sym2[0], kernel_arguments={"normal_dir_a": normal_dir, "tangent_dir_a": tangent_dir}, qbx_forced_limit=None)

        _k2 = -sym.S(dv2(dv2(knl)), sigma_sym2[1], kernel_arguments={"normal_dir_a": normal_dir}, qbx_forced_limit=None) + \
                    sym.S(dt2(dt2(knl)), sigma_sym2[1], kernel_arguments={"tangent_dir_a": tangent_dir}, qbx_forced_limit=None)

        _k3 = sym.S(AxisTargetDerivative(0, dv2(dv2(dv2(knl)))), sigma_sym2[0], kernel_arguments={"normal_dir_a": normal_dir}, qbx_forced_limit=None) + \
                    3*sym.S(AxisTargetDerivative(0, dv2(dt2(dt2(knl)))), sigma_sym2[0], kernel_arguments={"normal_dir_a": normal_dir, "tangent_dir_a": tangent_dir}, qbx_forced_limit=None)

    else:
        # Only k11
        sigma = sym.var("sigma")
        k1 = 0
        k1 = k1 + sym.S(dv(dv(dv(knl))), sigma, kernel_arguments={"normal_dir": normal_dir}, qbx_forced_limit='avg')
        k1 = k1 + 3*sym.S(dv(dt(dt(knl))), sigma, kernel_arguments={"normal_dir": normal_dir, "tangent_dir": tangent_dir}, qbx_forced_limit='avg')

        k2 = -sym.S(dv(dv(knl)), sigma, kernel_arguments={"normal_dir": normal_dir}, qbx_forced_limit='avg') + \
                    sym.S(dt(dt(knl)), sigma, kernel_arguments={"tangent_dir": tangent_dir}, qbx_forced_limit='avg')

        #k1 = sym.normal_derivative(2, sym.S(LaplaceKernel(2), sigma))#, kernel_arguments={"normal_dir": normal_dir})
        bound_op = bind(qbx, sym.normal_derivative(2, k1))

    #A = build_matrix(bound_op.scipy_op(queue, "sigma", dtype))

    #eigs = np.linalg.eig(A)[0]
    #pt.scatter(eigs.real, eigs.imag)

    op_u = op.operator(sym.var("sigma"))
    bound_op = bind(qbx, op_u)
    bvp_rhs = bind(density_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=[bc1,bc2])

    #bvp_rhs = bind(qbx, sym.make_sym_vector("bc",qbx.ambient_dim))(queue, bc=[bc1,bc2])

    """
    scipy_op = bound_op.scipy_op(queue, "sigma", dtype)
    #rand_charges = cl.array.to_device(queue, 1.5 + np.sin(np.linspace(0, 2*np.pi, scipy_op.shape[0])))
    one_charges = cl.array.to_device(queue, np.ones(scipy_op.shape[0]))
    bdry = scipy_op.matvec(one_charges)
    h = 1/1000.0

    bdry_limit_targets = ellipse(1, np.linspace(0, 1, 40))* (1-h)
    bdry_limit_op = bind((qbx, PointsTarget(bdry_limit_targets)), (_k1 + _k2))
    bdry_limit = bdry_limit_op(queue, sigma2=one_charges.reshape(2, scipy_op.shape[0]//2))

    test_normal = sym.make_sym_vector("nt",qbx.ambient_dim)
    representation_normal = sym.dd_axis(0, qbx.ambient_dim, _k1 + _k2)*test_normal[0] + sym.dd_axis(1, qbx.ambient_dim, _k1+_k2)*test_normal[1]
    representation_normal2 = sym.dd_axis(0, qbx.ambient_dim, _k1 + _k2)

    bdry_limit_op = bind((qbx, PointsTarget(bdry_limit_targets)), representation_normal)
    bdry_limit = bdry_limit_op(queue, sigma2=one_charges.reshape(2, scipy_op.shape[0]//2), nt=cl.array.to_device(queue, bdry_limit_targets/np.linalg.norm(bdry_limit_targets, 2, axis=0)))

    N = 200
    fplot = FieldPlotter(np.zeros(2), extent=2.0, npoints=N)
    fld_in_vol = bind((qbx, PointsTarget(fplot.points)), _k1+_k2)(queue, sigma2=one_charges.reshape(2, scipy_op.shape[0]//2))

    points = fplot.points
    #points = np.array([[0]*N, list(np.linspace(-1, 1, N))])
    fld_in_vol = bind((qbx, PointsTarget(points)), representation_normal)(queue, sigma2=one_charges.reshape(2, scipy_op.shape[0]//2), nt=cl.array.to_device(queue, points/np.linalg.norm(points, 2, axis=0)))

    fplot.show_scalar_in_matplotlib(fld_in_vol.get()) 
    import matplotlib.pyplot as pt 
    pt.colorbar() 
    pt.show()
    pt.title("normal derivative of u(x)")


    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=bdry_limit_targets[0,:], ys=bdry_limit_targets[1,:], zs=bdry_limit.get())

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    A = build_matrix(bound_op.scipy_op(queue, "sigma", dtype))

    eigs = np.linalg.eig(A)[0]
    pt.scatter(eigs.real, eigs.imag)

    # }}}

    # {{{ solve

    b = np.concatenate((bvp_rhs[0].get(), bvp_rhs[1].get()))

    u = np.linalg.solve(A, b)

    u0 = cl.array.to_device(queue, u[:len(u)//2])
    u1 = cl.array.to_device(queue, u[len(u)//2:])
    u_d = [u0, u1]



    point_source = PointPotentialSource(cl_ctx, [[0],[0]])
    log_points = list(np.logspace(-20, 0))
    test_targets = np.array([log_points, [0]*len(log_points)])
    pot_src = sym.IntG(
        # FIXME: qbx_forced_limit--really?
        knl, sym.var("sigma"), qbx_forced_limit=None, **knl_kwargs)
    """

    try:
        from pytential.solve import gmres
        gmres_result = gmres(
                bound_op.scipy_op(queue, "sigma", dtype, **concrete_knl_kwargs),
                bvp_rhs,
                tol=case.gmres_tol,
                progress=True,
                hard_failure=True,
                stall_iterations=50, no_progress_factor=1.05, require_monotonicity=True)
    except QBXTargetAssociationFailedException as e:
        bdry_vis = make_visualizer(queue, density_discr, case.target_order+3)

        bdry_vis.write_vtk_file("failed-targets-%s.vtu" % resolution, [
            ("failed_targets", e.failed_target_flags),
            ])
        raise

    print("gmres state:", gmres_result.state)
    weighted_u = gmres_result.solution


    #test_targets = ellipse(1, np.linspace(0, 1, 100))* (1-h)

    fplot = FieldPlotter(np.zeros(2), extent=2.0, npoints=1000)
    test_targets = fplot.points

    test_direct = bind((point_source, PointsTarget(test_targets)), pot_src)(
            queue, charges=source_charges_dev, **concrete_knl_kwargs)

    points_target = PointsTarget(test_targets)

    bound_tgt_op = bind((qbx, points_target), _k1 + _k2)

    test_via_bdry = bound_tgt_op(queue, sigma2=weighted_u)

    err = test_via_bdry - test_direct

    err = np.abs(err.get())

    print(err/test_direct.get())

    points = []
    for i, point in enumerate(fplot.points.T):
        if point[0]**2+3*point[1]**2>=1:
            err[i] = 1e-12
    max_err.append(np.max(np.abs(err)))
    print("====================================================================")
    print(max_err)

pt.imshow(np.abs(err).reshape(int(np.sqrt(len(err))), int(np.sqrt(len(err)))), norm=LogNorm())
pt.colorbar()

test_targets = np.array(points).T.copy()

fld_in_vol = bind((qbx, PointsTarget(fplot.points)), _k1+_k2)(queue, sigma2=one_charges.reshape(2, scipy_op.shape[0]//2))


fplot.show_scalar_in_matplotlib(fld_in_vol.get())



A = build_matrix(bound_op.scipy_op(queue, "sigma", dtype))

eigs = np.linalg.eig(A)[0]
pt.scatter(eigs.real, eigs.imag)
