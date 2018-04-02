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
from sumpy.visualization import FieldPlotter
from sumpy.symbolic import USE_SYMENGINE
from pytential import bind, sym

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
    def __init__(self, helmholtz_k, bc_type, loc_sign):
        self.helmholtz_k = helmholtz_k
        self.bc_type = bc_type
        self.loc_sign = loc_sign

    @property
    def k(self):
        return self.helmholtz_k

    def __str__(self):
        return ("name: %s, bc_type: %s, loc_sign: %s, "
                "helmholtz_k: %s, qbx_order: %d, target_order: %d"
            % (self.name, self.bc_type, self.loc_sign, self.helmholtz_k,
                self.qbx_order, self.target_order))

    fmm_backend = "sumpy"
    gmres_tol = 1e-14


class CurveIntEqTestCase(IntEqTestCase):
    resolutions = [40, 50, 60]

    def get_mesh(self, resolution, target_order):
        return make_curve_mesh(
                self.curve_func,
                np.linspace(0, 1, resolution+1),
                target_order)

    use_refinement = True

    inner_radius = 0.1
    outer_radius = 2

    qbx_order = 5
    target_order = qbx_order
    fmm_backend = None

    check_tangential_deriv = True
    check_gradient = False


class EllipseIntEqTestCase(CurveIntEqTestCase):
    name = "3-to-1 ellipse"

    def curve_func(self, x):
        return ellipse(3, x)


class Helmholtz3DIntEqTestCase(IntEqTestCase):
    fmm_backend = "fmmlib"
    use_refinement = False

    @property
    def target_order(self):
        return self.qbx_order

    check_tangential_deriv = False

    gmres_tol = 1e-7


class EllipsoidIntEqTestCase(Helmholtz3DIntEqTestCase):
    resolutions = [2, 0.8]
    name = "ellipsoid"

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("ellipsoid.step"), 2, order=2,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        from meshmode.mesh.processing import perform_flips
        # Flip elements--gmsh generates inside-out geometry.
        return perform_flips(mesh, np.ones(mesh.nelements))

    fmm_order = 13

    inner_radius = 0.4
    outer_radius = 5

    check_gradient = True


class SphereIntEqTestCase(IntEqTestCase):
    resolutions = [1, 2]
    name = "sphere"

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.generation import generate_icosphere
        from meshmode.mesh.refinement import refine_uniformly
        mesh = refine_uniformly(
                generate_icosphere(1, target_order),
                resolution)

        return mesh

    fmm_backend = "fmmlib"
    use_refinement = False

    fmm_tol = 1e-4

    inner_radius = 0.4
    outer_radius = 5

    qbx_order = 5
    target_order = 8
    check_gradient = False
    check_tangential_deriv = False

    gmres_tol = 1e-7


class MergedCubesIntEqTestCase(Helmholtz3DIntEqTestCase):
    resolutions = [1.4]
    name = "merged-cubes"

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("merged-cubes.step"), 2, order=2,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        from meshmode.mesh.processing import perform_flips
        # Flip elements--gmsh generates inside-out geometry.
        mesh = perform_flips(mesh, np.ones(mesh.nelements))

        return mesh

    use_refinement = True

    inner_radius = 0.4
    outer_radius = 12


class ManyEllipsoidIntEqTestCase(Helmholtz3DIntEqTestCase):
    resolutions = [2, 1]
    name = "ellipsoid"

    nx = 2
    ny = 2
    nz = 2

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        base_mesh = generate_gmsh(
                FileSource("ellipsoid.step"), 2, order=2,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        from meshmode.mesh.processing import perform_flips
        # Flip elements--gmsh generates inside-out geometry.
        base_mesh = perform_flips(base_mesh, np.ones(base_mesh.nelements))

        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
        from meshmode.mesh.tools import rand_rotation_matrix
        pitch = 10
        meshes = [
                affine_map(
                    base_mesh,
                    A=rand_rotation_matrix(3),
                    b=pitch*np.array([
                        (ix-self.nx//2),
                        (iy-self.ny//2),
                        (iz-self.ny//2)]))
                for ix in range(self.nx)
                for iy in range(self.ny)
                for iz in range(self.nz)
                ]

        mesh = merge_disjoint_meshes(meshes, single_group=True)
        return mesh

    inner_radius = 0.4
    # This should sit in the area just outside the middle ellipsoid
    outer_radius = 5


class ElliptiplaneIntEqTestCase(IntEqTestCase):
    name = "elliptiplane"

    resolutions = [0.2]

    fmm_backend = "fmmlib"
    use_refinement = True

    qbx_order = 3
    fmm_tol = 1e-4
    target_order = qbx_order
    check_gradient = False
    check_tangential_deriv = False

    # We're only expecting three digits based on FMM settings. Who are we
    # kidding?
    gmres_tol = 1e-5

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

    inner_radius = 0.2
    outer_radius = 12

# }}}


# {{{ test backend

def run_int_eq_test(cl_ctx, queue, case, resolution, visualize):
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
            from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder
            qbx_lpot_kwargs["fmm_level_to_order"] = SimpleExpansionOrderFinder(
                    case.fmm_tol)

        elif hasattr(case, "fmm_order"):
            qbx_lpot_kwargs["fmm_order"] = case.fmm_order
        else:
            qbx_lpot_kwargs["fmm_order"] = case.qbx_order + 5

    qbx = QBXLayerPotentialSource(
            pre_density_discr, fine_order=source_order, qbx_order=case.qbx_order,
            fmm_backend=case.fmm_backend, **qbx_lpot_kwargs)

    if case.use_refinement:
        if case.k != 0:
            refiner_extra_kwargs["kernel_length_scale"] = 5/case.k

        print("%d elements before refinement" % pre_density_discr.mesh.nelements)
        qbx, _ = qbx.with_refinement(**refiner_extra_kwargs)
        print("%d elements after refinement" % qbx.density_discr.mesh.nelements)

    density_discr = qbx.density_discr

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
            from meshmode.discretization.visualization import make_visualizer
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
            NeumannOperator)

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    if case.k:
        knl = HelmholtzKernel(mesh.ambient_dim)
        knl_kwargs = {"k": sym.var("k")}
        concrete_knl_kwargs = {"k": case.k}
    else:
        knl = LaplaceKernel(mesh.ambient_dim)
        knl_kwargs = {}
        concrete_knl_kwargs = {}

    if knl.is_complex_valued:
        dtype = np.complex128
    else:
        dtype = np.float64

    if case.bc_type == "dirichlet":
        op = DirichletOperator(knl, case.loc_sign, use_l2_weighting=False,
                kernel_arguments=knl_kwargs)
    elif case.bc_type == "neumann":
        op = NeumannOperator(knl, case.loc_sign, use_l2_weighting=False,
                 use_improved_operator=False, kernel_arguments=knl_kwargs)
    else:
        assert False

    op_u = op.operator(sym.var("u"))

    # }}}

    # {{{ set up test data

    if case.loc_sign < 0:
        test_src_geo_radius = case.outer_radius
        test_tgt_geo_radius = case.inner_radius
    else:
        test_src_geo_radius = case.inner_radius
        test_tgt_geo_radius = case.outer_radius

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

    test_direct = bind((point_source, PointsTarget(test_targets)), pot_src)(
            queue, charges=source_charges_dev, **concrete_knl_kwargs)

    if case.bc_type == "dirichlet":
        bc = bind((point_source, density_discr), pot_src)(
                queue, charges=source_charges_dev, **concrete_knl_kwargs)

    elif case.bc_type == "neumann":
        bc = bind(
                (point_source, density_discr),
                sym.normal_derivative(
                    qbx.ambient_dim, pot_src, where=sym.DEFAULT_TARGET)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)

    # }}}

    # {{{ solve

    bound_op = bind(qbx, op_u)

    rhs = bind(density_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bc)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "u", dtype, **concrete_knl_kwargs),
            rhs,
            tol=case.gmres_tol,
            progress=True,
            hard_failure=True)

    print("gmres state:", gmres_result.state)
    u = gmres_result.solution

    # }}}

    # {{{ build matrix for spectrum check

    if 0:
        from sumpy.tools import build_matrix
        mat = build_matrix(bound_op.scipy_op("u", dtype=dtype, k=case.k))
        w, v = la.eig(mat)
        if 0:
            pt.imshow(np.log10(1e-20+np.abs(mat)))
            pt.colorbar()
            pt.show()

        #assert abs(s[-1]) < 1e-13, "h
        #assert abs(s[-2]) > 1e-7
        #from pudb import set_trace; set_trace()

    # }}}

    # {{{ error check

    points_target = PointsTarget(test_targets)
    bound_tgt_op = bind((qbx, points_target),
            op.representation(sym.var("u")))

    test_via_bdry = bound_tgt_op(queue, u=u, k=case.k)

    err = test_direct-test_via_bdry

    err = err.get()
    test_direct = test_direct.get()
    test_via_bdry = test_via_bdry.get()

    # {{{ remove effect of net source charge

    if case.k == 0 and case.bc_type == "neumann" and case.loc_sign == -1:

        # remove constant offset in interior Laplace Neumann error
        tgt_ones = np.ones_like(test_direct)
        tgt_ones = tgt_ones/la.norm(tgt_ones)
        err = err - np.vdot(tgt_ones, err)*tgt_ones

    # }}}

    rel_err_2 = la.norm(err)/la.norm(test_direct)
    rel_err_inf = la.norm(err, np.inf)/la.norm(test_direct, np.inf)

    # }}}

    print("rel_err_2: %g rel_err_inf: %g" % (rel_err_2, rel_err_inf))

    # {{{ test gradient

    if case.check_gradient:
        bound_grad_op = bind((qbx, points_target),
                op.representation(
                    sym.var("u"),
                    map_potentials=lambda pot: sym.grad(mesh.ambient_dim, pot),
                    qbx_forced_limit=None))

        #print(bound_t_deriv_op.code)

        grad_from_src = bound_grad_op(
                queue, u=u, **concrete_knl_kwargs)

        grad_ref = (bind(
                (point_source, points_target),
                sym.grad(mesh.ambient_dim, pot_src)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)
                )

        grad_err = (grad_from_src - grad_ref)

        rel_grad_err_inf = (
                la.norm(grad_err[0].get(), np.inf)
                /
                la.norm(grad_ref[0].get(), np.inf))

        print("rel_grad_err_inf: %g" % rel_grad_err_inf)

    # }}}
    # {{{ test tangential derivative

    if case.check_tangential_deriv:
        bound_t_deriv_op = bind(qbx,
                op.representation(
                    sym.var("u"),
                    map_potentials=lambda pot: sym.tangential_derivative(2, pot),
                    qbx_forced_limit=case.loc_sign))

        #print(bound_t_deriv_op.code)

        tang_deriv_from_src = bound_t_deriv_op(
                queue, u=u, **concrete_knl_kwargs).as_scalar().get()

        tang_deriv_ref = (bind(
                (point_source, density_discr),
                sym.tangential_derivative(2, pot_src)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)
                .as_scalar().get())

        if 0:
            pt.plot(tang_deriv_ref.real)
            pt.plot(tang_deriv_from_src.real)
            pt.show()

        td_err = (tang_deriv_from_src - tang_deriv_ref)

        rel_td_err_inf = la.norm(td_err, np.inf)/la.norm(tang_deriv_ref, np.inf)

        print("rel_td_err_inf: %g" % rel_td_err_inf)

    else:
        rel_td_err_inf = None

    # }}}

    # {{{ 3D plotting

    if visualize and qbx.ambient_dim == 3:
        from meshmode.discretization.visualization import make_visualizer
        bdry_vis = make_visualizer(queue, density_discr, case.target_order+3)

        bdry_normals = bind(density_discr, sym.normal(qbx.ambient_dim))(queue)\
                .as_vector(dtype=object)

        bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
            ("u", u),
            ("bc", bc),
            ("bdry_normals", bdry_normals),
            ])

        from sumpy.visualization import make_field_plotter_from_bbox  # noqa
        from meshmode.mesh.processing import find_bounding_box

        fplot = make_field_plotter_from_bbox(
                find_bounding_box(mesh), h=(0.025, 0.025, 0.15)[:qbx.ambient_dim])

        qbx_tgt_tol = qbx.copy(target_association_tolerance=0.15)
        from pytential.target import PointsTarget
        from pytential.qbx import QBXTargetAssociationFailedException

        try:
            solved_pot = bind(
                    (qbx_tgt_tol, PointsTarget(fplot.points)),
                    op.representation(sym.var("u"))
                    )(queue, u=u, k=case.k)
        except QBXTargetAssociationFailedException as e:
            fplot.write_vtk_file(
                    "failed-targets.vts",
                    [
                        ("failed_targets", e.failed_target_flags.get(queue))
                        ])
            raise

        solved_pot = solved_pot.get()

        true_pot = bind((point_source, PointsTarget(fplot.points)), pot_src)(
                queue, charges=source_charges_dev, **concrete_knl_kwargs).get()

        #fplot.show_scalar_in_mayavi(solved_pot.real, max_val=5)
        fplot.write_vtk_file(
                "potential-%s.vts" % resolution,
                [
                    ("solved_pot", solved_pot),
                    ("true_pot", true_pot),
                    ("pot_diff", solved_pot-true_pot),
                    ]
                )

    # }}}

    # {{{ 2D plotting

    if 0:
        fplot = FieldPlotter(np.zeros(2),
                extent=1.25*2*max(test_src_geo_radius, test_tgt_geo_radius),
                npoints=200)

        #pt.plot(u)
        #pt.show()

        fld_from_src = bind((point_source, PointsTarget(fplot.points)),
                pot_src)(queue, charges=source_charges_dev, **concrete_knl_kwargs)
        fld_from_bdry = bind(
                (qbx, PointsTarget(fplot.points)),
                op.representation(sym.var("u"))
                )(queue, u=u, k=case.k)
        fld_from_src = fld_from_src.get()
        fld_from_bdry = fld_from_bdry.get()

        nodes = density_discr.nodes().get(queue=queue)

        def prep():
            pt.plot(point_sources[0], point_sources[1], "o",
                    label="Monopole 'Point Charges'")
            pt.plot(test_targets[0], test_targets[1], "v",
                    label="Observation Points")
            pt.plot(nodes[0], nodes[1], "k-",
                    label=r"$\Gamma$")

        from matplotlib.cm import get_cmap
        cmap = get_cmap()
        cmap._init()
        if 0:
            cmap._lut[(cmap.N*99)//100:, -1] = 0  # make last percent transparent?

        prep()
        if 1:
            pt.subplot(131)
            pt.title("Field error (loc_sign=%s)" % case.loc_sign)
            log_err = np.log10(1e-20+np.abs(fld_from_src-fld_from_bdry))
            log_err = np.minimum(-3, log_err)
            fplot.show_scalar_in_matplotlib(log_err, cmap=cmap)

            #from matplotlib.colors import Normalize
            #im.set_norm(Normalize(vmin=-6, vmax=1))

            cb = pt.colorbar(shrink=0.9)
            cb.set_label(r"$\log_{10}(\mathdefault{Error})$")

        if 1:
            pt.subplot(132)
            prep()
            pt.title("Source Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_src.real, max_val=3)

            pt.colorbar(shrink=0.9)
        if 1:
            pt.subplot(133)
            prep()
            pt.title("Solved Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_bdry.real, max_val=3)

            pt.colorbar(shrink=0.9)

        # total field
        #fplot.show_scalar_in_matplotlib(
        #fld_from_src.real+fld_from_bdry.real, max_val=0.1)

        #pt.colorbar()

        pt.legend(loc="best", prop=dict(size=15))
        from matplotlib.ticker import NullFormatter
        pt.gca().xaxis.set_major_formatter(NullFormatter())
        pt.gca().yaxis.set_major_formatter(NullFormatter())

        pt.gca().set_aspect("equal")

        if 0:
            border_factor_top = 0.9
            border_factor = 0.3

            xl, xh = pt.xlim()
            xhsize = 0.5*(xh-xl)
            pt.xlim(xl-border_factor*xhsize, xh+border_factor*xhsize)

            yl, yh = pt.ylim()
            yhsize = 0.5*(yh-yl)
            pt.ylim(yl-border_factor_top*yhsize, yh+border_factor*yhsize)

        #pt.savefig("helmholtz.pdf", dpi=600)
        pt.show()

        # }}}

    class Result(Record):
        pass

    return Result(
            h_max=qbx.h_max,
            rel_err_2=rel_err_2,
            rel_err_inf=rel_err_inf,
            rel_td_err_inf=rel_td_err_inf,
            gmres_result=gmres_result)

# }}}


# {{{ test frontend

@pytest.mark.parametrize("case", [
    EllipseIntEqTestCase(helmholtz_k=helmholtz_k, bc_type=bc_type,
        loc_sign=loc_sign)
    for helmholtz_k in [0, 1.2]
    for bc_type in ["dirichlet", "neumann"]
    for loc_sign in [-1, +1]
    ])
# Sample test run:
# 'test_integral_equation(cl._csc, EllipseIntEqTestCase(0, "dirichlet", +1), visualize=True)'  # noqa: E501
def test_integral_equation(ctx_getter, case, visualize=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    if case.fmm_backend == "fmmlib":
        pytest.importorskip("pyfmmlib")

    if USE_SYMENGINE and case.fmm_backend is None:
        pytest.skip("https://gitlab.tiker.net/inducer/sumpy/issues/25")

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from pytools.convergence import EOCRecorder
    print("qbx_order: %d, %s" % (case.qbx_order, case))

    eoc_rec_target = EOCRecorder()
    eoc_rec_td = EOCRecorder()

    for resolution in case.resolutions:
        result = run_int_eq_test(cl_ctx, queue, case, resolution,
                visualize=visualize)

        eoc_rec_target.add_data_point(result.h_max, result.rel_err_2)

        if result.rel_td_err_inf is not None:
            eoc_rec_td.add_data_point(result.h_max, result.rel_td_err_inf)

    if case.bc_type == "dirichlet":
        tgt_order = case.qbx_order
    elif case.bc_type == "neumann":
        tgt_order = case.qbx_order-1
    else:
        assert False

    print("TARGET ERROR:")
    print(eoc_rec_target)
    assert eoc_rec_target.order_estimate() > tgt_order - 1.3

    if case.check_tangential_deriv:
        print("TANGENTIAL DERIVATIVE ERROR:")
        print(eoc_rec_td)
        assert eoc_rec_td.order_estimate() > tgt_order - 2.3

# }}}


# You can test individual routines by typing
# $ python test_scalar_int_eq.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
