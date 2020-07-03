from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2017 Natalie Beams"

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
import pyopencl as cl
import pyopencl.clmath
import pytest

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytools.obj_array import make_obj_array
from sumpy.visualization import FieldPlotter

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from pytential import bind, sym, norm  # noqa
from pytential import GeometryCollection
from pytential.solve import gmres
import logging


def run_exterior_stokes_2d(ctx_factory, nelements,
        mesh_order=4, target_order=4, qbx_order=4,
        fmm_order=False, mu=1, circle_rad=1.5, visualize=False):

    # This program tests an exterior Stokes flow in 2D using the
    # compound representation given in Hsiao & Kress,
    # ``On an integral equation for the two-dimensional exterior Stokes problem,''
    # Applied Numerical Mathematics 1 (1985).

    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    ovsmp_target_order = 4*target_order

    # {{{ geometries

    from meshmode.mesh.generation import (  # noqa
            make_curve_mesh, starfish, ellipse, drop)
    mesh = make_curve_mesh(
            lambda t: circle_rad * ellipse(1, t),
            np.linspace(0, 1, nelements+1),
            target_order)
    coarse_density_discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    target_association_tolerance = 0.05
    qbx = QBXLayerPotentialSource(
            coarse_density_discr, fine_order=ovsmp_target_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            target_association_tolerance=target_association_tolerance,
            _expansions_in_tree_have_extent=True,
            )

    def circle_mask(test_points, radius):
        return (test_points[0, :]**2 + test_points[1, :]**2 > radius**2)

    def outside_circle(test_points, radius):
        mask = circle_mask(test_points, radius)
        return np.array([
            row[mask]
            for row in test_points])

    from pytential.target import PointsTarget
    nsamp = 30
    eval_points_1d = np.linspace(-3., 3., nsamp)
    eval_points = np.zeros((2, len(eval_points_1d)**2))
    eval_points[0, :] = np.tile(eval_points_1d, len(eval_points_1d))
    eval_points[1, :] = np.repeat(eval_points_1d, len(eval_points_1d))
    eval_points = outside_circle(eval_points, radius=circle_rad)
    point_targets = PointsTarget(eval_points)

    fplot = FieldPlotter(np.zeros(2), extent=6, npoints=100)
    plot_targets = PointsTarget(outside_circle(fplot.points, radius=circle_rad))

    places = GeometryCollection({
        sym.DEFAULT_SOURCE: qbx,
        sym.DEFAULT_TARGET: qbx.density_discr,
        "point_target": point_targets,
        "plot_target": plot_targets,
        })

    density_discr = places.get_discretization(sym.DEFAULT_SOURCE)

    normal = bind(places, sym.normal(2).as_vector())(actx)
    path_length = bind(places, sym.integral(2, 1, 1))(actx)

    # }}}

    # {{{ describe bvp

    from pytential.symbolic.stokes import StressletWrapper, StokesletWrapper
    dim = 2
    cse = sym.cse

    sigma_sym = sym.make_sym_vector("sigma", dim)
    meanless_sigma_sym = cse(sigma_sym - sym.mean(2, 1, sigma_sym))
    int_sigma = sym.Ones() * sym.integral(2, 1, sigma_sym)

    nvec_sym = sym.make_sym_vector("normal", dim)
    mu_sym = sym.var("mu")

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = 1

    stresslet_obj = StressletWrapper(dim=2)
    stokeslet_obj = StokesletWrapper(dim=2)
    bdry_op_sym = (
            -loc_sign * 0.5 * sigma_sym
            - stresslet_obj.apply(sigma_sym, nvec_sym, mu_sym,
                qbx_forced_limit="avg")
            + stokeslet_obj.apply(meanless_sigma_sym, mu_sym,
                qbx_forced_limit="avg") - (0.5/np.pi) * int_sigma)

    # }}}

    bound_op = bind(places, bdry_op_sym)

    # {{{ fix rhs and solve

    def fund_soln(x, y, loc, strength):
        #with direction (1,0) for point source
        r = actx.np.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        scaling = strength/(4*np.pi*mu)
        xcomp = (-actx.np.log(r) + (x - loc[0])**2/r**2) * scaling
        ycomp = ((x - loc[0])*(y - loc[1])/r**2) * scaling
        return [xcomp, ycomp]

    def rotlet_soln(x, y, loc):
        r = actx.np.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        xcomp = -(y - loc[1])/r**2
        ycomp = (x - loc[0])/r**2
        return [xcomp, ycomp]

    def fund_and_rot_soln(x, y, loc, strength):
        #with direction (1,0) for point source
        r = actx.np.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        scaling = strength/(4*np.pi*mu)
        xcomp = (
                (-actx.np.log(r) + (x - loc[0])**2/r**2) * scaling
                - (y - loc[1])*strength*0.125/r**2 + 3.3)
        ycomp = (
                ((x - loc[0])*(y - loc[1])/r**2) * scaling
                + (x - loc[0])*strength*0.125/r**2 + 1.5)
        return make_obj_array([xcomp, ycomp])

    from meshmode.dof_array import unflatten, flatten, thaw
    nodes = flatten(thaw(actx, density_discr.nodes()))
    fund_soln_loc = np.array([0.5, -0.2])
    strength = 100.
    bc = unflatten(actx, density_discr,
            fund_and_rot_soln(nodes[0], nodes[1], fund_soln_loc, strength))

    omega_sym = sym.make_sym_vector("omega", dim)
    u_A_sym_bdry = stokeslet_obj.apply(  # noqa: N806
            omega_sym, mu_sym, qbx_forced_limit=1)

    from pytential.utils import unflatten_from_numpy
    omega = unflatten_from_numpy(actx, make_obj_array([
            (strength/path_length)*np.ones(len(nodes[0])),
            np.zeros(len(nodes[0]))
            ]), density_discr)

    bvp_rhs = bind(places,
            sym.make_sym_vector("bc", dim) + u_A_sym_bdry)(actx,
                    bc=bc, mu=mu, omega=omega)
    gmres_result = gmres(
            bound_op.scipy_op(actx, "sigma", np.float64, mu=mu, normal=normal),
            bvp_rhs,
            x0=bvp_rhs,
            tol=1e-9, progress=True,
            stall_iterations=0,
            hard_failure=True)

    # }}}

    # {{{ postprocess/visualize

    sigma = gmres_result.solution
    sigma_int_val_sym = sym.make_sym_vector("sigma_int_val", 2)
    int_val = bind(places, sym.integral(2, 1, sigma_sym))(actx, sigma=sigma)
    int_val = -int_val/(2 * np.pi)
    print("int_val = ", int_val)

    u_A_sym_vol = stokeslet_obj.apply(  # noqa: N806
            omega_sym, mu_sym, qbx_forced_limit=2)
    representation_sym = (
            - stresslet_obj.apply(
                sigma_sym, nvec_sym, mu_sym, qbx_forced_limit=2)
            + stokeslet_obj.apply(
                meanless_sigma_sym, mu_sym, qbx_forced_limit=2)
            - u_A_sym_vol + sigma_int_val_sym)

    where = (sym.DEFAULT_SOURCE, "point_target")
    vel = bind(places, representation_sym, auto_where=where)(actx,
            sigma=sigma,
            mu=mu,
            normal=normal,
            sigma_int_val=int_val,
            omega=omega)
    print("@@@@@@@@")

    plot_vel = bind(places, representation_sym,
            auto_where=(sym.DEFAULT_SOURCE, "plot_target"))(actx,
                    sigma=sigma,
                    mu=mu,
                    normal=normal,
                    sigma_int_val=int_val,
                    omega=omega)

    def get_obj_array(obj_array):
        return make_obj_array([
                ary.get()
                for ary in obj_array
                ])

    exact_soln = fund_and_rot_soln(
            actx.from_numpy(eval_points[0]),
            actx.from_numpy(eval_points[1]),
            fund_soln_loc,
            strength)

    vel = get_obj_array(vel)
    err = vel-get_obj_array(exact_soln)

    # FIXME: Pointwise relative errors don't make sense!
    rel_err = err/(get_obj_array(exact_soln))

    if 0:
        print("@@@@@@@@")
        print("vel[0], err[0], rel_err[0] ***** vel[1], err[1], rel_err[1]: ")
        for i in range(len(vel[0])):
            print("%15.8e, %15.8e, %15.8e ***** %15.8e, %15.8e, %15.8e\n" % (
                             vel[0][i], err[0][i], rel_err[0][i],
                             vel[1][i], err[1][i], rel_err[1][i]))

        print("@@@@@@@@")

    l2_err = np.sqrt(
            (6./(nsamp-1))**2*np.sum(err[0]*err[0])
            + (6./(nsamp-1))**2*np.sum(err[1]*err[1]))
    l2_rel_err = np.sqrt(
            (6./(nsamp-1))**2*np.sum(rel_err[0]*rel_err[0])
            + (6./(nsamp-1))**2*np.sum(rel_err[1]*rel_err[1]))

    print("L2 error estimate: ", l2_err)
    print("L2 rel error estimate: ", l2_rel_err)
    print("max error at sampled points: ", max(abs(err[0])), max(abs(err[1])))
    print("max rel error at sampled points: ",
            max(abs(rel_err[0])), max(abs(rel_err[1])))

    if visualize:
        import matplotlib.pyplot as plt

        full_pot = np.zeros_like(fplot.points) * float("nan")
        mask = circle_mask(fplot.points, radius=circle_rad)

        for i, vel in enumerate(plot_vel):
            full_pot[i, mask] = vel.get()

        plt.quiver(
                fplot.points[0], fplot.points[1],
                full_pot[0], full_pot[1],
                linewidth=0.1)
        plt.savefig("exterior-2d-field.pdf")

    # }}}

    h_max = bind(places, sym.h_max(qbx.ambient_dim))(actx)
    return h_max, l2_err


@pytest.mark.slowtest
def test_exterior_stokes_2d(ctx_factory, qbx_order=3):
    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for nelements in [20, 50]:
        h_max, l2_err = run_exterior_stokes_2d(ctx_factory, nelements)
        eoc_rec.add_data_point(h_max, l2_err)

    print(eoc_rec)
    assert eoc_rec.order_estimate() >= qbx_order - 1


# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
