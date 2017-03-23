import numpy as np
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym, norm  # noqa

# {{{ set some constants for use below
# This program tests an exterior Stokes flow in 2D using the 
# compound representation given in Hsiao & Kress,
# ``On an integral equation for the two-dimensional exterior Stokes problem,''
# Applied Numerical Mathematics 1 (1985).


nelements = 50
mesh_order = 4
target_order = 8
ovsmp_target_order = 4*target_order
qbx_order = 4
fmm_order = 7
mu = 1

# }}}


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import (  # noqa
            make_curve_mesh, starfish, ellipse, drop)
    mesh = make_curve_mesh(
            lambda t: ellipse(1, t),
            np.linspace(0, 1, nelements+1),
            target_order)
    coarse_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    stick_out = 0.05
    qbx, _ = QBXLayerPotentialSource(
            coarse_density_discr, fine_order=ovsmp_target_order, qbx_order=qbx_order,
            fmm_order=fmm_order, target_stick_out_factor=stick_out
            ).with_refinement()

    density_discr = qbx.density_discr
    normal = bind(density_discr, sym.normal(2).as_vector())(queue)
    path_length = bind(density_discr, sym.integral(2, 1, 1))(queue)
    print("path_length = ", path_length)

    # {{{ describe bvp

    from pytential.symbolic.stokes import StressletWrapper, StokesletWrapper
    from pytools.obj_array import make_obj_array
    dim=2
    cse = sym.cse

    sigma_sym = sym.make_sym_vector("sigma", dim)
    sqrt_w = sym.sqrt_jac_q_weight(2)
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)
    meanless_sigma_sym = cse(sigma_sym - sym.mean(2, 1, sigma_sym))
    int_sigma = sym.Ones() * sym.integral(2, 1, sigma_sym)
    
    nvec_sym = sym.make_sym_vector("normal", dim)
    mu_sym = sym.var("mu")

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = 1

    stresslet_obj = StressletWrapper(dim=2)
    stokeslet_obj = StokesletWrapper(dim=2)
    bdry_op_sym = -loc_sign * 0.5 * sigma_sym - stresslet_obj.apply(sigma_sym, nvec_sym, mu_sym,
                            qbx_forced_limit='avg') + stokeslet_obj.apply(meanless_sigma_sym, mu_sym,
                                    qbx_forced_limit='avg') - (0.5/np.pi) * int_sigma 

    # }}}

    bound_op = bind(qbx, bdry_op_sym)

    # {{{ fix rhs and solve

    def fund_soln(x, y, loc, strength):
        #with direction (1,0) for point source
        r = cl.clmath.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        scaling = strength/(4*np.pi*mu)
        xcomp = (-cl.clmath.log(r) + (x - loc[0])**2/r**2) * scaling
        ycomp = ((x - loc[0])*(y - loc[1])/r**2) * scaling
        return [ xcomp, ycomp ]

    def rotlet_soln(x, y, loc):
        r = cl.clmath.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        xcomp = -(y - loc[1])/r**2
        ycomp = (x - loc[0])/r**2
        return [ xcomp, ycomp ] 

    def fund_and_rot_soln(x, y, loc, strength):
        #with direction (1,0) for point source
        r = cl.clmath.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        scaling = strength/(4*np.pi*mu)
        xcomp = (-cl.clmath.log(r) + (x - loc[0])**2/r**2) * scaling - (y - loc[1])*strength*0.125/r**2 + 3.3
        ycomp = ((x - loc[0])*(y - loc[1])/r**2) * scaling + (x - loc[0])*strength*0.125/r**2 + 1.5
        return [ xcomp, ycomp ]
       

    nodes = density_discr.nodes().with_queue(queue)
    fund_soln_loc = np.array([0.5, -0.2])
    strength = 100.
    bc = fund_and_rot_soln(nodes[0], nodes[1], fund_soln_loc, strength)
 
    omega_sym = sym.make_sym_vector("omega", dim)
    u_A_sym_bdry = stokeslet_obj.apply(omega_sym, mu_sym, qbx_forced_limit=1)

    omega = [cl.array.to_device(queue, (strength/path_length)*np.ones(len(nodes[0]))), cl.array.to_device(queue, np.zeros(len(nodes[0])))]
    bvp_rhs = bind(qbx, (sym.make_sym_vector("bc",dim) + u_A_sym_bdry))(queue, bc=bc, mu=mu, omega=omega)
    from pytential.solve import gmres
    gmres_result = gmres(
             bound_op.scipy_op(queue, "sigma", np.float64, mu=mu, normal=normal),
             bvp_rhs, tol=1e-9, progress=True,
             stall_iterations=0,
             hard_failure=True)

    # }}}


 # {{{ postprocess/visualize
    sigma = gmres_result.solution
    sigma_int_val_sym = sym.make_sym_vector("sigma_int_val", 2)
    int_val = bind(qbx, sym.integral(2, 1, sigma_sym))(queue, sigma=sigma)
    int_val = -int_val/(2 * np.pi)
    print("int_val = ", int_val)

    u_A_sym_vol = stokeslet_obj.apply(omega_sym, mu_sym, qbx_forced_limit=2)
    representation_sym = - stresslet_obj.apply(sigma_sym, nvec_sym, mu_sym, qbx_forced_limit=2) + stokeslet_obj.apply(
                                meanless_sigma_sym, mu_sym, qbx_forced_limit=2) - u_A_sym_vol + sigma_int_val_sym


    from sumpy.visualization import FieldPlotter
    nsamp = 30
    eval_points_1d = np.linspace(-3., 3., nsamp)
    eval_points = np.zeros((2, len(eval_points_1d)**2))
    eval_points[0,:] = np.tile(eval_points_1d, len(eval_points_1d))
    eval_points[1,:] = np.repeat(eval_points_1d, len(eval_points_1d))

    def circle_mask(test_points, radius):
        return (test_points[0,:]**2 + test_points[1,:]**2 > radius**2)

    def outside_circle(test_points, radius):
        mask = circle_mask(test_points, radius)
        return np.array([
            row[mask]
            for row in test_points])

    eval_points = outside_circle(eval_points, radius=1)
    from pytential.target import PointsTarget
    vel = bind(
            (qbx, PointsTarget(eval_points)),
            representation_sym)(queue, sigma=sigma, mu=mu, normal=normal, sigma_int_val=int_val, omega=omega)
    print("@@@@@@@@")

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=6, npoints=100)
    plot_pts = outside_circle(fplot.points, radius=1)
    plot_vel = bind(
          (qbx, PointsTarget(plot_pts)),
          representation_sym)(queue, sigma=sigma, mu=mu, normal=normal, sigma_int_val=int_val, omega=omega)

    def get_obj_array(obj_array):
        from pytools.obj_array import make_obj_array
        return make_obj_array([
                ary.get()
                for ary in obj_array
                ])
    exact_soln = fund_and_rot_soln(cl.array.to_device(queue, eval_points[0]), cl.array.to_device(
                                    queue, eval_points[1]), fund_soln_loc, strength) 
    vel = get_obj_array(vel)
    err = vel-get_obj_array(exact_soln)
    rel_err = err/(get_obj_array(exact_soln))

    print("@@@@@@@@")
    print("vel[0], err[0], rel_err[0] ***** vel[1], err[1], rel_err[1]: ")
    for i in range(len(vel[0])):
        print("%15.8e, %15.8e, %15.8e ***** %15.8e, %15.8e, %15.8e\n"%(
                        vel[0][i], err[0][i], rel_err[0][i], vel[1][i], err[1][i], rel_err[1][i]))

    print("@@@@@@@@")
    print("L2 error estimate: ", np.sqrt((6./(nsamp-1))**2*np.sum(err[0]*err[0]) + (6./(nsamp-1))**2*np.sum(err[1]*err[1])))
    print("L2 rel error estimate: ", np.sqrt((6./(nsamp-1))**2*np.sum(rel_err[0]*rel_err[0]) + (6./(nsamp-1))**2*np.sum(rel_err[1]*rel_err[1])))
    print("max error at sampled points: ", max(abs(err[0])), max(abs(err[1])))
    print("max rel error at sampled points: ", max(abs(rel_err[0])), max(abs(rel_err[1])))

    full_pot = np.zeros_like(fplot.points) * float("nan")
    mask = circle_mask(fplot.points, radius=1)

    for i, vel in enumerate(plot_vel):
        full_pot[i,mask] = vel.get()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.quiver(fplot.points[0], fplot.points[1], full_pot[0], full_pot[1], linewidth=0.1)
    plt.savefig("exterior-2d-field.pdf")

    # }}}



if __name__ == "__main__":
    main()
