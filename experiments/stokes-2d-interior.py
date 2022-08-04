import numpy as np
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential.target import PointsTarget
import matplotlib
matplotlib.use("Agg")


from pytential import bind, sym, norm  # noqa

# {{{ set some constants for use below

mesh_order = 4
target_order = 8
ovsmp_target_order = 4*target_order
qbx_order = 2
fmm_order = 7
mu = 3
# method has to be one of biharmonic/naive for 2D
method = "biharmonic"

# Test solution type -- either 'fundamental' or 'couette' (default is couette)
soln_type = 'couette'

# }}}


def main(nelements):
    import logging
    logging.basicConfig(level=logging.INFO)

    def get_obj_array(obj_array):
        from pytools.obj_array import make_obj_array
        return make_obj_array([
                ary.get()
                for ary in obj_array
                ])

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import (  # noqa
            make_curve_mesh, starfish, ellipse, drop)
    mesh = make_curve_mesh(
            lambda t: starfish(t),
            np.linspace(0, 1, nelements+1),
            target_order)
    coarse_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    target_association_tolerance = 0.05
    qbx, _ = QBXLayerPotentialSource(
            coarse_density_discr, fine_order=ovsmp_target_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            target_association_tolerance=target_association_tolerance,
            ).with_refinement()

    density_discr = qbx.density_discr
    nodes = density_discr.nodes().with_queue(queue)

    # Get normal vectors for the density discretization -- used in integration with stresslet
    mv_normal = bind(density_discr, sym.normal(2))(queue)
    normal = mv_normal.as_vector(object)


    # {{{ describe bvp

    from sumpy.kernel import LaplaceKernel
    from pytential.symbolic.stokes import StressletWrapper
    from pytools.obj_array import make_obj_array
    dim=2
    cse = sym.cse

    nvec_sym = sym.make_sym_vector("normal", dim)
    sigma_sym = sym.make_sym_vector("sigma", dim)
    mu_sym = sym.var("mu")
    sqrt_w = sym.sqrt_jac_q_weight(2)
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = -1

    # Create stresslet object
    stresslet_obj = StressletWrapper(dim=2, mu_sym=mu_sym, method=method)

    # Describe boundary operator
    bdry_op_sym = loc_sign * 0.5 * sigma_sym + sqrt_w * stresslet_obj.apply(inv_sqrt_w_sigma, nvec_sym, qbx_forced_limit='avg')

    # Bind to the qbx discretization
    bound_op = bind(qbx, bdry_op_sym)

    # }}}


    # {{{ fix rhs and solve

    def fund_soln(x, y, loc):
        #with direction (1,0) for point source
        r = cl.clmath.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        scaling = 1./(4*np.pi*mu)
        xcomp = (-cl.clmath.log(r) + (x - loc[0])**2/r**2) * scaling
        ycomp = ((x - loc[0])*(y - loc[1])/r**2) * scaling
        return [ xcomp, ycomp ]

    def couette_soln(x, y, dp, h):
        scaling = 1./(2*mu)
        xcomp = scaling * dp * ((y+(h/2.))**2 - h * (y+(h/2.)))
        ycomp = scaling * 0*y
        return [xcomp, ycomp]



    if soln_type == 'fundamental':
        pt_loc = np.array([2.0, 0.0])
        bc = fund_soln(nodes[0], nodes[1], pt_loc)
    else:
        dp = -10.
        h = 2.5
        bc = couette_soln(nodes[0], nodes[1], dp, h)

    # Get rhs vector
    bvp_rhs = bind(qbx, sqrt_w*sym.make_sym_vector("bc",dim))(queue, bc=bc)

    from pytential.linalg.gmres import gmres
    gmres_result = gmres(
             bound_op.scipy_op(queue, "sigma", np.float64, mu=mu, normal=normal),
             bvp_rhs, tol=1e-9, progress=True,
             stall_iterations=0,
             hard_failure=True)

    # }}}

    # {{{ postprocess/visualize
    sigma = gmres_result.solution

    # Describe representation of solution for evaluation in domain
    representation_sym = stresslet_obj.apply(inv_sqrt_w_sigma, nvec_sym, qbx_forced_limit=-2)

    from sumpy.visualization import FieldPlotter
    nsamp = 10
    eval_points_1d = np.linspace(-1., 1., nsamp)
    eval_points = np.zeros((2, len(eval_points_1d)**2))
    eval_points[0,:] = np.tile(eval_points_1d, len(eval_points_1d))
    eval_points[1,:] = np.repeat(eval_points_1d, len(eval_points_1d))


    gamma_sym = sym.var("gamma")
    inv_sqrt_w_gamma = cse(gamma_sym/sqrt_w)
    constant_laplace_rep = sym.D(LaplaceKernel(dim=2), inv_sqrt_w_gamma, qbx_forced_limit=None)
    sqrt_w_vec = bind(qbx, sqrt_w)(queue)

    def general_mask(test_points):
        const_density = bind((qbx, PointsTarget(test_points)), constant_laplace_rep)(queue, gamma=sqrt_w_vec).get()
        return (abs(const_density) > 0.1)

    def inside_domain(test_points):
        mask = general_mask(test_points)
        return np.array([
            row[mask]
            for row in test_points])

    def stride_hack(arr):
        from numpy.lib.stride_tricks import as_strided
        return np.array(as_strided(arr, strides=(8 * len(arr[0]), 8)))

    eval_points = inside_domain(eval_points)
    eval_points_dev = cl.array.to_device(queue, eval_points)

    # Evaluate the solution at the evaluation points
    vel = bind(
            (qbx, PointsTarget(eval_points_dev)),
            representation_sym)(queue, sigma=sigma, mu=mu, normal=normal)
    print("@@@@@@@@")
    vel = get_obj_array(vel)


    if soln_type == 'fundamental':
        exact_soln = fund_soln(eval_points_dev[0], eval_points_dev[1], pt_loc)
    else:
        exact_soln = couette_soln(eval_points_dev[0], eval_points_dev[1], dp, h)
    err = vel - get_obj_array(exact_soln)

    print("@@@@@@@@")

    print("L2 error estimate: ", np.sqrt((2./(nsamp-1))**2*np.sum(err[0]*err[0]) + (2./(nsamp-1))**2*np.sum(err[1]*err[1])))
    max_error_loc = [abs(err[0]).argmax(), abs(err[1]).argmax()]
    print("max error at sampled points: ", max(abs(err[0])), max(abs(err[1])))
    print("exact velocity at max error points: x -> ", err[0][max_error_loc[0]], ", y -> ", err[1][max_error_loc[1]])

    from pytential.symbolic.mappers import DerivativeTaker
    rep_pressure = stresslet_obj.apply_pressure(inv_sqrt_w_sigma, nvec_sym, qbx_forced_limit=-2)
    pressure = bind((qbx, PointsTarget(eval_points_dev)),
                     rep_pressure)(queue, sigma=sigma, mu=mu, normal=normal)
    pressure = pressure.get()
    print(f"pressure = {pressure}")

    x_dir_vecs = np.zeros((2,len(eval_points[0])))
    x_dir_vecs[0,:] = 1.0
    y_dir_vecs = np.zeros((2, len(eval_points[0])))
    y_dir_vecs[1,:] = 1.0
    x_dir_vecs = cl.array.to_device(queue, x_dir_vecs)
    y_dir_vecs = cl.array.to_device(queue, y_dir_vecs)
    dir_vec_sym = sym.make_sym_vector("force_direction", dim)
    rep_stress = stresslet_obj.apply_stress(inv_sqrt_w_sigma, nvec_sym, dir_vec_sym, qbx_forced_limit=-2)

    applied_stress_x = bind((qbx, PointsTarget(eval_points_dev)),
                             rep_stress)(queue, sigma=sigma, normal=normal, force_direction=x_dir_vecs, mu=mu)
    applied_stress_x = get_obj_array(applied_stress_x)
    applied_stress_y = bind((qbx, PointsTarget(eval_points_dev)),
                             rep_stress)(queue, sigma=sigma, normal=normal, force_direction=y_dir_vecs, mu=mu)
    applied_stress_y = get_obj_array(applied_stress_y)

    print(f"stress applied to x direction: {applied_stress_x}")
    print(f"stress applied to y direction: {applied_stress_y}")


    import matplotlib.pyplot as plt
    plt.quiver(eval_points[0], eval_points[1], vel[0], vel[1], linewidth=0.1)
    file_name = "field-n%s.pdf"%(nelements)
    plt.savefig(file_name)

    return (max(abs(err[0])), max(abs(err[1])))

    # }}}


if __name__ == "__main__":
    n_elements = np.array([30, 60, 120])
    max_errs_x = np.zeros(3)
    max_errs_y = np.zeros(3)


    for i in range(3):
        max_errs_x[i], max_errs_y[i] = main(n_elements[i])


    print("@@@@@@@@@@@@@@@@@@@@@")
    print("max errs in x for each mesh: ", max_errs_x)
    print("max errs in y for each mesh: ", max_errs_y)


