import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, NArmedStarfish, drop, n_gon, qbx_peanut,
        make_curve_mesh, starfish)
from pytential import bind, sym, norm

import logging
logger = logging.getLogger(__name__)


def get_starfish_mesh(nelements, target_order):
    return make_curve_mesh(
            NArmedStarfish(20, 0.8),
            np.linspace(0, 1, nelements+1),
            target_order)


WITH_EXTENTS = True
EXPANSION_STICKOUT_FACTOR = 0.5


def get_green_error(
        queue, mesh_getter, nelements, fmm_order, qbx_order, k=0):

    target_order = 12

    mesh = mesh_getter(nelements, target_order)

    d = mesh.ambient_dim

    # u_sym = sym.var("u")
    dn_u_sym = sym.var("dn_u")

    from sumpy.kernel import LaplaceKernel
    k_sym = LaplaceKernel(d)
    zero_op = (
            sym.S(k_sym, dn_u_sym, qbx_forced_limit=-1)
            # - sym.D(k_sym, u_sym, qbx_forced_limit="avg")
            # - 0.5*u_sym
            )

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)
    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order,
            qbx_order, fmm_order=fmm_order,
            _expansions_in_tree_have_extent=WITH_EXTENTS,
            _expansion_stick_out_factor=EXPANSION_STICKOUT_FACTOR,
            )

    lpot_source, _ = lpot_source.with_refinement()

    density_discr = lpot_source.density_discr

    # {{{ compute values of a solution to the PDE

    nodes_host = density_discr.nodes().get(queue)
    normal = bind(density_discr, sym.normal(d))(queue).as_vector(np.object)
    normal_host = [normal[j].get() for j in range(d)]

    center = np.array([3, 1, 2])[:d]
    diff = nodes_host - center[:, np.newaxis]
    dist_squared = np.sum(diff**2, axis=0)
    dist = np.sqrt(dist_squared)
    if d == 2:
        u = np.log(dist)
        grad_u = diff/dist_squared
    elif d == 3:
        u = 1/dist
        grad_u = -diff/dist**3
    else:
        assert False

    dn_u = 0
    for i in range(d):
        dn_u = dn_u + normal_host[i]*grad_u[i]

    # }}}

    u_dev = cl.array.to_device(queue, u)
    dn_u_dev = cl.array.to_device(queue, dn_u)

    bound_op = bind(lpot_source, zero_op)
    error = bound_op(
            queue, u=u_dev, dn_u=dn_u_dev, k=k)

    print(len(error), np.where(np.isnan(error.get())))
    return norm(density_discr, queue, error)


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    get_green_error(queue, get_starfish_mesh, 500, 20, 2, k=0)


if __name__ == "__main__":
    main()
