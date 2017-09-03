from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh, NArmedStarfish)
from pytential import bind, sym, norm

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


# {{{ ellipse eigenvalues

@pytest.mark.parametrize(["ellipse_aspect", "mode_nr", "qbx_order"], [
    # Run with FMM
    (1, 5, 3),
    (1, 6, 3),
    # (2, 5, 3), /!\ FIXME: Does not achieve sufficient FMM precision

    # Run without FMM
    (1, 5, 4),
    (1, 6, 4),
    (1, 7, 5),
    (2, 5, 4),
    (2, 6, 4),
    (2, 7, 5),
    ])
def test_ellipse_eigenvalues(ctx_getter, ellipse_aspect, mode_nr, qbx_order):
    logging.basicConfig(level=logging.INFO)

    print("ellipse_aspect: %s, mode_nr: %d, qbx_order: %d" % (
            ellipse_aspect, mode_nr, qbx_order))

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 8

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    from pytools.convergence import EOCRecorder

    s_eoc_rec = EOCRecorder()
    d_eoc_rec = EOCRecorder()
    sp_eoc_rec = EOCRecorder()

    if ellipse_aspect != 1:
        nelements_values = [60, 100, 150, 200]
    else:
        nelements_values = [30, 70]

    # See
    #
    # [1] G. J. Rodin and O. Steinbach, "Boundary Element Preconditioners
    # for Problems Defined on Slender Domains", SIAM Journal on Scientific
    # Computing, Vol. 24, No. 4, pg. 1450, 2003.
    # http://dx.doi.org/10.1137/S1064827500372067

    for nelements in nelements_values:
        mesh = make_curve_mesh(partial(ellipse, ellipse_aspect),
                np.linspace(0, 1, nelements+1),
                target_order)

        fmm_order = qbx_order + 3
        if fmm_order > 6:
            # FIXME: for now
            fmm_order = False

        pre_density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
        qbx, _ = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                qbx_order, fmm_order=fmm_order,
                _expansions_in_tree_have_extent=True,
                ).with_refinement()

        density_discr = qbx.density_discr
        nodes = density_discr.nodes().with_queue(queue)

        if 0:
            # plot geometry, centers, normals

            from pytential.qbx.utils import get_centers_on_side
            centers = get_centers_on_side(qbx, 1)

            nodes_h = nodes.get()
            centers_h = [centers[0].get(), centers[1].get()]
            pt.plot(nodes_h[0], nodes_h[1], "x-")
            pt.plot(centers_h[0], centers_h[1], "o")
            normal = bind(qbx, sym.normal())(queue).as_vector(np.object)
            pt.quiver(nodes_h[0], nodes_h[1],
                    normal[0].get(), normal[1].get())
            pt.gca().set_aspect("equal")
            pt.show()

        angle = cl.clmath.atan2(nodes[1]*ellipse_aspect, nodes[0])

        ellipse_fraction = ((1-ellipse_aspect)/(1+ellipse_aspect))**mode_nr

        # (2.6) in [1]
        J = cl.clmath.sqrt(  # noqa
                cl.clmath.sin(angle)**2
                + (1/ellipse_aspect)**2 * cl.clmath.cos(angle)**2)

        from sumpy.kernel import LaplaceKernel
        lap_knl = LaplaceKernel(2)

        # {{{ single layer

        sigma = cl.clmath.cos(mode_nr*angle)/J

        s_sigma_op = bind(qbx, sym.S(lap_knl, sym.var("sigma")))
        s_sigma = s_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        s_eigval = 1/(2*mode_nr) * (1 + (-1)**mode_nr * ellipse_fraction)

        # (2.12) in [1]
        s_sigma_ref = s_eigval*J*sigma

        if 0:
            #pt.plot(s_sigma.get(), label="result")
            #pt.plot(s_sigma_ref.get(), label="ref")
            pt.plot((s_sigma_ref-s_sigma).get(), label="err")
            pt.legend()
            pt.show()

        s_err = (
                norm(density_discr, queue, s_sigma - s_sigma_ref)
                /
                norm(density_discr, queue, s_sigma_ref))
        s_eoc_rec.add_data_point(qbx.h_max, s_err)

        # }}}

        # {{{ double layer

        sigma = cl.clmath.cos(mode_nr*angle)

        d_sigma_op = bind(qbx,
                sym.D(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
        d_sigma = d_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        d_eigval = -(-1)**mode_nr * 1/2*ellipse_fraction

        d_sigma_ref = d_eigval*sigma

        if 0:
            pt.plot(d_sigma.get(), label="result")
            pt.plot(d_sigma_ref.get(), label="ref")
            pt.legend()
            pt.show()

        if ellipse_aspect == 1:
            d_ref_norm = norm(density_discr, queue, sigma)
        else:
            d_ref_norm = norm(density_discr, queue, d_sigma_ref)

        d_err = (
                norm(density_discr, queue, d_sigma - d_sigma_ref)
                /
                d_ref_norm)
        d_eoc_rec.add_data_point(qbx.h_max, d_err)

        # }}}

        if ellipse_aspect == 1:
            # {{{ S'

            sigma = cl.clmath.cos(mode_nr*angle)

            sp_sigma_op = bind(qbx,
                    sym.Sp(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
            sp_sigma = sp_sigma_op(queue=queue, sigma=sigma)
            sp_eigval = 0

            sp_sigma_ref = sp_eigval*sigma

            sp_err = (
                    norm(density_discr, queue, sp_sigma - sp_sigma_ref)
                    /
                    norm(density_discr, queue, sigma))
            sp_eoc_rec.add_data_point(qbx.h_max, sp_err)

            # }}}

    print("Errors for S:")
    print(s_eoc_rec)
    required_order = qbx_order + 1
    assert s_eoc_rec.order_estimate() > required_order - 1.5

    print("Errors for D:")
    print(d_eoc_rec)
    required_order = qbx_order
    assert d_eoc_rec.order_estimate() > required_order - 1.5

    if ellipse_aspect == 1:
        print("Errors for S':")
        print(sp_eoc_rec)
        required_order = qbx_order
        assert sp_eoc_rec.order_estimate() > required_order - 1.5

# }}}


# You can test individual routines by typing
# $ python test_layer_pot_eigenvalues.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])


# vim: fdm=marker
