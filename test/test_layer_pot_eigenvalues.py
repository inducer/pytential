__copyright__ = "Copyright (C) 2013-7 Andreas Kloeckner"

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

import pytest
from functools import partial

import numpy as np

from arraycontext import flatten, unflatten
from pytential import bind, sym, norm
from pytential import GeometryCollection
import meshmode.mesh.generation as mgen

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ ellipse eigenvalues

@pytest.mark.parametrize(["ellipse_aspect", "mode_nr", "qbx_order", "force_direct"],
        [
            (1, 5, 3, False),
            (1, 6, 3, False),
            (2, 5, 3, False),
            (1, 5, 4, False),
            (1, 7, 5, False),
            (2, 7, 5, False),

            (2, 7, 5, True),
            ])
def test_ellipse_eigenvalues(actx_factory, ellipse_aspect, mode_nr, qbx_order,
        force_direct, visualize=False):
    logging.basicConfig(level=logging.INFO)

    print("ellipse_aspect: %s, mode_nr: %d, qbx_order: %d" % (
            ellipse_aspect, mode_nr, qbx_order))

    actx = actx_factory()

    ambient_dim = 2
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
    # https://dx.doi.org/10.1137/S1064827500372067

    for nelements in nelements_values:
        mesh = mgen.make_curve_mesh(partial(mgen.ellipse, ellipse_aspect),
                np.linspace(0, 1, nelements+1),
                target_order)

        fmm_order = 12
        if force_direct:
            fmm_order = False

        pre_density_discr = Discretization(
                actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
        qbx = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                qbx_order, fmm_order=fmm_order,
                _expansions_in_tree_have_extent=True,
                )
        places = GeometryCollection(qbx)

        density_discr = places.get_discretization(places.auto_source.geometry)
        nodes = actx.thaw(density_discr.nodes())

        if visualize:
            # plot geometry, centers, normals
            centers = bind(places,
                    sym.expansion_centers(qbx.ambient_dim, +1))(actx)
            normals = bind(places,
                    sym.normal(qbx.ambient_dim))(actx).as_vector(object)

            nodes_h = actx.to_numpy(
                    flatten(nodes, actx)).reshape(ambient_dim, -1)
            centers_h = actx.to_numpy(
                    flatten(centers, actx)).reshape(ambient_dim, -1)
            normals_h = actx.to_numpy(
                    flatten(normals, actx)).reshape(ambient_dim, -1)

            import matplotlib.pyplot as pt

            pt.plot(nodes_h[0], nodes_h[1], "x-")
            pt.plot(centers_h[0], centers_h[1], "o")
            pt.quiver(nodes_h[0], nodes_h[1], normals_h[0], normals_h[1])
            pt.gca().set_aspect("equal")
            pt.show()

        angle = actx.np.arctan2(nodes[1]*ellipse_aspect, nodes[0])

        ellipse_fraction = ((1-ellipse_aspect)/(1+ellipse_aspect))**mode_nr

        # (2.6) in [1]
        J = actx.np.sqrt(  # noqa
                actx.np.sin(angle)**2
                + (1/ellipse_aspect)**2 * actx.np.cos(angle)**2)

        from sumpy.kernel import LaplaceKernel
        lap_knl = LaplaceKernel(2)

        # {{{ single layer

        sigma_sym = sym.var("sigma")
        s_sigma_op = sym.S(lap_knl, sigma_sym, qbx_forced_limit=+1)

        sigma = actx.np.cos(mode_nr*angle)/J
        s_sigma = bind(places, s_sigma_op)(actx, sigma=sigma)

        # SIGN BINGO! :)
        s_eigval = 1/(2*mode_nr) * (1 + (-1)**mode_nr * ellipse_fraction)

        # (2.12) in [1]
        s_sigma_ref = s_eigval*J*sigma

        if 0:
            s_sigma_h = actx.to_numpy(flatten(s_sigma, actx))
            s_sigma_ref_h = actx.to_numpy(flatten(s_sigma_ref, actx))

            # pt.plot(s_sigma_h, label="Result")
            # pt.plot(s_sigma_ref_h, label="Reference")
            pt.plot(s_sigma_ref_h - s_sigma_h, label="Error")
            pt.legend()
            pt.show()

        h_max = actx.to_numpy(
                bind(places, sym.h_max(qbx.ambient_dim))(actx)
                )
        s_err = actx.to_numpy(
                norm(density_discr, s_sigma - s_sigma_ref)
                / norm(density_discr, s_sigma_ref))
        s_eoc_rec.add_data_point(h_max, s_err)

        # }}}

        # {{{ double layer

        d_sigma_op = sym.D(lap_knl, sigma_sym, qbx_forced_limit="avg")

        sigma = actx.np.cos(mode_nr*angle)
        d_sigma = bind(places, d_sigma_op)(actx, sigma=sigma)

        # SIGN BINGO! :)
        d_eigval = -(-1)**mode_nr * 1/2*ellipse_fraction

        d_sigma_ref = d_eigval*sigma

        if 0:
            pt.plot(actx.to_numpy(flatten(d_sigma, actx)), label="Result")
            pt.plot(actx.to_numpy(flatten(d_sigma_ref, actx)), label="Reference")
            pt.legend()
            pt.show()

        if ellipse_aspect == 1:
            d_ref_norm = norm(density_discr, sigma)
        else:
            d_ref_norm = norm(density_discr, d_sigma_ref)

        d_err = actx.to_numpy(
                norm(density_discr, d_sigma - d_sigma_ref) / d_ref_norm
                )
        d_eoc_rec.add_data_point(h_max, d_err)

        # }}}

        if ellipse_aspect == 1:
            # {{{ S'

            sp_sigma_op = sym.Sp(lap_knl, sym.var("sigma"), qbx_forced_limit="avg")

            sigma = actx.np.cos(mode_nr*angle)
            sp_sigma = bind(places, sp_sigma_op)(actx, sigma=sigma)
            sp_eigval = 0

            sp_sigma_ref = sp_eigval*sigma

            sp_err = actx.to_numpy(
                    norm(density_discr, sp_sigma - sp_sigma_ref)
                    / norm(density_discr, sigma))
            sp_eoc_rec.add_data_point(h_max, sp_err)

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


# {{{ sphere eigenvalues

@pytest.mark.parametrize(["mode_m", "mode_n", "qbx_order"], [
    (2, 3, 3),
    ])
@pytest.mark.parametrize("fmm_backend", [
    "sumpy",
    "fmmlib",
    ])
def test_sphere_eigenvalues(actx_factory, mode_m, mode_n, qbx_order,
        fmm_backend):
    special = pytest.importorskip("scipy.special")
    if fmm_backend == "fmmlib":
        pytest.importorskip("pyfmmlib")

    logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    target_order = 8

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    from pytools.convergence import EOCRecorder

    s_eoc_rec = EOCRecorder()
    d_eoc_rec = EOCRecorder()
    sp_eoc_rec = EOCRecorder()
    dp_eoc_rec = EOCRecorder()

    def rel_err(comp, ref):
        return actx.to_numpy(
                norm(density_discr, comp - ref) / norm(density_discr, ref)
                )

    for nrefinements in [0, 1]:
        from meshmode.mesh.generation import generate_sphere
        mesh = generate_sphere(1, target_order,
                uniform_refinement_rounds=nrefinements)

        pre_density_discr = Discretization(
                actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
        qbx = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                qbx_order, fmm_order=6,
                fmm_backend=fmm_backend,
                )
        places = GeometryCollection(qbx)

        density_discr = places.get_discretization(places.auto_source.geometry)
        nodes = actx.thaw(density_discr.nodes())
        r = actx.np.sqrt(nodes[0]*nodes[0] + nodes[1]*nodes[1] + nodes[2]*nodes[2])
        phi = actx.np.arccos(nodes[2]/r)
        theta = actx.np.arctan2(nodes[0], nodes[1])

        ymn = unflatten(theta,
                actx.from_numpy(
                    special.sph_harm(
                        mode_m, mode_n,
                        actx.to_numpy(flatten(theta, actx)),
                        actx.to_numpy(flatten(phi, actx)))),
                    actx, strict=False)

        from sumpy.kernel import LaplaceKernel
        lap_knl = LaplaceKernel(3)

        # {{{ single layer

        s_sigma_op = bind(places,
                sym.S(lap_knl, sym.var("sigma"), qbx_forced_limit=+1))
        s_sigma = s_sigma_op(actx, sigma=ymn)
        s_eigval = 1/(2*mode_n + 1)

        h_max = actx.to_numpy(
                bind(places, sym.h_max(qbx.ambient_dim))(actx)
                )
        s_eoc_rec.add_data_point(h_max, rel_err(s_sigma, s_eigval*ymn))

        # }}}

        # {{{ double layer

        d_sigma_op = bind(places,
                sym.D(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
        d_sigma = d_sigma_op(actx, sigma=ymn)
        d_eigval = -1/(2*(2*mode_n + 1))
        d_eoc_rec.add_data_point(h_max, rel_err(d_sigma, d_eigval*ymn))

        # }}}

        # {{{ S'

        sp_sigma_op = bind(places,
                 sym.Sp(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
        sp_sigma = sp_sigma_op(actx, sigma=ymn)
        sp_eigval = -1/(2*(2*mode_n + 1))

        sp_eoc_rec.add_data_point(h_max, rel_err(sp_sigma, sp_eigval*ymn))

        # }}}

        # {{{ D'

        dp_sigma_op = bind(places,
                sym.Dp(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
        dp_sigma = dp_sigma_op(actx, sigma=ymn)
        dp_eigval = -(mode_n*(mode_n+1))/(2*mode_n + 1)

        dp_eoc_rec.add_data_point(h_max, rel_err(dp_sigma, dp_eigval*ymn))

        # }}}

    print("Errors for S:")
    print(s_eoc_rec)
    required_order = qbx_order + 1
    assert s_eoc_rec.order_estimate() > required_order - 1.5

    print("Errors for D:")
    print(d_eoc_rec)
    required_order = qbx_order
    assert d_eoc_rec.order_estimate() > required_order - 0.5

    print("Errors for S':")
    print(sp_eoc_rec)
    required_order = qbx_order
    assert sp_eoc_rec.order_estimate() > required_order - 1.5

    print("Errors for D':")
    print(dp_eoc_rec)
    required_order = qbx_order
    assert dp_eoc_rec.order_estimate() > required_order - 1.5

# }}}


# You can test individual routines by typing
# $ python test_layer_pot_eigenvalues.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
