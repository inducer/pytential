from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2016 Matt Wala
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
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pytools import RecordWithoutPickling
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests
from pytential.qbx import QBXLayerPotentialSource

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut,
        make_curve_mesh, generate_icosphere, generate_torus)
from extra_curve_data import horseshoe


import logging
logger = logging.getLogger(__name__)

__all__ = ["pytest_generate_tests"]


RNG_SEED = 10
FAR_TARGET_DIST_FROM_SOURCE = 10


# {{{ source refinement checker

class ElementInfo(RecordWithoutPickling):
    """
    .. attribute:: element_nr
    .. attribute:: discr_slice
    """
    __slots__ = ["element_nr",
                 "discr_slice"]


def iter_elements(discr):
    discr_nodes_idx = 0
    element_nr = 0

    for discr_group in discr.groups:
        start = element_nr
        for element_nr in range(start, start + discr_group.nelements):
            yield ElementInfo(
                element_nr=element_nr,
                discr_slice=slice(discr_nodes_idx,
                   discr_nodes_idx + discr_group.nunit_nodes))

            discr_nodes_idx += discr_group.nunit_nodes


def run_source_refinement_test(ctx_getter, mesh, order, helmholtz_k=None):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    factory = InterpolatoryQuadratureSimplexGroupFactory(order)

    discr = Discretization(cl_ctx, mesh, factory)

    from pytential.qbx.refinement import (
            RefinerCodeContainer, refine_for_global_qbx)

    from pytential.qbx.utils import TreeCodeContainer

    lpot_source = QBXLayerPotentialSource(discr,
            qbx_order=order,  # not used in refinement
            fine_order=order)
    del discr

    expansion_disturbance_tolerance = 0.025
    refiner_extra_kwargs = {
            "expansion_disturbance_tolerance": expansion_disturbance_tolerance,
            }
    if helmholtz_k is not None:
        refiner_extra_kwargs["kernel_length_scale"] = 5/helmholtz_k

    lpot_source, conn = refine_for_global_qbx(
            lpot_source,
            RefinerCodeContainer(
                cl_ctx, TreeCodeContainer(cl_ctx)).get_wrangler(queue),
            factory, **refiner_extra_kwargs)

    from pytential.qbx.utils import get_centers_on_side

    discr_nodes = lpot_source.density_discr.nodes().get(queue)
    fine_discr_nodes = \
            lpot_source.quad_stage2_density_discr.nodes().get(queue)
    int_centers = get_centers_on_side(lpot_source, -1)
    int_centers = np.array([axis.get(queue) for axis in int_centers])
    ext_centers = get_centers_on_side(lpot_source, +1)
    ext_centers = np.array([axis.get(queue) for axis in ext_centers])
    expansion_radii = lpot_source._expansion_radii("nsources").get(queue)
    quad_res = lpot_source._coarsest_quad_resolution("npanels").get(queue)
    source_danger_zone_radii = \
            lpot_source._source_danger_zone_radii("npanels").get(queue)

    # {{{ check if satisfying criteria

    def check_disk_undisturbed_by_sources(centers_panel, sources_panel):
        if centers_panel.element_nr == sources_panel.element_nr:
            # Same panel
            return

        my_int_centers = int_centers[:, centers_panel.discr_slice]
        my_ext_centers = ext_centers[:, centers_panel.discr_slice]
        all_centers = np.append(my_int_centers, my_ext_centers, axis=-1)

        nodes = discr_nodes[:, sources_panel.discr_slice]

        # =distance(centers of panel 1, panel 2)
        dist = (
            la.norm((
                    all_centers[..., np.newaxis]
                    - nodes[:, np.newaxis, ...]).T,
                axis=-1)
            .min())

        # Criterion:
        # A center cannot be closer to another panel than to its originating
        # panel.

        rad = expansion_radii[centers_panel.discr_slice]
        assert (dist >= rad * (1-expansion_disturbance_tolerance)).all(), \
                (dist, rad, centers_panel.element_nr, sources_panel.element_nr)

    def check_sufficient_quadrature_resolution(centers_panel, sources_panel):
        dz_radius = source_danger_zone_radii[sources_panel.element_nr]

        my_int_centers = int_centers[:, centers_panel.discr_slice]
        my_ext_centers = ext_centers[:, centers_panel.discr_slice]
        all_centers = np.append(my_int_centers, my_ext_centers, axis=-1)

        nodes = fine_discr_nodes[:, sources_panel.discr_slice]

        # =distance(centers of panel 1, panel 2)
        dist = (
            la.norm((
                    all_centers[..., np.newaxis]
                    - nodes[:, np.newaxis, ...]).T,
                axis=-1)
            .min())

        # Criterion:
        # The quadrature contribution from each panel is as accurate
        # as from the center's own source panel.
        assert dist >= dz_radius, \
                (dist, dz_radius, centers_panel.element_nr, sources_panel.element_nr)

    def check_quad_res_to_helmholtz_k_ratio(panel):
        # Check wavenumber to panel size ratio.
        assert quad_res[panel.element_nr] * helmholtz_k <= 5

    for i, panel_1 in enumerate(iter_elements(lpot_source.density_discr)):
        for panel_2 in iter_elements(lpot_source.density_discr):
            check_disk_undisturbed_by_sources(panel_1, panel_2)
        for panel_2 in iter_elements(lpot_source.quad_stage2_density_discr):
            check_sufficient_quadrature_resolution(panel_1, panel_2)
        if helmholtz_k is not None:
            check_quad_res_to_helmholtz_k_ratio(panel_1)

    # }}}

# }}}


@pytest.mark.parametrize(("curve_name", "curve_f", "nelements"), [
    ("20-to-1 ellipse", partial(ellipse, 20), 100),
    ("horseshoe", horseshoe, 64),
    ])
def test_source_refinement_2d(ctx_getter, curve_name, curve_f, nelements):
    helmholtz_k = 10
    order = 8

    mesh = make_curve_mesh(curve_f, np.linspace(0, 1, nelements+1), order)
    run_source_refinement_test(ctx_getter, mesh, order, helmholtz_k)


@pytest.mark.parametrize(("surface_name", "surface_f", "order"), [
    ("sphere", partial(generate_icosphere, 1), 4),
    ("torus", partial(generate_torus, 3, 1, n_inner=10, n_outer=7), 6),
    ])
def test_source_refinement_3d(ctx_getter, surface_name, surface_f, order):
    mesh = surface_f(order=order)
    run_source_refinement_test(ctx_getter, mesh, order)


@pytest.mark.parametrize(("curve_name", "curve_f", "nelements"), [
    ("20-to-1 ellipse", partial(ellipse, 20), 100),
    ("horseshoe", horseshoe, 64),
    ])
def test_target_association(ctx_getter, curve_name, curve_f, nelements,
        visualize=False):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ generate lpot source

    order = 16

    # Make the curve mesh.
    mesh = make_curve_mesh(curve_f, np.linspace(0, 1, nelements+1), order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    factory = InterpolatoryQuadratureSimplexGroupFactory(order)

    discr = Discretization(cl_ctx, mesh, factory)

    lpot_source, conn = QBXLayerPotentialSource(discr,
            qbx_order=order,  # not used in target association
            fine_order=order).with_refinement()
    del discr

    from pytential.qbx.utils import get_interleaved_centers
    centers = np.array([ax.get(queue)
            for ax in get_interleaved_centers(queue, lpot_source)])

    # }}}

    # {{{ generate targets

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(cl_ctx, seed=RNG_SEED)
    nsources = lpot_source.density_discr.nnodes
    noise = rng.uniform(queue, nsources, dtype=np.float, a=0.01, b=1.0)
    tunnel_radius = \
            lpot_source._close_target_tunnel_radius("nsources").with_queue(queue)

    def targets_from_sources(sign, dist):
        from pytential import sym, bind
        dim = 2
        nodes = bind(lpot_source.density_discr, sym.nodes(dim))(queue)
        normals = bind(lpot_source.density_discr, sym.normal(dim))(queue)
        return (nodes + normals * sign * dist).as_vector(np.object)

    from pytential.target import PointsTarget

    int_targets = PointsTarget(targets_from_sources(-1, noise * tunnel_radius))
    ext_targets = PointsTarget(targets_from_sources(+1, noise * tunnel_radius))
    far_targets = PointsTarget(targets_from_sources(+1, FAR_TARGET_DIST_FROM_SOURCE))

    # Create target discretizations.
    target_discrs = (
        # On-surface targets, interior
        (lpot_source.density_discr, -1),
        # On-surface targets, exterior
        (lpot_source.density_discr, +1),
        # Interior close targets
        (int_targets, -2),
        # Exterior close targets
        (ext_targets, +2),
        # Far targets, should not need centers
        (far_targets, 0),
    )

    sizes = np.cumsum([discr.nnodes for discr, _ in target_discrs])

    (surf_int_slice,
     surf_ext_slice,
     vol_int_slice,
     vol_ext_slice,
     far_slice,
     ) = [slice(start, end) for start, end in zip(np.r_[0, sizes], sizes)]

    # }}}

    # {{{ run target associator and check

    from pytential.qbx.target_assoc import (
            TargetAssociationCodeContainer, associate_targets_to_qbx_centers)

    from pytential.qbx.utils import TreeCodeContainer

    code_container = TargetAssociationCodeContainer(
            cl_ctx, TreeCodeContainer(cl_ctx))

    target_assoc = (associate_targets_to_qbx_centers(
            lpot_source,
            code_container.get_wrangler(queue),
            target_discrs,
            target_association_tolerance=1e-10)
        .get(queue=queue))

    expansion_radii = lpot_source._expansion_radii("ncenters").get(queue)

    surf_targets = np.array(
            [axis.get(queue) for axis in lpot_source.density_discr.nodes()])
    int_targets = np.array([axis.get(queue) for axis in int_targets.nodes()])
    ext_targets = np.array([axis.get(queue) for axis in ext_targets.nodes()])

    def visualize_curve_and_assoc():
        import matplotlib.pyplot as plt
        from meshmode.mesh.visualization import draw_curve

        draw_curve(lpot_source.density_discr.mesh)

        targets = int_targets
        tgt_slice = surf_int_slice

        plt.plot(centers[0], centers[1], "+", color="orange")
        ax = plt.gca()

        for tx, ty, tcenter in zip(
                targets[0, tgt_slice],
                targets[1, tgt_slice],
                target_assoc.target_to_center[tgt_slice]):
            if tcenter >= 0:
                ax.add_artist(
                        plt.Line2D(
                            (tx, centers[0, tcenter]),
                            (ty, centers[1, tcenter]),
                            ))

        ax.set_aspect("equal")
        plt.show()

    if visualize:
        visualize_curve_and_assoc()

    # Checks that the targets match with centers on the appropriate side and
    # within the allowable distance.
    def check_close_targets(centers, targets, true_side,
                            target_to_center, target_to_side_result,
                            tgt_slice):
        targets_have_centers = (target_to_center >= 0).all()
        assert targets_have_centers

        assert (target_to_side_result == true_side).all()

        TOL = 1e-3
        dists = la.norm((targets.T - centers.T[target_to_center]), axis=1)
        assert (dists <= (1 + TOL) * expansion_radii[target_to_center]).all()

    # Center side order = -1, 1, -1, 1, ...
    target_to_center_side = 2 * (target_assoc.target_to_center % 2) - 1

    # interior surface
    check_close_targets(
        centers, surf_targets, -1,
        target_assoc.target_to_center[surf_int_slice],
        target_to_center_side[surf_int_slice],
        surf_int_slice)

    # exterior surface
    check_close_targets(
        centers, surf_targets, +1,
        target_assoc.target_to_center[surf_ext_slice],
        target_to_center_side[surf_ext_slice],
        surf_ext_slice)

    # interior volume
    check_close_targets(
        centers, int_targets, -1,
        target_assoc.target_to_center[vol_int_slice],
        target_to_center_side[vol_int_slice],
        vol_int_slice)

    # exterior volume
    check_close_targets(
        centers, ext_targets, +1,
        target_assoc.target_to_center[vol_ext_slice],
        target_to_center_side[vol_ext_slice],
        vol_ext_slice)

    # Checks that far targets are not assigned a center.
    assert (target_assoc.target_to_center[far_slice] == -1).all()

    # }}}


def test_target_association_failure(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ generate circle

    order = 5
    nelements = 40

    # Make the curve mesh.
    curve_f = partial(ellipse, 1)
    mesh = make_curve_mesh(curve_f, np.linspace(0, 1, nelements+1), order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    factory = InterpolatoryQuadratureSimplexGroupFactory(order)
    discr = Discretization(cl_ctx, mesh, factory)
    lpot_source = QBXLayerPotentialSource(discr,
            qbx_order=order,  # not used in target association
            fine_order=order)

    # }}}

    # {{{ generate targets and check

    close_circle = 0.999 * np.exp(
        2j * np.pi * np.linspace(0, 1, 500, endpoint=False))
    from pytential.target import PointsTarget
    close_circle_target = (
        PointsTarget(cl.array.to_device(
            queue, np.array([close_circle.real, close_circle.imag]))))

    targets = (
        (close_circle_target, 0),
        )

    from pytential.qbx.target_assoc import (
            TargetAssociationCodeContainer, associate_targets_to_qbx_centers,
            QBXTargetAssociationFailedException)

    from pytential.qbx.utils import TreeCodeContainer

    code_container = TargetAssociationCodeContainer(
            cl_ctx, TreeCodeContainer(cl_ctx))

    with pytest.raises(QBXTargetAssociationFailedException):
        associate_targets_to_qbx_centers(
            lpot_source,
            code_container.get_wrangler(queue),
            targets,
            target_association_tolerance=1e-10)

    # }}}


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
