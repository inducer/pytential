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

import pytest
from functools import partial
from dataclasses import dataclass

import numpy as np
import numpy.linalg as la

from arraycontext import flatten
from pytential import GeometryCollection, bind, sym
from pytential.qbx import QBXLayerPotentialSource
import meshmode.mesh.generation as mgen

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

from extra_curve_data import horseshoe
from extra_int_eq_data import QuadSpheroidTestCase

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


RNG_SEED = 10
FAR_TARGET_DIST_FROM_SOURCE = 10


# {{{ source refinement checker

@dataclass
class ElementInfo:
    element_nr: int
    discr_slice: slice


def iter_elements(discr):
    discr_nodes_idx = 0
    element_nr = 0

    for discr_group in discr.groups:
        start = element_nr
        for element_nr in range(start, start + discr_group.nelements):
            yield ElementInfo(
                element_nr=element_nr,
                discr_slice=slice(discr_nodes_idx,
                    discr_nodes_idx + discr_group.nunit_dofs))

            discr_nodes_idx += discr_group.nunit_dofs


def run_source_refinement_test(actx_factory, mesh, order,
        helmholtz_k=None, surface_name="surface", visualize=False):
    actx = actx_factory()

    # {{{ initial geometry

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureGroupFactory
    discr = Discretization(actx, mesh, InterpolatoryQuadratureGroupFactory(order))

    lpot_source = QBXLayerPotentialSource(discr,
            qbx_order=order,  # not used in refinement
            fine_order=order)
    places = GeometryCollection(lpot_source)

    logger.info("nelements: %d", discr.mesh.nelements)
    logger.info("ndofs: %d", discr.ndofs)

    # }}}

    # {{{ refined geometry

    def _visualize_quad_resolution(_places, dd, suffix):
        if dd.discr_stage is None:
            vis_discr = lpot_source.density_discr
        else:
            vis_discr = _places.get_discretization(dd.geometry, dd.discr_stage)

        stretch = bind(_places,
                sym._simplex_mapping_max_stretch_factor(
                    _places.ambient_dim, with_elementwise_max=False),
                auto_where=dd)(actx)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, vis_discr, order, force_equidistant=True)
        vis.write_vtk_file(
                f"global-qbx-source-refinement-{surface_name}-{order}-{suffix}.vtu",
                [("stretch", stretch)],
                overwrite=True, use_high_order=True)

    kernel_length_scale = 5 / helmholtz_k if helmholtz_k else None
    expansion_disturbance_tolerance = 0.025

    from pytential.qbx.refinement import refine_geometry_collection
    places = refine_geometry_collection(places,
            kernel_length_scale=kernel_length_scale,
            expansion_disturbance_tolerance=expansion_disturbance_tolerance,
            visualize=False)

    if visualize:
        dd = places.auto_source
        _visualize_quad_resolution(places, dd.copy(discr_stage=None), "original")
        _visualize_quad_resolution(places, dd.to_stage1(), "stage1")
        _visualize_quad_resolution(places, dd.to_stage2(), "stage2")

    # }}}

    dd = places.auto_source
    ambient_dim = places.ambient_dim
    stage1_density_discr = places.get_discretization(dd.geometry)

    stage1_density_nodes = actx.to_numpy(
            flatten(stage1_density_discr.nodes(), actx)
            ).reshape(ambient_dim, -1)

    quad_stage2_density_discr = places.get_discretization(
            dd.geometry, sym.QBX_SOURCE_QUAD_STAGE2)
    quad_stage2_density_nodes = actx.to_numpy(
            flatten(quad_stage2_density_discr.nodes(), actx)
            ).reshape(ambient_dim, -1)

    int_centers = actx.to_numpy(flatten(
        bind(places, sym.expansion_centers(ambient_dim, -1))(actx), actx)
        ).reshape(ambient_dim, -1)
    ext_centers = actx.to_numpy(flatten(
        bind(places, sym.expansion_centers(ambient_dim, +1))(actx), actx)
        ).reshape(ambient_dim, -1)
    expansion_radii = actx.to_numpy(flatten(
        bind(places, sym.expansion_radii(ambient_dim))(actx), actx)
        )

    dd = dd.copy(granularity=sym.GRANULARITY_ELEMENT)
    source_danger_zone_radii = actx.to_numpy(flatten(
            bind(
                places,
                sym._source_danger_zone_radii(ambient_dim, dofdesc=dd.to_stage2())
                )(actx), actx)
            )
    quad_res = actx.to_numpy(flatten(
            bind(places,
                sym._quad_resolution(ambient_dim, dofdesc=dd))(actx), actx)
            )

    # {{{ check if satisfying criteria

    def check_disk_undisturbed_by_sources(centers_panel, sources_panel):
        if centers_panel.element_nr == sources_panel.element_nr:
            # Same panel
            return

        my_int_centers = int_centers[:, centers_panel.discr_slice]
        my_ext_centers = ext_centers[:, centers_panel.discr_slice]
        all_centers = np.append(my_int_centers, my_ext_centers, axis=-1)

        nodes = stage1_density_nodes[:, sources_panel.discr_slice]

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
        assert np.all(dist >= rad * (1 - expansion_disturbance_tolerance)), (
                dist, rad, centers_panel.element_nr, sources_panel.element_nr)

    def check_sufficient_quadrature_resolution(centers_panel, sources_panel):
        dz_radius = source_danger_zone_radii[sources_panel.element_nr]

        my_int_centers = int_centers[:, centers_panel.discr_slice]
        my_ext_centers = ext_centers[:, centers_panel.discr_slice]
        all_centers = np.append(my_int_centers, my_ext_centers, axis=-1)

        nodes = quad_stage2_density_nodes[:, sources_panel.discr_slice]

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

    for panel_1 in iter_elements(stage1_density_discr):
        for panel_2 in iter_elements(stage1_density_discr):
            check_disk_undisturbed_by_sources(panel_1, panel_2)
        for panel_2 in iter_elements(quad_stage2_density_discr):
            check_sufficient_quadrature_resolution(panel_1, panel_2)
        if helmholtz_k is not None:
            check_quad_res_to_helmholtz_k_ratio(panel_1)

    # }}}

# }}}


@pytest.mark.parametrize(("curve_name", "curve_f", "nelements"), [
    ("20-to-1 ellipse", partial(mgen.ellipse, 20), 100),
    ("horseshoe", horseshoe, 64),
    ])
def test_source_refinement_2d(actx_factory,
        curve_name, curve_f, nelements, visualize=False):
    helmholtz_k = 10
    order = 8

    mesh = mgen.make_curve_mesh(curve_f, np.linspace(0, 1, nelements+1), order)
    run_source_refinement_test(actx_factory, mesh, order,
            helmholtz_k=helmholtz_k,
            surface_name=curve_name,
            visualize=visualize)


@pytest.mark.parametrize(("surface_name", "surface_f", "order"), [
    ("sphere", partial(mgen.generate_sphere, 1), 4),
    ("torus", partial(mgen.generate_torus, 3, 1, n_minor=10, n_major=7), 6),
    ("spheroid-quad", lambda order: QuadSpheroidTestCase().get_mesh(2, order), 4),
    ])
def test_source_refinement_3d(actx_factory,
        surface_name, surface_f, order, visualize=False):
    mesh = surface_f(order=order)
    run_source_refinement_test(actx_factory, mesh, order,
            surface_name=surface_name,
            visualize=visualize)


@pytest.mark.parametrize(("curve_name", "curve_f", "nelements"), [
    ("20-to-1 ellipse", partial(mgen.ellipse, 20), 100),
    ("horseshoe", horseshoe, 64),
    ])
def test_target_association(actx_factory, curve_name, curve_f, nelements,
        visualize=False):
    actx = actx_factory()

    # {{{ generate lpot source

    order = 16

    # Make the curve mesh.
    mesh = mgen.make_curve_mesh(curve_f, np.linspace(0, 1, nelements+1), order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    factory = InterpolatoryQuadratureSimplexGroupFactory(order)
    discr = Discretization(actx, mesh, factory)

    lpot_source = QBXLayerPotentialSource(discr,
            qbx_order=order,  # not used in target association
            fine_order=order)
    places = GeometryCollection(lpot_source)

    # }}}

    # {{{ generate targets

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(actx.context, seed=RNG_SEED)

    ambient_dim = places.ambient_dim
    dd = places.auto_source.to_stage1()

    centers = actx.to_numpy(flatten(
        bind(places,
            sym.interleaved_expansion_centers(ambient_dim, dofdesc=dd))(actx),
        actx)).reshape(ambient_dim, -1)

    density_discr = places.get_discretization(dd.geometry)

    noise = actx.to_numpy(
            rng.uniform(actx.queue, density_discr.ndofs,
                dtype=np.float64, a=0.01, b=1.0)
            )

    tunnel_radius = actx.to_numpy(flatten(
        bind(places, sym._close_target_tunnel_radii(ambient_dim, dofdesc=dd))(actx),
        actx))

    def targets_from_sources(sign, dist, dim=2):
        nodes = actx.to_numpy(flatten(
                bind(places, sym.nodes(dim, dofdesc=dd))(actx).as_vector(), actx)
                ).reshape(dim, -1)
        normals = actx.to_numpy(flatten(
                bind(places, sym.normal(dim, dofdesc=dd))(actx).as_vector(), actx)
                ).reshape(dim, -1)

        return actx.from_numpy(nodes + normals * sign * dist)

    from pytential.target import PointsTarget
    int_targets = PointsTarget(targets_from_sources(-1, noise * tunnel_radius))
    ext_targets = PointsTarget(targets_from_sources(+1, noise * tunnel_radius))
    far_targets = PointsTarget(targets_from_sources(+1, FAR_TARGET_DIST_FROM_SOURCE))

    # Create target discretizations.
    target_discrs = (
        # On-surface targets, interior
        (density_discr, -1),
        # On-surface targets, exterior
        (density_discr, +1),
        # Interior close targets
        (int_targets, -2),
        # Exterior close targets
        (ext_targets, +2),
        # Far targets, should not need centers
        (far_targets, 0),
    )

    sizes = np.cumsum([discr.ndofs for discr, _ in target_discrs])

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
            actx, TreeCodeContainer(actx))

    target_assoc = (
            associate_targets_to_qbx_centers(
                places,
                places.auto_source,
                code_container.get_wrangler(actx),
                target_discrs,
                target_association_tolerance=1e-10)
            ).get(queue=actx.queue)

    expansion_radii = actx.to_numpy(flatten(
            bind(places, sym.expansion_radii(ambient_dim,
                granularity=sym.GRANULARITY_CENTER))(actx), actx)
            )
    surf_targets = actx.to_numpy(
            flatten(density_discr.nodes(), actx)
            ).reshape(ambient_dim, -1)
    int_targets = actx.to_numpy(int_targets.nodes())
    ext_targets = actx.to_numpy(ext_targets.nodes())

    def visualize_curve_and_assoc():
        import matplotlib.pyplot as plt
        from meshmode.mesh.visualization import draw_curve

        draw_curve(density_discr.mesh)

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
                            tgt_slice, tol=1.0e-3):
        targets_have_centers = np.all(target_to_center >= 0)
        assert targets_have_centers
        assert np.all(target_to_side_result == true_side)

        dists = la.norm((targets.T - centers.T[target_to_center]), axis=1)
        assert np.all(dists <= (1 + tol) * expansion_radii[target_to_center])

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
    assert np.all(target_assoc.target_to_center[far_slice] == -1)

    # }}}


def test_target_association_failure(actx_factory):
    actx = actx_factory()

    # {{{ generate circle

    order = 5
    nelements = 40

    # Make the curve mesh.
    curve_f = partial(mgen.ellipse, 1)
    mesh = mgen.make_curve_mesh(curve_f, np.linspace(0, 1, nelements+1), order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    factory = InterpolatoryQuadratureSimplexGroupFactory(order)
    discr = Discretization(actx, mesh, factory)
    lpot_source = QBXLayerPotentialSource(discr,
            qbx_order=order,  # not used in target association
            fine_order=order)
    places = GeometryCollection(lpot_source)

    # }}}

    # {{{ generate targets and check

    close_circle = 0.999 * np.exp(
        2j * np.pi * np.linspace(0, 1, 500, endpoint=False))
    from pytential.target import PointsTarget
    close_circle_target = (
            PointsTarget(
                actx.from_numpy(
                    np.array([close_circle.real, close_circle.imag]))))

    targets = (
        (close_circle_target, 0),
        )

    from pytential.qbx.target_assoc import (
            TargetAssociationCodeContainer, associate_targets_to_qbx_centers,
            QBXTargetAssociationFailedException)

    from pytential.qbx.utils import TreeCodeContainer

    code_container = TargetAssociationCodeContainer(
            actx, TreeCodeContainer(actx))

    with pytest.raises(QBXTargetAssociationFailedException):
        associate_targets_to_qbx_centers(
            places,
            places.auto_source,
            code_container.get_wrangler(actx),
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
