# pyright: reportIncompatibleVariableOverride=none, reportAssignmentType=none
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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pytential import sym
from pytools import memoize_method
from sumpy.kernel import Kernel
from meshmode.mesh import MeshElementGroup, SimplexElementGroup
from meshmode.discretization import ElementGroupFactory
from meshmode.discretization.poly_element import InterpolatoryQuadratureGroupFactory

import logging
logger = logging.getLogger(__name__)


# {{{ make_circular_point_group

def make_circular_point_group(ambient_dim, npoints, radius,
        center=None, func=lambda x: x):
    if center is None:
        center = np.array([0., 0.])
    t = func(np.linspace(0, 1, npoints, endpoint=False)) * (2 * np.pi)
    center = np.asarray(center)
    result = np.zeros((ambient_dim, npoints))
    result[:2, :] = center[:, np.newaxis] + radius*np.vstack((np.cos(t), np.sin(t)))
    return result


def make_source_and_target_points(
        actx, side, inner_radius, outer_radius, ambient_dim,
        nsources=10, ntargets=20):
    if side == -1:
        test_src_geo_radius = outer_radius
        test_tgt_geo_radius = inner_radius
    elif side == +1:
        test_src_geo_radius = inner_radius
        test_tgt_geo_radius = outer_radius
    elif side == "scat":
        test_src_geo_radius = outer_radius
        test_tgt_geo_radius = outer_radius
    else:
        raise ValueError(f"unknown side: {side}")

    from pytential.source import PointPotentialSource
    point_sources = make_circular_point_group(
            ambient_dim, nsources, test_src_geo_radius,
            func=lambda x: x**1.5)
    point_source = PointPotentialSource(
            actx.freeze(actx.from_numpy(point_sources)))

    from pytential.target import PointsTarget
    test_targets = make_circular_point_group(
            ambient_dim, ntargets, test_tgt_geo_radius)
    point_target = PointsTarget(
        actx.freeze(actx.from_numpy(test_targets)))

    return point_source, point_target

# }}}


# {{{ IntegralEquationTestCase

@dataclass
class IntegralEquationTestCase:
    name: str | None = None
    ambient_dim: int | None = None

    # operator
    knl_class_or_helmholtz_k: type[Kernel] | float = 0
    knl_kwargs: dict[str, Any] = field(default_factory=dict)
    bc_type: str = "dirichlet"
    side: int = -1

    # qbx
    qbx_order: int | None = None
    source_ovsmp: int = 4
    target_order: int | None = None
    use_refinement: bool = True
    group_cls: type[MeshElementGroup] = SimplexElementGroup
    group_factory_cls: type[ElementGroupFactory] = InterpolatoryQuadratureGroupFactory

    # fmm
    fmm_backend: str | None = "sumpy"
    fmm_order: int | None = None
    fmm_tol: float | None = None
    disable_fft: bool = False

    # solver
    gmres_tol: float = 1.0e-14

    # test case
    resolutions: list[int] | None = None
    inner_radius: float | None = None
    outer_radius: float | None = None
    check_tangential_deriv: bool = True
    check_gradient: bool = False

    box_extent_norm: str | None = None
    from_sep_smaller_crit: str | None = None

    # {{{ symbolic

    @property
    @memoize_method
    def knl_class(self):
        if isinstance(self.knl_class_or_helmholtz_k, type):
            return self.knl_class_or_helmholtz_k

        if self.knl_class_or_helmholtz_k == 0:
            from sumpy.kernel import LaplaceKernel
            return LaplaceKernel
        else:
            from sumpy.kernel import HelmholtzKernel
            return HelmholtzKernel

    @property
    @memoize_method
    def knl_concrete_kwargs(self):
        if isinstance(self.knl_class_or_helmholtz_k, type):
            return self.knl_kwargs

        kwargs = self.knl_kwargs.copy()
        if self.knl_class_or_helmholtz_k != 0:
            kwargs["k"] = self.knl_class_or_helmholtz_k

        return kwargs

    @property
    @memoize_method
    def knl_sym_kwargs(self):
        return {k: sym.var(k) for k in self.knl_concrete_kwargs}

    def get_operator(self, ambient_dim):
        sign = +1 if self.side in [+1, "scat"] else -1
        knl = self.knl_class(ambient_dim)

        if self.bc_type == "dirichlet":
            from pytential.symbolic.pde.scalar import DirichletOperator
            op = DirichletOperator(knl, sign,
                    use_l2_weighting=True,
                    kernel_arguments=self.knl_sym_kwargs)
        elif self.bc_type == "neumann":
            from pytential.symbolic.pde.scalar import NeumannOperator
            op = NeumannOperator(knl, sign,
                    use_l2_weighting=True,
                    use_improved_operator=False,
                    kernel_arguments=self.knl_sym_kwargs)
        elif self.bc_type == "clamped_plate":
            from pytential.symbolic.pde.scalar import BiharmonicClampedPlateOperator
            op = BiharmonicClampedPlateOperator(knl, sign)
        else:
            raise ValueError(f"unknown bc_type: '{self.bc_type}'")

        return op

    # }}}

    # {{{ geometry

    def get_mesh(self, resolution, mesh_order):
        raise NotImplementedError

    def get_discretization(self, actx, resolution, mesh_order):
        mesh = self.get_mesh(resolution, mesh_order)
        return self._get_discretization(actx, mesh)

    def _get_discretization(self, actx, mesh):
        from meshmode.discretization import Discretization
        return Discretization(actx, mesh,
                self.group_factory_cls(self.target_order))

    def get_layer_potential(self, actx, resolution, mesh_order):
        pre_density_discr = self.get_discretization(actx, resolution, mesh_order)

        from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder
        fmm_kwargs = {}
        if self.fmm_backend is None:
            fmm_kwargs["fmm_order"] = False
        else:
            if self.fmm_tol is not None:
                fmm_kwargs["fmm_order"] = SimpleExpansionOrderFinder(self.fmm_tol)
            elif self.fmm_order is not None:
                fmm_kwargs["fmm_order"] = self.fmm_order
            else:
                fmm_kwargs["fmm_order"] = self.qbx_order + 5

        if self.disable_fft:
            from pytential.qbx import NonFFTExpansionFactory
            fmm_kwargs["expansion_factory"] = NonFFTExpansionFactory()

        from pytential.qbx import QBXLayerPotentialSource
        return QBXLayerPotentialSource(
                pre_density_discr,
                fine_order=self.source_ovsmp * self.target_order,
                qbx_order=self.qbx_order,
                fmm_backend=self.fmm_backend, **fmm_kwargs,

                _disable_refinement=not self.use_refinement,
                _box_extent_norm=self.box_extent_norm,
                _from_sep_smaller_crit=self.from_sep_smaller_crit,
                _from_sep_smaller_min_nsources_cumul=30,
                )

    # }}}

    def __str__(self):
        from dataclasses import fields
        attrs = {f.name: getattr(self, f.name) for f in fields(self)}

        width = len(max(attrs, key=len))
        fmt = f"%{width}s : %s"

        header = {
                "class": type(self).__name__,
                "name": attrs.get("name", self.name),
                "-" * width: "-" * width
                }

        return "\n".join([
            "\t%s" % "\n\t".join(fmt % (k, v) for k, v in header.items()),
            "\t%s" % "\n\t".join(fmt % (k, v) for k, v in sorted(attrs.items())),
            ])

# }}}


# {{{ 2d curves

@dataclass
class CurveTestCase(IntegralEquationTestCase):
    ambient_dim: int = 2

    # qbx
    qbx_order: int = 5
    target_order: int = 5

    # fmm
    fmm_backend: str | None = None

    # test case
    curve_fn: Callable[[np.ndarray], np.ndarray] | None = None
    inner_radius: float = 0.1
    outer_radius: float = 2
    resolutions: list[int] = field(default_factory=lambda: [40, 50, 60])

    def _curve_fn(self, t):
        return self.curve_fn(t)

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import make_curve_mesh
        return make_curve_mesh(
                self._curve_fn,
                np.linspace(0, 1, resolution + 1),
                mesh_order)


@dataclass
class EllipseTestCase(CurveTestCase):
    name: str = "ellipse"
    aspect_ratio: float = 3.0
    radius: float = 1.0

    def _curve_fn(self, t):
        from meshmode.mesh.generation import ellipse
        return self.radius * ellipse(self.aspect_ratio, t)


@dataclass
class CircleTestCase(EllipseTestCase):
    name: str = "circle"
    aspect_ratio: float = 1.0
    radius: float = 1.0


@dataclass
class WobbleCircleTestCase(CurveTestCase):
    name: str = "wobble-circle"
    resolutions: list[int] = field(default_factory=lambda: [2000, 3000, 4000])

    def _curve_fn(self, t):
        from meshmode.mesh.generation import WobblyCircle
        return WobblyCircle.random(30, seed=30)(t)


@dataclass
class StarfishTestCase(CurveTestCase):
    name: str = "starfish"
    n_arms: int = 5
    amplitude: float = 0.25

    # NOTE: these are valid for the (n_arms, amplitude) above
    inner_radius: float = 0.25
    outer_radius: float = 2.0

    resolutions: list[int] = field(default_factory=lambda: [30, 50, 70, 90])

    def _curve_fn(self, t):
        from meshmode.mesh.generation import NArmedStarfish
        return NArmedStarfish(self.n_arms, self.amplitude)(t)

# }}}


# {{{ 3d surfaces

@dataclass
class Helmholtz3DTestCase(IntegralEquationTestCase):
    ambient_dim: int = 3

    # qbx
    use_refinement: bool = False

    # fmm
    fmm_backend: str | None = "fmmlib"

    # solver
    gmres_tol: float = 1.0e-7

    # test case
    check_tangential_deriv: bool = False


@dataclass
class HelmholtzEllisoidTestCase(Helmholtz3DTestCase):
    name: str = "ellipsoid"

    # qbx
    qbx_order: int = 5
    fmm_order: int = 13

    # test case
    resolutions: list[int] = field(
        default_factory=lambda: [2.0, 0.8])     # type: ignore[list-item]
    inner_radius: float = 0.4
    outer_radius: float = 5.0
    check_gradient: bool = True

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("ellipsoid.step"), 2, order=mesh_order,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        # flip elements -- gmsh generates inside-out geometries
        from meshmode.mesh.processing import perform_flips
        return perform_flips(mesh, np.ones(mesh.nelements))


@dataclass
class SphereTestCase(IntegralEquationTestCase):
    name: str = "sphere"
    ambient_dim: int = 3

    # qbx
    qbx_order: int = 5
    target_order: int = 8
    use_refinement: bool = False

    # fmm
    fmm_backend: str | None = "fmmlib"
    fmm_tol: float = 1.0e-4

    # solver
    gmres_tol: float = 1.0e-7

    # test case
    resolutions: list[int] = field(default_factory=lambda: [1, 2])
    check_gradient: bool = False
    check_tangential_deriv: bool = False

    radius: float = 1.0
    inner_radius: float = 0.4
    outer_radius: float = 5.0

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import generate_sphere
        return generate_sphere(self.radius, mesh_order,
                uniform_refinement_rounds=resolution,
                group_cls=self.group_cls)


@dataclass
class SpheroidTestCase(SphereTestCase):
    name: str = "spheroid"
    aspect_ratio: float = 2.0

    def get_mesh(self, resolution, mesh_order):
        mesh = super().get_mesh(resolution, mesh_order)

        from meshmode.mesh.processing import affine_map
        return affine_map(mesh, A=np.diag([
            self.radius, self.radius, self.radius / self.aspect_ratio,
            ]))


@dataclass
class QuadSpheroidTestCase(SphereTestCase):
    name: str = "quadspheroid"
    aspect_ratio: float = 2.0

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh import TensorProductElementGroup
        from meshmode.mesh.generation import generate_sphere
        mesh = generate_sphere(1.0, mesh_order,
                uniform_refinement_rounds=resolution,
                group_cls=TensorProductElementGroup)

        if abs(self.aspect_ratio - 1.0) > 1.0e-14:
            from meshmode.mesh.processing import affine_map
            mesh = affine_map(mesh, A=np.diag([1.0, 1.0, 1/self.aspect_ratio]))

        return mesh


@dataclass
class GMSHSphereTestCase(SphereTestCase):
    name: str = "gmsphere"

    radius: float = 1.5
    resolutions: list[int] = field(
        default_factory=lambda: [0.4])      # type: ignore[list-item]

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.io import ScriptSource
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        if issubclass(self.group_cls, SimplexElementGroup):
            script = ScriptSource(
                """
                Mesh.CharacteristicLengthMax = %(length)g;
                Mesh.HighOrderOptimize = 1;
                Mesh.Algorithm = 1;

                SetFactory("OpenCASCADE");
                Sphere(1) = {0, 0, 0, %(radius)g};
                """ % {"length": resolution, "radius": self.radius},
                "geo")
        elif issubclass(self.group_cls, TensorProductElementGroup):
            script = ScriptSource(
                """
                Mesh.CharacteristicLengthMax = %(length)g;
                Mesh.HighOrderOptimize = 1;
                Mesh.Algorithm = 6;

                SetFactory("OpenCASCADE");
                Sphere(1) = {0, 0, 0, %(radius)g};
                Recombine Surface "*" = 0.0001;
                """ % {"length": resolution, "radius": self.radius},
                "geo")
        else:
            raise TypeError

        from meshmode.mesh.io import generate_gmsh
        return generate_gmsh(
                script,
                order=mesh_order,
                dimensions=2,
                force_ambient_dim=3,
                target_unit="MM",
                )


@dataclass
class TorusTestCase(IntegralEquationTestCase):
    name: str = "torus"
    ambient_dim: int = 3

    # qbx
    qbx_order: int = 4
    target_order: int = 7
    use_refinement: bool = True

    # geometry
    r_major: float = 10.0
    r_minor: float = 2.0

    # test case
    resolutions: list[int] = field(default_factory=lambda: [0, 1, 2])

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import generate_torus
        mesh = generate_torus(self.r_major, self.r_minor, order=mesh_order,
                              group_cls=self.group_cls)

        from meshmode.mesh.refinement import refine_uniformly
        return refine_uniformly(mesh, resolution)


@dataclass
class MergedCubesTestCase(Helmholtz3DTestCase):
    name: str = "merged_cubes"

    # qbx
    use_refinement: bool = True

    # test case
    resolutions: list[int] = field(
        default_factory=lambda: [1.4])  # type: ignore[list-item]
    inner_radius: float = 0.4
    outer_radius: float = 12.0

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("merged-cubes.step"), 2, order=mesh_order,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        # Flip elements--gmsh generates inside-out geometry.
        from meshmode.mesh.processing import perform_flips
        return perform_flips(mesh, np.ones(mesh.nelements))


@dataclass
class ManyEllipsoidTestCase(Helmholtz3DTestCase):
    name: str = "ellipsoid"

    # test case
    resolutions: list[int] = field(default_factory=lambda: [2, 1])
    inner_radius: float = 0.4
    outer_radius: float = 5.0

    # repeated geometry
    pitch: int = 10
    nx: int = 2
    ny: int = 2
    nz: int = 2

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        base_mesh = generate_gmsh(
                FileSource("ellipsoid.step"), 2, order=mesh_order,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        from meshmode.mesh.processing import perform_flips
        base_mesh = perform_flips(base_mesh, np.ones(base_mesh.nelements))

        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
        from meshmode.mesh.tools import rand_rotation_matrix
        meshes = [
                affine_map(
                    base_mesh,
                    A=rand_rotation_matrix(3),
                    b=self.pitch*np.array([
                        (i_x-self.nx//2),
                        (i_y-self.ny//2),
                        (i_z-self.ny//2)]))
                for i_x in range(self.nx)
                for i_y in range(self.ny)
                for i_z in range(self.nz)
                ]

        return merge_disjoint_meshes(meshes, single_group=True)

# }}}


# {{{ fancy geometries

@dataclass
class EllipticPlaneTestCase(IntegralEquationTestCase):
    name: str = "elliptic_plane"
    ambient_dim: int = 3

    # qbx
    qbx_order: int = 3
    target_order: int = 3
    use_refinement: bool = True

    # fmm
    fmm_backend: str | None = "fmmlib"
    fmm_tol: float = 1.0e-4

    # test case
    resolutions: list[int] = field(
        default_factory=lambda: [0.1])  # type: ignore[list-item]
    inner_radius: float = 0.2
    outer_radius: float = 12   # was '-13' in some large-scale run (?)
    check_gradient: bool = False
    check_tangential_deriv: bool = False

    # solver
    # NOTE: we're only expecting three digits based on FMM settings
    gmres_tol: float = 1.0e-5

    # to match the scheme given in the GIGAQBX3D paper
    box_extent_norm: str = "l2"
    from_sep_smaller_crit: str = "static_l2"

    def get_mesh(self, resolution, mesh_order):
        from pytools import download_from_web_if_not_present

        download_from_web_if_not_present(
                "https://raw.githubusercontent.com/inducer/geometries/master/"
                "surface-3d/elliptiplane.brep")

        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("elliptiplane.brep"), 2, order=mesh_order,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        # now centered at origin and extends to -1,1

        from meshmode.mesh.processing import perform_flips
        return perform_flips(mesh, np.ones(mesh.nelements))


@dataclass
class BetterPlaneTestCase(IntegralEquationTestCase):
    name: str = "better_plane"
    ambient_dim: int = 3

    # qbx
    qbx_order: int = 3
    target_order: int = 6

    # fmm
    fmm_backend: str | None = "fmmlib"
    fmm_tol: float = 1.0e-4
    use_refinement: bool = True

    # test case
    resolutions: list[int] = field(
        default_factory=lambda: [0.2])  # type: ignore[list-item]
    inner_radius: float = 0.2
    outer_radius: float = 15
    check_gradient: bool = False
    check_tangential_deriv: bool = False

    # solver
    gmres_tol: float = 1.0e-5

    # other stuff
    visualize_geometry: bool = True
    vis_grid_spacing: tuple[float, float, float] = field(
        default_factory=lambda: (0.025, 0.2, 0.025))
    vis_extend_factor: float = 0.2

    # refine_on_helmholtz_k = False
    # scaled_max_curvature_threshold = 1
    expansion_disturbance_tolerance: float = 0.3

    def get_mesh(self, resolution, target_order):
        from pytools import download_from_web_if_not_present

        download_from_web_if_not_present(
                "https://raw.githubusercontent.com/inducer/geometries/a869fc3/"
                "surface-3d/betterplane.brep")

        # {{{ source

        from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
        mesh = generate_gmsh(
                ScriptWithFilesSource("""
                    Merge "betterplane.brep";

                    Mesh.CharacteristicLengthMax = %(lcmax)f;
                    Mesh.ElementOrder = 2;
                    Mesh.CharacteristicLengthExtendFromBoundary = 0;

                    // 2D mesh optimization
                    // Mesh.Lloyd = 1;

                    l_superfine() = Unique(Abs(Boundary{ Surface{
                        27, 25, 17, 13, 18  }; }));
                    l_fine() = Unique(Abs(Boundary{ Surface{ 2, 6, 7}; }));
                    l_coarse() = Unique(Abs(Boundary{ Surface{ 14, 16  }; }));

                    // p() = Unique(Abs(Boundary{ Line{l_fine()}; }));
                    // Characteristic Length{p()} = 0.05;

                    Field[1] = Distance;
                    Field[1].NNodesByEdge = 100;
                    Field[1].EdgesList = {l_superfine()};

                    Field[2] = Threshold;
                    Field[2].IField = 1;
                    Field[2].LcMin = 0.075;
                    Field[2].LcMax = %(lcmax)f;
                    Field[2].DistMin = 0.1;
                    Field[2].DistMax = 0.4;

                    Field[3] = Distance;
                    Field[3].NNodesByEdge = 100;
                    Field[3].EdgesList = {l_fine()};

                    Field[4] = Threshold;
                    Field[4].IField = 3;
                    Field[4].LcMin = 0.1;
                    Field[4].LcMax = %(lcmax)f;
                    Field[4].DistMin = 0.15;
                    Field[4].DistMax = 0.4;

                    Field[5] = Distance;
                    Field[5].NNodesByEdge = 100;
                    Field[5].EdgesList = {l_coarse()};

                    Field[6] = Threshold;
                    Field[6].IField = 5;
                    Field[6].LcMin = 0.15;
                    Field[6].LcMax = %(lcmax)f;
                    Field[6].DistMin = 0.2;
                    Field[6].DistMax = 0.4;

                    Field[7] = Min;
                    Field[7].FieldsList = {2, 4, 6};

                    Background Field = 7;
                    """ % {
                        "lcmax": resolution,
                        }, ["betterplane.brep"]), 2)

        # }}}

        from meshmode.mesh.processing import perform_flips
        return perform_flips(mesh, np.ones(mesh.nelements))

# }}}

# vim: fdm=marker
