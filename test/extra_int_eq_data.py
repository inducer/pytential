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

from pytential import sym
from pytools import RecordWithoutPickling, memoize_method

import logging
logger = logging.getLogger(__name__)


# {{{ make_circular_point_group

def make_circular_point_group(ambient_dim, npoints, radius,
        center=np.array([0., 0.]), func=lambda x: x):
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

class IntegralEquationTestCase(RecordWithoutPickling):
    name = "unknown"

    # operator
    knl_class_or_helmholtz_k = 0
    knl_kwargs = {}
    bc_type = "dirichlet"
    side = -1

    # qbx
    qbx_order = None
    source_ovsmp = 4
    target_order = None
    use_refinement = True

    # fmm
    fmm_backend = "sumpy"
    fmm_order = None
    fmm_tol = None

    # solver
    gmres_tol = 1.0e-14

    # test case
    resolutions = None
    inner_radius = None
    outer_radius = None
    check_tangential_deriv = True
    check_gradient = False

    def __init__(self, **kwargs):
        import inspect
        members = inspect.getmembers(type(self), lambda m: not inspect.isroutine(m))
        members = dict(
                m for m in members
                if (not m[0].startswith("__")
                    and m[0] != "fields"
                    and not isinstance(m[1], property))
                )

        for k, v in kwargs.items():
            if k not in members:
                raise KeyError(f"unknown keyword argument '{k}'")
            members[k] = v

        super().__init__(**members)

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
        knl = self.knl_class(ambient_dim)   # noqa: pylint:disable=E1102

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

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        return Discretization(actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(self.target_order))

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

        from pytential.qbx import QBXLayerPotentialSource
        return QBXLayerPotentialSource(
                pre_density_discr,
                fine_order=self.source_ovsmp * self.target_order,
                qbx_order=self.qbx_order,
                fmm_backend=self.fmm_backend, **fmm_kwargs,

                _disable_refinement=not self.use_refinement,
                _box_extent_norm=getattr(self, "box_extent_norm", None),
                _from_sep_smaller_crit=getattr(self, "from_sep_smaller_crit", None),
                _from_sep_smaller_min_nsources_cumul=30,
                )

    # }}}

    def __str__(self):
        if not self.__class__.fields:
            return f"{type(self).__name__}()"

        width = len(max(list(self.__class__.fields), key=len))
        fmt = f"%{width}s : %s"

        attrs = {k: getattr(self, k) for k in self.__class__.fields}
        header = {
                "class": type(self).__name__,
                "name": attrs.pop("name"),
                "-" * width: "-" * width
                }

        return "\n".join([
            "\t%s" % "\n\t".join(fmt % (k, v) for k, v in header.items()),
            "\t%s" % "\n\t".join(fmt % (k, v) for k, v in sorted(attrs.items())),
            ])

# }}}


# {{{ 2d curves

class CurveTestCase(IntegralEquationTestCase):
    # qbx
    qbx_order = 5
    target_order = 5

    # fmm
    fmm_backend = None

    # test case
    curve_fn = None
    inner_radius = 0.1
    outer_radius = 2
    resolutions = [40, 50, 60]

    def _curve_fn(self, t):
        return self.curve_fn(t)     # pylint:disable=not-callable

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import make_curve_mesh
        return make_curve_mesh(
                self._curve_fn,
                np.linspace(0, 1, resolution + 1),
                mesh_order)


class EllipseTestCase(CurveTestCase):
    name = "ellipse"
    aspect_ratio = 3.0
    radius = 1.0

    def _curve_fn(self, t):
        from meshmode.mesh.generation import ellipse
        return self.radius * ellipse(self.aspect_ratio, t)


class CircleTestCase(EllipseTestCase):
    name = "circle"
    aspect_ratio = 1.0
    radius = 1.0

# }}}


# {{{ 3d surfaces

class Helmholtz3DTestCase(IntegralEquationTestCase):
    # qbx
    use_refinement = False

    # fmm
    fmm_backend = "fmmlib"

    # solver
    gmres_tol = 1.0e-7

    # test case
    check_tangential_deriv = False


class HelmholtzEllisoidTestCase(Helmholtz3DTestCase):
    name = "ellipsoid"

    # qbx
    qbx_order = 5
    fmm_order = 13

    # test case
    resolutions = [2, 0.8]
    inner_radius = 0.4
    outer_radius = 5.0
    check_gradient = True

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


class SphereTestCase(IntegralEquationTestCase):
    name = "sphere"

    # qbx
    qbx_order = 5
    target_order = 8
    use_refinement = False

    # fmm
    fmm_backend = "fmmlib"
    fmm_tol = 1.0e-4

    # solver
    gmres_tol = 1.0e-7

    # test case
    resolutions = [1, 2]
    inner_radius = 0.4
    outer_radius = 5.0
    check_gradient = False
    check_tangential_deriv = False

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import generate_icosphere
        return generate_icosphere(1.0, mesh_order,
                uniform_refinement_rounds=resolution)


class TorusTestCase(IntegralEquationTestCase):
    name = "torus"

    # qbx
    qbx_order = 4
    target_order = 7
    use_refinement = True

    # geometry
    r_major = 10.0
    r_minor = 2.0

    # test case
    resolutions = [0, 1, 2]

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import generate_torus
        mesh = generate_torus(self.r_major, self.r_minor, order=mesh_order)

        from meshmode.mesh.refinement import refine_uniformly
        return refine_uniformly(mesh, resolution)


class MergedCubesTestCase(Helmholtz3DTestCase):
    name = "merged_cubes"

    # qbx
    use_refinement = True

    # test case
    resolutions = [1.4]
    inner_radius = 0.4
    outer_radius = 12.0

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


class ManyEllipsoidTestCase(Helmholtz3DTestCase):
    name = "ellipsoid"

    # test case
    resolutions = [2, 1]
    inner_radius = 0.4
    outer_radius = 5.0

    # repeated geometry
    pitch = 10
    nx = 2
    ny = 2
    nz = 2

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
                        (ix-self.nx//2),
                        (iy-self.ny//2),
                        (iz-self.ny//2)]))
                for ix in range(self.nx)
                for iy in range(self.ny)
                for iz in range(self.nz)
                ]

        return merge_disjoint_meshes(meshes, single_group=True)

# }}}


# {{{ fancy geometries

class EllipticPlaneTestCase(IntegralEquationTestCase):
    name = "elliptic_plane"

    # qbx
    qbx_order = 3
    target_order = 3
    use_refinement = True

    # fmm
    fmm_backend = "fmmlib"
    fmm_tol = 1.0e-4

    # test case
    resolutions = [0.1]
    inner_radius = 0.2
    outer_radius = 12   # was '-13' in some large-scale run (?)
    check_gradient = False
    check_tangential_deriv = False

    # solver
    # NOTE: we're only expecting three digits based on FMM settings
    gmres_tol = 1.0e-5

    # to match the scheme given in the GIGAQBX3D paper
    box_extent_norm = "l2"
    from_sep_smaller_crit = "static_l2"

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


class BetterPlaneTestCase(IntegralEquationTestCase):
    name = "better_plane"

    # qbx
    qbx_order = 3
    target_order = 6

    # fmm
    fmm_backend = "fmmlib"
    fmm_tol = 1.0e-4
    use_refinement = True

    # test case
    resolutions = [0.2]
    inner_radius = 0.2
    outer_radius = 15
    check_gradient = False
    check_tangential_deriv = False

    # solver
    gmres_tol = 1.0e-5

    # other stuff
    visualize_geometry = True
    vis_grid_spacing = (0.025, 0.2, 0.025)
    vis_extend_factor = 0.2

    # refine_on_helmholtz_k = False
    # scaled_max_curvature_threshold = 1
    expansion_disturbance_tolerance = 0.3

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
