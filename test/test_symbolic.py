__copyright__ = "Copyright (C) 2017 Matt Wala"

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
import numpy.linalg as la

from arraycontext import thaw, flatten, unflatten
import meshmode.mesh.generation as mgen
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory
from pytential import bind, sym

from meshmode import _acf           # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ discretization getters

def get_ellipse_with_ref_mean_curvature(actx, nelements, aspect=1):
    order = 4
    mesh = mgen.make_curve_mesh(
            partial(mgen.ellipse, aspect),
            np.linspace(0, 1, nelements+1),
            order)

    discr = Discretization(actx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(order))

    nodes = thaw(discr.nodes(), actx)

    a = 1
    b = 1/aspect
    t = actx.np.arctan2(nodes[1] * aspect, nodes[0])

    return discr, a*b / ((a*actx.np.sin(t))**2 + (b*actx.np.cos(t))**2)**(3/2)


def get_torus_with_ref_mean_curvature(actx, h):
    order = 4
    r_minor = 1.0
    r_major = 3.0

    from meshmode.mesh.generation import generate_torus
    mesh = generate_torus(r_major, r_minor,
            n_major=h, n_minor=h, order=order)
    discr = Discretization(actx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(order))

    nodes = thaw(discr.nodes(), actx)

    # copied from meshmode.mesh.generation.generate_torus
    a = r_major
    b = r_minor

    u = actx.np.arctan2(nodes[1], nodes[0])
    from pytools.obj_array import flat_obj_array
    rvec = flat_obj_array(actx.np.cos(u), actx.np.sin(u), 0*u)
    rvec = sum(nodes * rvec) - a
    cosv = actx.np.cos(actx.np.arctan2(nodes[2], rvec))

    return discr, (a + 2.0 * b * cosv) / (2 * b * (a + b * cosv))

# }}}


# {{{ test_mean_curvature

@pytest.mark.parametrize(("discr_name",
        "resolutions",
        "discr_and_ref_mean_curvature_getter"), [
    ("unit_circle", [16, 32, 64],
        get_ellipse_with_ref_mean_curvature),
    ("2-to-1 ellipse", [16, 32, 64],
        partial(get_ellipse_with_ref_mean_curvature, aspect=2)),
    ("torus", [8, 10, 12, 16],
        get_torus_with_ref_mean_curvature),
    ])
def test_mean_curvature(actx_factory, discr_name, resolutions,
        discr_and_ref_mean_curvature_getter, visualize=False):
    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for r in resolutions:
        discr, ref_mean_curvature = \
                discr_and_ref_mean_curvature_getter(actx, r)
        mean_curvature = bind(
            discr, sym.mean_curvature(discr.ambient_dim))(actx)

        h = 1.0 / r
        from meshmode.dof_array import flat_norm
        h_error = flat_norm(mean_curvature - ref_mean_curvature, np.inf)
        eoc.add_data_point(h, actx.to_numpy(h_error))
    print(eoc)

    order = min([g.order for g in discr.groups])
    assert eoc.order_estimate() > order - 1.1

# }}}


# {{{ test_tangential_onb

def test_tangential_onb(actx_factory):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_torus
    mesh = generate_torus(5, 2, order=3)

    discr = Discretization(
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(3))

    tob = sym.tangential_onb(mesh.ambient_dim)
    nvecs = tob.shape[1]

    # make sure tangential_onb is mutually orthogonal and normalized
    orth_check = bind(discr, sym.make_obj_array([
        np.dot(tob[:, i], tob[:, j]) - (1 if i == j else 0)
        for i in range(nvecs) for j in range(nvecs)])
        )(actx)

    for orth_i in orth_check:
        assert actx.to_numpy(
                actx.np.all(actx.np.abs(orth_i) < 1e-13)
                )

    # make sure tangential_onb is orthogonal to normal
    orth_check = bind(discr, sym.make_obj_array([
        np.dot(tob[:, i], sym.normal(mesh.ambient_dim).as_vector())
        for i in range(nvecs)])
        )(actx)

    for orth_i in orth_check:
        assert actx.to_numpy(
                actx.np.all(actx.np.abs(orth_i) < 1e-13)
                )

# }}}


# {{{ test_expr_pickling

def test_expr_pickling():
    import pickle
    from sumpy.kernel import LaplaceKernel, AxisTargetDerivative

    ops_for_testing = [
        sym.d_dx(
            2,
            sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=-2)
        ),
        sym.D(
            AxisTargetDerivative(0, LaplaceKernel(2)),
            sym.var("sigma"),
            qbx_forced_limit=-2
        )
    ]

    for op in ops_for_testing:
        pickled_op = pickle.dumps(op)
        after_pickle_op = pickle.loads(pickled_op)

        assert op == after_pickle_op

# }}}


# {{{ test basic layer potentials

@pytest.mark.parametrize("lpot_class", [
    sym.S, sym.Sp, sym.Spp, sym.D, sym.Dp
    ])
def test_layer_potential_construction(lpot_class, ambient_dim=2):
    from sumpy.kernel import LaplaceKernel

    kernel_sym = LaplaceKernel(ambient_dim)
    density_sym = sym.var("sigma")
    lpot_sym = lpot_class(kernel_sym, density_sym, qbx_forced_limit=None)

    assert lpot_sym is not None

# }}}


# {{{ test interpolation

@pytest.mark.parametrize(("name", "source_discr_stage", "target_granularity"), [
    ("default_explicit", sym.QBX_SOURCE_STAGE1, sym.GRANULARITY_NODE),
    ("stage2", sym.QBX_SOURCE_STAGE2, sym.GRANULARITY_NODE),
    ("stage2_center", sym.QBX_SOURCE_STAGE2, sym.GRANULARITY_CENTER),
    ("quad", sym.QBX_SOURCE_QUAD_STAGE2, sym.GRANULARITY_NODE)
    ])
def test_interpolation(actx_factory, name, source_discr_stage, target_granularity):
    actx = actx_factory()

    nelements = 32
    target_order = 7
    qbx_order = 4

    where = sym.as_dofdesc("test_interpolation")
    from_dd = sym.DOFDescriptor(
            geometry=where.geometry,
            discr_stage=source_discr_stage,
            granularity=sym.GRANULARITY_NODE)
    to_dd = sym.DOFDescriptor(
            geometry=where.geometry,
            discr_stage=sym.QBX_SOURCE_QUAD_STAGE2,
            granularity=target_granularity)

    mesh = mgen.make_curve_mesh(mgen.starfish,
            np.linspace(0.0, 1.0, nelements + 1),
            target_order)
    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(discr,
            fine_order=4 * target_order,
            qbx_order=qbx_order,
            fmm_order=False)

    from pytential import GeometryCollection
    places = GeometryCollection(qbx, auto_where=where)

    sigma_sym = sym.var("sigma")
    op_sym = sym.sin(sym.interp(from_dd, to_dd, sigma_sym))
    bound_op = bind(places, op_sym, auto_where=where)

    def discr_and_nodes(stage):
        density_discr = places.get_discretization(where.geometry, stage)
        return density_discr, actx.to_numpy(
                flatten(density_discr.nodes(), actx)
                ).reshape(density_discr.ambient_dim, -1)

    _, target_nodes = discr_and_nodes(sym.QBX_SOURCE_QUAD_STAGE2)
    source_discr, source_nodes = discr_and_nodes(source_discr_stage)

    sigma_target = np.sin(la.norm(target_nodes, axis=0))
    sigma_dev = unflatten(
            thaw(source_discr.nodes()[0], actx),
            actx.from_numpy(la.norm(source_nodes, axis=0)), actx)
    sigma_target_interp = actx.to_numpy(
            flatten(bound_op(actx, sigma=sigma_dev), actx)
            )

    if name in ("default", "default_explicit", "stage2", "quad"):
        error = la.norm(sigma_target_interp - sigma_target) / la.norm(sigma_target)
        assert error < 1.0e-10
    elif name in ("stage2_center",):
        assert len(sigma_target_interp) == 2 * len(sigma_target)
    else:
        raise ValueError(f"unknown test case name: {name}")

# }}}


# {{{ test node reductions

def test_node_reduction(actx_factory):
    actx = actx_factory()

    # {{{ build discretization

    target_order = 4
    nelements = 32

    mesh = mgen.make_curve_mesh(mgen.starfish,
            np.linspace(0.0, 1.0, nelements + 1),
            target_order)
    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    # }}}

    # {{{ test

    # create a shuffled [1, nelements + 1] array
    ary = []
    el_nr_base = 0
    for grp in discr.groups:
        x = 1 + np.arange(el_nr_base, grp.nelements)
        np.random.shuffle(x)

        ary.append(actx.freeze(actx.from_numpy(x.reshape(-1, 1))))
        el_nr_base += grp.nelements

    from meshmode.dof_array import DOFArray
    ary = DOFArray(actx, tuple(ary))

    for func, expected in [
            (sym.NodeSum, nelements * (nelements + 1) // 2),
            (sym.NodeMax, nelements),
            (sym.NodeMin, 1),
            ]:
        r = bind(discr, func(sym.var("x")))(actx, x=ary)
        assert abs(actx.to_numpy(r) - expected) < 1.0e-15, r

    # }}}

# }}}


# {{{ test_prepare_expr

def principal_curvatures(ambient_dim, dim=None, dofdesc=None):
    from pytential import sym
    s_op = sym.shape_operator(ambient_dim, dim=dim, dofdesc=dofdesc)

    from pytential.symbolic.primitives import _small_mat_eigenvalues
    kappa1, kappa2 = _small_mat_eigenvalues(s_op)

    from pytools.obj_array import make_obj_array
    return make_obj_array([
        sym.cse(kappa1, "principal_curvature_0", sym.cse_scope.DISCRETIZATION),
        sym.cse(kappa2, "principal_curvature_1", sym.cse_scope.DISCRETIZATION),
        ])


def principal_directions(ambient_dim, dim=None, dofdesc=None):
    from pytential import sym
    s_op = sym.shape_operator(ambient_dim, dim=dim, dofdesc=dofdesc)

    (s11, s12), (_, s22) = s_op
    k1, k2 = principal_curvatures(ambient_dim, dim=dim, dofdesc=dofdesc)

    from pytools.obj_array import make_obj_array
    d1 = sym.cse(make_obj_array([s12, -(s11 - k1)]))
    d2 = sym.cse(make_obj_array([-(s22 - k2), s12]))

    form1 = sym.first_fundamental_form(ambient_dim, dim=dim, dofdesc=dofdesc)
    return make_obj_array([
        sym.cse(
            d1 / sym.sqrt(d1 @ (form1 @ d1)),
            "principal_direction_0", sym.cse_scope.DISCRETIZATION),
        sym.cse(
            d2 / sym.sqrt(d2 @ (form1 @ d2)),
            "principal_direction_1", sym.cse_scope.DISCRETIZATION),
        ])


def test_derivative_binder_expr():
    logging.basicConfig(level=logging.INFO)

    ambient_dim = 3
    dim = ambient_dim - 1

    from pytential.symbolic.mappers import DerivativeBinder
    d1, d2 = principal_directions(ambient_dim, dim=dim)
    expr = (d1 @ d2 + d1 @ d1) / (d2 @ d2)

    nruns = 4
    for i in range(nruns):
        from pytools import ProcessTimer
        with ProcessTimer() as pd:
            new_expr = DerivativeBinder()(expr)

        logger.info("time: [%04d/%04d] bind [%s] (%s)",
                i, nruns, pd, expr is new_expr)

        assert expr is new_expr

# }}}


# {{{ test_stretch_factor

# {{{ twisted mesh

def make_twisted_mesh(order, cls):
    #  2       3             5
    #   o------o-----------o
    #   |      |           |
    #   |      |           |
    #   |      |           |
    #   o------o-----------o
    #  0       1            4
    #
    #
    vertices = np.array([
        [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
        [4, -1, 0], [4, 1, 0],
        ], dtype=np.float64).T

    import meshmode.mesh as mm
    if issubclass(cls, mm.SimplexElementGroup):
        vertex_indices = np.array([
            (0, 1, 2), (1, 3, 2),
            (1, 4, 3), (4, 5, 3),
            ], dtype=np.int32)
    elif issubclass(cls, mm.TensorProductElementGroup):
        vertex_indices = np.array([
            (0, 1, 2, 3), (1, 4, 3, 5),
            ], dtype=np.int32)
    else:
        raise ValueError

    from meshmode.mesh.generation import make_group_from_vertices
    grp = make_group_from_vertices(
            vertices, vertex_indices, order,
            unit_nodes=None,
            group_cls=cls)

    def wobble(x):
        result = np.empty_like(x)
        result[0] = x[0] + 0.5 * np.sin(x[1])
        result[1] = x[1] + 0.5 * np.cos(x[0])
        result[2] = x[2] + 0.5 * np.cos(x[1]) + 0.5 * np.sin(x[0])
        # result[2] = np.sin(x[1]) * np.sin(x[0])

        return result

    from meshmode.mesh.processing import map_mesh
    mesh = mm.Mesh(vertices, [grp], is_conforming=True)
    return map_mesh(mesh, wobble)

# }}}


# {{{ torus elements

def make_torus_mesh(order, cls, a=2.0, b=1.0, n_major=12, n_minor=6):
    # NOTE: this is the torus construction before
    #   https://github.com/inducer/meshmode/pull/288
    # which caused very discontinuous stretch factors on simplices
    u, v = np.mgrid[0:2*np.pi:2*np.pi/n_major, 0:2*np.pi:2*np.pi/n_minor]

    x = np.cos(u) * (a + b*np.cos(v))
    y = np.sin(u) * (a + b*np.cos(v))
    z = b * np.sin(v)
    del u, v

    vertices = (
            np.vstack((x[np.newaxis], y[np.newaxis], z[np.newaxis]))
            .transpose(0, 2, 1).copy().reshape(3, -1))

    def idx(i, j):
        return (i % n_major) + (j % n_minor) * n_major

    import meshmode.mesh as mm
    # i, j = 0, 0
    i, j = 0, n_minor // 3
    if issubclass(cls, mm.SimplexElementGroup):
        vertex_indices = [
                (idx(i, j), idx(i+1, j), idx(i, j+1)),
                (idx(i+1, j), idx(i+1, j+1), idx(i, j+1)),
                ]
    elif issubclass(cls, mm.TensorProductElementGroup):
        vertex_indices = [(idx(i, j), idx(i+1, j), idx(i, j+1), idx(i+1, j+1))]
    else:
        raise TypeError(f"unsupported 'group_cls': {cls}")

    from meshmode.mesh.generation import make_group_from_vertices
    vertex_indices = np.array(vertex_indices, dtype=np.int32)
    grp = make_group_from_vertices(
            vertices, vertex_indices, order,
            group_cls=cls)

    # NOTE: project the nodes back to the torus surface
    u = np.arctan2(grp.nodes[1], grp.nodes[0])
    v = np.arctan2(
            grp.nodes[2],
            grp.nodes[0] * np.cos(u) + grp.nodes[1] * np.sin(u) - a)

    nodes = np.empty_like(grp.nodes)
    nodes[0] = np.cos(u) * (a + b*np.cos(v))
    nodes[1] = np.sin(u) * (a + b*np.cos(v))
    nodes[2] = b * np.sin(v)

    return mm.Mesh(vertices, [grp.copy(nodes=nodes)], is_conforming=True)

# }}}


# {{{ gmsh sphere

def make_gmsh_sphere(order: int, cls: type):
    from meshmode.mesh.io import ScriptSource
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if issubclass(cls, SimplexElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = 0.05;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 1;

            SetFactory("OpenCASCADE");
            Sphere(1) = {0, 0, 0, 0.5};
            """,
            "geo")
    elif issubclass(cls, TensorProductElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = 0.05;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 6;

            SetFactory("OpenCASCADE");
            Sphere(1) = {0, 0, 0, 0.5};
            Recombine Surface "*" = 0.0001;
            """,
            "geo")
    else:
        raise TypeError

    from meshmode.mesh.io import generate_gmsh
    return generate_gmsh(
            script,
            order=order,
            dimensions=2,
            force_ambient_dim=3,
            target_unit="MM",
            )

# }}}


# {{{ gmsh torus

def make_gmsh_torus(order: int, cls: type):
    from meshmode.mesh.io import ScriptSource
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if issubclass(cls, SimplexElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = 0.05;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 1;

            SetFactory("OpenCASCADE");
            Torus(1) = {0, 0, 0, 1, 0.5, 2*Pi};
            """,
            "geo")
    elif issubclass(cls, TensorProductElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = 0.05;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 7;

            SetFactory("OpenCASCADE");
            Torus(1) = {0, 0, 0, 1, 0.5, 2*Pi};
            Recombine Surface "*" = 0.0001;
            """,
            "geo")
    else:
        raise TypeError

    from meshmode.mesh.io import generate_gmsh
    return generate_gmsh(
            script,
            order=order,
            dimensions=2,
            force_ambient_dim=3,
            target_unit="MM",
            )

# }}}


# {{{ symbolic

def metric_from_form1(form1, metric_type: str):
    from pytential.symbolic.primitives import _small_sym_mat_eigenvalues
    s0, s1 = _small_sym_mat_eigenvalues(4 * form1)

    if metric_type == "singvals":
        return np.array([sym.sqrt(s0), sym.sqrt(s1)], dtype=object)
    elif metric_type == "det":
        return np.array([s0 * s1], dtype=object)
    elif metric_type == "trace":
        return np.array([s0 + s1], dtype=object)
    elif metric_type == "norm":
        return np.array([sym.sqrt(s0**2 + s1**2)], dtype=object)
    elif metric_type == "aspect":
        return np.array([
            (s0 * s1)**(2/3) / (s0**2 + s1**2),
            ], dtype=object)
    elif metric_type == "condition":
        import pymbolic.primitives as prim
        return np.array([
            prim.Max((s0, s1)) / prim.Min((s0, s1))
            ], dtype=object)
    else:
        raise ValueError(f"unknown metric type: '{metric_type}'")


def make_simplex_stretch_factors(ambient_dim: int, metric_type: str):
    from pytential.symbolic.primitives import \
            _equilateral_parametrization_derivative_matrix
    equi_pder = _equilateral_parametrization_derivative_matrix(ambient_dim)
    equi_form1 = sym.cse(equi_pder.T @ equi_pder, "pd_mat_jtj")

    return metric_from_form1(equi_form1, metric_type)


def make_quad_stretch_factors(ambient_dim: int, metric_type: str):
    pder = sym.parametrization_derivative_matrix(ambient_dim, ambient_dim - 1)
    form1 = sym.cse(pder.T @ pder, "pd_mat_jtj")

    return metric_from_form1(form1, metric_type)

# }}}


@pytest.mark.parametrize("order", [4, 8])
def test_stretch_factor(actx_factory, order,
        mesh_name="torus", metric_type="singvals",
        visualize=False):
    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if mesh_name == "torus":
        quad_mesh = make_torus_mesh(order, TensorProductElementGroup)
        simplex_mesh = make_torus_mesh(order, SimplexElementGroup)
    elif mesh_name == "twisted":
        quad_mesh = make_twisted_mesh(order, TensorProductElementGroup)
        simplex_mesh = make_twisted_mesh(order, SimplexElementGroup)
    elif mesh_name == "sphere":
        from meshmode.mesh.generation import generate_sphere
        simplex_mesh = generate_sphere(1, order=order,
                uniform_refinement_rounds=1,
                group_cls=SimplexElementGroup)
        quad_mesh = generate_sphere(1, order=order,
                uniform_refinement_rounds=1,
                group_cls=TensorProductElementGroup)
    elif mesh_name == "gmsh_sphere":
        simplex_mesh = make_gmsh_sphere(order, cls=SimplexElementGroup)
        quad_mesh = make_gmsh_sphere(order, cls=TensorProductElementGroup)
    elif mesh_name == "gmsh_torus":
        simplex_mesh = make_gmsh_torus(order, cls=SimplexElementGroup)
        quad_mesh = make_gmsh_torus(order, cls=TensorProductElementGroup)
    else:
        raise ValueError(f"unknown mesh: '{mesh_name}'")

    ambient_dim = 3
    assert simplex_mesh.ambient_dim == ambient_dim
    assert quad_mesh.ambient_dim == ambient_dim

    from meshmode.discretization import Discretization
    import meshmode.discretization.poly_element as mpoly
    simplex_discr = Discretization(actx, simplex_mesh,
            mpoly.InterpolatoryEquidistantGroupFactory(order))
    quad_discr = Discretization(actx, quad_mesh,
            mpoly.InterpolatoryEquidistantGroupFactory(order))

    print(f"simplex_discr.ndofs: {simplex_discr.ndofs}")
    print(f"quad_discr.ndofs:  {quad_discr.ndofs}")

    sym_simplex = make_simplex_stretch_factors(ambient_dim, metric_type)
    sym_quad = make_quad_stretch_factors(ambient_dim, metric_type)

    s = bind(simplex_discr, sym_simplex)(actx)
    q = bind(quad_discr, sym_quad)(actx)

    def print_bounds(x, name):
        for i, si in enumerate(x):
            print("{}{} [{:.12e}, {:.12e}]".format(
                name, i,
                actx.to_numpy(actx.np.min(si))[()],
                actx.to_numpy(actx.np.min(si))[()]
                ), end=" ")
        print()

    print_bounds(s, "s")
    print_bounds(q, "q")

    if not visualize:
        return

    suffix = f"{mesh_name}_{metric_type}_{order:02d}"

    # {{{ plot vtk

    s_pder = bind(simplex_discr, sym.parametrization_derivative_matrix(3, 2))(actx)
    q_pder = bind(quad_discr, sym.parametrization_derivative_matrix(3, 2))(actx)

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, simplex_discr, order, force_equidistant=True)
    vis.write_vtk_file(f"simplex_{suffix}.vtu",
            [(f"s{i}", si) for i, si in enumerate(s)]
            + [(f"J_{i}_{j}", pder) for (i, j), pder in np.ndenumerate(s_pder)],
            overwrite=True, use_high_order=True)

    vis = make_visualizer(actx, quad_discr, order, force_equidistant=True)
    vis.write_vtk_file(f"quad_{suffix}.vtu",
            [(f"q{i}", qi) for i, qi in enumerate(q)]
            + [(f"J_{i}_{j}", pder) for (i, j), pder in np.ndenumerate(q_pder)],
            overwrite=True, use_high_order=True)

    # }}}

    if s.size != 2:
        return

    s0, s1 = s
    q0, q1 = q

    # {{{ plot reference simplex

    if quad_discr.mesh.nelements <= 2:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 10), dpi=300)

        xi = simplex_discr.groups[0].unit_nodes
        for name, sv in zip(["s0", "s1"], [s0, s1]):
            sv = actx.to_numpy(sv[0])[0].ravel()

            ax = fig.gca()
            im = ax.tricontourf(xi[0], xi[1], sv, levels=32)
            fig.colorbar(im, ax=ax)
            fig.savefig(f"simplex_{suffix}_{name}")
            fig.clf()

    # }}}

    # {{{ plot reference quad

    if quad_discr.mesh.nelements <= 2:
        xi = quad_discr.groups[0].unit_nodes
        for name, sv in zip(["q0", "q1"], [q0, q1]):
            sv = actx.to_numpy(sv[0])[0]

            ax = fig.gca()
            im = ax.tricontourf(xi[0], xi[1], sv, levels=32)
            fig.colorbar(im, ax=ax)
            fig.savefig(f"quad_{suffix}_{name}")
            fig.clf()

    # }}}

# }}}


# You can test individual routines by typing
# $ python test_symbolic.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
