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

from arraycontext import flatten, unflatten
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

from pytential.utils import (  # noqa: F401
        pytest_teardown_function as teardown_function)

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

    nodes = actx.thaw(discr.nodes())

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

    nodes = actx.thaw(discr.nodes())

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
    logger.info("eoc:\n%s", eoc)

    order = min(g.order for g in discr.groups)
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
            actx.thaw(source_discr.nodes()[0]),
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

    # create a shuffled [1, ndofs + 1] array
    rng = np.random.default_rng(seed=42)

    def randrange_like(xi, offset):
        x = offset + np.arange(1, xi.size + 1)
        rng.shuffle(x)

        return actx.from_numpy(x.reshape(xi.shape))

    from meshmode.dof_array import DOFArray
    base_node_nrs = np.cumsum([0] + [grp.ndofs for grp in discr.groups])
    ary = DOFArray(actx, tuple([
        randrange_like(xi, offset)
        for xi, offset in zip(discr.nodes()[0], base_node_nrs)
        ]))

    n = discr.ndofs
    for func, expected in [
            (sym.NodeSum, n * (n + 1) // 2),
            (sym.NodeMax, n),
            (sym.NodeMin, 1),
            ]:
        r = actx.to_numpy(
            bind(discr, func(sym.var("x")))(actx, x=ary)
            )
        assert r == expected, (r, expected)

    arys = tuple([rng.random(size=xi.shape) for xi in ary])
    x = DOFArray(actx, tuple([actx.from_numpy(xi) for xi in arys]))

    from meshmode.dof_array import flat_norm
    for func, np_func in [
            (sym.ElementwiseSum, np.sum),
            (sym.ElementwiseMax, np.max)
            ]:
        expected = DOFArray(actx, tuple([
            actx.from_numpy(np.tile(np_func(xi, axis=1, keepdims=True), xi.shape[1]))
            for xi in arys
            ]))
        r = bind(
            discr, func(sym.var("x"), dofdesc=sym.GRANULARITY_NODE)
            )(actx, x=x)
        assert actx.to_numpy(flat_norm(r - expected)) < 1.0e-15

        expected = DOFArray(actx, tuple([
            actx.from_numpy(np_func(xi, axis=1, keepdims=True))
            for xi in arys
            ]))
        r = bind(
            discr, func(sym.var("x"), dofdesc=sym.GRANULARITY_ELEMENT)
            )(actx, x=x)
        assert actx.to_numpy(flat_norm(r - expected)) < 1.0e-15

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


# {{{ test_mapper_kernel_transformation_remover

def _make_operator(ambient_dim: int, op_name: str, k: float, *, side: int = +1):
    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    if k == 0:
        kernel = LaplaceKernel(ambient_dim)
        kernel_arguments = {}
    else:
        kernel = HelmholtzKernel(ambient_dim)
        kernel_arguments = {"k": sym.var("k")}

    import pytential.symbolic.pde.scalar as ops
    if op_name == "dirichlet":
        op: ops.L2WeightedPDEOperator = ops.DirichletOperator(
                kernel, side, use_l2_weighting=True,
                kernel_arguments=kernel_arguments)
    elif op_name == "neumann":
        op = ops.NeumannOperator(
                kernel, side, use_l2_weighting=True,
                kernel_arguments=kernel_arguments)
    else:
        raise ValueError(f"unknown operator: '{op_name}'")

    return op


@pytest.mark.parametrize("op_name", ["dirichlet", "neumann"])
@pytest.mark.parametrize("k", [0, 5])
def test_mapper_kernel_transformation_remover(op_name, k):
    ambient_dim = 3
    op = _make_operator(ambient_dim, op_name, k)
    expr = op.operator(op.get_density_var("sigma"))

    from pytential.linalg.direct_solver_symbolic import (
            OperatorCollector, KernelTransformationRemover)
    expr_without_transformations = KernelTransformationRemover()(expr)
    intgs = OperatorCollector()(expr_without_transformations)

    def is_base_kernel(knl):
        return knl.get_base_kernel() == knl

    for intg in intgs:
        assert is_base_kernel(intg.target_kernel)
        assert all(is_base_kernel(kernel) for kernel in intg.source_kernels)

# }}}


# {{{ test_mapper_int_g_term_collector

@pytest.mark.parametrize("op_name", ["dirichlet", "neumann"])
def test_mapper_int_g_term_collector(op_name, k=0):
    ambient_dim = 3
    op = _make_operator(ambient_dim, op_name, k)
    expr = op.operator(op.get_density_var("sigma"))

    from pytential.linalg.direct_solver_symbolic import IntGTermCollector
    expr_only_intgs = IntGTermCollector()(expr)

    # FIXME: how to check this did something?
    sigma = sym.cse(op.get_density_var("sigma") / op.get_sqrt_weight())
    if op_name == "dirichlet":
        expected_expr = -1 * sym.D(op.kernel, sigma)
    elif op_name == "neumann":
        int_g = sym.S(op.kernel, sigma, qbx_forced_limit="avg")
        expected_expr = sym.div([int_g] * ambient_dim)
    else:
        raise ValueError(f"unknown operator name: {op_name}")

    assert expr_only_intgs == expected_expr

# }}}


# {{{ test_mapper_dof_descriptor_replacer

@pytest.mark.parametrize("op_name", ["dirichlet", "neumann"])
@pytest.mark.parametrize("k", [0, 5])
def test_mapper_dof_descriptor_replacer(op_name, k):
    ambient_dim = 3
    op = _make_operator(ambient_dim, op_name, k)
    expr = op.operator(op.get_density_var("sigma"))

    from pytential.symbolic.mappers import ToTargetTagger
    from pytential.linalg.direct_solver_symbolic import DOFDescriptorReplacer
    source_dd = sym.as_dofdesc(sym.DEFAULT_SOURCE)
    target_dd = sym.as_dofdesc(sym.DEFAULT_TARGET)
    tagged_expr = ToTargetTagger(source_dd, target_dd)(expr)

    source_new_dd = sym.as_dofdesc("source")
    target_new_dd = sym.as_dofdesc("target")
    replaced_expr = DOFDescriptorReplacer(source_new_dd, target_new_dd)(tagged_expr)

    from testlib import DOFDescriptorCollector
    collector = DOFDescriptorCollector()
    assert collector(tagged_expr) == {source_dd, target_dd}
    assert collector(replaced_expr) == {source_new_dd, target_new_dd}

# }}}


# {{{ test_derivative_with_spatial_constant

def test_derivative_with_spatial_constant():
    ambient_dim = 3

    from sumpy.kernel import LaplaceKernel
    knl = LaplaceKernel(ambient_dim)
    density = sym.var("sigma")

    sym.d_dx(ambient_dim,
            sym.SpatialConstant("kappa")
            * sym.D(knl, density, qbx_forced_limit="avg"))

    from sumpy.kernel import LaplaceKernel
    sym.d_dx(ambient_dim,
            (3+sym.SpatialConstant("kappa"))
            * sym.D(knl, density, qbx_forced_limit="avg"))

    from pytential.symbolic.mappers import _DerivativeTakerUnsupoortedProductError
    with pytest.raises(_DerivativeTakerUnsupoortedProductError):
        sym.d_dx(ambient_dim,
                (3+sym.Variable("kappa"))
                * sym.D(knl, density, qbx_forced_limit="avg"))

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
