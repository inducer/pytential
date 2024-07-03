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

import pytest
from functools import partial

import numpy as np
import numpy.linalg as la

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


# {{{ test_gmres

def test_gmres():
    rng = np.random.default_rng(seed=42)

    n = 200
    A = (  # noqa
            n * (np.eye(n) + 2j * np.eye(n))
            + rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))

    true_sol = rng.normal(size=n) + 1j * rng.normal(size=n)
    b = np.dot(A, true_sol)

    A_func = lambda x: np.dot(A, x)  # noqa
    A_func.shape = A.shape
    A_func.dtype = A.dtype

    from pytential.linalg.gmres import gmres, ResidualPrinter
    tol = 1e-6
    sol = gmres(A_func, b, callback=ResidualPrinter(),
            maxiter=5*n, tol=tol).solution

    assert la.norm(true_sol - sol) / la.norm(sol) < tol

# }}}


# {{{ test_interpolatory_error_reporting

def test_interpolatory_error_reporting(actx_factory):
    actx = actx_factory()

    h = 0.2
    from meshmode.mesh.io import generate_gmsh, FileSource
    mesh = generate_gmsh(
            FileSource("circle.step"), 2, order=4, force_ambient_dim=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h],
            target_unit="mm",
            )

    logger.info("%d elements" % mesh.nelements)

    # {{{ discretizations and connections

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory

    vol_discr = Discretization(actx, mesh,
            QuadratureSimplexGroupFactory(5))

    vol_x = actx.thaw(vol_discr.nodes())

    # }}}

    from pytential import integral
    one = 1 + 0*vol_x[0]
    from meshmode.discretization import NoninterpolatoryElementGroupError
    with pytest.raises(NoninterpolatoryElementGroupError):
        logger.info("AREA integral %g exact %g",
                    actx.to_numpy(integral(vol_discr, one)).item(),
                    0.25**2*np.pi)

# }}}


# {{{ test_geometry_collection_caching

def test_geometry_collection_caching(actx_factory):
    # NOTE: checks that the on-demand caching works properly in
    # the `GeometryCollection`. This is done by constructing a few separated
    # spheres, putting a few `QBXLayerPotentialSource`s on them and requesting
    # the `nodes` on each `discr_stage`.
    actx = actx_factory()

    ndim = 2
    nelements = 1024
    target_order = 7
    qbx_order = 4
    ngeometry = 3

    # construct discretizations
    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from meshmode.mesh.processing import affine_map
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discrs = []
    radius = 1.0
    for k in range(ngeometry):
        if k == 0:
            mesh = make_curve_mesh(partial(ellipse, radius),
                    np.linspace(0.0, 1.0, nelements + 1),
                    target_order)
        else:
            mesh = affine_map(discrs[0].mesh,
                    b=np.array([3 * k * radius, 0]))

        discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))
        discrs.append(discr)

    # construct qbx source
    from pytential.qbx import QBXLayerPotentialSource

    lpots = [
        QBXLayerPotentialSource(density_discr,
            fine_order=2 * target_order,
            qbx_order=qbx_order,
            fmm_order=False)
        for density_discr in discrs
        ]
    sources = [f"source_{k}" for k in range(ngeometry)]

    # construct a geometry collection
    from pytential import GeometryCollection
    places = GeometryCollection(dict(zip(sources, lpots)), auto_where=sources[0])
    logger.info("%s", places.places)

    # check on-demand refinement
    from pytential import bind, sym
    discr_stages = [sym.QBX_SOURCE_STAGE1,
            sym.QBX_SOURCE_STAGE2,
            sym.QBX_SOURCE_QUAD_STAGE2]

    for k in range(ngeometry):
        for discr_stage in discr_stages:
            with pytest.raises(KeyError):
                discr = places._get_discr_from_cache(sources[k], discr_stage)

            dofdesc = sym.DOFDescriptor(sources[k], discr_stage=discr_stage)
            bind(places, sym.nodes(ndim, dofdesc=dofdesc))(actx)

            discr = places._get_discr_from_cache(sources[k], discr_stage)
            assert discr is not None

# }}}


# {{{ test_geometry_collection_merge

def _check_cache_state(places, include_cs_prefixes, exclude_cse_prefixes):
    dd = places.auto_source

    # check that the refined geometries are still here
    from pytential import sym
    try:
        places._get_discr_from_cache(
                dd.geometry, sym.QBX_SOURCE_QUAD_STAGE2)
    except KeyError:
        return False

    # check that the connections are still here too
    try:
        places._get_conn_from_cache(
                dd.geometry, sym.QBX_SOURCE_STAGE2, sym.QBX_SOURCE_QUAD_STAGE2)
    except KeyError:
        return False

    # check that the normal is in there
    from pytential.symbolic.execution import EvaluationMapperCSECacheKey
    cache = places._get_cache(EvaluationMapperCSECacheKey)
    if not any(prefix.startswith("normal_") for (prefix, _) in cache):
        return False

    # check that any additional data is in there
    for cse_prefix in include_cs_prefixes:
        if not any(prefix.startswith(cse_prefix) for (prefix, _) in cache):
            return False

    for cse_prefix in exclude_cse_prefixes:
        if any(prefix.startswith(cse_prefix) for (prefix, _) in cache):
            return False

    return True


def _add_geometry_to_collection(actx, places, geometry, dofdesc=None):
    if dofdesc is None:
        dofdesc = places.auto_source
    ambient_dim = places.ambient_dim

    from pytential.collection import add_geometry_to_collection
    new_places = add_geometry_to_collection(places, {"geometry": geometry})

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(ambient_dim)

    from pytential import bind, sym
    sym_density = sym.nodes(ambient_dim).as_vector()[0]
    sym_op = sym.D(kernel, sym_density, qbx_forced_limit=None)

    extra_cse_prefixes = ("expansion_radii", "weights_area_elements")
    assert _check_cache_state(new_places, (), extra_cse_prefixes)

    r = bind(new_places, sym_op,
            auto_where=(dofdesc, "geometry"))(actx)
    assert r is not None

    assert _check_cache_state(new_places, extra_cse_prefixes, ())


def test_add_geometry_to_collection(actx_factory):
    """
    Test case of `add_geometry_to_collection`. Verifies that
    * cse_scope.DISCRETIZATION caches stick around
    * refinement / connection caches stick around
    * caches added to the new collection don't polute the original one
    """

    actx = actx_factory()

    from extra_int_eq_data import StarfishTestCase, make_source_and_target_points
    case = StarfishTestCase()

    from pytential import GeometryCollection
    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    from pytential import bind, sym
    from pytential.qbx.refinement import refine_geometry_collection
    places = refine_geometry_collection(
            places,
            refine_discr_stage=sym.QBX_SOURCE_QUAD_STAGE2,
            )

    sources, targets = make_source_and_target_points(
            actx, case.side, case.inner_radius, case.outer_radius,
            places.ambient_dim,
            nsources=64, ntargets=64)

    # compute the normal so that it gets cached in the original collection
    normal = bind(places, sym.normal(places.ambient_dim).as_vector())(actx)
    assert normal is not None

    # add some more geometries and see if the normal gets recomputed
    _add_geometry_to_collection(actx, places, sources)
    _add_geometry_to_collection(actx, places, targets)

# }}}


# You can test individual routines by typing
# $ python test_tools.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
