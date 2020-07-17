from __future__ import division

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

from functools import partial

from meshmode.array_context import PyOpenCLArrayContext
import pytest

import numpy as np
import numpy.linalg as la
import pyopencl as cl

import logging
logger = logging.getLogger(__name__)

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def test_gmres():
    n = 200
    A = (  # noqa
            n * (np.eye(n) + 2j * np.eye(n))
            + np.random.randn(n, n) + 1j * np.random.randn(n, n))

    true_sol = np.random.randn(n) + 1j * np.random.randn(n)
    b = np.dot(A, true_sol)

    A_func = lambda x: np.dot(A, x)  # noqa
    A_func.shape = A.shape
    A_func.dtype = A.dtype

    from pytential.solve import gmres, ResidualPrinter
    tol = 1e-6
    sol = gmres(A_func, b, callback=ResidualPrinter(),
            maxiter=5*n, tol=tol).solution

    assert la.norm(true_sol - sol) / la.norm(sol) < tol


def test_interpolatory_error_reporting(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

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

    from meshmode.dof_array import thaw
    vol_x = thaw(actx, vol_discr.nodes())

    # }}}

    from pytential import integral
    one = 1 + 0*vol_x[0]
    from meshmode.discretization import NoninterpolatoryElementGroupError
    with pytest.raises(NoninterpolatoryElementGroupError):
        print("AREA", integral(vol_discr, one), 0.25**2*np.pi)


def test_geometry_collection_caching(ctx_factory):
    # NOTE: checks that the on-demand caching works properly in
    # the `GeometryCollection`. This is done by constructing a few separated
    # spheres, putting a few `QBXLayerPotentialSource`s on them and requesting
    # the `nodes` on each `discr_stage`.
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

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

    lpots = []
    sources = ["source_{}".format(k) for k in range(ngeometry)]
    for k, density_discr in enumerate(discrs):
        qbx = QBXLayerPotentialSource(density_discr,
            fine_order=2 * target_order,
            qbx_order=qbx_order,
            fmm_order=False)
        lpots.append(qbx)

    # construct a geometry collection
    from pytential import GeometryCollection
    places = GeometryCollection(dict(zip(sources, lpots)))
    print(places.places)

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
