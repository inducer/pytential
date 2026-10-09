__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
Copyright (C) 2023 Drew Anderson
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


from typing import Tuple

import numpy as np
import pyopencl as cl
from pytools.obj_array import make_obj_array  # noqa: F401
from pytential import bind, sym, GeometryCollection
from sumpy.kernel import HelmholtzKernel, AxisSourceDerivative

from meshmode import _acf  # noqa: F401
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory,])


def dyadic_layer_pot(
    helm_k, vec, *, transpose: bool, source_curl: bool = False,
    **kwargs
) -> np.ndarray:
    k_sym = HelmholtzKernel(3)

    if transpose:
        outer = lambda i, j: j  # noqa: E731
        inner = lambda i, j: i  # noqa: E731
    else:
        outer = lambda i, j: i  # noqa: E731
        inner = lambda i, j: j  # noqa: E731

    def helmholtz_vec(source_derivative_axes: Tuple[int, ...]):
        knl = k_sym
        for axis in source_derivative_axes:
            knl = AxisSourceDerivative(axis, knl)

        return sym.int_g_vec(knl, vec, kernel_arguments={"k": helm_k}, **kwargs)

    from pytools import levi_civita

    # FIXME Could/should use Schwarz's theorem to optimize, but probably
    # only off-surface where potential is smooth.
    if source_curl:
        return make_obj_array(
            [
                sum(
                    # levi_civita((ell, m, n))
                    levi_civita((ell, outer(m, n), inner(m, n)))
                    * (
                        helmholtz_vec((m,))[n]
                        + 1 / helm_k**2
                        * sum(
                            helmholtz_vec((m, outer(n, j), inner(n, j)))[j]
                            for j in range(3)
                        )
                    )
                    for m in range(3)
                    for n in range(3)
                )
                for ell in range(3)
            ]
        )
    else:
        return make_obj_array(
            [
                helmholtz_vec(())[i]
                + 1 / helm_k**2
                * sum(helmholtz_vec((outer(i, j), inner(i, j)))[j] for j in range(3))
                for i in range(3)
            ]
        )


def dyadicgreensfunction(targets, sources, strengths, k, cl_ctx, queue, actx):
    from sumpy.p2p import P2P
    from sumpy.kernel import HelmholtzKernel, AxisTargetDerivative

    dimension = len(sources)
    if not isinstance(targets, np.ndarray):
        try:
            newtargets = actx.to_numpy(targets)
            targets = newtargets
        except Exception:
            raise Exception(
                "Invalid array type for targets. Use an ndarray or cltaggable array"
            )

    hknl = HelmholtzKernel(dim=dimension, allow_evanescent=True)
    p2p = P2P(
        cl_ctx,
        [hknl]
        + [AxisTargetDerivative(i, hknl) for i in range(0, dimension)]
        + [
            AxisTargetDerivative(j, AxisTargetDerivative(jj, hknl))
            for j in range(0, dimension)
            for jj in range(j, dimension)
        ],
        exclude_self=False,
        value_dtypes=np.complex128,
    )

    evt, garrs = p2p(queue, targets, sources, strengths, k=complex(k))
    dim1 = len(garrs[0])

    dgfarray = np.zeros((dim1, 3, 3), dtype=np.complex128)
    curldgfarray = np.zeros((dim1, 3, 3), dtype=np.complex128)

    for m in range(0, 3):
        dgfarray[:, m, m] = garrs[0]
        for n in range(0, 3):
            dgfarray[:, m, n] += garrs[m + n + 4 + ((m > 0) and (n > 0))] / k**2

    constantarray = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    for w in range(0, 3):
        for x in range(0, 3):
            curldgfarray[:, w, x] = constantarray[w][x] * garrs[4 - (w + x)]

    return [dgfarray, curldgfarray]


def test_dyadic_green(actx_factory):
    import logging

    logging.basicConfig(level=logging.INFO)  # INFO for more progress info

    actx = actx_factory()

    # helm_k = sym.var("k")
    # trace_e = sym.make_sym_vector("trace_e", 3)
    # trace_curl_e = sym.make_sym_vector("trace_curl_e", 3)

    # normal = sym.normal(3).as_vector()

    # nxe = sym.cross(normal, trace_e)
    # nxcurl_e = sym.cross(normal, trace_curl_e)

    # # Monk, Finite element methods for Maxwell's equations (2003)
    # # https://doi.org/10.1093/acprof:oso/9780198508885.001.0001
    # # Theorem 12.2.
    # zero_op = (
    #     dyadic_layer_pot(helm_k, nxe, transpose=True)
    #     + dyadic_layer_pot(helm_k, nxcurl_e, transpose=True, source_curl=True)
    #     - trace_e
    # )

    from meshmode.mesh.generation import generate_sphere
    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )
    from meshmode.dof_array import DOFArray
    from pytential.target import PointsTarget

    target_order = 5
    k = 1.0

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for res in [0, 1, 2, 3]:
        mesh = generate_sphere(
                1, target_order, uniform_refinement_rounds=res)

        pre_density_discr = Discretization(
            actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

        qbx = QBXLayerPotentialSource(
            pre_density_discr,
            fine_order=target_order,
            qbx_order=1,
            fmm_order=False,
            target_association_tolerance=0.005,
            # fmm_backend="fmmlib",
        )

        # create coordinates for layer potential targets outside sigma

        outer_mesh = generate_sphere(2, 1)
        coords_outside_sigma = outer_mesh.groups[0].nodes

        # take x, y, and z coordinate arrays and stick them together, changing
        # array type as needed
        x_coords_outside_sigma = actx.from_numpy(coords_outside_sigma[0])
        y_coords_outside_sigma = actx.from_numpy(coords_outside_sigma[1])
        z_coords_outside_sigma = actx.from_numpy(coords_outside_sigma[2])
        targets_outside_sigma = cl.array.stack(
            [
                x_coords_outside_sigma.ravel(),
                y_coords_outside_sigma.ravel(),
                z_coords_outside_sigma.ravel(),
            ]
        )

        places = GeometryCollection(
            {
                "qbx": qbx,
                "targets": PointsTarget(targets_outside_sigma),
            },
            auto_where="qbx",
        )

        # create target arrays on our curve sigma
        density_discr = places.get_discretization("qbx")
        nodes = actx.thaw(density_discr.nodes())

        (x_coords_on_sigma,) = nodes[0]
        (y_coords_on_sigma,) = nodes[1]
        (z_coords_on_sigma,) = nodes[2]
        targets_on_sigma = cl.array.stack(
            [
                x_coords_on_sigma.ravel(),
                y_coords_on_sigma.ravel(),
                z_coords_on_sigma.ravel(),
            ]
        )

        # set up and run dgf to generate greens function at desired points
        # (targets)

        sources = np.array([[0.0], [0.0], [0.0]])
        strengths = np.array([[1.0]])

        G_arr_outside_sigma, G_curl_arr_outside_sigma = dyadicgreensfunction(
            targets_outside_sigma, sources, strengths, k, actx.context, actx.queue,
            actx)
        G_arr_on_sigma, G_curl_arr_on_sigma = dyadicgreensfunction(
            targets_on_sigma, sources, strengths, k, actx.context, actx.queue, actx
        )

        # need to change shape of data to set up for layer potentials. Note arrays
        # are not contiguous at first
        in_vec_d_data = G_arr_on_sigma[:, :, 0]
        in_vec_d_row_0 = np.ascontiguousarray(in_vec_d_data[:, 0]).reshape(
            x_coords_on_sigma.shape
        )
        in_vec_d_row_1 = np.ascontiguousarray(in_vec_d_data[:, 1]).reshape(
            x_coords_on_sigma.shape
        )
        in_vec_d_row_2 = np.ascontiguousarray(in_vec_d_data[:, 2]).reshape(
            x_coords_on_sigma.shape
        )
        in_vec_d = make_obj_array(
            [
                DOFArray(actx, (actx.from_numpy(in_vec_d_row_0),)),
                DOFArray(actx, (actx.from_numpy(in_vec_d_row_1),)),
                DOFArray(actx, (actx.from_numpy(in_vec_d_row_2),)),
            ]
        )

        in_vec_s_data = G_curl_arr_on_sigma[:, :, 0]
        in_vec_s_row_0 = np.ascontiguousarray(in_vec_s_data[:, 0]).reshape(
            x_coords_on_sigma.shape
        )
        in_vec_s_row_1 = np.ascontiguousarray(in_vec_s_data[:, 1]).reshape(
            x_coords_on_sigma.shape
        )
        in_vec_s_row_2 = np.ascontiguousarray(in_vec_s_data[:, 2]).reshape(
            x_coords_on_sigma.shape
        )
        in_vec_s = make_obj_array(
            [
                DOFArray(actx, (actx.from_numpy(in_vec_s_row_0),)),
                DOFArray(actx, (actx.from_numpy(in_vec_s_row_1),)),
                DOFArray(actx, (actx.from_numpy(in_vec_s_row_2),)),
            ]
        )

        # function to define what layer potentials are being evaluated

        def op(**kwargs):
            in_vec_d = sym.n_cross(sym.make_sym_vector("in_vec_d", 3))
            in_vec_s = sym.n_cross(sym.make_sym_vector("in_vec_s", 3))
            return dyadic_layer_pot(
                sym.SpatialConstant("k"),
                in_vec_d,
                transpose=True,
                source_curl=True,
                **kwargs
            ) + dyadic_layer_pot(
                sym.SpatialConstant("k"),
                in_vec_s,
                transpose=True,
                source_curl=False,
                **kwargs
            )

        # run layer potentials
        bound_layer_pot_val = bind(
            places, op(source="qbx", target="targets", qbx_forced_limit=None)
        )
        layer_pot_val = bound_layer_pot_val(
                actx, in_vec_d=in_vec_d, in_vec_s=in_vec_s, k=k)

        # get norm

        # rearrange the dyadic greens function outside sigma to easily compare
        true_soln = G_arr_outside_sigma[:, :, 0]
        true_soln_row_0 = true_soln[:, 0]
        true_soln_row_1 = true_soln[:, 1]
        true_soln_row_2 = true_soln[:, 2]

        # then calculate max norm over everything
        norm_row_0 = np.linalg.norm(
            actx.to_numpy(layer_pot_val[0]) - true_soln_row_0, np.inf
        )
        norm_row_1 = np.linalg.norm(
            actx.to_numpy(layer_pot_val[1]) - true_soln_row_1, np.inf
        )
        norm_row_2 = np.linalg.norm(
            actx.to_numpy(layer_pot_val[2]) - true_soln_row_2, np.inf
        )

        err = max(norm_row_0, norm_row_1, norm_row_2)
        eoc_rec.add_data_point(2**-res, err)

    print(eoc_rec)

    assert eoc_rec.order_estimate() > target_order + 1


# You can test individual routines by typing
# $ python test_maxwell_green.py 'test_routine()'

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])

# vim: fdm=marker
