from __future__ import annotations


__copyright__ = "Copyright (C) 2026 Shawn/Chaoqi Lin"

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

from typing import TYPE_CHECKING

import numpy as np

from arraycontext import flatten, pytest_generate_tests_for_array_contexts
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import InterpolatoryQuadratureGroupFactory
from meshmode.mesh import TensorProductElementGroup
from meshmode.mesh.generation import generate_sphere
from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.kernel import DirectionalSourceDerivative, LaplaceKernel
from sumpy.qbx import LayerPotential

from pytential import GeometryCollection, bind, sym
from pytential.array_context import PytestPyOpenCLArrayContextFactory
from pytential.qbx import QBXLayerPotentialSource


if TYPE_CHECKING:
    from arraycontext.context import ArrayContextFactory


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
])


def test_no_aliasing_in_kernel_arguments(actx_factory: ArrayContextFactory):
    """Disagreement between sumpy and pytential would previously arise because
    pytential would upsample an already-computed normal, as opposed to recomputing
    the normal on the upsampled grid. Doing the former leads to avoidable
    aliasing error that would limit pytential to attaining about 5 digits
    in this specific test.
    """
    actx = actx_factory()

    order = 4
    qbx_order = 4
    level = 3
    ambient_dim = 3
    base_knl = LaplaceKernel(3)
    dlp_knl = DirectionalSourceDerivative(base_knl, dir_vec_name="dsource_vec")

    mesh = generate_sphere(1.0, order, uniform_refinement_rounds=level,
                           group_cls=TensorProductElementGroup)
    pre_density_discr = Discretization(
        actx, mesh, InterpolatoryQuadratureGroupFactory(order))

    fine_orders = [order, order*2, order*4]
    rows = []

    for fine_order in fine_orders:
        qbx = QBXLayerPotentialSource(
            pre_density_discr, fine_order=fine_order,
            qbx_order=qbx_order, fmm_order=False)
        places = GeometryCollection({"qbx": qbx}, auto_where="qbx")

        target_discr = places.get_discretization("qbx", sym.QBX_SOURCE_STAGE1)
        source_discr = places.get_discretization(
            "qbx", sym.QBX_SOURCE_QUAD_STAGE2)
        source_dd = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_QUAD_STAGE2)

        # --- pytential sym.D ---
        sigma = target_discr.zeros(actx) + 1
        bound_op = bind(places, sym.D(base_knl, sym.var("sigma"),
                          qbx_forced_limit=-1))
        result_pyt = bound_op(actx, sigma=sigma)
        err_pyt = float(np.max(np.abs(
            actx.to_numpy(result_pyt[0]).ravel() - (-1.0))))

        # --- sumpy direct ---
        expn = LineTaylorLocalExpansion(base_knl, qbx_order)
        lpot = LayerPotential(
            expansion=expn,
            source_kernels=(dlp_knl,),
            target_kernels=(dlp_knl,),
        )

        targets = actx.thaw(target_discr.nodes())
        sources = actx.thaw(source_discr.nodes())
        normals_src = bind(places, sym.normal(
            ambient_dim, dofdesc=source_dd))(actx).as_vector(object)
        waa = bind(places, sym.weights_and_area_elements(
            ambient_dim=ambient_dim, dim=ambient_dim - 1,
            dofdesc=source_dd))(actx)
        expansion_radii = bind(
            places, sym.expansion_radii(ambient_dim))(actx)
        centers_in = bind(
            places, sym.expansion_centers(ambient_dim, -1))(actx)

        targets_h = actx.to_numpy(
            flatten(targets, actx)).reshape(ambient_dim, -1)
        sources_h = actx.to_numpy(
            flatten(sources, actx)).reshape(ambient_dim, -1)
        centers_h = actx.to_numpy(
            flatten(centers_in, actx)).reshape(ambient_dim, -1)
        radii_h = actx.to_numpy(flatten(expansion_radii, actx))
        waa_h = actx.to_numpy(flatten(waa, actx))
        normals_h = actx.to_numpy(
            flatten(normals_src, actx)).reshape(ambient_dim, -1)

        result_sumpy = lpot(actx,
            targets=actx.from_numpy(targets_h),
            sources=actx.from_numpy(sources_h),
            centers=actx.from_numpy(centers_h),
            strengths=(actx.from_numpy(waa_h),),
            expansion_radii=actx.from_numpy(radii_h),
            dsource_vec=actx.from_numpy(normals_h),
        )
        err_sumpy = float(np.max(np.abs(
            actx.to_numpy(result_sumpy[0]).ravel() - (-1.0))))

        pyt_flat = actx.to_numpy(result_pyt[0]).ravel()
        sumpy_flat = actx.to_numpy(result_sumpy[0]).ravel()
        diff = float(np.max(np.abs(pyt_flat - sumpy_flat)))

        rows.append((fine_order, err_sumpy, err_pyt, diff))

        if diff >= 1e-14:
            header = (f"  {'fine_order':>10s}  {'err sumpy':>12s}  "
                      f"{'err pytential':>14s}  {'|sumpy-pyt|':>12s}")
            lines = [header, f"  {'-'*54}"]
            for fine_order, err_sumpy, err_pytential, diff in rows:
                lines.append(
                    f"  {fine_order:10d}  {err_sumpy:12.2e}  "
                    f"{err_pytential:14.2e}  {diff:12.2e}")
            table = "\n".join(lines)
            raise AssertionError(
                f"DLP results disagree at fine_order={fine_order}\n{table}")


if __name__ == "__main__":
    import sys

    from pytential.array_context import _acf  # noqa: F401

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
