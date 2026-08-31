__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

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

import logging
from dataclasses import dataclass

import numpy as np

from pytools.convergence import EOCRecorder

from pytential import GeometryCollection, sym
from pytential.array_context import PyOpenCLArrayContext


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Timings:
    build: float
    matvec: float


def run_hmatrix_matvec(
        actx: PyOpenCLArrayContext,
        places: GeometryCollection, *,
        dofdesc: sym.DOFDescriptor) -> None:
    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(places.ambient_dim)
    sym_u = sym.var("u")
    sym_op = -0.5 * sym_u + sym.D(kernel, sym_u, qbx_forced_limit="avg")

    density_discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    u = actx.thaw(density_discr.nodes()[0])

    def build_hmat():
        from pytential.linalg.hmatrix import build_hmatrix_by_proxy
        return build_hmatrix_by_proxy(
            actx, places, sym_op, sym_u,
            domains=[dofdesc],
            context={},
            auto_where=dofdesc,
            id_eps=1.0e-10,
            _tree_kind="adaptive-level-restricted",
            _approx_nproxy=64,
            _proxy_radius_factor=1.15).get_forward()

    # warmup
    from pytools import ProcessTimer
    with ProcessTimer() as pt:
        hmat = build_hmat()
        actx.queue.finish()

    logger.info("build(warmup): %s", pt)

    # build
    with ProcessTimer() as pt:
        hmat = build_hmat()
        actx.queue.finish()

    t_build = pt.wall_elapsed
    logger.info("build: %s", pt)

    # matvec
    with ProcessTimer() as pt:
        du = hmat @ u
        assert du is not None
        actx.queue.finish()

    t_matvec = pt.wall_elapsed
    logger.info("matvec: %s", pt)

    return Timings(t_build, t_matvec)


def run_scaling_study(
        ambient_dim: int, *,
        target_order: int = 4,
        source_ovsmp: int = 4,
        qbx_order: int = 4,
        ) -> None:
    dd = sym.DOFDescriptor(f"d{ambient_dim}", discr_stage=sym.QBX_SOURCE_STAGE2)

    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    eoc_build = EOCRecorder()
    eoc_matvec = EOCRecorder()

    import meshmode.discretization.poly_element as mpoly
    import meshmode.mesh.generation as mgen

    resolutions = [64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072]

    for n in resolutions:
        mesh = mgen.make_curve_mesh(
            mgen.NArmedStarfish(5, 0.25),
            np.linspace(0, 1, n),
            order=target_order)

        from meshmode.discretization import Discretization
        pre_density_discr = Discretization(actx, mesh,
            mpoly.InterpolatoryQuadratureGroupFactory(target_order))

        from pytential.qbx import QBXLayerPotentialSource
        qbx = QBXLayerPotentialSource(
            pre_density_discr,
            fine_order=source_ovsmp * target_order,
            qbx_order=qbx_order,
            fmm_order=False, fmm_backend=None,
            )
        places = GeometryCollection(qbx, auto_where=dd.geometry)
        density_discr = places.get_discretization(dd.geometry, dd.discr_stage)

        logger.info("ndofs:     %d", density_discr.ndofs)
        logger.info("nelements: %d", density_discr.mesh.nelements)

        timings = run_hmatrix_matvec(actx, places, dofdesc=dd)
        eoc_build.add_data_point(density_discr.ndofs, timings.build)
        eoc_matvec.add_data_point(density_discr.ndofs, timings.matvec)

    for name, eoc in [("build", eoc_build), ("matvec", eoc_matvec)]:
        logger.info("%s\n%s",
            name, eoc.pretty_print(
                abscissa_label="dofs",
                error_label=f"{name} (s)",
                abscissa_format="%d",
                error_format="%.3fs",
                eoc_format="%.2f",
                )
            )
        visualize_eoc(f"scaling-study-hmatrix-{name}", eoc, 1)


def visualize_eoc(
        filename: str, eoc: EOCRecorder, order: int,
        overwrite: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.gca()

    h, error = np.array(eoc.history).T  # type: ignore[no-untyped-call]
    ax.loglog(h, error, "o-")

    max_h = np.max(h)
    min_e = np.min(error)
    max_e = np.max(error)
    min_h = np.exp(np.log(max_h) + np.log(min_e / max_e) / order)

    ax.loglog(
        [max_h, min_h], [max_e, min_e], "k-", label=rf"$\mathcal{{O}}(h^{order})$"
    )

    # }}}

    ax.grid(True, which="major", linestyle="-", alpha=0.75)
    ax.grid(True, which="minor", linestyle="--", alpha=0.5)

    ax.set_xlabel("$N$")
    ax.set_ylabel("$T~(s)$")

    import pathlib
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    run_scaling_study(ambient_dim=2)
