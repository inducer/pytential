from __future__ import division, absolute_import, print_function

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


import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pytools import RecordWithoutPickling
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut,
        make_curve_mesh)

import logging
logger = logging.getLogger(__name__)

__all__ = ["pytest_generate_tests"]


class ElementInfo(RecordWithoutPickling):
    """
    .. attribute:: element_nr
    .. attribute:: neighbors
    .. attribute:: discr_slice
    .. attribute:: mesh_slice
    .. attribute:: element_group
    .. attribute:: mesh_element_group
    """
    __slots__ = ["element_nr",
                 "neighbors",
                 "discr_slice"]


def iter_elements(discr):
    discr_nodes_idx = 0
    element_nr = 0
    adjacency = discr.mesh.nodal_adjacency

    for discr_group in discr.groups:
        start = element_nr
        for element_nr in range(start, start + discr_group.nelements):
            yield ElementInfo(
                element_nr=element_nr,
                neighbors=list(adjacency.neighbors[
                    slice(*adjacency.neighbors_starts[
                        element_nr:element_nr+2])]),
                discr_slice=slice(discr_nodes_idx,
                   discr_nodes_idx + discr_group.nunit_nodes))

            discr_nodes_idx += discr_group.nunit_nodes


from extra_curve_data import horseshoe


@pytest.mark.parametrize(("curve_name", "curve_f", "nelements"), [
    ("20-to-1 ellipse", partial(ellipse, 20), 100),
    ("horseshoe", horseshoe, 50),
    ])
def test_global_lpot_source_refinement(ctx_getter, curve_name, curve_f, nelements):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    order = 16
    helmholtz_k = 10

    mesh = make_curve_mesh(curve_f, np.linspace(0, 1, nelements+1), order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    factory = InterpolatoryQuadratureSimplexGroupFactory(order)

    discr = Discretization(cl_ctx, mesh, factory)

    from pytential.qbx.refinement import (
        NewQBXLayerPotentialSource, QBXLayerPotentialSourceRefiner)

    lpot_source = NewQBXLayerPotentialSource(discr, order)
    del discr
    refiner = QBXLayerPotentialSourceRefiner(cl_ctx)

    lpot_source, conn = refiner(lpot_source, factory, helmholtz_k)

    discr_nodes = lpot_source.density_discr.nodes().get(queue)
    int_centers = lpot_source.centers(-1)
    int_centers = np.array([axis.get(queue) for axis in int_centers])
    ext_centers = lpot_source.centers(+1)
    ext_centers = np.array([axis.get(queue) for axis in ext_centers])
    panel_sizes = lpot_source.panel_sizes("nelements").get(queue)

    def check_panel(panel):
        # Check 2-to-1 panel to neighbor size ratio.
        for neighbor in panel.neighbors:
            assert panel_sizes[panel.element_nr] / panel_sizes[neighbor] <= 2, \
                (panel_sizes[panel.element_nr], panel_sizes[neighbor])

        # Check wavenumber to panel size ratio.
        assert panel_sizes[panel.element_nr] * helmholtz_k <= 5

    def check_panel_pair(panel_1, panel_2):
        h_1 = panel_sizes[panel_1.element_nr]
        h_2 = panel_sizes[panel_2.element_nr]

        if panel_1.element_nr == panel_2.element_nr:
            # Same panel
            return

        panel_1_centers = int_centers[:, panel_1.discr_slice]
        panel_2_nodes = discr_nodes[:, panel_2.discr_slice]

        # =distance(centers of panel 1, panel 2)
        dist = (
            la.norm((
                    panel_1_centers[..., np.newaxis] -
                    panel_2_nodes[:, np.newaxis, ...]).T,
                axis=-1)
            .min())

        # Criterion 1:
        # A center cannot be closer to another panel than to its originating
        # panel.

        assert dist >= h_1 / 2, (dist, h_1, panel_1.element_nr, panel_2.element_nr)

        # Criterion 2:
        # A center cannot be closer to another panel than that panel's
        # centers - unless the panels are adjacent, to allow for refinement.

        if panel_2.element_nr in panel_1.neighbors:
            return

        assert dist >= h_2 / 2, (dist, h_2, panel_1.element_nr, panel_2.element_nr)

    for panel_1 in iter_elements(lpot_source.density_discr):
        check_panel(panel_1)
        for panel_2 in iter_elements(lpot_source.density_discr):
            check_panel_pair(panel_1, panel_2)


# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
