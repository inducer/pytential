from __future__ import annotations


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

import extra_matrix_data as extra
import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.mesh.generation import NArmedStarfish

from pytential import GeometryCollection
from pytential.array_context import PytestPyOpenCLArrayContextFactory
from pytential.utils import pytest_teardown_function as teardown_function  # noqa: F401


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])

CLUSTER_TEST_CASES = [
        extra.CurveTestCase(
            name="starfish",
            target_order=4,
            curve_fn=NArmedStarfish(5, 0.25),
            resolutions=[64]),
        extra.TorusTestCase(
            target_order=4,
            resolutions=[1])
        ]


# {{{ test_cluster_tree

@pytest.mark.parametrize(("case", "tree_kind"), [
    (CLUSTER_TEST_CASES[0], None),
    (CLUSTER_TEST_CASES[0], "adaptive"),
    (CLUSTER_TEST_CASES[0], "adaptive-level-restricted"),
    (CLUSTER_TEST_CASES[1], "adaptive"),
    ])
def test_cluster_tree(actx_factory, case, tree_kind, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)

    from dataclasses import replace
    actx = actx_factory()
    case = replace(case, tree_kind=tree_kind)
    logger.info("\n%s", case)

    discr = case.get_discretization(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(discr, auto_where=case.name)

    srcindex, ctree = case.get_cluster_index(actx, places)
    assert srcindex.nclusters == ctree.nclusters

    from pytential.linalg.cluster import split_array
    rng = np.random.default_rng(42)
    x = split_array(rng.random(srcindex.indices.shape), srcindex)

    logger.info("nclusters %4d nlevels %4d", srcindex.nclusters, ctree.nlevels)

    if visualize and ctree._tree is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10), dpi=300)

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(ctree._tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.draw_box_numbers()
        plotter.set_bounding_box()

        fig.savefig("test_cluster_tree")

    from pytential.linalg.cluster import cluster, uncluster
    for clevel in ctree.levels:
        logger.info("======== Level %d", clevel.level)
        logger.info("box_ids        %s", clevel.box_ids)
        logger.info("sizes          %s", np.diff(srcindex.starts))
        logger.info("parent_map     %s", clevel.parent_map)

        assert srcindex.nclusters == clevel.nclusters

        next_srcindex = cluster(srcindex, clevel)
        for i, ppm in enumerate(clevel.parent_map):
            partition = np.concatenate([srcindex.cluster_indices(j) for j in ppm])

            assert partition.size == next_srcindex.cluster_size(i)
            assert np.allclose(partition, next_srcindex.cluster_indices(i))

        y = cluster(x, clevel)
        z = uncluster(y, srcindex, clevel)
        assert all(np.allclose(xi, zi) for xi, zi in zip(x, z, strict=True))

        srcindex = next_srcindex
        x = y

# }}}


if __name__ == "__main__":
    import sys

    from pytential.array_context import _acf  # noqa: F401

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
