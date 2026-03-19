from __future__ import annotations


__copyright__ = "Copyright (C) 2018 Andreas Kloeckner"

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
from typing import TYPE_CHECKING

import numpy as np

from pytools import log_process


if TYPE_CHECKING:
    from collections.abc import Callable

    from meshmode.mesh import Mesh, MeshElementGroup
    from meshmode.mesh.refinement import Refiner


logger = logging.getLogger(__name__)


__doc__ = """
Tools for Iterative Refinement
-------------------------------

.. autofunction:: warp_and_refine_until_resolved
.. autofunction:: refine_mesh_and_get_urchin_warper
.. autofunction:: generate_urchin
"""


# {{{ warp_and_refine_until_resolved

def _mode_order(mode_id: tuple[int, ...], is_tensor_product: bool) -> int:
    """Return the "order" of a mode for the purpose of identifying tail modes.

    For simplex elements, the order is the total polynomial degree
    (sum of indices). For tensor product elements, the order is the maximum
    per-axis degree, which better captures high-frequency content in each
    axis direction.
    """
    if is_tensor_product:
        return max(mode_id)
    else:
        return sum(mode_id)


@log_process(logger)
def warp_and_refine_until_resolved(
        unwarped_mesh_or_refiner: Mesh | Refiner,
        warp_callable: Callable[[Mesh], Mesh],
        est_rel_interp_tolerance: float) -> Mesh:
    """Given an original ("unwarped") :class:`meshmode.mesh.Mesh` and a
    warping function *warp_callable* that takes and returns a mesh and a
    tolerance to which the mesh should be resolved by the mapping polynomials,
    this function will iteratively refine the *unwarped_mesh* until relative
    interpolation error estimates on the warped version are smaller than
    *est_rel_interp_tolerance* on each element.

    Unlike :func:`meshmode.mesh.generation.warp_and_refine_until_resolved`,
    this version supports both :class:`meshmode.mesh.SimplexElementGroup` and
    :class:`meshmode.mesh.TensorProductElementGroup` element groups.

    :arg unwarped_mesh_or_refiner: Either a :class:`meshmode.mesh.Mesh` or
        a :class:`~meshmode.mesh.refinement.RefinerWithoutAdjacency`. In the
        latter case, the refiner's current mesh is used as the starting mesh.
    :arg warp_callable: A callable that takes a :class:`meshmode.mesh.Mesh` and
        returns a warped version of it.
    :arg est_rel_interp_tolerance: The relative interpolation error tolerance
        below which all elements are considered resolved.
    :returns: The refined, unwarped mesh.
    """
    import numpy.linalg as la

    import modepy as mp
    from meshmode.mesh import ModepyElementGroup, TensorProductElementGroup
    from meshmode.mesh.refinement import RefinerWithoutAdjacency

    if isinstance(unwarped_mesh_or_refiner, RefinerWithoutAdjacency):
        refiner = unwarped_mesh_or_refiner
        unwarped_mesh = refiner.get_current_mesh()
    else:
        from meshmode.mesh import Mesh
        if isinstance(unwarped_mesh_or_refiner, Mesh):
            unwarped_mesh = unwarped_mesh_or_refiner
            refiner = RefinerWithoutAdjacency(unwarped_mesh)
        else:
            raise TypeError(
                f"unsupported type: '{type(unwarped_mesh_or_refiner).__name__}'")

    iteration = 0

    while True:
        refine_flags = np.zeros(unwarped_mesh.nelements, dtype=bool)

        warped_mesh = warp_callable(unwarped_mesh)
        vertices = warped_mesh.vertices
        assert vertices is not None

        # test whether there are invalid values in warped mesh
        if not np.all(np.isfinite(vertices)):
            raise FloatingPointError("Warped mesh contains non-finite vertices "
                                     "(NaN or Inf)")

        for group in warped_mesh.groups:
            if not np.all(np.isfinite(group.nodes)):
                raise FloatingPointError("Warped mesh contains non-finite nodes "
                                         "(NaN or Inf)")

        for base_element_nr, egrp in zip(
                warped_mesh.base_element_nrs, warped_mesh.groups,
                strict=True):
            if not isinstance(egrp, ModepyElementGroup):
                raise TypeError(
                    f"Unsupported element group type: '{type(egrp).__name__}'")

            is_tp = isinstance(egrp, TensorProductElementGroup)

            # Build the orthonormal basis for this element's space and shape.
            # This works for both SimplexElementGroup (shape=Simplex(dim),
            # space=PN(dim)) and TensorProductElementGroup (shape=Hypercube(dim),
            # space=QN(dim)).
            basis = mp.orthonormal_basis_for_space(egrp.space, egrp.shape)
            vdm = mp.vandermonde(basis.functions, egrp.unit_nodes)
            vdm_inv = la.inv(vdm)

            # Identify high-order "tail" modes. For simplex elements we use the
            # total polynomial degree (sum of mode indices), matching the
            # behaviour of simplex_interp_error_coefficient_estimator_matrix.
            # For tensor product elements we use the max per-axis degree so that
            # modes that are high-order along *any* axis are included in the
            # tail estimate.
            order_vector = np.array([
                _mode_order(mode_id, is_tp) for mode_id in basis.mode_ids
                ])
            max_order = np.max(order_vector)

            n_tail_orders = 1 if warped_mesh.dim > 1 else 2
            interp_err_est_mat = vdm_inv[order_vector > max_order - n_tail_orders]

            mapping_coeffs = np.einsum("ij,dej->dei", vdm_inv, egrp.nodes)
            mapping_norm_2 = np.sqrt(np.sum(mapping_coeffs**2, axis=-1))

            interp_error_coeffs = np.einsum(
                    "ij,dej->dei", interp_err_est_mat, egrp.nodes)
            interp_error_norm_2 = np.sqrt(np.sum(interp_error_coeffs**2, axis=-1))

            # max over dimensions
            est_rel_interp_error = np.max(interp_error_norm_2/mapping_norm_2, axis=0)

            refine_flags[base_element_nr:base_element_nr + egrp.nelements] = (
                est_rel_interp_error > est_rel_interp_tolerance)

        nrefined_elements = np.sum(refine_flags.astype(np.int32))
        if nrefined_elements == 0:
            break

        logger.info("warp_and_refine_until_resolved: "
                "iteration %d -> splitting %d/%d elements",
                iteration, nrefined_elements, unwarped_mesh.nelements)

        unwarped_mesh = refiner.refine(refine_flags)
        iteration += 1

    return unwarped_mesh

# }}}


# {{{ urchin

def refine_mesh_and_get_urchin_warper(
        order: int,
        m: int,
        n: int,
        est_rel_interp_tolerance: float,
        min_rad: float = 0.2,
        uniform_refinement_rounds: int = 0,
        group_cls: type[MeshElementGroup] | None = None,
        ) -> tuple[Refiner, Callable[[Mesh], Mesh]]:
    """Create a locally-refined sphere mesh and a callable to warp it into an
    "urchin" shape covered by a spherical harmonic.

    Unlike :func:`meshmode.mesh.generation.refine_mesh_and_get_urchin_warper`,
    this version supports both :class:`meshmode.mesh.SimplexElementGroup` and
    :class:`meshmode.mesh.TensorProductElementGroup` via the *group_cls*
    parameter, by using :func:`warp_and_refine_until_resolved` from this
    module.

    :arg order: order of the (simplex or tensor-product) elements.
    :arg m: order of the spherical harmonic :math:`Y^m_n`.
    :arg n: order of the spherical harmonic :math:`Y^m_n`.
    :arg est_rel_interp_tolerance: a tolerance for the relative
        interpolation error estimates on the warped version of the mesh.
    :arg min_rad: minimum radius of the urchin shape.
    :arg uniform_refinement_rounds: number of uniform refinement rounds
        to perform before adaptive refinement.
    :arg group_cls: a :class:`~meshmode.mesh.MeshElementGroup` subclass.
        Defaults to :class:`meshmode.mesh.SimplexElementGroup`.
    :returns: a tuple ``(refiner, warp_mesh)``, where *refiner* is a
        :class:`~meshmode.mesh.refinement.RefinerWithoutAdjacency` (from
        which the unwarped mesh may be obtained), and *warp_mesh* is a
        callable taking and returning a mesh that warps the unwarped mesh
        into a smooth shape covered by a spherical harmonic of order
        :math:`(m, n)`.
    """
    from scipy.special import sph_harm_y

    from meshmode.mesh import make_mesh
    from meshmode.mesh.generation import generate_sphere
    from meshmode.mesh.refinement import RefinerWithoutAdjacency

    def sph_harm(
            m_: int, n_: int, pts: np.ndarray) -> np.ndarray:
        assert abs(m_) <= n_
        x, y, z = pts
        r = np.sqrt(np.sum(pts**2, axis=0))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        # NOTE: This matches the spherical harmonic convention in the QBX3D paper:
        # https://arxiv.org/abs/1805.06106
        return sph_harm_y(n_, m_, theta, phi)

    def map_coords(pts: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(pts**2, axis=0))

        sph = sph_harm(m, n, pts).real
        scaled = min_rad + (sph - lo)/(hi-lo)
        new_rad = scaled

        return pts * new_rad / r

    def warp_mesh(mesh: Mesh) -> Mesh:
        from dataclasses import replace
        groups = [
            replace(grp, nodes=map_coords(grp.nodes))
            for grp in mesh.groups]

        return make_mesh(
                map_coords(mesh.vertices),
                groups,
                node_vertex_consistency_tolerance=False,
                is_conforming=mesh.is_conforming,
                )

    unwarped_mesh = generate_sphere(1, order=order, group_cls=group_cls)

    refiner = RefinerWithoutAdjacency(unwarped_mesh)
    for _ in range(uniform_refinement_rounds):
        refiner.refine_uniformly()

    nodes_sph = sph_harm(m, n, unwarped_mesh.groups[0].nodes).real
    lo = np.min(nodes_sph)
    hi = np.max(nodes_sph)
    del nodes_sph

    unwarped_mesh = warp_and_refine_until_resolved(
                refiner,
                warp_mesh,
                est_rel_interp_tolerance)

    return refiner, warp_mesh


def generate_urchin(
        order: int,
        m: int,
        n: int,
        est_rel_interp_tolerance: float,
        min_rad: float = 0.2,
        group_cls: type[MeshElementGroup] | None = None,
        ) -> Mesh:
    """Generate a refined mesh of a smooth shape covered by a spherical harmonic.

    Unlike :func:`meshmode.mesh.generation.generate_urchin`, this version
    supports both :class:`meshmode.mesh.SimplexElementGroup` and
    :class:`meshmode.mesh.TensorProductElementGroup` via the *group_cls*
    parameter.

    :arg order: order of the (simplex or tensor-product) elements.
    :arg m: order of the spherical harmonic :math:`Y^m_n`.
    :arg n: order of the spherical harmonic :math:`Y^m_n`.
    :arg est_rel_interp_tolerance: a tolerance for the relative
        interpolation error estimates on the warped version of the mesh.
    :arg min_rad: minimum radius of the urchin shape.
    :arg group_cls: a :class:`~meshmode.mesh.MeshElementGroup` subclass.
        Defaults to :class:`meshmode.mesh.SimplexElementGroup`.
    :returns: a refined :class:`~meshmode.mesh.Mesh` of a smooth shape
        covered by a spherical harmonic of order :math:`(m, n)`.
    """
    refiner, warper = refine_mesh_and_get_urchin_warper(
            order, m, n, est_rel_interp_tolerance,
            min_rad=min_rad,
            uniform_refinement_rounds=0,
            group_cls=group_cls,
            )

    return warper(refiner.get_current_mesh())

# }}}
