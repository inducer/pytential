from __future__ import division, absolute_import

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2018 Matt Wala
Copyright (C) 2019 Hao Gao
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

from six.moves import range
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: F401
from pyopencl.array import take
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.tools import dtype_to_ctype
from mako.template import Template
from pymbolic import var
from pytools import memoize_method

from boxtree.cost import (
    FMMTranslationCostModel, AbstractFMMCostModel, PythonFMMCostModel, CLFMMCostModel
)
from abc import abstractmethod

import logging
logger = logging.getLogger(__name__)


# {{{ translation cost model

class QBXTranslationCostModel(FMMTranslationCostModel):
    """Provides modeled costs for individual translations or evaluations."""

    def __init__(self, ncoeffs_qbx, ncoeffs_fmm_by_level, uses_point_and_shoot):
        self.ncoeffs_qbx = ncoeffs_qbx
        FMMTranslationCostModel.__init__(
            self, ncoeffs_fmm_by_level, uses_point_and_shoot
        )

    def p2qbxl(self):
        return var("c_p2qbxl") * self.ncoeffs_qbx

    def p2p_tsqbx(self):
        # This term should be linear in the QBX order, which is the
        # square root of the number of QBX coefficients.
        return var("c_p2p_tsqbx") * self.ncoeffs_qbx ** (1 / 2)

    def qbxl2p(self):
        return var("c_qbxl2p") * self.ncoeffs_qbx

    def m2qbxl(self, level):
        return var("c_m2qbxl") * self.e2e_cost(
            self.ncoeffs_fmm_by_level[level],
            self.ncoeffs_qbx)

    def l2qbxl(self, level):
        return var("c_l2qbxl") * self.e2e_cost(
            self.ncoeffs_fmm_by_level[level],
            self.ncoeffs_qbx)

# }}}


# {{{ translation cost model factories

def pde_aware_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation operators that make use of the
    knowledge that the potential satisfies a PDE.
    """
    p_qbx = var("p_qbx")
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])

    uses_point_and_shoot = False

    ncoeffs_fmm = (p_fmm + 1) ** (dim - 1)
    ncoeffs_qbx = (p_qbx + 1) ** (dim - 1)

    if dim == 3:
        uses_point_and_shoot = True

    return QBXTranslationCostModel(
        ncoeffs_qbx=ncoeffs_qbx,
        ncoeffs_fmm_by_level=ncoeffs_fmm,
        uses_point_and_shoot=uses_point_and_shoot)


def taylor_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation based on Taylor expansions
    in Cartesian coordinates.
    """
    p_qbx = var("p_qbx")
    p_fmm = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])

    ncoeffs_fmm = (p_fmm + 1) ** dim
    ncoeffs_qbx = (p_qbx + 1) ** dim

    return QBXTranslationCostModel(
        ncoeffs_qbx=ncoeffs_qbx,
        ncoeffs_fmm_by_level=ncoeffs_fmm,
        uses_point_and_shoot=False)

# }}}


# {{{ cost model

class AbstractQBXCostModel(AbstractFMMCostModel):
    def __init__(
            self,
            translation_cost_model_factory=pde_aware_translation_cost_model):
        AbstractFMMCostModel.__init__(
            self, translation_cost_model_factory
        )

    """

    @abstractmethod
    def process_l2qbxl(self):
        pass

    @abstractmethod
    def process_eval_qbxl(self):
        pass

    """

    @abstractmethod
    def process_form_qbxl(self, p2qbxl_cost, geo_data,
                          ndirect_sources_per_target_box):
        pass

    @abstractmethod
    def process_eval_target_specific_qbxl(self, p2p_tsqbx_cost, geo_data,
                                          ndirect_sources_per_target_box):
        pass

    @abstractmethod
    def process_m2qbxl(self, geo_data, m2qbxl_cost):
        """
        :arg geo_data: TODO
        :arg m2qbxl_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape (nlevels,) where the ith entry represents the evaluation cost
            from multipole expansion at level i to a QBX center.
        :return:
        """
        pass


class CLQBXCostModel(AbstractQBXCostModel, CLFMMCostModel):
    def __init__(self, queue,
                 translation_cost_model_factory=pde_aware_translation_cost_model):
        self.queue = queue
        AbstractQBXCostModel.__init__(self, translation_cost_model_factory)

    @memoize_method
    def _fill_array_with_index_knl(self, idx_dtype, array_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                ${idx_t} *index,
                ${array_t} *array,
                ${array_t} val
            """).render(
                idx_t=dtype_to_ctype(idx_dtype),
                array_t=dtype_to_ctype(array_dtype)
            ),
            Template(r"""
                array[index[i]] = val;
            """).render(),
            name="fill_array_with_index"
        )

    def _fill_array_with_index(self, array, index, value):
        idx_dtype = index.dtype
        array_dtype = array.dtype
        knl = self._fill_array_with_index_knl(idx_dtype, array_dtype)
        knl(index, array, value, queue=self.queue)

    @memoize_method
    def count_global_qbx_centers_knl(self, box_id_dtype, particle_id_dtype):
        return ElementwiseKernel(
            self.queue.context,
            Template(r"""
                ${particle_id_t} *nqbx_centers_itgt_box,
                char *global_qbx_center_mask,
                ${box_id_t} *target_boxes,
                ${particle_id_t} *box_target_starts,
                ${particle_id_t} *box_target_counts_nonchild
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype)
            ),
            Template(r"""
                ${box_id_t} global_box_id = target_boxes[i];
                ${particle_id_t} start = box_target_starts[global_box_id];
                ${particle_id_t} end = start + box_target_counts_nonchild[
                    global_box_id
                ];

                ${particle_id_t} nqbx_centers = 0;
                for(${particle_id_t} iparticle = start; iparticle < end; iparticle++)
                    if(global_qbx_center_mask[iparticle])
                        nqbx_centers++;

                nqbx_centers_itgt_box[i] = nqbx_centers;
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype)
            ),
            name="count_global_qbx_centers"
        )

    def get_nqbx_centers_per_tgt_box(self, geo_data):
        """
        :arg geo_data: TODO
        :return: a :class:`pyopencl.array.Array` of shape (ntarget_boxes,) where the
            ith entry represents the number of `geo_data.global_qbx_centers` in
            target_boxes[i].
        """
        traversal = geo_data.traversal()
        tree = geo_data.tree()
        global_qbx_centers = geo_data.global_qbx_centers()

        # Build a mask of whether a target is a global qbx center
        global_qbx_centers_tree_order = take(
            tree.sorted_target_ids, global_qbx_centers, queue=self.queue
        )
        global_qbx_center_mask = cl.array.zeros(
            self.queue, tree.ntargets, dtype=np.int8
        )
        self._fill_array_with_index(
            global_qbx_center_mask, global_qbx_centers_tree_order, 1
        )

        # Each target box enumerate its target list and count the number of global
        # qbx centers
        ntarget_boxes = len(traversal.target_boxes)
        nqbx_centers_itgt_box = cl.array.empty(
            self.queue, ntarget_boxes, dtype=tree.particle_id_dtype
        )

        count_global_qbx_centers_knl = self.count_global_qbx_centers_knl(
            tree.box_id_dtype, tree.particle_id_dtype
        )
        count_global_qbx_centers_knl(
            nqbx_centers_itgt_box,
            global_qbx_center_mask,
            traversal.target_boxes,
            tree.box_target_starts,
            tree.box_target_counts_nonchild
        )

        return nqbx_centers_itgt_box

    def process_form_qbxl(self, p2qbxl_cost, geo_data,
                          ndirect_sources_per_target_box):
        nqbx_centers_itgt_box = self.get_nqbx_centers_per_tgt_box(geo_data)

        return (nqbx_centers_itgt_box
                * ndirect_sources_per_target_box
                * p2qbxl_cost)

    def process_eval_target_specific_qbxl(self, p2p_tsqbx_cost, geo_data,
                                          ndirect_sources_per_target_box):
        pass

    def process_m2qbxl(self, geo_data, m2qbxl_cost):
        pass


class PythonQBXCostModel(AbstractQBXCostModel, PythonFMMCostModel):
    def process_form_qbxl(self, p2qbxl_cost, geo_data,
                          ndirect_sources_per_target_box):
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        traversal = geo_data.traversal()

        np2qbxl = np.zeros(len(traversal.target_boxes), dtype=np.float64)

        for tgt_icenter in global_qbx_centers:
            itgt_box = qbx_center_to_target_box[tgt_icenter]
            np2qbxl[itgt_box] += ndirect_sources_per_target_box[itgt_box]

        return np2qbxl * p2qbxl_cost

    def process_eval_target_specific_qbxl(self, p2p_tsqbx_cost, geo_data,
                                          ndirect_sources_per_target_box):
        center_to_targets_starts = geo_data.center_to_tree_targets().starts
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        traversal = geo_data.traversal()

        neval_tsqbx = np.zeros(len(traversal.target_boxes), dtype=np.float64)
        for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
            start, end = center_to_targets_starts[tgt_icenter:tgt_icenter + 2]
            itgt_box = qbx_center_to_target_box[tgt_icenter]
            neval_tsqbx[itgt_box] += (
                    ndirect_sources_per_target_box[itgt_box] * (end - start)
            )

        return neval_tsqbx * p2p_tsqbx_cost

    def process_m2qbxl(self, geo_data, m2qbxl_cost):
        traversal = geo_data.traversal()
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_center_to_target_box_source_level = \
            geo_data.qbx_center_to_target_box_source_level()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()

        ntarget_boxes = len(traversal.target_boxes)
        nm2qbxl = np.zeros(ntarget_boxes, dtype=np.float64)

        for isrc_level, sep_smaller_list in enumerate(
                traversal.from_sep_smaller_by_level):
            for tgt_icenter in global_qbx_centers:
                icontaining_tgt_box = qbx_center_to_target_box_source_level[
                    isrc_level][tgt_icenter]

                if icontaining_tgt_box == -1:
                    continue

                start = sep_smaller_list.starts[icontaining_tgt_box]
                stop = sep_smaller_list.starts[icontaining_tgt_box + 1]

                containing_tgt_box = qbx_center_to_target_box(tgt_icenter)

                nm2qbxl[containing_tgt_box] += (
                        (stop - start) * m2qbxl_cost[isrc_level])

        return nm2qbxl

# }}}

# vim: foldmethod=marker
