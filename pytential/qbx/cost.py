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

import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: F401
from pyopencl.array import take
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.tools import dtype_to_ctype
from mako.template import Template
from pymbolic import var, evaluate
from pytools import memoize_method
from functools import partial

from boxtree.cost import (
    FMMTranslationCostModel, AbstractFMMCostModel as BaseAbstractFMMCostModel,
    FMMCostModel, _PythonFMMCostModel
)
from abc import abstractmethod

Template = partial(Template, strict_undefined=True)

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. note::

   This module is experimental. Its interface is subject to change until this
   notice is removed.

This module helps predict the running time of each step of QBX, as an extension of
the similar module :mod:`boxtree.cost` in boxtree.

:class:`QBXTranslationCostModel` describes the translation or evaluation cost of a
single operation. For example, *m2qbxl* describes the cost for translating a single
multipole expansion to a QBX local expansion.

:class:`AbstractQBXCostModel` uses :class:`QBXTranslationCostModel` and
kernel-specific calibration parameter to compute the total cost of each step of QBX
in each box. :class:`QBXCostModel` is one implementation of
:class:`AbstractQBXCostModel` using OpenCL.

:file:`examples/cost.py` in the source distribution demonstrates how the calibration
and evaluation are performed.

Translation Cost of a Single Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: QBXTranslationCostModel

.. autofunction:: make_pde_aware_translation_cost_model

.. autofunction:: make_taylor_translation_cost_model

Cost Model Classes
^^^^^^^^^^^^^^^^^^

.. autoclass:: AbstractQBXCostModel

.. autoclass:: QBXCostModel

Calibration (Generate Calibration Parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: AbstractQBXCostModel.estimate_kernel_specific_calibration_params

Evaluating
^^^^^^^^^^

.. automethod:: AbstractQBXCostModel.qbx_cost_per_stage

.. automethod:: AbstractQBXCostModel.qbx_cost_per_box

To get the cost from `BoundExpression`, refer to
:meth:`pytential.symbolic.execution.BoundExpression.cost_per_stage` and
:meth:`pytential.symbolic.execution.BoundExpression.cost_per_box`.

Utilities
^^^^^^^^^

.. automethod:: boxtree.cost.AbstractFMMCostModel.aggregate_over_boxes

.. automethod:: AbstractQBXCostModel.get_unit_calibration_params
"""


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

def make_pde_aware_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation operators that make use of the
    knowledge that the potential satisfies a PDE.
    """
    p_qbx = var("p_qbx")
    p_fmm = np.array([var(f"p_fmm_lev{i}") for i in range(nlevels)])

    uses_point_and_shoot = False

    ncoeffs_fmm = (p_fmm + 1) ** (dim - 1)
    ncoeffs_qbx = (p_qbx + 1) ** (dim - 1)

    if dim == 3:
        uses_point_and_shoot = True

    return QBXTranslationCostModel(
        ncoeffs_qbx=ncoeffs_qbx,
        ncoeffs_fmm_by_level=ncoeffs_fmm,
        uses_point_and_shoot=uses_point_and_shoot)


def make_taylor_translation_cost_model(dim, nlevels):
    """Create a cost model for FMM translation based on Taylor expansions
    in Cartesian coordinates.
    """
    p_qbx = var("p_qbx")
    p_fmm = np.array([var(f"p_fmm_lev{i}") for i in range(nlevels)])

    ncoeffs_fmm = (p_fmm + 1) ** dim
    ncoeffs_qbx = (p_qbx + 1) ** dim

    return QBXTranslationCostModel(
        ncoeffs_qbx=ncoeffs_qbx,
        ncoeffs_fmm_by_level=ncoeffs_fmm,
        uses_point_and_shoot=False)

# }}}


# {{{ abstract cost model

class AbstractQBXCostModel(BaseAbstractFMMCostModel):
    """An interface to obtain both QBX operation counts and calibrated (e.g. in
    seconds) cost estimates.

    * To obtain operation counts only, use :meth:`get_unit_calibration_params`
      with :meth:`qbx_cost_per_stage` or :meth:`qbx_cost_per_box`.

    * To calibrate the model, pass operation counts per stage together with timing
      data to :meth:`estimate_kernel_specific_calibration_params`.

    * To evaluate the calibrated models, pass the kernel-specific calibration
      parameters from :meth:`estimate_kernel_specific_calibration_params` to
      :meth:`qbx_cost_per_stage` or :meth:`qbx_cost_per_box`.
    """

    @abstractmethod
    def process_form_qbxl(self, queue, geo_data, p2qbxl_cost,
                          ndirect_sources_per_target_box):
        """
        :arg queue: a :class:`pyopencl.CommandQueue` object.
        :arg geo_data: a :class:`pytential.qbx.geometry.QBXFMMGeometryData` object.
        :arg p2qbxl_cost: a :class:`numpy.float64` constant representing the cost of
            adding a source to a QBX local expansion.
        :arg ndirect_sources_per_target_box: a :class:`numpy.ndarray` or
            :class:`pyopencl.array.Array` of shape ``(ntarget_boxes,)``, with the
            *i*th entry representing the number of direct evaluation sources (list 1,
            list 3 close and list 4 close) for ``target_boxes[i]``.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            ``(ntarget_boxes,)``, with the *i*th entry representing the cost of
            adding all direct evaluation sources to QBX local expansions of centers
            in ``target_boxes[i]``.
        """
        pass

    @abstractmethod
    def process_m2qbxl(self, queue, geo_data, m2qbxl_cost):
        """
        :arg geo_data: a :class:`pytential.qbx.geometry.QBXFMMGeometryData` object.
        :arg m2qbxl_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape ``(nlevels,)`` where the *i*th entry represents the translation
            cost from multipole expansion at level *i* to a QBX center.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            ``(ntarget_boxes,)``, with the *i*th entry representing the cost of
            translating multipole expansions of list 3 boxes at all source levels to
            all QBX centers in ``target_boxes[i]``.
        """
        pass

    @abstractmethod
    def process_l2qbxl(self, queue, geo_data, l2qbxl_cost):
        """
        :arg geo_data: a :class:`pytential.qbx.geometry.QBXFMMGeometryData` object.
        :arg l2qbxl_cost: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array`
            of shape ``(nlevels,)`` where each entry represents the translation
            cost from a box local expansion to a QBX local expansion.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            ``(ntarget_boxes,)``, with each entry representing the cost of
            translating box local expansions to all QBX local expansions.
        """
        pass

    @abstractmethod
    def process_eval_qbxl(self, queue, geo_data, qbxl2p_cost):
        """
        :arg geo_data: a :class:`pytential.qbx.geometry.QBXFMMGeometryData` object.
        :arg qbxl2p_cost: a :class:`numpy.float64` constant, representing the
            evaluation cost of a target from its QBX local expansion.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            ``(ntarget_boxes,)``, with the *i*th entry representing the cost of
            evaluating all targets associated with QBX centers in ``target_boxes[i]``
            from QBX local expansions.
        """
        pass

    @abstractmethod
    def process_eval_target_specific_qbxl(self, queue, geo_data, p2p_tsqbx_cost,
                                          ndirect_sources_per_target_box):
        """
        :arg geo_data: a :class:`pytential.qbx.geometry.QBXFMMGeometryData` object.
        :arg p2p_tsqbx_cost: a :class:`numpy.float64` constant representing the
            evaluation cost of a target from a direct evaluation source of the target
            box containing the expansion center.
        :arg ndirect_sources_per_target_box: a :class:`numpy.ndarray` or
            :class:`pyopencl.array.Array` of shape ``(ntarget_boxes,)``, with the
            *i*th entry representing the number of direct evaluation sources
            (list 1, list 3 close and list 4 close) for ``target_boxes[i]``.
        :return: a :class:`numpy.ndarray` or :class:`pyopencl.array.Array` of shape
            ``(ntarget_boxes,)``, with the *i*th entry representing the evaluation
            cost of all targets associated with centers in ``target_boxes[i]`` from
            the direct evaluation sources of ``target_boxes[i]``.
        """
        pass

    def qbx_cost_factors_for_kernels_from_model(
            self, queue, nlevels, xlat_cost, context):
        """Evaluate translation cost factors from symbolic model. The result of this
        function can be used for process_* methods in this class.

        This method overwrite the method in parent
        :class:`boxtree.cost.AbstractFMMCostModel` to support operations specific to
        QBX.

        :arg queue: If not None, the cost factor arrays will be transferred to device
            using this queue.
        :arg nlevels: the number of tree levels.
        :arg xlat_cost: a :class:`QBXTranslationCostModel`.
        :arg context: a :class:`dict` mapping from the symbolic names of parameters
            to their values, serving as context when evaluating symbolic expressions
            in *xlat_cost*.
        :return: a :class:`dict`, mapping from stage names to the translation costs
            of those stages in FMM and QBX.
        """
        cost_factors = self.fmm_cost_factors_for_kernels_from_model(
            queue, nlevels, xlat_cost, context
        )

        cost_factors.update({
            "p2qbxl_cost": evaluate(xlat_cost.p2qbxl(), context=context),
            "m2qbxl_cost": np.array([
                evaluate(xlat_cost.m2qbxl(ilevel), context=context)
                for ilevel in range(nlevels)
            ]),
            "l2qbxl_cost": np.array([
                evaluate(xlat_cost.l2qbxl(ilevel), context=context)
                for ilevel in range(nlevels)
            ]),
            "qbxl2p_cost": evaluate(xlat_cost.qbxl2p(), context=context),
            "p2p_tsqbx_cost": evaluate(xlat_cost.p2p_tsqbx(), context=context)
        })

        if queue:
            cost_factors = self.cost_factors_to_dev(cost_factors, queue)

        return cost_factors

    @staticmethod
    def gather_metadata(geo_data, fmm_level_to_order):
        lpot_source = geo_data.lpot_source
        tree = geo_data.tree()

        metadata = {
            "p_qbx": lpot_source.qbx_order,
            "nlevels": tree.nlevels,
            "nsources": tree.nsources,
            "ntargets": tree.ntargets,
            "ncenters": geo_data.ncenters
        }

        for level in range(tree.nlevels):
            metadata[f"p_fmm_lev{level}"] = fmm_level_to_order[level]

        return metadata

    def qbx_cost_per_box(self, queue, geo_data, kernel, kernel_arguments,
                         calibration_params):
        # FIXME: This should support target filtering.
        lpot_source = geo_data.lpot_source
        use_tsqbx = lpot_source._use_target_specific_qbx
        tree = geo_data.tree()
        traversal = geo_data.traversal()
        nqbtl = geo_data.non_qbx_box_target_lists()
        box_target_counts_nonchild = nqbtl.box_target_counts_nonchild
        target_boxes = traversal.target_boxes

        # FIXME: We can avoid using *kernel* and *kernel_arguments* if we talk
        # to the wrangler to obtain the FMM order (see also
        # https://gitlab.tiker.net/inducer/boxtree/issues/25)
        fmm_level_to_order = [
            lpot_source.fmm_level_to_order(
                kernel.get_base_kernel(), kernel_arguments, tree, ilevel
            ) for ilevel in range(tree.nlevels)
        ]

        # {{{ Construct parameters

        params = calibration_params.copy()
        params.update(dict(p_qbx=lpot_source.qbx_order))

        for ilevel in range(tree.nlevels):
            params[f"p_fmm_lev{ilevel}"] = fmm_level_to_order[ilevel]

        # }}}

        xlat_cost = self.translation_cost_model_factory(
            tree.dimensions, tree.nlevels
        )

        translation_cost = self.qbx_cost_factors_for_kernels_from_model(
            queue, tree.nlevels, xlat_cost, params
        )

        ndirect_sources_per_target_box = \
            self.get_ndirect_sources_per_target_box(queue, traversal)

        # get FMM cost per box from parent class
        result = self.cost_per_box(
            queue, traversal, fmm_level_to_order,
            calibration_params,
            ndirect_sources_per_target_box=ndirect_sources_per_target_box,
            box_target_counts_nonchild=box_target_counts_nonchild
        )

        if use_tsqbx:
            result[target_boxes] += self.process_eval_target_specific_qbxl(
                queue, geo_data, translation_cost["p2p_tsqbx_cost"],
                ndirect_sources_per_target_box
            )
        else:
            result[target_boxes] += self.process_form_qbxl(
                queue, geo_data, translation_cost["p2qbxl_cost"],
                ndirect_sources_per_target_box
            )

        result[target_boxes] += self.process_m2qbxl(
            queue, geo_data, translation_cost["m2qbxl_cost"]
        )

        result[target_boxes] += self.process_l2qbxl(
            queue, geo_data, translation_cost["l2qbxl_cost"]
        )

        result[target_boxes] += self.process_eval_qbxl(
            queue, geo_data, translation_cost["qbxl2p_cost"]
        )

        metadata = self.gather_metadata(geo_data, fmm_level_to_order)

        return result, metadata

    def qbx_cost_per_stage(self, queue, geo_data, kernel, kernel_arguments,
                           calibration_params):
        # FIXME: This should support target filtering.
        lpot_source = geo_data.lpot_source
        use_tsqbx = lpot_source._use_target_specific_qbx
        tree = geo_data.tree()
        traversal = geo_data.traversal()
        nqbtl = geo_data.non_qbx_box_target_lists()
        box_target_counts_nonchild = nqbtl.box_target_counts_nonchild

        # FIXME: We can avoid using *kernel* and *kernel_arguments* if we talk
        # to the wrangler to obtain the FMM order (see also
        # https://gitlab.tiker.net/inducer/boxtree/issues/25)
        fmm_level_to_order = [
            lpot_source.fmm_level_to_order(
                kernel.get_base_kernel(), kernel_arguments, tree, ilevel
            ) for ilevel in range(tree.nlevels)
        ]

        # {{{ Construct parameters

        params = calibration_params.copy()
        params.update(dict(p_qbx=lpot_source.qbx_order))

        for ilevel in range(tree.nlevels):
            params[f"p_fmm_lev{ilevel}"] = fmm_level_to_order[ilevel]

        # }}}

        xlat_cost = self.translation_cost_model_factory(
            tree.dimensions, tree.nlevels
        )

        translation_cost = self.qbx_cost_factors_for_kernels_from_model(
            queue, tree.nlevels, xlat_cost, params
        )

        ndirect_sources_per_target_box = \
            self.get_ndirect_sources_per_target_box(queue, traversal)

        # get FMM per-stage cost from parent class
        result = self.cost_per_stage(
            queue, traversal, fmm_level_to_order,
            calibration_params,
            ndirect_sources_per_target_box=ndirect_sources_per_target_box,
            box_target_counts_nonchild=box_target_counts_nonchild
        )

        if use_tsqbx:
            result["eval_target_specific_qbx_locals"] = self.aggregate_over_boxes(
                self.process_eval_target_specific_qbxl(
                    queue, geo_data, translation_cost["p2p_tsqbx_cost"],
                    ndirect_sources_per_target_box=ndirect_sources_per_target_box
                )
            )
        else:
            result["form_global_qbx_locals"] = self.aggregate_over_boxes(
                self.process_form_qbxl(
                    queue, geo_data, translation_cost["p2qbxl_cost"],
                    ndirect_sources_per_target_box
                )
            )

        result["translate_box_multipoles_to_qbx_local"] = self.aggregate_over_boxes(
            self.process_m2qbxl(queue, geo_data, translation_cost["m2qbxl_cost"])
        )

        result["translate_box_local_to_qbx_local"] = self.aggregate_over_boxes(
            self.process_l2qbxl(queue, geo_data, translation_cost["l2qbxl_cost"])
        )

        result["eval_qbx_expansions"] = self.aggregate_over_boxes(
            self.process_eval_qbxl(queue, geo_data, translation_cost["qbxl2p_cost"])
        )

        metadata = self.gather_metadata(geo_data, fmm_level_to_order)

        return result, metadata

    @staticmethod
    def get_unit_calibration_params():
        calibration_params = BaseAbstractFMMCostModel.get_unit_calibration_params()

        calibration_params.update(dict(
            c_p2qbxl=1.0,
            c_p2p_tsqbx=1.0,
            c_qbxl2p=1.0,
            c_m2qbxl=1.0,
            c_l2qbxl=1.0
        ))

        return calibration_params

    _QBX_STAGE_TO_CALIBRATION_PARAMETER = {
        "form_global_qbx_locals": "c_p2qbxl",
        "translate_box_multipoles_to_qbx_local": "c_m2qbxl",
        "translate_box_local_to_qbx_local": "c_l2qbxl",
        "eval_qbx_expansions": "c_qbxl2p",
        "eval_target_specific_qbx_locals": "c_p2p_tsqbx"
    }

    def estimate_calibration_params(self, model_results, timing_results,
                                    time_field_name="wall_elapsed",
                                    additional_stage_to_param_names=()):
        stage_to_param_names = self._QBX_STAGE_TO_CALIBRATION_PARAMETER.copy()
        stage_to_param_names.update(additional_stage_to_param_names)

        return super().estimate_calibration_params(
            model_results, timing_results, time_field_name=time_field_name,
            additional_stage_to_param_names=stage_to_param_names
        )

    def estimate_kernel_specific_calibration_params(
            self, model_results, timing_results, time_field_name="wall_elapsed"):
        """Get kernel-specific calibration parameters from samples of model costs and
        real costs.

        :arg model_results: a :class:`list` of modeled costs. Each model cost can be
            obtained from `BoundExpression.cost_per_stage` with "constant_one" for
            argument `calibration_params`.
        :arg timing_results: a :class:`list` of timing data. Each timing data can be
            obtained from `BoundExpression.eval`.
        :arg time_field_name: a :class:`str`, the field name from the timing result.
            Usually this can be ``"wall_elapsed"`` or ``"process_elapsed"``.
        :return: a :class:`dict` which maps kernels to calibration parameters.
        """
        cost_per_kernel = {}
        params_per_kernel = {}

        assert len(model_results) == len(timing_results)

        for icase in range(len(model_results)):
            model_cost = model_results[icase]
            real_cost = timing_results[icase]

            for insn in real_cost:
                assert (insn in model_cost)

                knls = frozenset(knl for knl in insn.source_kernels)

                if knls not in cost_per_kernel:
                    cost_per_kernel[knls] = {
                        "model_costs": [],
                        "real_costs": []
                    }

                cost_per_kernel[knls]["model_costs"].append(model_cost[insn])
                cost_per_kernel[knls]["real_costs"].append(real_cost[insn])

        for knls in cost_per_kernel:
            params_per_kernel[knls] = self.estimate_calibration_params(
                cost_per_kernel[knls]["model_costs"],
                cost_per_kernel[knls]["real_costs"],
                time_field_name=time_field_name
            )

        return params_per_kernel


class QBXCostModel(AbstractQBXCostModel, FMMCostModel):
    """This class is an implementation of interface :class:`AbstractQBXCostModel`
    using :mod:`pyopencl`.
    """
    def __init__(
            self,
            translation_cost_model_factory=make_pde_aware_translation_cost_model):
        """
        :arg translation_cost_model_factory: a function, which takes tree dimension
            and the number of tree levels as arguments, returns an object of
            :class:`QBXTranslationCostModel`.
        """
        FMMCostModel.__init__(self, translation_cost_model_factory)

    @memoize_method
    def _fill_array_with_index_knl(self, context, idx_dtype, array_dtype):
        return ElementwiseKernel(
            context,
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

    def _fill_array_with_index(self, queue, array, index, value):
        idx_dtype = index.dtype
        array_dtype = array.dtype
        knl = self._fill_array_with_index_knl(queue.context, idx_dtype, array_dtype)
        knl(index, array, value, queue=queue)

    @memoize_method
    def count_global_qbx_centers_knl(self, context, box_id_dtype, particle_id_dtype):
        return ElementwiseKernel(
            context,
            Template(r"""
                ${particle_id_t} *nqbx_centers_itgt_box,
                ${particle_id_t} *global_qbx_center_weight,
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
                    nqbx_centers += global_qbx_center_weight[iparticle];

                nqbx_centers_itgt_box[i] = nqbx_centers;
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype)
            ),
            name="count_global_qbx_centers"
        )

    def get_nqbx_centers_per_tgt_box(self, queue, geo_data, weights=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue` object.
        :arg geo_data: a :class:`pytential.qbx.geometry.QBXFMMGeometryData` object.
        :arg weights: a :class:`pyopencl.array.Array` of shape ``(ncenters,)`` with
            particle_id_dtype, the weight of each center in user order.
        :return: a :class:`pyopencl.array.Array` of shape ``(ntarget_boxes,)`` with
            type *particle_id_dtype* where the *i*th entry represents the number of
            `geo_data.global_qbx_centers` in ``target_boxes[i]``, optionally weighted
            by *weights*.
        """
        traversal = geo_data.traversal()
        tree = geo_data.tree()
        global_qbx_centers = geo_data.global_qbx_centers()
        ncenters = geo_data.ncenters

        # Build a mask (weight) of whether a target is a global qbx center
        global_qbx_centers_tree_order = take(
            tree.sorted_target_ids, global_qbx_centers, queue=queue
        )
        global_qbx_center_weight = cl.array.zeros(
            queue, tree.ntargets, dtype=tree.particle_id_dtype
        )

        self._fill_array_with_index(
            queue, global_qbx_center_weight, global_qbx_centers_tree_order, 1
        )

        if weights is not None:
            assert weights.dtype == tree.particle_id_dtype
            global_qbx_center_weight[tree.sorted_target_ids[:ncenters]] *= weights

        # Each target box enumerate its target list and add the weight of global
        # qbx centers
        ntarget_boxes = len(traversal.target_boxes)
        nqbx_centers_itgt_box = cl.array.empty(
            queue, ntarget_boxes, dtype=tree.particle_id_dtype
        )

        count_global_qbx_centers_knl = self.count_global_qbx_centers_knl(
            queue.context, tree.box_id_dtype, tree.particle_id_dtype
        )
        count_global_qbx_centers_knl(
            nqbx_centers_itgt_box,
            global_qbx_center_weight,
            traversal.target_boxes,
            tree.box_target_starts,
            tree.box_target_counts_nonchild,
            queue=queue
        )

        return nqbx_centers_itgt_box

    def process_form_qbxl(self, queue, geo_data, p2qbxl_cost,
                          ndirect_sources_per_target_box):
        nqbx_centers_itgt_box = self.get_nqbx_centers_per_tgt_box(queue, geo_data)

        return (nqbx_centers_itgt_box
                * ndirect_sources_per_target_box
                * p2qbxl_cost)

    @memoize_method
    def process_m2qbxl_knl(self, context, box_id_dtype, particle_id_dtype):
        return ElementwiseKernel(
            context,
            Template(r"""
                ${box_id_t} *idx_to_itgt_box,
                ${particle_id_t} *nqbx_centers_itgt_box,
                ${box_id_t} *ssn_starts,
                double *nm2qbxl,
                double m2qbxl_cost
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype)
            ),
            Template(r"""
                // get the index of current box in target_boxes
                ${box_id_t} itgt_box = idx_to_itgt_box[i];
                // get the number of expansion centers in current box
                ${particle_id_t} nqbx_centers = nqbx_centers_itgt_box[itgt_box];
                // get the number of list 3 boxes of the current box in a particular
                // level
                ${box_id_t} nlist3_boxes = ssn_starts[i + 1] - ssn_starts[i];
                // calculate the cost
                nm2qbxl[itgt_box] += (nqbx_centers * nlist3_boxes * m2qbxl_cost);
            """).render(
                box_id_t=dtype_to_ctype(box_id_dtype),
                particle_id_t=dtype_to_ctype(particle_id_dtype)
            ),
            name="process_m2qbxl"
        )

    def process_m2qbxl(self, queue, geo_data, m2qbxl_cost):
        tree = geo_data.tree()
        traversal = geo_data.traversal()
        ntarget_boxes = len(traversal.target_boxes)
        nqbx_centers_itgt_box = self.get_nqbx_centers_per_tgt_box(queue, geo_data)

        process_m2qbxl_knl = self.process_m2qbxl_knl(
            queue.context, tree.box_id_dtype, tree.particle_id_dtype
        )

        nm2qbxl = cl.array.zeros(queue, ntarget_boxes, dtype=np.float64)

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):
            process_m2qbxl_knl(
                ssn.nonempty_indices,
                nqbx_centers_itgt_box,
                ssn.starts,
                nm2qbxl,
                m2qbxl_cost[isrc_level].get(queue).reshape(-1)[0],
                queue=queue
            )

        return nm2qbxl

    def process_l2qbxl(self, queue, geo_data, l2qbxl_cost):
        tree = geo_data.tree()
        traversal = geo_data.traversal()
        nqbx_centers_itgt_box = self.get_nqbx_centers_per_tgt_box(queue, geo_data)

        # l2qbxl_cost_itgt_box = l2qbxl_cost[tree.box_levels[traversal.target_boxes]]
        l2qbxl_cost_itgt_box = take(
            l2qbxl_cost,
            take(tree.box_levels, traversal.target_boxes, queue=queue),
            queue=queue
        )

        return nqbx_centers_itgt_box * l2qbxl_cost_itgt_box

    def process_eval_qbxl(self, queue, geo_data, qbxl2p_cost):
        center_to_targets_starts = geo_data.center_to_tree_targets().starts
        center_to_targets_starts = center_to_targets_starts.with_queue(queue)
        weights = center_to_targets_starts[1:] - center_to_targets_starts[:-1]

        nqbx_targets_itgt_box = self.get_nqbx_centers_per_tgt_box(
            queue, geo_data, weights=weights
        )

        return nqbx_targets_itgt_box * qbxl2p_cost

    def process_eval_target_specific_qbxl(self, queue, geo_data, p2p_tsqbx_cost,
                                          ndirect_sources_per_target_box):
        center_to_targets_starts = geo_data.center_to_tree_targets().starts
        center_to_targets_starts = center_to_targets_starts.with_queue(queue)
        weights = center_to_targets_starts[1:] - center_to_targets_starts[:-1]

        nqbx_targets_itgt_box = self.get_nqbx_centers_per_tgt_box(
            queue, geo_data, weights=weights
        )

        return (nqbx_targets_itgt_box
                * ndirect_sources_per_target_box
                * p2p_tsqbx_cost)

    def qbx_cost_factors_for_kernels_from_model(
            self, queue, nlevels, xlat_cost, context):
        if not isinstance(queue, cl.CommandQueue):
            raise TypeError(
                "An OpenCL command queue must be supplied for cost model")

        return AbstractQBXCostModel.qbx_cost_factors_for_kernels_from_model(
            self, queue, nlevels, xlat_cost, context
        )


class _PythonQBXCostModel(AbstractQBXCostModel, _PythonFMMCostModel):
    def __init__(
            self,
            translation_cost_model_factory=make_pde_aware_translation_cost_model):
        """This cost model is a redundant implementation used for testing. It should
        not be used outside of tests for :mod:`pytential`.

        :arg translation_cost_model_factory: a function, which takes tree dimension
            and the number of tree levels as arguments, returns an object of
            :class:`TranslationCostModel`.
        """
        _PythonFMMCostModel.__init__(self, translation_cost_model_factory)

    def process_form_qbxl(self, queue, geo_data, p2qbxl_cost,
                          ndirect_sources_per_target_box):
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        traversal = geo_data.traversal()

        np2qbxl = np.zeros(len(traversal.target_boxes), dtype=np.float64)

        for tgt_icenter in global_qbx_centers:
            itgt_box = qbx_center_to_target_box[tgt_icenter]
            np2qbxl[itgt_box] += ndirect_sources_per_target_box[itgt_box]

        return np2qbxl * p2qbxl_cost

    def process_eval_target_specific_qbxl(self, queue, geo_data, p2p_tsqbx_cost,
                                          ndirect_sources_per_target_box):
        center_to_targets_starts = geo_data.center_to_tree_targets().starts
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        traversal = geo_data.traversal()

        neval_tsqbx = np.zeros(len(traversal.target_boxes), dtype=np.float64)
        for tgt_icenter in global_qbx_centers:
            start, end = center_to_targets_starts[tgt_icenter:tgt_icenter + 2]
            itgt_box = qbx_center_to_target_box[tgt_icenter]
            neval_tsqbx[itgt_box] += (
                    ndirect_sources_per_target_box[itgt_box] * (end - start)
            )

        return neval_tsqbx * p2p_tsqbx_cost

    def process_m2qbxl(self, queue, geo_data, m2qbxl_cost):
        traversal = geo_data.traversal()
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()

        ntarget_boxes = len(traversal.target_boxes)
        nm2qbxl = np.zeros(ntarget_boxes, dtype=np.float64)

        for isrc_level, sep_smaller_list in enumerate(
                traversal.from_sep_smaller_by_level):

            qbx_center_to_target_box_source_level = \
                geo_data.qbx_center_to_target_box_source_level(isrc_level)

            for tgt_icenter in global_qbx_centers:
                icontaining_tgt_box = qbx_center_to_target_box_source_level[
                    tgt_icenter
                ]

                if icontaining_tgt_box == -1:
                    continue

                start = sep_smaller_list.starts[icontaining_tgt_box]
                stop = sep_smaller_list.starts[icontaining_tgt_box + 1]

                containing_tgt_box = qbx_center_to_target_box[tgt_icenter]

                nm2qbxl[containing_tgt_box] += (
                        (stop - start) * m2qbxl_cost[isrc_level])

        return nm2qbxl

    def process_l2qbxl(self, queue, geo_data, l2qbxl_cost):
        tree = geo_data.tree()
        traversal = geo_data.traversal()
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()

        ntarget_boxes = len(traversal.target_boxes)
        nl2qbxl = np.zeros(ntarget_boxes, dtype=np.float64)

        for tgt_icenter in global_qbx_centers:
            itgt_box = qbx_center_to_target_box[tgt_icenter]
            tgt_ibox = traversal.target_boxes[itgt_box]
            nl2qbxl[itgt_box] += l2qbxl_cost[tree.box_levels[tgt_ibox]]

        return nl2qbxl

    def process_eval_qbxl(self, queue, geo_data, qbxl2p_cost):
        traversal = geo_data.traversal()
        global_qbx_centers = geo_data.global_qbx_centers()
        center_to_targets_starts = geo_data.center_to_tree_targets().starts
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()

        ntarget_boxes = len(traversal.target_boxes)
        neval_qbxl = np.zeros(ntarget_boxes, dtype=np.float64)

        for src_icenter in global_qbx_centers:
            start, end = center_to_targets_starts[src_icenter:src_icenter+2]
            icontaining_tgt_box = qbx_center_to_target_box[src_icenter]
            neval_qbxl[icontaining_tgt_box] += (end - start)

        return neval_qbxl * qbxl2p_cost

    def qbx_cost_per_box(self, queue, geo_data, kernel, kernel_arguments,
                         calibration_params):
        """This function transfers *geo_data* to host if necessary
        """
        from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
        from pytential.qbx.geometry import QBXFMMGeometryData

        if not isinstance(geo_data, ToHostTransferredGeoDataWrapper):
            assert isinstance(geo_data, QBXFMMGeometryData)
            geo_data = ToHostTransferredGeoDataWrapper(geo_data)

        return AbstractQBXCostModel.qbx_cost_per_box(
            self, queue, geo_data, kernel, kernel_arguments, calibration_params
        )

    def qbx_cost_per_stage(self, queue, geo_data, kernel, kernel_arguments,
                           calibration_params):
        """This function additionally transfers geo_data to host if necessary
        """
        from pytential.qbx.utils import ToHostTransferredGeoDataWrapper
        from pytential.qbx.geometry import QBXFMMGeometryData

        if not isinstance(geo_data, ToHostTransferredGeoDataWrapper):
            assert isinstance(geo_data, QBXFMMGeometryData)
            geo_data = ToHostTransferredGeoDataWrapper(geo_data)

        return AbstractQBXCostModel.qbx_cost_per_stage(
            self, queue, geo_data, kernel, kernel_arguments, calibration_params
        )

    def qbx_cost_factors_for_kernels_from_model(
            self, queue, nlevels, xlat_cost, context):
        return AbstractQBXCostModel.qbx_cost_factors_for_kernels_from_model(
            self, None, nlevels, xlat_cost, context
        )

# }}}

# vim: foldmethod=marker
