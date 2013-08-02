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


import numpy as np  # noqa
import pyopencl as cl  # noqa
import pyopencl.array  # noqa
from sumpy.fmm import SumpyExpansionWranglerCodeContainer, SumpyExpansionWrangler


import logging
logger = logging.getLogger(__name__)


# {{{ expansion wrangler

class QBXExpansionWranglerCodeContainer(SumpyExpansionWranglerCodeContainer):
    def __init__(self, cl_context,
            multipole_expansion, local_expansion, qbx_local_expansion, out_kernels):
        SumpyExpansionWranglerCodeContainer.__init__(self,
                cl_context, multipole_expansion, local_expansion, out_kernels)

        self.qbx_local_expansion = qbx_local_expansion

        # TODO:
        #self.l2qbxl = E2EFromParent(cl_context,
                #local_expansion, qbx_local_expansion)

    def get_wrangler(self, queue, geo_data, dtype, extra_kwargs={}):
        return QBXExpansionWrangler(self, queue, geo_data,
                dtype, extra_kwargs)


class QBXExpansionWrangler(SumpyExpansionWrangler):
    def __init__(self, code_container, queue, geo_data,
            dtype, extra_kwargs):
        SumpyExpansionWrangler.__init__(self,
                code_container, queue, geo_data.tree(),
                dtype, extra_kwargs)
        self.geo_data = geo_data

    def box_source_list_kwargs(self):
        return dict(
                box_source_starts=self.tree.box_point_source_starts,
                box_source_counts_nonchild=
                self.tree.box_point_source_counts_nonchild,
                sources=self.tree.point_sources)

    def box_target_list_kwargs(self):
        # This only covers the non-QBX targets.

        nqftl = self.geo_data.non_qbx_box_target_lists()
        return dict(
                box_target_starts=nqftl.box_target_starts,
                box_target_counts_nonchild=
                nqftl.box_targets_counts_nonchild,
                targets=nqftl.targets)

    def reorder_src_weights(self, src_weights):
        return src_weights.with_queue(self.queue)[self.tree.user_point_source_ids]

    def reorder_potentials(self, potentials):
        raise NotImplementedError

# }}}


# {{{ FMM top-level

def drive_fmm(geo_data, expansion_wrangler, src_weights):
    """Top-level driver routine for a fast multipole calculation.

    In part, this is intended as a template for custom FMMs, in the sense that
    you may copy and paste its
    `source code <https://github.com/inducer/boxtree/blob/master/boxtree/fmm.py>`_
    as a starting point.

    Nonetheless, many common applications (such as point-to-point FMMs) can be
    covered by supplying the right *expansion_wrangler* to this routine.

    :arg traversal: A :class:`QBXFMMGeometryData` instance.
    :arg expansion_wrangler: An object exhibiting the
        :class:`ExpansionWranglerInterface`.
    :arg src_weights: Source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.

    Returns the potentials computed by *expansion_wrangler*.

    See also :func:`boxtree.fmm.drive_fmm`.
    """
    traversal = geo_data.traversal()
    tree = traversal.tree

    wrangler = expansion_wrangler

    # FIXME: last stage:
    # - translate to QBX
    # - do to-QBX list1
    # - do conventional to-target eval

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    logger.info("start qbx fmm")

    logger.debug("reorder source weights")

    src_weights = wrangler.reorder_src_weights(src_weights)

    # {{{ "Step 2.1:" Construct local multipoles

    logger.debug("construct local multipoles")

    mpole_exps = wrangler.form_multipoles(
            traversal.source_boxes,
            src_weights)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    logger.debug("propagate multipoles upward")

    for lev in xrange(tree.nlevels-1, -1, -1):
        start_parent_box, end_parent_box = \
                traversal.level_start_source_parent_box_nrs[lev:lev+2]
        wrangler.coarsen_multipoles(
                traversal.source_parent_boxes[start_parent_box:end_parent_box],
                mpole_exps)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")

    potentials = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights)

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")

    local_exps = wrangler.multipole_to_local(
            traversal.target_or_target_parent_boxes,
            traversal.sep_siblings_starts,
            traversal.sep_siblings_lists,
            mpole_exps)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    potentials = potentials + wrangler.eval_multipoles(
            traversal.target_boxes,
            traversal.sep_smaller_starts,
            traversal.sep_smaller_lists,
            mpole_exps)

    # these potentials are called beta in [1]

    if traversal.sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                "('list 3 close')")

        potentials = potentials + wrangler.eval_direct(
                traversal.target_boxes,
                traversal.sep_close_smaller_starts,
                traversal.sep_close_smaller_lists,
                src_weights)

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger mpoles ("list 4")

    logger.debug("form locals for separated bigger mpoles ('list 4 far')")

    local_exps = local_exps + wrangler.form_locals(
            traversal.target_or_target_parent_boxes,
            traversal.sep_bigger_starts,
            traversal.sep_bigger_lists,
            src_weights)

    if traversal.sep_close_bigger_starts is not None:
        logger.debug("evaluate separated close bigger interactions directly "
                "('list 4 close')")

        potentials = potentials + wrangler.eval_direct(
                traversal.target_or_target_parent_boxes,
                traversal.sep_close_bigger_starts,
                traversal.sep_close_bigger_lists,
                src_weights)

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    logger.debug("propagate local_exps downward")

    for lev in xrange(1, tree.nlevels):
        start_box, end_box = \
                traversal.level_start_target_or_target_parent_box_nrs[lev:lev+2]
        wrangler.refine_locals(
                traversal.target_or_target_parent_boxes[start_box:end_box],
                local_exps)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    logger.debug("evaluate locals")

    potentials = potentials + wrangler.eval_locals(
            traversal.target_boxes,
            local_exps)

    # }}}

    1/0
    logger.debug("reorder potentials")
    result = wrangler.reorder_potentials(potentials)

    logger.info("qbx fmm complete")

    return result

# }}}


# vim: foldmethod=marker
