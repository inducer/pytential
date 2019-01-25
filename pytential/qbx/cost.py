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

import numpy as np
from six.moves import range
from pymbolic import var

from boxtree.cost import (
    FMMTranslationCostModel, AbstractFMMCostModel
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
        return var("c_p2p_tsqbx") * self.ncoeffs_qbx ** (1/2)

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
    def process_eval_target_specific_qbxl(self):
        pass
    """

    @abstractmethod
    def process_form_qbxl(self):
        pass

    @abstractmethod
    def process_m2qbxl(self):
        pass

    @abstractmethod
    def process_l2qbxl(self):
        pass

    @abstractmethod
    def process_eval_qbxl(self):
        pass

# }}}

# vim: foldmethod=marker
