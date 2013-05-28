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




import numpy as np
#import numpy.linalg as la
import modepy as mp
from pytools import memoize_method




# {{{ QBX grid element group types

class QBXGridElementGroupBase:
    """Container for the :class:`QBXGrid` data corresponding to
    one :class:`pytential.mesh.ElementGroup`.

    .. attribute :: grid
    .. attribute :: mesh_el_group
    .. attribute :: node_nr_base
    """

    def __init__(self, grid, mesh_el_group, order, node_nr_base):
        """
        :arg grid: an instance of :class:`QBXGridBase`
        :arg mesh_el_group: an instance of :class:`pytential.mesh.ElementGroup`
        """
        self.grid = grid
        self.mesh_el_group = mesh_el_group
        self.order = order
        self.node_nr_base = node_nr_base

    @property
    def nelements(self):
        return self.mesh_el_group.nelements

    @property
    def nunit_nodes(self):
        return self.unit_nodes.shape[-1]

    @property
    def nnodes(self):
        return self.nunit_nodes * self.nelements

    @memoize_method
    def _from_mesh_interp_matrix(self):
        meg = self.mesh_el_group
        return mp.resampling_matrix(
                mp.simplex_onb(meg.dim, meg.order),
                self.unit_nodes,
                meg.unit_nodes)

    def _nodes(self):
        # Not cached, because the global nodes array is what counts.
        # This is just used to build that.

        return np.tensordot(
                self.mesh_el_group.nodes,
                self._from_mesh_interp_matrix(),
                (-1, -1))

    def view(self, global_array):
        return global_array[..., self.node_nr_base:self.node_nr_base+self.nnodes] \
                .reshape(
                        global_array.shape[:-1]
                        + (self.nelements, self.nunit_nodes))




class QBXGridInterpolationElementGroup(QBXGridElementGroupBase):
    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 1:
            return mp.LegendreGaussQuadrature(self.order)
        else:
            return mp.VioreanuRokhlinSimplexQuad(self.order, dims)

    @property
    @memoize_method
    def unit_nodes(self):
        return self._quadrature_rule().nodes

    @property
    @memoize_method
    def weights(self):
        return self._quadrature_rule().weights


class QBXGridQuadratureElementGroup(QBXGridElementGroupBase):
    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 1:
            return mp.LegendreGaussQuadrature(self.order)
        else:
            return mp.XiaoGimbutasSimplexQuadrature(self.order, dims)

    @property
    @memoize_method
    def unit_nodes(self):
        return self._quadrature_rule().nodes

    @memoize_method
    def weights(self):
        return self._quadrature_rule().weights

# }}}

# {{{ QBX grid types

class QBXGridBase(object):

    def __init__(self, mesh, order, group_class):
        """
        :arg order: If *is_quadrature_only* is *False*, the maximum total degree
            that can be exactly represented on this grid.
            If *is_quadrature_only* is *True*, the maximum total degree
            that can be exactly integrated by this grid.
        """
        self.mesh = mesh


    @property
    @memoize_method
    def nodes(self):
        result = np.empty((self.mesh.ambient_dim, self.nnodes))
        for grp in self.groups:
            grp.view(result)[:] = grp._nodes()
        return result

# }}}

# vim: fdm=marker
