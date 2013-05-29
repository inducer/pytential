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
from pytools import memoize_method, memoize_method_nested
from pytential.discretization import Discretization

import pyopencl as cl

import loopy as lp
import modepy as mp



__doc__ = """A composite polynomial discretization without
any specific opinion on how to evaluate layer potentials.
"""




# {{{ element group base

class PolynomialElementGroupBase(object):
    """Container for the :class:`QBXGrid` data corresponding to
    one :class:`pytential.mesh.MeshElementGroup`.

    .. attribute :: grid
    .. attribute :: mesh_el_group
    .. attribute :: node_nr_base
    """

    def __init__(self, grid, mesh_el_group, order, node_nr_base):
        """
        :arg grid: an instance of :class:`QBXGridBase`
        :arg mesh_el_group: an instance of :class:`pytential.mesh.MeshElementGroup`
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

# }}}

# {{{ element group

class PolynomialElementGroup(PolynomialElementGroupBase):
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

# }}}

# {{{ discretization

class PolynomialElementDiscretizationBase(Discretization):
    """An (unstructured) composite polynomial discretization.

    .. attribute :: mesh
    .. attribute :: groups
    .. attribute :: nnodes

    .. autoattribute :: nodes
    """

    def __init__(self, cl_ctx, mesh, order, real_dtype=np.float64):
        """
        :arg order: A polynomial-order-like parameter passed unmodified to
            :attr:`group_class`. See subclasses for more precise definition.
        """

        self.cl_context = cl_ctx

        self.mesh = mesh
        self.nnodes = 0
        self.groups = []
        for mg in mesh.groups:
            ng = self.group_class(self, mg, order, self.nnodes)
            self.groups.append(ng)
            self.nnodes += ng.nnodes

        self.real_dtype = np.dtype(real_dtype)
        self.complex_dtype = (self.real_dtype.type(0) + 1j).dtype

    @property
    def dim(self):
        return self.mesh.dim

    @property
    def ambient_dim(self):
        return self.mesh.ambient_dim

    def _empty(self, dtype):
        return cl.array.empty(self.cl_context, self.nnodes, dtype=dtype)

    @memoize_method
    def _diff_matrices(self, grp):
        meg = grp.mesh_el_group
        result = mp.differentiation_matrices(
                mp.simplex_onb(self.dim, meg.order),
                mp.grad_simplex_onb(self.dim, meg.order),
                meg.unit_nodes, grp.unit_nodes)
        if not isinstance(result, tuple):
            return (result,)
        else:
            return result

    def get_parametrization_derivative_component(self, queue, ambient_axis, ref_axis):
        @memoize_method_nested
        def knl():
            knl = lp.make_kernel(self.cl_context.devices[0],
                "{[k,i,j]: 0<=k<nelements and 0<=i<ndiscr_nodes and 0<=j<nmesh_nodes}",
                "result[k,i] = sum(j, diff_mat[i, j] * coord[k, j])")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = self._empty(self.real_dtype)

        for grp in self.groups:
            meg = grp.mesh_el_group
            knl()(queue,
                    diff_mat=self._diff_matrices(grp)[ref_axis],
                    result=grp.view(result), coord=meg.nodes[ambient_axis],
                    nelements=meg.nelements, ndiscr_nodes=grp.nunit_nodes,
                    nmesh_nodes=meg.nunit_nodes)

        1/0
        return result




class PolynomialElementDiscretization(PolynomialElementDiscretizationBase):
    def __init__(self, cl_ctx, mesh, poly_order, real_dtype=np.float64):
        """
        :arg poly_order: The total degree of polynomial representable
            on each element in each element group.
            (this is intended to later vary across element groups)
        """
        PolynomialElementDiscretizationBase.__init__(self, cl_ctx, mesh, poly_order,
                real_dtype)

    group_class = PolynomialElementGroup

# }}}


# vim: fdm=marker
