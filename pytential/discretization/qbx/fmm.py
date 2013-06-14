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
import pyopencl as cl
import pyopencl.array  # noqa
from pytools import memoize_method, Record
import loopy as lp


# {{{ code getter

class QBXFMMCodeGetter(object):
    def __init__(self, cl_context, ambient_dim):
        self.cl_context = cl_context
        self.ambient_dim = ambient_dim

    @property
    @memoize_method
    def pick_centers(self):
        knl = lp.make_kernel(self.cl_context.devices[0],
            """{[dim,k,i]:
                0<=dim<ndims and
                0<=k<nelements and
                0<=i<nout_nodes}""",
            """
                centers[dim, k, i] = all_centers[dim, k, kept_center_indices[i]]
                radii[k, i] = all_radii[k, kept_center_indices[i]]
                """,
            [
                lp.GlobalArg("all_centers", None,
                    shape="ndims,nelements,nunit_nodes"),
                lp.GlobalArg("all_radii", None, shape="nelements,nunit_nodes"),
                lp.ValueArg("nunit_nodes", np.int32),
                "..."
                ],
            default_offset=lp.auto, name="center_pick",
            defines=dict(ndims=self.ambient_dim))

        knl = lp.tag_data_axes(knl, "centers,all_centers", "sep, C, C")

        knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
        return lp.tag_inames(knl, dict(k="g.0", dim="ilp"))

    @property
    @memoize_method
    def build_tree(self):
        from boxtree import TreeBuilder
        return TreeBuilder(self.cl_context)

    @property
    @memoize_method
    def build_traversal(self):
        from boxtree.traversal import FMMTraversalBuilder
        return FMMTraversalBuilder(self.cl_context)

# }}}


# {{{ geometry data

class CenterInfo(Record):
    """
    .. attribute:: centers
    .. attribute:: sides

        -1 for inside, +1 for outside

    .. attribute:: radii
    """


class QBXFMMGeometryData(object):
    def __init__(self, code_getter, source_discr, target_discrs):
        """
        :arg targets: a tuple of
            :class:`pytential.discretization.Discretization`
            or
            :class:`pytential.discretization.target.TargetBase`
            instances
        """

        self.code_getter = code_getter
        self.source_discr = source_discr
        self.target_discrs = target_discrs

    @memoize_method
    def kept_center_indices(self, el_group):
        # FIXME: Be more careful about which nodes to keep
        return np.arange(0, el_group.nunit_nodes, 8)

    @memoize_method
    def center_info(self):
        ncenters = 0
        for el_group in self.source_discr.groups:
            kept_indices = self.kept_center_indices(el_group)
            # two: one for positive side, one for negative side
            ncenters += 2 * len(kept_indices) * el_group.nelements

        dtype = self.source_discr.nodes().dtype

        from pytential import sym, bind
        from pytools.obj_array import make_obj_array
        with cl.CommandQueue(self.source_discr.cl_context) as queue:
            radii_sym = sym.cse(2*sym.area_element(), "radii")
            all_radii, all_pos_centers, all_neg_centers = bind(self.source_discr,
                    make_obj_array([
                        radii_sym,
                        sym.Nodes() + radii_sym*sym.normal(),
                        sym.Nodes() - radii_sym*sym.normal()
                        ]))(queue)

            # The centers are returned from the above as multivectors.
            all_pos_centers = all_pos_centers.as_vector(np.object)
            all_neg_centers = all_neg_centers.as_vector(np.object)

            # -1 for inside, +1 for outside
            center_sides = cl.array.empty(
                    self.source_discr.cl_context, ncenters, np.int8)
            radii = cl.array.empty(
                    self.source_discr.cl_context, ncenters, dtype)
            centers = make_obj_array([
                cl.array.empty(self.source_discr.cl_context, ncenters, dtype)
                for i in xrange(self.source_discr.ambient_dim)])

            ibase = 0
            for el_group in self.source_discr.groups:
                kept_center_indices = self.kept_center_indices(el_group)
                group_len = len(kept_indices) * el_group.nelements

                for side, all_centers in [
                        (+1, all_pos_centers),
                        (-1, all_neg_centers),
                        ]:

                    center_sides[ibase:ibase + group_len].fill(side, queue=queue)

                    radii_view = radii[ibase:ibase + group_len] \
                            .reshape(el_group.nelements, len(kept_indices))
                    centers_view = make_obj_array([
                            centers_i[ibase:ibase + group_len]
                            .reshape((el_group.nelements, len(kept_indices)))
                            for centers_i in centers
                        ])
                    all_centers_view = make_obj_array([
                        el_group.view(pos_centers_i)
                        for pos_centers_i in all_centers
                        ])
                    self.code_getter.pick_centers(queue,
                            centers=centers_view,
                            all_centers=all_centers_view,
                            radii=radii_view,
                            all_radii=el_group.view(all_radii),
                            kept_center_indices=kept_center_indices)

                    ibase += group_len

            assert ibase == ncenters

        return CenterInfo(
                center_sides=center_sides,
                radii=radii,
                centers=centers)

    def tree(self):
        pass

# }}}

# vim: foldmethod=marker
