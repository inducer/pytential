from __future__ import division

__copyright__ = "Copyright (C) 2010,2012,2013 Andreas Kloeckner, Michael Tom"

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
from pytools import Record

class ElementGroup(Record):
    """A group of elements sharing a common reference element.

    .. attribute:: ref_element

        An instance of :class:`pytential.ref_element.RefElementBase`.

    .. attribute:: vertex_indices

        An array *(nelements, ref_element.nvertices)* of (mesh-wide) vertex indices.

    .. attribute:: nodes

        An array of node coordinates with shape
        *(mesh.ambient_dims, nelements, ref_element.nnodes)*.

    .. attribute:: element_nr_base

        Lowest element number in this element group.
    """

    def __init__(self, ref_element, vertex_indices, nodes, element_nr_base=0):
        Record.__init__(self,
                ref_element=ref_element,
                vertex_indices=vertex_indices,
                nodes=nodes,
                element_nr_base=element_nr_base)

class Mesh(Record):
    """
    .. attribute:: vertices

        An array of vertex coordinates with shape
        *(ambient_dims, nvertices)*

    .. attribute:: groups

        A list of :class:`ElementGroup` instances.
    """

    def __init__(self, vertices, groups):
        Record.__init__(self, vertices=vertices, groups=groups)

    @property
    def ambient_dims(self):
        return self.vertices.shape[0]




# vim: foldmethod=marker
