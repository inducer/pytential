from __future__ import division

__copyright__ = "Copyright (C) 2010-2013 Andreas Kloeckner"

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

__doc__ = """
Target discretizations are a simpler version of the full
:class:`pytential.discretization.Discretization` interface.
They do not provide any evaluation of integrals, norms, or
layer potentials, but are instead only geared towards being
used as evaluation targets.

.. autoclass:: TargetBase

.. autoclass:: PointsTarget

"""


class TargetBase(object):
    """
    .. attribute:: ambient_dim
    .. method:: nodes

        Shape: ``[ambient_dim, ndofs]``
    .. attribute:: ndofs
    """


class PointsTarget(TargetBase):
    """The point of this class is to act as a container for some target points
    while presenting enough of the :class:`meshmode.discretization.Discretization`
    interface to not necessitate a lot of special cases in that code path.
    """

    def __init__(self, nodes, normals=None):
        self._nodes = nodes

    @property
    def ambient_dim(self):
        return self._nodes.shape[0]

    def preprocess_optemplate(self, name, discretizations, expr):
        return expr

    def nodes(self):
        """Shape: ``[ambient_dim, ndofs]``
        """

        return self._nodes

    @property
    def ndofs(self):
        for coord_ary in self._nodes:
            return coord_ary.shape[0]

# vim: foldmethod=marker
