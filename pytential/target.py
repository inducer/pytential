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
:class:`meshmode.discretization.Discretization` interface.
They do not provide any evaluation of integrals, norms, or
layer potentials, but are instead only geared towards being
used as evaluation targets.

.. autoclass:: TargetBase

.. autoclass:: PointsTarget

.. class:: Array

    See :class:`arraycontext.Array`.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from arraycontext.context import Array
from pytools import T

if TYPE_CHECKING:
    from pytential.collection import GeometryCollection


class TargetBase(ABC):
    """
    .. autoproperty:: ambient_dim
    .. autoproperty:: ndofs
    .. automethod:: nodes
    """

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Ambient dimension of the points in the target geometry."""

    @property
    @abstractmethod
    def ndofs(self) -> int:
        """Number of points (DOFs) in the target geometry."""

    @abstractmethod
    def nodes(self) -> Array:
        """
        :returns: an array of points of shape ``[ambient_dim, ndofs]`` that
            form the target geometry.
        """


class PointsTarget(TargetBase):
    """A generic container for a set of target points.

    The point of this class is to act as a container for some target points
    while presenting enough of the :class:`meshmode.discretization.Discretization`
    interface to not necessitate a lot of special cases in that code path.

    .. attribute:: normals
        :type: Optional[Array]

        An array of the same shape as :meth:`TargetBase.nodes` that gives the
        normals at each of the target points.

    .. automethod:: preprocess_optemplate
    """

    def __init__(self, nodes: Array, normals: Array | None = None) -> None:
        self._nodes = nodes
        self.normals = normals

    @property
    def ambient_dim(self) -> int:
        adim = self._nodes.shape[0]
        assert isinstance(adim, int)
        return adim

    @property
    def ndofs(self) -> int:
        # NOTE: arraycontext.Array is not iterable theoretically
        for coord_ary in self._nodes:   # type: ignore[attr-defined]
            return coord_ary.shape[0]

        raise AttributeError

    def nodes(self) -> Array:
        return self._nodes

    def preprocess_optemplate(self,
                name: str,
                discretizations: "GeometryCollection",
                # FIXME: replace this with a pymbolic TypeVar bound to an actual
                # expression when that gets in
                expr: T) -> T:
        """See :class:`~pytential.source.PotentialSource`."""
        return expr


# vim: foldmethod=marker
