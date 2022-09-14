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

from typing import Any, Hashable, Optional, Union

__doc__ = """
.. autoclass:: DEFAULT_SOURCE
.. autoclass:: DEFAULT_TARGET

.. autoclass:: QBX_SOURCE_STAGE1
.. autoclass:: QBX_SOURCE_STAGE2
.. autoclass:: QBX_SOURCE_QUAD_STAGE2

.. autoclass:: GRANULARITY_NODE
.. autoclass:: GRANULARITY_CENTER
.. autoclass:: GRANULARITY_ELEMENT

.. autoclass:: DOFDescriptor
.. autofunction:: as_dofdesc

.. class:: DiscretizationStages

    A :class:`~typing.Union` of all the allowed discretization stages.

.. class:: DOFGranularities

    A :class:`~typing.Union` of all the allowed DOF granularity types.

.. class:: DOFDescriptorLike

    Types convertible to a :class:`~pytential.symbolic.dof_desc.DOFDescriptor`
    by :func:`~pytential.symbolic.dof_desc.as_dofdesc`.

"""


# {{{ discretizations

class _UNNAMED_SOURCE:                   # noqa: N801
    """Symbolic identifier for an unnamed source. This is for internal
    use only."""


class _UNNAMED_TARGET:                   # noqa: N801
    """Symbolic identifier for an unnamed target. This is for internal
    use only."""


class DEFAULT_SOURCE:                   # noqa: N801
    """Symbolic identifier for the default source. Geometries with
    this value get replaced with the default source given to
    :func:`pytential.bind`."""


class DEFAULT_TARGET:                   # noqa: N801
    """Symbolic identifier for the default target. Geometries with
    this value get replaced with the default target given to
    :func:`pytential.bind`."""


class TAG_WITH_DEFAULT_SOURCE:          # noqa: N801
    """Symbolic identifier for a source that is tagged with
    the default source."""


class TAG_WITH_DEFAULT_TARGET:          # noqa: N801
    """Symbolic identifier for a source that is tagged with
    the default target."""


class QBX_SOURCE_STAGE1:                # noqa: N801
    """Symbolic identifier for the Stage 1 discretization of a
    :class:`pytential.qbx.QBXLayerPotentialSource`.
    """


class QBX_SOURCE_STAGE2:                # noqa: N801
    """Symbolic identifier for the Stage 2 discretization of a
    :class:`pytential.qbx.QBXLayerPotentialSource`.
    """


class QBX_SOURCE_QUAD_STAGE2:           # noqa: N801
    """Symbolic identifier for the upsampled Stage 2 discretization of a
    :class:`pytential.qbx.QBXLayerPotentialSource`.
    """


# }}}


# {{{ granularity

class GRANULARITY_NODE:                 # noqa: N801
    """DOFs are per node."""


class GRANULARITY_CENTER:               # noqa: N801
    """DOFs interleaved per expansion center (two per node, one on each side)."""


class GRANULARITY_ELEMENT:              # noqa: N801
    """DOFs per discretization element."""


# }}}


# {{{ DOFDescriptor

class _NoArgSentinel:
    pass


class DOFDescriptor:
    """A data structure specifying the meaning of a vector of degrees of freedom
    that is handled by :mod:`pytential` (a "DOF vector"). In particular, using
    :attr:`geometry`, this data structure describes the geometric object on which
    the (scalar) function described by the DOF vector exists. Using
    :attr:`granularity`, the data structure describes how the geometric object
    is discretized (e.g. conventional nodal data, per-element scalars, etc.)

    .. attribute:: geometry

        An identifier for the geometry on which the DOFs exist. This can be a
        simple string or any other hashable identifier for the geometric object.
        The geometric objects are generally subclasses of
        :class:`~pytential.source.PotentialSource`,
        :class:`~pytential.target.TargetBase` or
        :class:`~meshmode.discretization.Discretization`.

    .. attribute:: discr_stage

        Specific to a :class:`pytential.source.LayerPotentialSourceBase`,
        this describes on which of the discretizations the
        DOFs are defined. Can be one of :class:`QBX_SOURCE_STAGE1`,
        :class:`QBX_SOURCE_STAGE2` or :class:`QBX_SOURCE_QUAD_STAGE2`.

    .. attribute:: granularity

        Describes the level of granularity of the DOF vector.
        Can be one of :class:`GRANULARITY_NODE` (one DOF per node),
        :class:`GRANULARITY_CENTER` (two DOFs per node, one per side) or
        :class:`GRANULARITY_ELEMENT` (one DOF per element).

    .. automethod:: copy
    .. automethod:: to_stage1
    .. automethod:: to_stage2
    .. automethod:: to_quad_stage2
    """

    def __init__(self,
            geometry: Optional[Hashable] = None,
            discr_stage: Optional["DiscretizationStages"] = None,
            granularity: Optional["DOFGranularities"] = None):
        if granularity is None:
            granularity = GRANULARITY_NODE

        if not (discr_stage is None
                or discr_stage == QBX_SOURCE_STAGE1
                or discr_stage == QBX_SOURCE_STAGE2
                or discr_stage == QBX_SOURCE_QUAD_STAGE2):
            raise ValueError(f"unknown discr stage tag: '{discr_stage}'")

        if not (granularity == GRANULARITY_NODE
                or granularity == GRANULARITY_CENTER
                or granularity == GRANULARITY_ELEMENT):
            raise ValueError(f"unknown granularity: '{granularity}'")

        self.geometry = geometry
        self.discr_stage = discr_stage
        self.granularity = granularity

    def copy(self,
            geometry: Optional[Hashable] = None,
            discr_stage: Optional["DiscretizationStages"] = _NoArgSentinel,
            granularity: Optional["DOFGranularities"] = None) -> "DOFDescriptor":
        if isinstance(geometry, DOFDescriptor):
            discr_stage = geometry.discr_stage \
                    if discr_stage is _NoArgSentinel else discr_stage
            geometry = geometry.geometry

        return type(self)(
                geometry=(self.geometry
                    if geometry is None else geometry),
                granularity=(self.granularity
                    if granularity is None else granularity),
                discr_stage=(self.discr_stage
                    if discr_stage is _NoArgSentinel else discr_stage),
                )

    def to_stage1(self) -> "DOFDescriptor":
        return self.copy(discr_stage=QBX_SOURCE_STAGE1)

    def to_stage2(self) -> "DOFDescriptor":
        return self.copy(discr_stage=QBX_SOURCE_STAGE2)

    def to_quad_stage2(self) -> "DOFDescriptor":
        return self.copy(discr_stage=QBX_SOURCE_QUAD_STAGE2)

    def __hash__(self) -> int:
        return hash((type(self),
            self.geometry, self.discr_stage, self.granularity))

    def __eq__(self, other: Any) -> bool:
        return (type(self) is type(other)
                and self.geometry == other.geometry
                and self.discr_stage == other.discr_stage
                and self.granularity == other.granularity)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        discr_stage = self.discr_stage \
                if self.discr_stage is None else self.discr_stage.__name__
        granularity = self.granularity.__name__
        return "{}(geometry={}, stage={}, granularity={})".format(
                type(self).__name__, self.geometry, discr_stage, granularity)

    def __str__(self) -> str:
        name = []
        if self.geometry is None:
            name.append("?")
        elif self.geometry in (_UNNAMED_SOURCE, DEFAULT_SOURCE):
            name.append("s")
        elif self.geometry in (_UNNAMED_TARGET, DEFAULT_TARGET):
            name.append("t")
        else:
            name.append(
                    self.geometry.__name__
                    if isinstance(self.geometry, type)
                    else str(self.geometry))

        if self.discr_stage == QBX_SOURCE_STAGE2:
            name.append("stage2")
        elif self.discr_stage == QBX_SOURCE_QUAD_STAGE2:
            name.append("quads2")

        if self.granularity == GRANULARITY_CENTER:
            name.append("center")
        elif self.granularity == GRANULARITY_ELEMENT:
            name.append("element")

        return "/".join(name)


def as_dofdesc(desc: "DOFDescriptorLike") -> "DOFDescriptor":
    if isinstance(desc, DOFDescriptor):
        return desc

    if (desc == QBX_SOURCE_STAGE1
            or desc == QBX_SOURCE_STAGE2
            or desc == QBX_SOURCE_QUAD_STAGE2):
        return DOFDescriptor(discr_stage=desc)

    if (desc == GRANULARITY_NODE
            or desc == GRANULARITY_CENTER
            or desc == GRANULARITY_ELEMENT):
        return DOFDescriptor(granularity=desc)

    return DOFDescriptor(geometry=desc)

# }}}


# {{{ type annotations

DiscretizationStages = Union[
        QBX_SOURCE_STAGE1,
        QBX_SOURCE_STAGE2,
        QBX_SOURCE_QUAD_STAGE2,
        ]

DOFGranularities = Union[
        GRANULARITY_NODE,
        GRANULARITY_CENTER,
        GRANULARITY_ELEMENT,
        ]

DOFDescriptorLike = Union[
    DOFDescriptor,
    Hashable
    ]

# }}}
