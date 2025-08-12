from __future__ import annotations


__copyright__ = """
Copyright (C) 2016 Matt Wala
Copyright (C) 2019 Alexandru Fikl
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

from typing import TYPE_CHECKING, cast

import numpy as np  # noqa: F401
from typing_extensions import override

import loopy as lp
from meshmode.discretization import Discretization
from meshmode.discretization.connection import (
    DiscretizationConnection,
    IdentityDiscretizationConnection,
)
from meshmode.dof_array import DOFArray
from pytools import memoize_in

from pytential.symbolic import dof_desc


if TYPE_CHECKING:
    from collections.abc import Sequence

    from arraycontext import Array, ArrayOrContainerOrScalarT

    from pytential.collection import GeometryCollection
    from pytential.symbolic.dof_desc import DOFDescriptor, DOFDescriptorLike

__doc__ = """

Connections
-----------

.. autoclass:: CenterGranularityConnection
.. autoclass:: DOFConnection
.. autofunction:: connection_from_dds

"""


# {{{ granularity connections

class CenterGranularityConnection(DiscretizationConnection):
    """A :class:`~meshmode.discretization.connection.DiscretizationConnection`
    used to transport from node data
    (:class:`~pytential.symbolic.primitives.GRANULARITY_NODE`) to expansion
    centers (:class:`~pytential.symbolic.primitives.GRANULARITY_CENTER`).

    .. attribute:: discr
    .. automethod:: __call__
    """

    def __init__(self, discr: Discretization) -> None:
        super().__init__(discr, discr, is_surjective=False)

    def _interleave_dof_arrays(self, ary1: DOFArray, ary2: DOFArray) -> DOFArray:
        if not isinstance(ary1, DOFArray) or not isinstance(ary2, DOFArray):
            raise TypeError("non-array passed to connection")

        if ary1.array_context is not ary2.array_context:
            raise ValueError("array context of the two arguments must match")

        if ary1.array_context is None:
            raise ValueError("cannot transport frozen arrays")

        actx = ary1.array_context

        @memoize_in(actx, (CenterGranularityConnection, "interleave"))
        def prg():
            from arraycontext import make_loopy_program
            t_unit = make_loopy_program(
                    "{[iel, idof]: 0 <= iel < nelements and 0 <= idof < nunit_dofs}",
                    """
                    result[iel, 2*idof] = ary1[iel, idof]
                    result[iel, 2*idof + 1] = ary2[iel, idof]
                    """, [
                        lp.GlobalArg("ary1", shape="(nelements, nunit_dofs)"),
                        lp.GlobalArg("ary2", shape="(nelements, nunit_dofs)"),
                        lp.GlobalArg("result", shape="(nelements, 2*nunit_dofs)"),
                        ...
                        ],
                    name="interleave")

            from meshmode.transform_metadata import (
                ConcurrentDOFInameTag,
                ConcurrentElementInameTag,
            )
            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        discr = self.from_discr
        results: list[Array] = []
        for grp, subary1, subary2 in zip(discr.groups, ary1, ary2, strict=True):
            if subary1.dtype != subary2.dtype:
                raise ValueError("dtype mismatch in inputs: "
                    f"'{subary1.dtype.name}' and '{subary2.dtype.name}'")

            assert subary1.shape[0] == grp.nelements
            assert subary1.shape == subary2.shape

            result = actx.call_loopy(
                    prg(),
                    ary1=subary1, ary2=subary2,
                    nelements=subary1.shape[0],
                    nunit_dofs=subary1.shape[1])["result"]
            results.append(result)

        return DOFArray(actx, tuple(results))

    @override
    def __call__(self, arys: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        r"""
        :param arys: a pair of :class:`~arraycontext.ArrayContainer`-like
            classes. Can also be a single element, in which case it is
            interleaved with itself. This function vectorizes over all the
            :class:`~meshmode.dof_array.DOFArray` leaves of the container.

        :returns: an interleaved :class:`~arraycontext.ArrayContainer`.
            If *arys* was a pair of arrays :math:`(x, y)`, they are
            interleaved as :math:`[x_1, y_1, x_2, y_2, \ddots, x_n, y_n]`.
        """
        if isinstance(arys, list | tuple):
            ary1, ary2 = cast("Sequence[ArrayOrContainerOrScalarT]", arys)
        else:
            ary1, ary2 = arys, arys

        if type(ary1) is not type(ary2):
            raise TypeError("cannot interleave arrays of different types: "
                    f"'{type(ary1).__name__}' and '{type(ary2).__name__}'")

        from meshmode.dof_array import rec_multimap_dof_array_container
        return rec_multimap_dof_array_container(
                self._interleave_dof_arrays,
                ary1, ary2)

# }}}


# {{{ dof connection

class DOFConnection(DiscretizationConnection):
    """An interpolation operation for converting a DOF vector between
    different DOF types, as described by
    :class:`~pytential.symbolic.dof_desc.DOFDescriptor`.

    .. attribute:: connections

        A list of :class:`~meshmode.discretization.connection.DiscretizationConnection`s
        used to transport from the given source to the target.

    .. attribute:: from_dd

        A :class:`~pytential.symbolic.dof_desc.DOFDescriptor` for the
        DOF type of the incoming array.

    .. attribute:: to_dd

        A :class:`~pytential.symbolic.dof_desc.DOFDescriptor` for the
        DOF type of the outgoing array.

    .. attribute:: from_discr
    .. attribute:: to_discr

    .. automethod:: __call__
    """

    from_dd: DOFDescriptor
    to_dd: DOFDescriptor
    connections: tuple[DiscretizationConnection, ...]

    def __init__(self,
                 connections: Sequence[DiscretizationConnection],
                 from_dd: DOFDescriptorLike | None = None,
                 to_dd: DOFDescriptorLike | None = None) -> None:
        self.from_dd = dof_desc.as_dofdesc(from_dd)
        self.to_dd = dof_desc.as_dofdesc(to_dd)
        self.connections = tuple(connections)

        if not self.connections:
            raise ValueError(
                "no connections given (use 'IdentityDiscretizationConnection')")

        from meshmode.discretization.connection import DiscretizationConnection
        for conn in self.connections:
            if not isinstance(conn, DiscretizationConnection):
                raise ValueError(f"unsupported connection type: {type(conn)}")

        from_discr = self.connections[0].from_discr
        to_discr = self.connections[-1].to_discr
        super().__init__(from_discr, to_discr, is_surjective=False)

    @override
    def __call__(self, ary: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        for conn in self.connections:
            ary = conn(ary)

        return ary


def connection_from_dds(places: GeometryCollection,
                        from_dd: DOFDescriptorLike,
                        to_dd: DOFDescriptorLike) -> DiscretizationConnection:
    """
    :arg places: a :class:`~pytential.collection.GeometryCollection`
        or an argument taken by its constructor.
    :arg from_dd: a descriptor for the incoming degrees of freedom. This
        can be a :class:`~pytential.symbolic.dof_desc.DOFDescriptor`
        or an identifier that can be transformed into one by
        :func:`~pytential.symbolic.dof_desc.as_dofdesc`.
    :arg to_dd: a descriptor for the outgoing degrees of freedom.

    :return: a :class:`DOFConnection` transporting between the two
        kinds of DOF vectors.
    """

    from_dd = dof_desc.as_dofdesc(from_dd)
    to_dd = dof_desc.as_dofdesc(to_dd)

    from pytential import GeometryCollection
    if not isinstance(places, GeometryCollection):
        places = GeometryCollection(places)

    lpot = places.get_geometry(from_dd.geometry)
    from_discr = places.get_discretization(from_dd.geometry, from_dd.discr_stage)
    to_discr = places.get_discretization(to_dd.geometry, to_dd.discr_stage)

    assert isinstance(from_discr, Discretization)
    assert isinstance(to_discr, Discretization)

    if from_dd.geometry != to_dd.geometry:
        raise ValueError("cannot interpolate between different geometries")

    if from_dd.granularity is not dof_desc.GRANULARITY_NODE:
        raise ValueError("can only interpolate from `GRANULARITY_NODE`")

    connections: list[DiscretizationConnection] = []
    if from_dd.discr_stage is not to_dd.discr_stage:
        from pytential.qbx import QBXLayerPotentialSource
        if not isinstance(lpot, QBXLayerPotentialSource):
            raise ValueError("can only interpolate on a "
                    "`QBXLayerPotentialSource`")

        if to_dd.discr_stage is not dof_desc.QBX_SOURCE_QUAD_STAGE2:
            # TODO: can probably extend this to project from a QUAD_STAGE2
            # using L2ProjectionInverseDiscretizationConnection
            raise ValueError("can only interpolate to `QBX_SOURCE_QUAD_STAGE2`")

        # FIXME: would be nice if these were ordered by themselves
        stage_name_to_index_map = {
                None: 0,
                dof_desc.QBX_SOURCE_STAGE1: 1,
                dof_desc.QBX_SOURCE_STAGE2: 2,
                dof_desc.QBX_SOURCE_QUAD_STAGE2: 3
                }
        stage_index_to_name_map = {
                i: name for name, i in stage_name_to_index_map.items()}

        from_stage = stage_name_to_index_map[from_dd.discr_stage]
        to_stage = stage_name_to_index_map[to_dd.discr_stage]

        for istage in range(from_stage, to_stage):
            conn = places._get_conn_from_cache(
                    from_dd.geometry,
                    stage_index_to_name_map[istage],
                    stage_index_to_name_map[istage + 1])
            connections.append(conn)

    if from_dd.granularity is not to_dd.granularity:
        if to_dd.granularity is dof_desc.GRANULARITY_NODE:
            pass
        elif to_dd.granularity is dof_desc.GRANULARITY_CENTER:
            connections.append(CenterGranularityConnection(to_discr))
        elif to_dd.granularity is dof_desc.GRANULARITY_ELEMENT:
            raise ValueError("Creating a connection to element granularity "
                    "is not allowed. Use Elementwise{Max,Min,Sum}.")
        else:
            raise ValueError(f"invalid to_dd granularity: {to_dd.granularity}")

    if from_dd.granularity is not to_dd.granularity:
        if connections:
            conn = DOFConnection(connections, from_dd=from_dd, to_dd=to_dd)
        else:
            conn = IdentityDiscretizationConnection(from_discr)
    else:
        from meshmode.discretization.connection import ChainedDiscretizationConnection
        conn = ChainedDiscretizationConnection(connections,
                from_discr=from_discr)

    return conn

# }}}

# vim: foldmethod=marker
