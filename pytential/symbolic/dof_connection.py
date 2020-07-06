# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

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

import six
from meshmode.array_context import PyOpenCLArrayContext  # noqa
from meshmode.dof_array import DOFArray
import numpy as np
from pytools import memoize_in

import loopy as lp


__doc__ = """

Connections
-----------

.. autoclass:: GranularityConnection
.. autoclass:: CenterGranularityConnection
.. autoclass:: DOFConnection
.. autofunction:: connection_from_dds

"""


# {{{ granularity connections

class GranularityConnection(object):
    """Abstract interface for transporting a DOF between different levels
    of granularity.

    .. attribute:: discr
    .. automethod:: __call__
    """

    def __init__(self, discr):
        self.discr = discr

    @property
    def from_discr(self):
        return self.discr

    @property
    def to_discr(self):
        return self.discr

    @property
    def array_context(self):
        return self.discr._setup_actx

    def __call__(self, ary):
        raise NotImplementedError()


class CenterGranularityConnection(GranularityConnection):
    """A :class:`GranularityConnection` used to transport from node data
    (:class:`~pytential.symbolic.primitives.GRANULARITY_NODE`) to expansion
    centers (:class:`~pytential.symbolic.primitives.GRANULARITY_CENTER`).

    .. attribute:: discr
    .. automethod:: __call__
    """

    def __init__(self, discr):
        super(CenterGranularityConnection, self).__init__(discr)

    def _interleave_dof_arrays(self, ary1, ary2):
        if not isinstance(ary1, DOFArray) or not isinstance(ary2, DOFArray):
            raise TypeError("non-array passed to connection")

        @memoize_in(self.array_context,
                 (CenterGranularityConnection, "interleave"))
        def prg():
            from meshmode.array_context import make_loopy_program
            return make_loopy_program(
                    """{[iel, idof]: 0<=iel<nelements and 0<=idof<nunit_dofs}""",
                    """
                    dst[iel, 2*idof] = src1[iel, idof]
                    dst[iel, 2*idof + 1] = src2[iel, idof]
                    """,
                    [
                        lp.GlobalArg("src1", shape="(nelements, nunit_dofs)"),
                        lp.GlobalArg("src2", shape="(nelements, nunit_dofs)"),
                        lp.GlobalArg("dst", shape="(nelements, 2*nunit_dofs)"),
                        "...",
                        ],
                    name="interleave")

        results = []
        for grp, src1, src2 in zip(self.discr.groups, ary1, ary2):
            if src1.dtype != src2.dtype:
                raise ValueError("dtype mismatch in inputs")
            result = self.array_context.empty(
                    (grp.nelements, 2 * grp.nunit_dofs), dtype=src1.dtype)
            self.array_context.call_loopy(
                    prg(), src1=src1, src2=src2, dst=result,
                    nelements=grp.nelements, nunit_dofs=grp.nunit_dofs)
            results.append(result)
        return DOFArray.from_list(self.array_context, results)

    def __call__(self, arys):
        r"""
        :arg arys: either a single :class:`~meshmode.dof_array.DOFArray`
            or a list/tuple with exactly 2 entries that are both
            :class:`~meshmode.dof_array.DOFArray`\ s.
            Additionally, this function vectorizes over object arrays of
            :class:`~meshmode.dof_array.DOFArrays`\ s.

        :return: an interleaved array or list of :class:`pyopencl.array.Array`s.
            If *vecs* was a pair of arrays :math:`(x, y)`, they are
            interleaved as :math:`[x_1, y_1, x_2, y_2, \ddots, x_n, y_n]`.
            A single array is simply interleaved with itself.

        """
        if isinstance(arys, np.ndarray):
            arys = (arys, arys)
        if isinstance(arys, (list, tuple)):
            assert len(arys) == 2
        else:
            raise ValueError("cannot interleave arrays")

        if isinstance(arys[0], DOFArray):
            return self._interleave_dof_arrays(*arys)
        else:
            from pytools.obj_array import obj_array_vectorize_n_args
            return obj_array_vectorize_n_args(self._interleave_dof_arrays, *arys)

# }}}


# {{{ dof connection

class DOFConnection(object):
    """An interpolation operation for converting a DOF vector between
    different DOF types, as described by
    :class:`~pytential.symbolic.primitives.DOFDescriptor`.

    .. attribute:: connections

        A list of
        :class:`~meshmode.discretization.connection.DiscretizationConnection`s
        and :class:`GranularityConnection`s used to transport from the given
        source to the target.

    .. attribute:: from_dd

        A :class:`~pytential.symbolic.primitives.DOFDescriptor` for the
        DOF type of the incoming array.

    .. attribute:: to_dd

        A :class:`~pytential.symbolic.primitives.DOFDescriptor` for the
        DOF type of the outgoing array.

    .. attribute:: from_discr
    .. attribute:: to_discr

    .. automethod:: __call__
    """

    def __init__(self, connections, from_dd=None, to_dd=None):
        self.from_dd = from_dd
        self.to_dd = to_dd
        self.connections = connections

        from meshmode.discretization.connection import DiscretizationConnection
        for conn in self.connections:
            if not isinstance(conn,
                    (DiscretizationConnection, GranularityConnection)):
                raise ValueError('unsupported connection type: {}'
                        .format(type(conn)))

        if self.connections:
            self.from_discr = self.connections[0].from_discr
            self.to_discr = self.connections[-1].to_discr

    def __call__(self, ary):
        for conn in self.connections:
            ary = conn(ary)

        return ary


def connection_from_dds(places, from_dd, to_dd):
    """
    :arg places: a :class:`~pytential.symbolic.execution.GeometryCollection`
        or an argument taken by its constructor.
    :arg from_dd: a descriptor for the incoming degrees of freedom. This
        can be a :class:`~pytential.symbolic.primitives.DOFDescriptor`
        or an identifier that can be transformed into one by
        :func:`~pytential.symbolic.primitives.as_dofdesc`.
    :arg to_dd: a descriptor for the outgoing degrees of freedom.

    :return: a :class:`DOFConnection` transporting between the two
        kinds of DOF vectors.
    """

    from pytential import sym
    from_dd = sym.as_dofdesc(from_dd)
    to_dd = sym.as_dofdesc(to_dd)

    from pytential import GeometryCollection
    if not isinstance(places, GeometryCollection):
        places = GeometryCollection(places)

    lpot = places.get_geometry(from_dd.geometry)
    from_discr = places.get_discretization(from_dd.geometry, from_dd.discr_stage)
    to_discr = places.get_discretization(to_dd.geometry, to_dd.discr_stage)

    if from_dd.geometry != to_dd.geometry:
        raise ValueError("cannot interpolate between different geometries")

    if from_dd.granularity is not sym.GRANULARITY_NODE:
        raise ValueError("can only interpolate from `GRANULARITY_NODE`")

    connections = []
    if from_dd.discr_stage is not to_dd.discr_stage:
        from pytential.qbx import QBXLayerPotentialSource
        if not isinstance(lpot, QBXLayerPotentialSource):
            raise ValueError("can only interpolate on a "
                    "`QBXLayerPotentialSource`")

        if to_dd.discr_stage is not sym.QBX_SOURCE_QUAD_STAGE2:
            # TODO: can probably extend this to project from a QUAD_STAGE2
            # using L2ProjectionInverseDiscretizationConnection
            raise ValueError("can only interpolate to "
                "`QBX_SOURCE_QUAD_STAGE2`")

        # FIXME: would be nice if these were ordered by themselves
        stage_name_to_index_map = {
                None: 0,
                sym.QBX_SOURCE_STAGE1: 1,
                sym.QBX_SOURCE_STAGE2: 2,
                sym.QBX_SOURCE_QUAD_STAGE2: 3
                }
        stage_index_to_name_map = dict([(i, name) for name, i in
                    six.iteritems(stage_name_to_index_map)])

        from_stage = stage_name_to_index_map[from_dd.discr_stage]
        to_stage = stage_name_to_index_map[to_dd.discr_stage]

        for istage in range(from_stage, to_stage):
            conn = places._get_conn_from_cache(from_dd.geometry,
                    stage_index_to_name_map[istage],
                    stage_index_to_name_map[istage + 1])
            connections.append(conn)

    if from_dd.granularity is not to_dd.granularity:
        if to_dd.granularity is sym.GRANULARITY_NODE:
            pass
        elif to_dd.granularity is sym.GRANULARITY_CENTER:
            connections.append(CenterGranularityConnection(to_discr))
        elif to_dd.granularity is sym.GRANULARITY_ELEMENT:
            raise ValueError("Creating a connection to element granularity "
                    "is not allowed. Use Elementwise{Max,Min,Sum}.")
        else:
            raise ValueError("invalid to_dd granularity: %s" % to_dd.granularity)

    if from_dd.granularity is not to_dd.granularity:
        conn = DOFConnection(connections, from_dd=from_dd, to_dd=to_dd)
    else:
        from meshmode.discretization.connection import \
                ChainedDiscretizationConnection
        conn = ChainedDiscretizationConnection(connections,
                from_discr=from_discr)

    return conn

# }}}

# vim: foldmethod=marker
