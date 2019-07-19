# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__copyright__ = """
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

import numpy as np

import pyopencl as cl
import pyopencl.array # noqa
from pytools import memoize

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


# {{{ granularity connections

def mesh_el_view(mesh, group_nr, global_array):
    """Return a view of *global_array* of shape
    ``(..., mesh.groups[group_nr].nelements)``
    where *global_array* is of shape ``(..., nelements)``,
    where *nelements* is the global (per-mesh) element count.
    """

    group = mesh.groups[group_nr]

    return global_array[
        ..., group.element_nr_base:group.element_nr_base + group.nelements] \
        .reshape(
            global_array.shape[:-1]
            + (group.nelements,))


class GranularityConnection(object):
    """Abstract interface for transporting a DOF between different levels
    of granularity.

    .. attribute:: discr
    .. automethod:: __call__
    """

    def __init__(self, discr):
        self.discr = discr

    def __call__(self, queue, vec):
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

    @memoize
    def kernel(self):
        knl = lp.make_kernel(
            "[srclen, dstlen] -> {[i]: 0 <= i < srclen}",
            """
            dst[2*i] = src1[i]
            dst[2*i + 1] = src2[i]
            """,
            [
                lp.GlobalArg("src1", shape="srclen"),
                lp.GlobalArg("src2", shape="srclen"),
                lp.GlobalArg("dst", shape="dstlen"),
                "..."
            ],
            name="node_interleaver_knl",
            assumptions="2*srclen = dstlen",
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.split_iname(knl, "i", 128,
                inner_tag="l.0", outer_tag="g.0")
        return knl

    def __call__(self, queue, vecs):
        r"""
        :arg vecs: a single :class:`pyopencl.array.Array` or a pair of arrays.
        :return: an interleaved array or list of :class:`pyopencl.array.Array`s.
            If *vecs* was a pair of arrays :math:`(x, y)`, they are
            interleaved as :math:`[x_1, y_1, x_2, y_2, \ddots, x_n, y_n]`.
            A single array is simply interleaved with itself.
        """

        if isinstance(vecs, cl.array.Array):
            vecs = [[vecs], [vecs]]
        elif isinstance(vecs, (list, tuple)):
            assert len(vecs) == 2
        else:
            raise ValueError('cannot interleave arrays')

        result = []
        for src1, src2 in zip(vecs[0], vecs[1]):
            if not isinstance(src1, cl.array.Array) \
                    or not isinstance(src2, cl.array.Array):
                raise TypeError('non-array passed to connection')

            if src1.shape != (self.discr.nnodes,) \
                    or src2.shape != (self.discr.nnodes,):
                raise ValueError('invalid shape of incoming array')

            axis = cl.array.empty(queue, 2 * len(src1), src1.dtype)
            self.kernel()(queue,
                    src1=src1, src2=src2, dst=axis)
            result.append(axis)

        return result[0] if len(result) == 1 else result


class ElementGranularityConnection(GranularityConnection):
    """A :class:`GranularityConnection` used to transport from node data
    (:class:`~pytential.symbolic.primitives.GRANULARITY_NODE`) to per-element
    data (:class:`~pytential.symbolic.primitives.GRANULARITY_ELEMENT`).

    .. attribute:: discr
    .. automethod:: __call__
    """

    def __init__(self, discr):
        super(ElementGranularityConnection, self).__init__(discr)

    @memoize
    def kernel(self):
        knl = lp.make_kernel(
            "{[i, k]: 0 <= i < nelements}",
            "result[i] = a[i, 0]",
            [
                lp.GlobalArg("a",
                    shape=("nelements", "nunit_nodes"), dtype=None),
                lp.ValueArg("nunit_nodes", dtype=np.int32),
                "..."
            ],
            name="node_element_subsample",
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.split_iname(knl, "i", 128,
                inner_tag="l.0", outer_tag="g.0")
        return knl

    def __call__(self, queue, vec):
        """
        :arg vec: a vector with degrees of freedom corresponding to
            nodes in :attr:`discr`.
        :return: an :class:`pyopencl.array.Array` of size
            ``(discr.mesh.nelements,)``. The reduction is performed by
            picking the first node in each element according to its
            group numbering.
        """

        if not isinstance(vec, cl.array.Array):
            raise TypeError('non-array passed to connection')

        if vec.shape != (self.discr.nnodes,):
            raise ValueError('invalid shape of incoming array')

        result = cl.array.empty(queue, self.discr.mesh.nelements, vec.dtype)
        for igrp, group in enumerate(self.discr.groups):
            self.kernel()(queue,
                a=group.view(vec),
                result=mesh_el_view(self.discr.mesh, igrp, result))

        return result

# }}}


# {{{ dof connection

class DOFConnection(object):
    """A class used to transport between DOF types.

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
            conn = self.connections[0]
            if isinstance(conn, DiscretizationConnection):
                self.from_discr = conn.from_discr
            elif isinstance(conn, GranularityConnection):
                self.from_discr = conn.discr

            conn = self.connections[-1]
            if isinstance(conn, DiscretizationConnection):
                self.to_discr = conn.to_discr
            elif isinstance(conn, GranularityConnection):
                self.to_discr = conn.discr

    def __call__(self, queue, vec):
        for conn in self.connections:
            vec = conn(queue, vec)

        return vec


def connection_from_dds(places, from_dd, to_dd):
    """
    :arg places: a :class:`~pytential.symbolic.execution.GeometryCollection`
        or an argument taken by its constructor.
    :arg from_dd: a descriptor for the incomming degrees of freedom. This
        can be a :class:`~pytential.symbolic.primitives.DOFDescriptor`
        or an identifier that can be transformed into one by
        :func:`~pytential.symbolic.primitives.as_dofdesc`.
    :arg to_dd: a descriptor for the outgoing degrees of freedom.

    :return: a :class:`DOFConnection` transporting between the two
        of DOFs.
    """

    from pytential import sym
    from_dd = sym.as_dofdesc(from_dd)
    to_dd = sym.as_dofdesc(to_dd)

    from pytential.symbolic.execution import GeometryCollection
    if not isinstance(places, GeometryCollection):
        places = GeometryCollection(places)
    from_discr = places[from_dd]

    if not ((from_dd.where in (sym.DEFAULT_SOURCE,)
            and to_dd.where in (sym.DEFAULT_SOURCE, sym.DEFAULT_TARGET))
            or from_dd.where == to_dd.where):
        raise ValueError("cannot interpolate between different domains")

    if from_dd.granularity != sym.GRANULARITY_NODE:
        raise ValueError("can only interpolate from `GRANULARITY_NODE`")

    from pytential.qbx import QBXLayerPotentialSource
    if (not isinstance(from_discr, QBXLayerPotentialSource)
            and from_dd.discr != to_dd.discr):
        raise ValueError("can only interpolate on a `QBXLayerPotentialSource`")

    connections = []
    if from_dd.discr != to_dd.discr:
        if to_dd.discr != sym.QBX_SOURCE_QUAD_STAGE2:
            # TODO: can probably extend this to project from a QUAD_STAGE2
            # using L2ProjectionInverseDiscretizationConnection
            raise RuntimeError("can only interpolate to "
                "`QBX_SOURCE_QUAD_STAGE2`")

        if from_dd.discr == sym.QBX_SOURCE_QUAD_STAGE2:
            pass
        elif from_dd.discr == sym.QBX_SOURCE_STAGE2:
            connections.append(
                    from_discr.refined_interp_to_ovsmp_quad_connection)
        else:
            connections.append(from_discr.resampler)

    if from_dd.granularity != to_dd.granularity:
        to_discr = places.get_discretization(to_dd)

        if to_dd.granularity == sym.GRANULARITY_NODE:
            pass
        elif to_dd.granularity == sym.GRANULARITY_CENTER:
            connections.append(CenterGranularityConnection(to_discr))
        elif to_dd.granularity == sym.GRANULARITY_ELEMENT:
            connections.append(ElementGranularityConnection(to_discr))

    return DOFConnection(connections, from_dd=from_dd, to_dd=to_dd)

# }}}

# vim: foldmethod=marker
