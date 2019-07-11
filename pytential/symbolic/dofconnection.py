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
            assumptions="2*srclen = dstlen",
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.split_iname(knl, "i", 128,
                inner_tag="l.0", outer_tag="g.0")
        return knl

    def __call__(self, queue, vecs):
        r"""
        :arg vecs: a single :class:`pyopencl.array.Array` or a pair of
            arrays. Given a pair of arrays :math:`x` and :math:`y`, they are
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
    (:class:`~pytential.symbolic.primitives.GRANULARITY_NODE`) to expansion
    center (:class:`~pytential.symbolic.primitives.GRANULARITY_ELEMENT`).

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
            name="subsample_to_elements",
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            )

        knl = lp.split_iname(knl, "i", 128,
                inner_tag="l.0", outer_tag="g.0")
        return knl

    def __call__(self, queue, vec):
        from pytential.qbx.utils import mesh_el_view

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

    .. attribute:: lpot_source

        A :class:`~pytential.source.LayerPotentialSourceBase`.

    .. attribute:: source

        A :class:`~pytential.symbolic.primitives.DOFDescriptor` for the
        DOF type of the incoming array.

    .. attribute:: target

        A :class:`~pytential.symbolic.primitives.DOFDescriptor` for the
        DOF type of the outgoing array.

    .. attribute:: connections

        A list of
        :class:`~meshmode.discretization.connection.DiscretizationConnection`s
        and :class:`GranularityConnection`s used to transport from the given
        source to the target.

    .. automethod:: __call__
    """

    def __init__(self, places, source, target):
        from pytential import sym
        source = sym.as_dofdesc(source)
        target = sym.as_dofdesc(target)

        if not ((source.where == sym.DEFAULT_SOURCE
                and target.where == sym.DEFAULT_TARGET)
                or source.where == target.where):
            raise ValueError('cannot interpolate between different domains')
        if source.granularity != sym.GRANULARITY_NODE:
            raise ValueError('can only interpolate from `GRANULARITY_NODE`')

        from pytential.symbolic.execution import GeometryCollection
        if not isinstance(places, GeometryCollection):
            places = GeometryCollection(places)
        lpot_source = places[source]

        connections = []
        if target.discr != source.discr:
            if target.discr != sym.QBX_SOURCE_QUAD_STAGE2:
                # TODO: can probably extend this to project from a QUAD_STAGE2
                # using L2ProjectionInverseDiscretizationConnection
                raise RuntimeError("can only interpolate to "
                    "`QBX_SOURCE_QUAD_STAGE2`")

            if source.discr == sym.QBX_SOURCE_STAGE2:
                connections.append(
                        lpot_source.refined_interp_to_ovsmp_quad_connection)
            elif source.discr == sym.QBX_SOURCE_QUAD_STAGE2:
                pass
            else:
                connections.append(lpot_source.resampler)

        if target.granularity != source.granularity:
            discr = places.get_discretization(target)
            if target.granularity == sym.GRANULARITY_CENTER:
                connections.append(CenterGranularityConnection(discr))
            elif target.granularity == sym.GRANULARITY_ELEMENT:
                connections.append(ElementGranularityConnection(discr))

        self.lpot_source = lpot_source
        self.source = source
        self.target = target
        self.connections = connections

    def __call__(self, queue, vec):
        for conn in self.connections:
            vec = conn(queue, vec)

        return vec

# }}}

# vim: foldmethod=marker
