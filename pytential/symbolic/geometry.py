from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2019 Alexandru Fikl"

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
import pyopencl as cl

from pytools import Record, memoize_method

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: GeometryCollection

.. automethod:: refine_geometry_collection
"""


# {{{ geometry collection

class GeometryCollection(object):
    """A mapping from symbolic identifiers ("place IDs", typically strings)
    to 'geometries', where a geometry can be a
    :class:`pytential.source.PotentialSource`
    or a :class:`pytential.target.TargetBase`.
    This class is meant to hold a specific combination of sources and targets
    serve to host caches of information derived from them, e.g. FMM trees
    of subsets of them, as well as related common subexpressions such as
    metric terms.

    .. automethod:: get_discretization
    .. automethod:: get_geometry
    .. automethod:: copy

    .. method:: get_cache
    """

    def __init__(self, places, auto_where=None):
        """
        :arg places: a scalar, tuple of or mapping of symbolic names to
            geometry objects. Supported objects are
            :class:`~pytential.source.PotentialSource`,
            :class:`~potential.target.TargetBase` and
            :class:`~meshmode.discretization.Discretization`.
        :arg auto_where: location identifier for each geometry object, used
            to denote specific discretizations, e.g. in the case where
            *places* is a :class:`~pytential.source.LayerPotentialSourceBase`.
            By default, we assume
            :class:`~pytential.symbolic.primitives.DEFAULT_SOURCE` and
            :class:`~pytential.symbolic.primitives.DEFAULT_TARGET` for
            sources and targets, respectively.
        """

        from pytential import sym
        from pytential.target import TargetBase
        from pytential.source import PotentialSource
        from pytential.qbx import QBXLayerPotentialSource
        from meshmode.discretization import Discretization

        # {{{ define default source and target descriptors

        if isinstance(auto_where, (list, tuple)):
            auto_source, auto_target = auto_where
        else:
            auto_source, auto_target = auto_where, None

        if auto_source is None:
            auto_source = sym.DEFAULT_SOURCE
        if auto_target is None:
            auto_target = sym.DEFAULT_TARGET

        auto_source = sym.as_dofdesc(auto_source)
        auto_target = sym.as_dofdesc(auto_target)

        # }}}

        # {{{ construct dict

        self.places = {}
        self.caches = {}

        if isinstance(places, QBXLayerPotentialSource):
            self.places[auto_source.geometry] = places
            auto_target = auto_source
        elif isinstance(places, TargetBase):
            self.places[auto_target.geometry] = places
            auto_source = auto_target
        if isinstance(places, (Discretization, PotentialSource)):
            self.places[auto_source.geometry] = places
            self.places[auto_target.geometry] = places
        elif isinstance(places, tuple):
            source_discr, target_discr = places
            self.places[auto_source.geometry] = source_discr
            self.places[auto_target.geometry] = target_discr
        else:
            self.places = places.copy()

        self.auto_where = (auto_source, auto_target)

        for p in six.itervalues(self.places):
            if not isinstance(p, (PotentialSource, TargetBase, Discretization)):
                raise TypeError("Must pass discretization, targets or "
                        "layer potential sources as 'places'.")

        # }}}

    @property
    def auto_source(self):
        return self.auto_where[0]

    @property
    def auto_target(self):
        return self.auto_where[1]

    @property
    @memoize_method
    def ambient_dim(self):
        from pytools import single_valued
        ambient_dim = [p.ambient_dim for p in six.itervalues(self.places)]
        return single_valued(ambient_dim)

    def _refined_discretization_stage(self, lpot, dofdesc, refiner=None):
        if lpot._disable_refinement:
            return lpot.density_discr

        from pytential import sym
        if dofdesc.discr_stage is None:
            dofdesc = dofdesc.to_stage1()

        cache = self.get_cache('qbx_refined_discrs')
        key = (dofdesc.geometry, dofdesc.discr_stage)
        if key in cache:
            return cache[key]

        if refiner is None:
            refiner = _make_qbx_refiner(self, dofdesc.geometry)

        def _rec_refine(queue, dd):
            cache = self.get_cache('qbx_refined_discrs')
            key = (dd.geometry, dd.discr_stage)
            if key in cache:
                return cache[key]

            if dd.discr_stage == sym.QBX_SOURCE_STAGE1:
                method = getattr(refiner, 'refine_for_stage1')
                prev_discr_stage = None
            elif dd.discr_stage == sym.QBX_SOURCE_STAGE2:
                method = getattr(refiner, 'refine_for_stage2')
                prev_discr_stage = sym.QBX_SOURCE_STAGE1
            elif dd.discr_stage == sym.QBX_SOURCE_QUAD_STAGE2:
                method = getattr(refiner, 'refine_for_quad_stage2')
                prev_discr_stage = sym.QBX_SOURCE_STAGE2
            else:
                raise ValueError('unknown discr stage: {}'.format(dd.discr_stage))

            discr, conn = method(self, dd,
                    lpot.refiner_code_container.get_wrangler(queue))
            cache[key] = discr

            cache = self.get_cache('qbx_refined_connections')
            key = (dd.geometry, prev_discr_stage, dd.discr_stage)
            cache[key] = conn

            return discr

        with cl.CommandQueue(lpot.cl_context) as queue:
            return _rec_refine(queue, dofdesc)

    def get_connection(self, from_dd, to_dd):
        from pytential import sym
        from_dd = sym.as_dofdesc(from_dd)
        to_dd = sym.as_dofdesc(to_dd)

        if from_dd.geometry != to_dd.geometry:
            raise KeyError('no connections between different geometries')

        lpot = self.get_geometry(from_dd)
        if from_dd.discr_stage is not None:
            self._refined_discretization_stage(lpot, from_dd)
        if to_dd.discr_stage is not None:
            self._refined_discretization_stage(lpot, to_dd)

        key = (from_dd.geometry, from_dd.discr_stage, to_dd.discr_stage)
        cache = self.get_cache('qbx_refined_connections')
        if key in cache:
            return cache[key]
        else:
            raise KeyError('connection not in the collection')

    def get_discretization(self, dofdesc):
        """
        :arg dofdesc: a :class:`~pytential.symbolic.primitives.DOFDescriptor`
            specifying the desired discretization.

        :return: a geometry object in the collection corresponding to the
            key *dofdesc*. If it is a
            :class:`~pytential.source.LayerPotentialSourceBase`, we look for
            the corresponding :class:`~meshmode.discretization.Discretization`
            in its attributes instead.
        """
        from pytential import sym
        dofdesc = sym.as_dofdesc(dofdesc)
        key = (dofdesc.geometry, dofdesc.discr_stage)

        if key in self.places:
            discr = self.places[key]
        elif dofdesc.geometry in self.places:
            discr = self.places[dofdesc.geometry]
        else:
            raise KeyError('discretization not in the collection: {}'.format(
                dofdesc.geometry))

        from pytential.qbx import QBXLayerPotentialSource
        from pytential.source import LayerPotentialSourceBase

        if isinstance(discr, QBXLayerPotentialSource):
            return self._refined_discretization_stage(discr, dofdesc)
        elif isinstance(discr, LayerPotentialSourceBase):
            return discr.density_discr
        else:
            return discr

    def get_geometry(self, dofdesc):
        from pytential import sym
        dofdesc = sym.as_dofdesc(dofdesc)
        return self.places[dofdesc.geometry]

    def copy(self, places=None, auto_where=None):
        if places is None:
            places = {}

        new_places = self.places.copy()
        new_places.update(places)

        return GeometryCollection(
                new_places,
                auto_where=(self.auto_where
                    if auto_where is None else auto_where))

    def get_cache(self, name):
        return self.caches.setdefault(name, {})

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, repr(self.places))

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, str(self.places))

# }}}


# {{{ refinement

class QBXGeometryRefinerData(Record):
    """Holds refinement parameters and forwards calls to low-level methods
    in :module:`pytential.qbx.refinement`.

    .. attribute:: target_order
    .. attribute:: kernel_length_scale
    .. attribute:: scaled_max_curvature_threshold
    .. attribute:: expansion_disturbance_tolerance
    .. attribute:: force_stage2_uniform_refinement_rounds
    .. attribute:: maxiter

    .. attribute:: debug
    .. attribute:: visualize

    .. method:: refine_for_stage1
    .. method:: refine_for_stage2
    .. method:: refine_for_quad_stage2

    """

    @property
    @memoize_method
    def _group_factory(self):
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        return InterpolatoryQuadratureSimplexGroupFactory(self.target_order)

    def refine_for_stage1(self, places, source_name, wrangler):
        from pytential.qbx.refinement import refine_qbx_stage1
        return refine_qbx_stage1(places, source_name, wrangler,
                self._group_factory,
                kernel_length_scale=self.kernel_length_scale,
                scaled_max_curvature_threshold=(
                    self.scaled_max_curvature_threshold),
                expansion_disturbance_tolerance=(
                    self.expansion_disturbance_tolerance),
                maxiter=self.maxiter,
                debug=self.debug,
                visualize=self.visualize)

    def refine_for_stage2(self, places, source_name, wrangler):
        from pytential.qbx.refinement import refine_qbx_stage2
        return refine_qbx_stage2(places, source_name, wrangler,
                self._group_factory,
                force_stage2_uniform_refinement_rounds=(
                    self.force_stage2_uniform_refinement_rounds),
                expansion_disturbance_tolerance=(
                    self.expansion_disturbance_tolerance),
                maxiter=self.maxiter,
                debug=self.debug,
                visualize=self.visualize)

    def refine_for_quad_stage2(self, places, source_name, wrangler):
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                QuadratureSimplexGroupFactory

        lpot = places.get_geometry(source_name)
        discr = places.get_discretization(source_name.to_stage2())

        quad_stage2_density_discr = Discretization(lpot.cl_context,
                discr.mesh,
                QuadratureSimplexGroupFactory(lpot.fine_order),
                lpot.real_dtype)

        from meshmode.discretization.connection import make_same_mesh_connection
        to_quad_stage2_conn = make_same_mesh_connection(
                quad_stage2_density_discr, discr)

        return quad_stage2_density_discr, to_quad_stage2_conn


def _make_qbx_refiner(places, source_name,
        target_order=None, kernel_length_scale=None,
        scaled_max_curvature_threshold=None,
        expansion_disturbance_tolerance=None,
        force_stage2_uniform_refinement_rounds=None,
        maxiter=None, debug=None, visualize=False):
    cache = places.get_cache('qbx_refiner_data')
    if source_name in cache:
        return cache[source_name]

    lpot = places.get_geometry(source_name)
    if target_order is None:
        target_order = lpot.density_discr.groups[0].order

    if expansion_disturbance_tolerance is None:
        expansion_disturbance_tolerance = 0.025

    if force_stage2_uniform_refinement_rounds is None:
        force_stage2_uniform_refinement_rounds = 0

    if debug is None:
        debug = lpot.debug

    if maxiter is None:
        maxiter = 10

    r = QBXGeometryRefinerData(
            target_order=target_order,
            kernel_length_scale=kernel_length_scale,
            scaled_max_curvature_threshold=(
                scaled_max_curvature_threshold),
            expansion_disturbance_tolerance=(
                expansion_disturbance_tolerance),
            force_stage2_uniform_refinement_rounds=(
                force_stage2_uniform_refinement_rounds),
            maxiter=maxiter, debug=debug, visualize=visualize)
    cache[source_name] = r

    return r


def refine_geometry_collection(places,
        refine_for_global_qbx=False,
        target_order=None, kernel_length_scale=None,
        scaled_max_curvature_threshold=None,
        expansion_disturbance_tolerance=None,
        force_stage2_uniform_refinement_rounds=None,
        maxiter=None, debug=None, visualize=False):
    from pytential import sym
    from pytential.qbx import QBXLayerPotentialSource

    if refine_for_global_qbx:
        discr_stage = sym.QBX_SOURCE_QUAD_STAGE2
    else:
        discr_stage = sym.QBX_SOURCE_STAGE1

    for geometry in places.places:
        lpot = places.get_geometry(geometry)
        if not isinstance(lpot, QBXLayerPotentialSource):
            continue

        dd = sym.as_dofdesc(geometry).copy(discr_stage=discr_stage)
        refiner = _make_qbx_refiner(places, dd.geometry,
                target_order=target_order,
                kernel_length_scale=kernel_length_scale,
                scaled_max_curvature_threshold=(
                    scaled_max_curvature_threshold),
                expansion_disturbance_tolerance=(
                    expansion_disturbance_tolerance),
                force_stage2_uniform_refinement_rounds=(
                    force_stage2_uniform_refinement_rounds),
                maxiter=maxiter, debug=debug, visualize=visualize)

        places._refined_discretization_stage(lpot, dd, refiner=refiner)

    return places

# }}}

# vim: foldmethod=marker
