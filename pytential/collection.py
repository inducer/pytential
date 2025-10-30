from __future__ import annotations


__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
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

from collections.abc import Hashable, Mapping
from typing import Any

from constantdict import constantdict
from typing_extensions import override

from meshmode.discretization import Discretization
from meshmode.discretization.connection.direct import DiscretizationConnection

import pytential.symbolic.primitives as sym
from pytential.qbx import QBXLayerPotentialSource
from pytential.source import (
    LayerPotentialSourceBase,
    PointPotentialSource,
    PotentialSource,
)
from pytential.symbolic.dof_desc import (
    DiscretizationStage,
    DOFDescriptor,
    DOFDescriptorLike,
    GeometryId,
)
from pytential.target import PointsTarget, TargetBase


__doc__ = """
.. class:: AutoWhereLike

    Types accepted for ``auto_where`` arguments to aid in determining where an
    expression is evaluated.

.. class:: GeometryLike

    Types accepted by the :class:`GeometryCollection`.

.. autoclass:: GeometryCollection
.. autofunction:: add_geometry_to_collection
"""

GeometryLike = TargetBase | PotentialSource | Discretization
AutoWhereLike = DOFDescriptorLike | tuple[DOFDescriptorLike, DOFDescriptorLike]


class NotADiscretizationError(TypeError):
    pass


def _is_valid_identifier(name: str) -> bool:
    import keyword
    return name.isidentifier() and not keyword.iskeyword(name)


class _GeometryCollectionDiscretizationCacheKey:
    """Serves as a unique key for the discretization cache in
    :meth:`GeometryCollection._get_cache`.
    """


class _GeometryCollectionConnectionCacheKey:
    """Serves as a unique key for the connection cache in
    :meth:`GeometryCollection._get_cache`.
    """


# {{{ geometry collection

class GeometryCollection:
    """A mapping from symbolic identifiers ("place IDs", typically strings)
    to 'geometries', where a geometry can be a
    :class:`~pytential.source.PotentialSource`, a
    :class:`~pytential.target.TargetBase` or a
    :class:`~meshmode.discretization.Discretization`.

    This class is meant to hold a specific combination of sources and targets
    serve to host caches of information derived from them, e.g. FMM trees
    of subsets of them, as well as related common subexpressions such as
    metric terms.

    Refinement of :class:`pytential.qbx.QBXLayerPotentialSource` entries is
    performed on demand, i.e. on calls to :meth:`get_discretization` with
    a specific *discr_stage*. To perform refinement explicitly, call
    :func:`pytential.qbx.refinement.refine_geometry_collection`,
    which allows more customization of the refinement process through
    parameters.

    .. automethod:: __init__

    .. attribute:: auto_source

        Default :class:`~pytential.symbolic.dof_desc.DOFDescriptor` for the
        source geometry.

    .. attribute:: auto_target

        Default :class:`~pytential.symbolic.dof_desc.DOFDescriptor` for the
        target geometry.

    .. automethod:: get_geometry
    .. automethod:: get_target_or_discretization
    .. automethod:: get_discretization
    .. automethod:: get_connection

    .. automethod:: copy
    .. automethod:: merge

    """

    ambient_dim: int
    places: Mapping[GeometryId, GeometryLike]
    auto_where: tuple[DOFDescriptor, DOFDescriptor]

    _caches: dict[Hashable, Any]

    def __init__(self,
            places: (
                GeometryLike
                | tuple[GeometryLike, GeometryLike]
                | Mapping[GeometryId, GeometryLike]
                ),
            auto_where: AutoWhereLike | None = None) -> None:
        r"""
        :arg places: a scalar, tuple of or mapping of symbolic names to
            geometry objects. Supported objects are
            :class:`~pytential.source.PotentialSource`,
            :class:`~pytential.target.TargetBase` and
            :class:`~meshmode.discretization.Discretization`. If this is
            a mapping, the keys that are strings must be valid Python identifiers.
            The tuple should contain only two entries, denoting the source and
            target geometries for layer potential evaluation, identified by
            *auto_where*.

        :arg auto_where: a single or a tuple of two
            :class:`~pytential.symbolic.dof_desc.DOFDescriptor`\ s, or values
            that can be converted to one using
            :func:`~pytential.symbolic.dof_desc.as_dofdesc`. The two
            descriptors are used to define the default source and target
            geometries for layer potential evaluations.
            By default, they are set to unspecified, but unique objects.
        """

        # {{{ construct dict

        from pytential.symbolic.execution import _prepare_auto_where
        auto_source, auto_target = _prepare_auto_where(auto_where)

        places_dict: Mapping[GeometryId, GeometryLike]
        if isinstance(places, QBXLayerPotentialSource):
            places_dict = {auto_source.geometry: places}
            auto_target = auto_source
        elif isinstance(places, TargetBase):
            places_dict = {auto_target.geometry: places}
            auto_source = auto_target
        if isinstance(places, Discretization | PotentialSource):
            places_dict = {
                auto_source.geometry: places,
                auto_target.geometry: places
            }
        elif isinstance(places, tuple):
            source_discr, target_discr = places
            places_dict = {
                auto_source.geometry: source_discr,
                auto_target.geometry: target_discr
            }
        else:
            assert isinstance(places, Mapping)
            places_dict = places

        # }}}

        # {{{ validate

        # check auto_where
        if auto_source.geometry not in places_dict:
            raise ValueError("'auto_where' source geometry is not in the "
                f"collection: '{auto_source.geometry}'")

        if auto_target.geometry not in places_dict:
            raise ValueError("'auto_where' target geometry is not in the "
                f"collection: '{auto_target.geometry}'")

        # check allowed identifiers
        for name in places_dict:
            if not isinstance(name, str):
                continue
            if not _is_valid_identifier(name):
                raise ValueError(f"'{name}' is not a valid identifier")

        # check allowed types
        for p in places_dict.values():
            if not isinstance(p, PotentialSource | TargetBase | Discretization):
                raise TypeError(
                    "Values in 'places' must be discretization, targets "
                    f"or layer potential sources, got '{type(p).__name__}'")

        # check ambient_dim
        from pytools import is_single_valued

        ambient_dims = [p.ambient_dim for p in places_dict.values()]
        if not is_single_valued(ambient_dims):
            raise RuntimeError("All 'places' must have the same ambient dimension.")

        # }}}

        self.ambient_dim = ambient_dims[0]
        self.places = constantdict(places_dict)
        self.auto_where = (auto_source, auto_target)

        self._caches = {}

    @property
    def auto_source(self) -> sym.DOFDescriptor:
        return self.auto_where[0]

    @property
    def auto_target(self) -> sym.DOFDescriptor:
        return self.auto_where[1]

    # {{{ cache handling

    def _get_cache(self, name: Hashable) -> dict[Hashable, Any]:
        return self._caches.setdefault(name, {})

    def _get_discr_from_cache(self,
                              geometry: GeometryId,
                              discr_stage: DiscretizationStage) -> Discretization:
        cache = self._get_cache(_GeometryCollectionDiscretizationCacheKey)
        key = (geometry, discr_stage)

        if key not in cache:
            raise KeyError(
                    f"cached discretization does not exist on '{geometry}' "
                    f"for stage '{discr_stage}'")

        result = cache[key]
        assert isinstance(result, Discretization)

        return result

    def _add_discr_to_cache(self,
                            discr: Discretization,
                            geometry: GeometryId,
                            discr_stage: DiscretizationStage) -> None:
        cache = self._get_cache(_GeometryCollectionDiscretizationCacheKey)
        key = (geometry, discr_stage)

        if key in cache:
            raise RuntimeError("trying to overwrite the discretization cache of "
                    f"'{geometry}' for stage '{discr_stage}'")

        cache[key] = discr

    def _get_conn_from_cache(self,
                             geometry: GeometryId,
                             from_stage: DiscretizationStage | None,
                             to_stage: DiscretizationStage | None
                             ) -> DiscretizationConnection:
        cache = self._get_cache(_GeometryCollectionConnectionCacheKey)
        key = (geometry, from_stage, to_stage)

        if key not in cache:
            raise KeyError("cached connection does not exist on "
                    f"'{geometry}' from stage '{from_stage}' to '{to_stage}'")

        result = cache[key]
        assert isinstance(result, DiscretizationConnection)

        return result

    def _add_conn_to_cache(self,
                           conn: DiscretizationConnection,
                           geometry: GeometryId,
                           from_stage: DiscretizationStage,
                           to_stage: DiscretizationStage) -> None:
        cache = self._get_cache(_GeometryCollectionConnectionCacheKey)
        key = (geometry, from_stage, to_stage)

        if key in cache:
            raise RuntimeError("trying to overwrite the connection cache of "
                    f"'{geometry}' from stage '{from_stage}' to '{to_stage}'")

        cache[key] = conn

    def _get_qbx_discretization(self,
                                geometry: GeometryId,
                                discr_stage: DiscretizationStage) -> Discretization:
        lpot_source = self.get_geometry(geometry)
        assert isinstance(lpot_source, LayerPotentialSourceBase)

        try:
            discr = self._get_discr_from_cache(geometry, discr_stage)
        except KeyError:
            dofdesc = sym.DOFDescriptor(geometry, discr_stage)

            from pytential.qbx.refinement import refiner_code_container
            wrangler = refiner_code_container(lpot_source._setup_actx).get_wrangler()

            from pytential.qbx.refinement import _refine_for_global_qbx
            # NOTE: this adds the required discretizations to the cache
            _refine_for_global_qbx(self, dofdesc, wrangler, _copy_collection=False)
            discr = self._get_discr_from_cache(geometry, discr_stage)

        return discr

    # }}}

    def get_connection(self,
                       from_dd: DOFDescriptorLike,
                       to_dd: DOFDescriptorLike) -> DiscretizationConnection:
        """Construct a connection from *from_dd* to *to_dd* geometries.

        :returns: an object compatible with the
            :class:`~meshmode.discretization.connection.DiscretizationConnection`
            interface.
        """

        from pytential.symbolic.dof_connection import connection_from_dds
        return connection_from_dds(self, from_dd, to_dd)

    def _get_discretization_or_geometry(
            self, geometry: GeometryId,
            discr_stage: DiscretizationStage | None = None
            ) -> GeometryLike:
        """
        If a specific QBX stage discretization is requested, refinement is
        performed on demand and cached for subsequent calls.
        """
        if discr_stage is None:
            discr_stage = sym.QBX_SOURCE_STAGE1
        result = self.get_geometry(geometry)

        if isinstance(result, QBXLayerPotentialSource):
            return self._get_qbx_discretization(geometry, discr_stage)
        elif isinstance(result, LayerPotentialSourceBase):
            return result.density_discr
        else:
            return result

    def get_source_or_discretization(
            self, geometry: GeometryId,
            discr_stage: DiscretizationStage | None = None
            ) -> Discretization | PointPotentialSource:
        """
        If a specific QBX stage discretization is requested, refinement is
        performed on demand and cached for subsequent calls.
        """
        result = self._get_discretization_or_geometry(geometry, discr_stage)

        if not isinstance(result, (Discretization, PointPotentialSource)):
            raise TypeError(f"'{geometry}' denotes neither target "
                            "nor discretization")
        return result

    def get_target_or_discretization(
            self, geometry: GeometryId,
            discr_stage: DiscretizationStage | None = None
            ) -> Discretization | TargetBase | PointPotentialSource:
        """
        If a specific QBX stage discretization is requested, refinement is
        performed on demand and cached for subsequent calls.
        """
        result = self._get_discretization_or_geometry(geometry, discr_stage)

        if not isinstance(result, (Discretization, TargetBase, PointPotentialSource)):
            raise TypeError(f"'{geometry}' denotes neither target "
                            "nor discretization")
        return result

    def get_discretization(
            self, geometry: GeometryId,
            discr_stage: DiscretizationStage | None = None
            ) -> Discretization:
        """
        If a specific QBX stage discretization is requested, refinement is
        performed on demand and cached for subsequent calls.
        """
        result = self.get_target_or_discretization(geometry, discr_stage)
        if not isinstance(result, Discretization):
            raise NotADiscretizationError(str(geometry))
        return result

    def get_geometry(self, geometry: GeometryId) -> GeometryLike:
        """
        :arg geometry: the identifier of the geometry in the collection.
        """

        try:
            return self.places[geometry]
        except KeyError:
            raise KeyError(f"geometry not in the collection: '{geometry}'") from None

    def copy(
            self,
            places: Mapping[GeometryId, GeometryLike] | None = None,
            auto_where: AutoWhereLike | None = None,
            ) -> GeometryCollection:
        """Get a shallow copy of the geometry collection."""
        return type(self)(
                places=self.places if places is None else places,
                auto_where=self.auto_where if auto_where is None else auto_where)

    def merge(
            self,
            places: GeometryCollection | Mapping[GeometryId, GeometryLike],
            ) -> GeometryCollection:
        """Merges two geometry collections and returns the new collection.

        :arg places: a mapping or :class:`GeometryCollection` to
            merge with the current collection. If it is empty, a copy of the
            current collection is returned.
        """

        new_places = dict(self.places)
        if places:
            new_places.update(
                places.places if isinstance(places, GeometryCollection)
                else places)

        return self.copy(places=new_places)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.places!r})"

    @override
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.places!r})"


# }}}


# {{{ add_geometry_to_collection

def add_geometry_to_collection(
        places: GeometryCollection,
        geometries: Mapping[GeometryId, GeometryLike]) -> GeometryCollection:
    """Adds a mapping of geometries to an existing collection.

    This function is similar to :meth:`GeometryCollection.merge`, but it makes
    an attempt to maintain the caches in *places*. In particular, a shallow
    copy of the following are passed to the new collection

    * Any cached discretizations from
      :func:`~pytential.qbx.refinement.refine_geometry_collection`.
    * Any cached expressions marked with `cse_scope.DISCRETIZATION` from the
      evaluation mapper.

    This allows adding new targets to the collection without recomputing the
    source geometry data.
    """
    for key, geometry in geometries.items():
        if key in places.places:
            raise ValueError(f"geometry '{key}' already in the collection")

        if not isinstance(geometry, PointsTarget | PointPotentialSource):
            raise TypeError(
                    f"Cannot add a geometry of type '{type(geometry).__name__}' "
                    "to the existing collection. Construct a new collection "
                    "instead.")

    from pytential.symbolic.execution import EvaluationMapperCSECacheKey

    known_cache_keys = (
            EvaluationMapperCSECacheKey,
            _GeometryCollectionConnectionCacheKey,
            _GeometryCollectionDiscretizationCacheKey,
            )

    # copy over all the caches
    new_places = places.merge(geometries)
    for key in places._caches:
        if key not in known_cache_keys:
            from warnings import warn
            warn(f"GeometryCollection cache key '{key}' is not known and will "
                 "be dropped in the new collection.",
                 stacklevel=2)
            continue

        new_cache = new_places._get_cache(key)
        new_cache.update(places._get_cache(key))

    return new_places

# }}}
