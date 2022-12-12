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

from typing import Dict, Hashable, Optional, Tuple, Union

from pytential import sym
from pytential.symbolic.execution import EvaluationMapperCSECacheKey
from pytential.symbolic.dof_desc import DOFDescriptorLike, DiscretizationStages

from pytential.target import TargetBase
from pytential.source import PotentialSource
from pytential.qbx import QBXLayerPotentialSource
from meshmode.discretization import Discretization

__doc__ = """
.. class:: GeometryLike

    Types accepted by the :class:`GeometryCollection`.

.. autoclass:: GeometryCollection

.. autofunction:: add_geometry_to_collection
"""

GeometryLike = Union[TargetBase, PotentialSource, Discretization]
AutoWhereLike = Union[
        "DOFDescriptorLike",
        Tuple["DOFDescriptorLike", "DOFDescriptorLike"]
        ]


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


_KNOWN_GEOMETRY_COLLECTION_CACHE_KEYS = (
        EvaluationMapperCSECacheKey,
        _GeometryCollectionConnectionCacheKey,
        _GeometryCollectionDiscretizationCacheKey,
        )


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
    .. automethod:: get_discretization
    .. automethod:: get_connection

    .. automethod:: copy
    .. automethod:: merge

    """

    def __init__(self,
            places: Union[
                "GeometryLike",
                Tuple["GeometryLike", "GeometryLike"],
                Dict[Hashable, "GeometryLike"]
                ],
            auto_where: Optional[AutoWhereLike] = None) -> None:
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

        places_dict = {}

        from pytential.symbolic.execution import _prepare_auto_where
        auto_source, auto_target = _prepare_auto_where(auto_where)
        if isinstance(places, QBXLayerPotentialSource):
            places_dict[auto_source.geometry] = places
            auto_target = auto_source
        elif isinstance(places, TargetBase):
            places_dict[auto_target.geometry] = places
            auto_source = auto_target
        if isinstance(places, (Discretization, PotentialSource)):
            places_dict[auto_source.geometry] = places
            places_dict[auto_target.geometry] = places
        elif isinstance(places, tuple):
            source_discr, target_discr = places
            places_dict[auto_source.geometry] = source_discr
            places_dict[auto_target.geometry] = target_discr
        else:
            places_dict = places

        import immutables
        self.places = immutables.Map(places_dict)
        self.auto_where = (auto_source, auto_target)

        self._caches = {}

        # }}}

        # {{{ validate

        # check auto_where
        if auto_source.geometry not in self.places:
            raise ValueError("'auto_where' source geometry is not in the "
                f"collection: '{auto_source.geometry}'")

        if auto_target.geometry not in self.places:
            raise ValueError("'auto_where' target geometry is not in the "
                f"collection: '{auto_target.geometry}'")

        # check allowed identifiers
        for name in self.places:
            if not isinstance(name, str):
                continue
            if not _is_valid_identifier(name):
                raise ValueError(f"'{name}' is not a valid identifier")

        # check allowed types
        for p in self.places.values():
            if not isinstance(p, (PotentialSource, TargetBase, Discretization)):
                raise TypeError(
                    "Values in 'places' must be discretization, targets "
                    f"or layer potential sources, got '{type(p).__name__}'")

        # check ambient_dim
        from pytools import is_single_valued
        ambient_dims = [p.ambient_dim for p in self.places.values()]
        if not is_single_valued(ambient_dims):
            raise RuntimeError("All 'places' must have the same ambient dimension.")

        self.ambient_dim = ambient_dims[0]

        # }}}

    @property
    def auto_source(self) -> sym.DOFDescriptor:
        return self.auto_where[0]

    @property
    def auto_target(self) -> sym.DOFDescriptor:
        return self.auto_where[1]

    # {{{ cache handling

    def _get_cache(self, name):
        return self._caches.setdefault(name, {})

    def _get_discr_from_cache(self, geometry, discr_stage):
        cache = self._get_cache(_GeometryCollectionDiscretizationCacheKey)
        key = (geometry, discr_stage)

        if key not in cache:
            raise KeyError(
                    f"cached discretization does not exist on '{geometry}' "
                    f"for stage '{discr_stage}'")

        return cache[key]

    def _add_discr_to_cache(self, discr, geometry, discr_stage):
        cache = self._get_cache(_GeometryCollectionDiscretizationCacheKey)
        key = (geometry, discr_stage)

        if key in cache:
            raise RuntimeError("trying to overwrite the discretization cache of "
                    f"'{geometry}' for stage '{discr_stage}'")

        cache[key] = discr

    def _get_conn_from_cache(self, geometry, from_stage, to_stage):
        cache = self._get_cache(_GeometryCollectionConnectionCacheKey)
        key = (geometry, from_stage, to_stage)

        if key not in cache:
            raise KeyError("cached connection does not exist on "
                    f"'{geometry}' from stage '{from_stage}' to '{to_stage}'")

        return cache[key]

    def _add_conn_to_cache(self, conn, geometry, from_stage, to_stage):
        cache = self._get_cache(_GeometryCollectionConnectionCacheKey)
        key = (geometry, from_stage, to_stage)

        if key in cache:
            raise RuntimeError("trying to overwrite the connection cache of "
                    f"'{geometry}' from stage '{from_stage}' to '{to_stage}'")

        cache[key] = conn

    def _get_qbx_discretization(self, geometry, discr_stage):
        lpot_source = self.get_geometry(geometry)

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
            from_dd: "DOFDescriptorLike",
            to_dd: "DOFDescriptorLike"):
        """Construct a connection from *from_dd* to *to_dd* geometries.

        :returns: an object compatible with the
            :class:`~meshmode.discretization.connection.DiscretizationConnection`
            interface.
        """

        from pytential.symbolic.dof_connection import connection_from_dds
        return connection_from_dds(self, from_dd, to_dd)

    def get_discretization(
            self, geometry: Hashable,
            discr_stage: Optional["DiscretizationStages"] = None
            ) -> "GeometryLike":
        """Get the geometry or discretization in the collection.

        If a specific QBX stage discretization is requested, refinement is
        performed on demand and cached for subsequent calls.

        :arg geometry: the identifier of the geometry in the collection.
        :arg discr_stage: if the geometry is a
            :class:`~pytential.source.LayerPotentialSourceBase`, this denotes
            the QBX stage of the returned discretization. Can be one of
            :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_STAGE1` (default),
            :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_STAGE2` or
            :class:`~pytential.symbolic.dof_desc.QBX_SOURCE_QUAD_STAGE2`.

        :returns: a geometry object in the collection or a
            :class:`~meshmode.discretization.Discretization` corresponding to
            *discr_stage*.
        """
        if discr_stage is None:
            discr_stage = sym.QBX_SOURCE_STAGE1
        discr = self.get_geometry(geometry)

        from pytential.qbx import QBXLayerPotentialSource
        from pytential.source import LayerPotentialSourceBase

        if isinstance(discr, QBXLayerPotentialSource):
            return self._get_qbx_discretization(geometry, discr_stage)
        elif isinstance(discr, LayerPotentialSourceBase):
            return discr.density_discr
        else:
            return discr

    def get_geometry(self, geometry: Hashable) -> "GeometryLike":
        """
        :arg geometry: the identifier of the geometry in the collection.
        """

        try:
            return self.places[geometry]
        except KeyError:
            raise KeyError(f"geometry not in the collection: '{geometry}'")

    def copy(
            self,
            places: Optional[Dict[Hashable, "GeometryLike"]] = None,
            auto_where: Optional[AutoWhereLike] = None,
            ) -> "GeometryCollection":
        """Get a shallow copy of the geometry collection."""
        return type(self)(
                places=self.places if places is None else places,
                auto_where=self.auto_where if auto_where is None else auto_where)

    def merge(
            self,
            places: Union["GeometryCollection", Dict[Hashable, "GeometryLike"]],
            ) -> "GeometryCollection":
        """Merges two geometry collections and returns the new collection.

        :arg places: a :class:`dict` or :class:`GeometryCollection` to
            merge with the current collection. If it is empty, a copy of the
            current collection is returned.
        """

        new_places = self.places
        if places:
            if isinstance(places, GeometryCollection):
                places = places.places
            new_places = new_places.update(places)

        return self.copy(places=new_places)

    def __repr__(self):
        return f"{type(self).__name__}({self.places!r})"

    def __str__(self):
        return f"{type(self).__name__}({self.places!r})"


# }}}


# {{{ add_geometry_to_collection

def add_geometry_to_collection(
        places: GeometryCollection,
        geometries: Dict[Hashable, "GeometryLike"]) -> GeometryCollection:
    """Adds a :class:`dict` of geometries to an existing collection.

    This function is similar to :meth:`GeometryCollection.merge`, but it makes
    an attempt to maintain the caches in *places*. In particular, a shallow
    copy of the following are pased to the new collection

    * Any cached discretizations from
      :func:`~pytential.qbx.refinement.refine_geometry_collection`.
    * Any cached expressions marked with `cse_scope.DISCRETIZATION` from the
      evaluation mapper.

    This allows adding new targets to the collection without recomputing the
    source geometry data.
    """
    from pytential.source import PointPotentialSource
    from pytential.target import PointsTarget
    for key, geometry in geometries.items():
        if key in places.places:
            raise ValueError(f"geometry '{key}' already in the collection")

        if not isinstance(geometry, (PointsTarget, PointPotentialSource)):
            raise TypeError(
                    f"Cannot add a geometry of type '{type(geometry).__name__}' "
                    "to the existing collection. Construct a new collection "
                    "instead.")

    # copy over all the caches
    new_places = places.merge(geometries)
    for key in places._caches:
        if key not in _KNOWN_GEOMETRY_COLLECTION_CACHE_KEYS:
            from warnings import warn
            warn(f"GeometryCollection cache key '{key}' is not known and will "
                    "be dropped in the new collection.")
            continue

        new_cache = new_places._get_cache(key)
        new_cache.update(places._get_cache(key))

    return new_places

# }}}
