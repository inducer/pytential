Discretizations
===============

QBX discretization
------------------

To compute a layer potential as an an end user, create a
:class:`meshmode.discretization.Discretization`
with a :class:`InterpolatoryQuadratureSimplexGroupFactory`
as a discretization for the density.

Then create :class:`pytential.qbx.QBXLayerPotentialSource`,
:func:`pytential.bind` a layer potential operator to it,
and you can start computing.

.. automodule:: pytential.qbx

Unregularized discretization
----------------------------

.. automodule:: pytential.unregularized

Sources
-------

.. automodule:: pytential.source

Targets
-------

.. automodule:: pytential.target
