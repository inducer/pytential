Discretizations
===============

To create a discretization as an end user, this function, along with perhaps
:class:`pytential.discretization.PointsTarget`, is likely all an end user will
need:

.. currentmodule:: pytential.discretization.qbx

.. autofunction:: make_upsampling_qbx_discr

The rest of this chapter documents the interface exposed by each discretization
component.

See :mod:`meshmode.discretization` and :mod:`meshmode.discretization.poly_element`
for base classes used by these discretizations.

QBX discretization
------------------

.. automodule:: pytential.discretization.qbx

Upsampling discretization wrapper
---------------------------------

.. automodule:: pytential.discretization.upsampling

Target discretizations
----------------------

.. automodule:: pytential.discretization.target
