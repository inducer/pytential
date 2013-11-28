Discretizations
===============

To create a discretization as an end user, this function, along with perhaps
:class:`pytential.discretization.PointsTarget`, is likely all an end user will
need:

.. currentmodule:: pytential.discretization.qbx

.. autofunction:: make_upsampling_qbx_discr

The rest of this chapter documents the interface exposed by each discretization
component.

Abstract interface
------------------

.. automodule:: pytential.discretization

Composite polynomial discretization
-----------------------------------

.. automodule:: pytential.discretization.poly_element

QBX discretization
------------------

.. automodule:: pytential.discretization.qbx

Upsampling discretization wrapper
---------------------------------

.. automodule:: pytential.discretization.upsampling

Target discretizations
----------------------

.. automodule:: pytential.discretization.target
