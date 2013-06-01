Discretizations
===============

To create a discretization as an end user, this function is likely all you
need:

.. currentmodule:: pytential.discretization.qbx

.. autofunction:: make_upsampling_qbx_discr

The rest of this chapter mostly documents interfaces intended for internal use.

Abstract interface
------------------

.. automodule:: pytential.discretization

.. autoclass:: Discretization

Composite polynomial discretization
-----------------------------------

.. automodule:: pytential.discretization.poly_element

.. autoclass:: PolynomialElementDiscretization

QBX discretization
------------------

.. automodule:: pytential.discretization.qbx

.. autoclass:: QBXDiscretization

Upsampling discretization wrapper
---------------------------------

.. automodule:: pytential.discretization.upsampling

.. autoclass:: UpsampleToSourceDiscretization
