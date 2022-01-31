Symbolic operator representation
================================

Based on :mod:`pymbolic`.

Basic objects
-------------

.. automodule:: pytential.symbolic.primitives

Binding an operator to a discretization
---------------------------------------

.. currentmodule:: pytential

.. autoclass:: GeometryCollection

.. autofunction:: bind

PDE operators
-------------

Scalar PDEs
^^^^^^^^^^^

.. automodule:: pytential.symbolic.pde.scalar

Maxwell's equations
^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.pde.maxwell

Stokes' equations
^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.stokes

Scalar Beltrami equations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.pde.beltrami

Internal affairs
----------------

Mappers
^^^^^^^

.. automodule:: pytential.symbolic.mappers

How a symbolic operator gets executed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.execution

.. automodule:: pytential.symbolic.compiler
