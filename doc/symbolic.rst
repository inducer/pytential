Symbolic operator representation
================================

Based on :mod:`pymbolic`.

DOF Description
---------------

.. automodule:: pytential.symbolic.dof_desc

Basic objects
-------------

.. automodule:: pytential.symbolic.primitives

Binding an operator to a discretization
---------------------------------------

.. currentmodule:: pytential

.. autofunction:: bind

PDE operators
-------------

Scalar PDEs
^^^^^^^^^^^

.. automodule:: pytential.symbolic.pde.scalar

Maxwell's equations
^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.pde.maxwell

Elasticity equations
^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.elasticity

Stokes' equations
^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.stokes

Scalar Beltrami equations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.pde.beltrami

Rewriting expressions with IntGs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.pde.system_utils

Internal affairs
----------------

Mappers
^^^^^^^

.. automodule:: pytential.symbolic.mappers

How a symbolic operator gets executed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytential.symbolic.execution

.. automodule:: pytential.symbolic.compiler

Rewriting expressions with IntGs internals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytential.symbolic.pde.system_utils.convert_target_transformation_to_source
.. automethod:: pytential.symbolic.pde.system_utils.convert_int_g_to_base

