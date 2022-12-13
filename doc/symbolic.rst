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

Internals
^^^^^^^^^

.. autoclass:: pytential.symbolic.elasticity.ElasticityWrapperYoshida
.. autoclass:: pytential.symbolic.elasticity.ElasticityDoubleLayerWrapperYoshida

Rewriting expressions with ``IntG``\ s
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Rewriting expressions with ``IntG``\ s internals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytential.symbolic.pde.system_utils.convert_target_transformation_to_source
.. automethod:: pytential.symbolic.pde.system_utils.rewrite_int_g_using_base_kernel

