Linear Algebra Routines
=======================

In the linear algebra parts of :mod:`pytential`, the following naming
scheme is used:

* ``block`` refers to a piece of a vector operator, e.g. the :math:`S_{xx}`
  component of the Stokeslet.
* ``cluster`` refers to a piece of a ``block`` as used by the recursive
  proxy-based skeletonization of the direct solver algorithms. Clusters
  are represented by a :class:`~pytential.linalg.TargetAndSourceClusterList`.

GMRES
-----

.. automodule:: pytential.linalg.gmres

.. _direct_solver:

Hierarchical Direct Solver
--------------------------

.. warning::

    All the classes and routines in this module are experimental and the
    API can change at any point.

.. automodule:: pytential.linalg.proxy
.. automodule:: pytential.linalg.utils

Internal Functionality
----------------------

.. automodule:: pytential.linalg.direct_solver_symbolic

.. vim: sw=4:tw=75:fdm=marker
