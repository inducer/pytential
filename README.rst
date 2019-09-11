pytential: 2D/3D Layer Potential Evaluation
===========================================

.. image:: https://gitlab.tiker.net/inducer/pytential/badges/master/pipeline.svg
   :target: https://gitlab.tiker.net/inducer/pytential/commits/master
.. image:: https://badge.fury.io/py/pytential.png
    :target: http://pypi.python.org/pypi/pytential

pytential helps you accurately evaluate layer
potentials (and, sooner or later, volume potentials).
It also knows how to set up meshes and solve integral
equations.

See `here <https://documen.tician.de/pytential/misc.html#installing-pytential>`_
for easy, self-contained installation instructions for Linux and macOS.

It relies on

* `numpy <http://pypi.python.org/pypi/numpy>`_ for arrays
* `boxtree <http://pypi.python.org/pypi/boxtree>`_ for FMM tree building
* `sumpy <http://pypi.python.org/pypi/sumpy>`_ for expansions and analytical routines
* `modepy <http://pypi.python.org/pypi/modepy>`_ for modes and nodes on simplices
* `meshmode <http://pypi.python.org/pypi/meshmode>`_ for high order discretizations
* `loopy <http://pypi.python.org/pypi/loo.py>`_ for fast array operations
* `pytest <http://pypi.python.org/pypi/pytest>`_ for automated testing

and, indirectly,

* `PyOpenCL <http://pypi.python.org/pypi/pyopencl>`_ as computational infrastructure

.. image:: https://badge.fury.io/py/pytential.png
    :target: http://pypi.python.org/pypi/pytential

Resources:

* `installation instructions <https://documen.tician.de/pytential/misc.html#installing-pytential>`_
* `documentation <http://documen.tician.de/pytential>`_
* `wiki home page <http://wiki.tiker.net/Pytential>`_
* `source code via git <http://github.com/inducer/pytential>`_
