pytential: 2D/3D Layer Potential Evaluation
===========================================

.. image:: https://gitlab.tiker.net/inducer/pytential/badges/master/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pytential/commits/master
.. image:: https://github.com/inducer/pytential/workflows/CI/badge.svg?branch=master&event=push
    :alt: Github Build Status
    :target: https://github.com/inducer/pytential/actions?query=branch%3Amaster+workflow%3ACI+event%3Apush
.. image:: https://badge.fury.io/py/pytential.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/pytential/

pytential helps you accurately evaluate layer
potentials (and, sooner or later, volume potentials).
It also knows how to set up meshes and solve integral
equations.

See `here <https://documen.tician.de/pytential/misc.html#installing-pytential>`_
for easy, self-contained installation instructions for Linux and macOS.

It relies on

* `numpy <http://pypi.org/project/numpy>`_ for arrays
* `boxtree <http://pypi.org/project/boxtree>`_ for FMM tree building
* `sumpy <http://pypi.org/project/sumpy>`_ for expansions and analytical routines
* `modepy <http://pypi.org/project/modepy>`_ for modes and nodes on simplices
* `meshmode <http://pypi.org/project/meshmode>`_ for high order discretizations
* `loopy <http://pypi.org/project/loopy>`_ for fast array operations
* `pytest <http://pypi.org/project/pytest>`_ for automated testing

and, indirectly,

* `PyOpenCL <http://pypi.org/project/pyopencl>`_ as computational infrastructure

.. image:: https://badge.fury.io/py/pytential.png
    :target: http://pypi.org/project/pytential

Resources:

* `installation instructions <https://documen.tician.de/pytential/misc.html#installing-pytential>`_
* `documentation <http://documen.tician.de/pytential>`_
* `wiki home page <http://wiki.tiker.net/Pytential>`_
* `source code via git <http://github.com/inducer/pytential>`_
