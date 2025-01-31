pytential: 2D/3D Layer Potential Evaluation
===========================================

.. image:: https://gitlab.tiker.net/inducer/pytential/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pytential/commits/main
.. image:: https://github.com/inducer/pytential/actions/workflows/ci.yml/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/pytential/actions/workflows/ci.yml
.. image:: https://badge.fury.io/py/pytential.svg
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/pytential/

pytential helps you accurately evaluate layer potentials (and, sooner or later,
volume potentials). It also knows how to set up meshes and solve integral
equations.

See `here <https://documen.tician.de/pytential/misc.html#installing-pytential>`__
for easy, self-contained installation instructions for Linux and macOS.

It relies on

* `boxtree <https://pypi.org/project/boxtree>`__ for FMM tree building
* `sumpy <https://pypi.org/project/sumpy>`__ for expansions and analytical routines
* `modepy <https://pypi.org/project/modepy>`__ for modes and nodes on simplices
* `meshmode <https://pypi.org/project/meshmode>`__ for high order discretizations
* `loopy <https://pypi.org/project/loopy>`__ for fast array operations
* `pytest <https://pypi.org/project/pytest>`__ for automated testing

and, indirectly,

* `PyOpenCL <https://pypi.org/project/pyopencl>`__ as computational infrastructure

.. image:: https://badge.fury.io/py/pyopencl.svg
    :target: https://pypi.org/project/pyopencl

Resources:

* `installation instructions <https://documen.tician.de/pytential/misc.html#installing-pytential>`__
* `documentation <https://documen.tician.de/pytential>`__
* `wiki home page <https://wiki.tiker.net/Pytential>`__
* `source code via git <https://github.com/inducer/pytential>`__
