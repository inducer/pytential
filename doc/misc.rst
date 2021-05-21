Installation and Usage
======================

Installing :mod:`pytential`
---------------------------

This set of instructions is intended for 64-bit Linux and macOS computers.

#.  Make sure your system has the basics to build software.

    On Debian derivatives (Ubuntu and many more),
    installing ``build-essential`` should do the trick.

    On macOS, run ``xcode-select --install`` to install build tools.

    Everywhere else, just making sure you have the ``g++`` package should be
    enough.

#.  Install `miniforge <https://github.com/conda-forge/miniforge>`_::

        curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
        bash ./Miniforge3-*.sh

#.  ``export CONDA=/WHERE/YOU/INSTALLED/miniforge3``

    If you accepted the default location, this should work:

    ``export CONDA=$HOME/miniforge3``

#.  ``$CONDA/bin/conda create -n inteq``

#.  ``source $CONDA/bin/activate inteq``

Then, on Linux:

#.  ``conda install cython git pip pocl islpy pyopencl sympy pyfmmlib pytest``

#.  Type the following command::

        hash -r; for i in pymbolic cgen genpy gmsh_interop modepy pyvisfile loopy boxtree sumpy arraycontext meshmode pytential; do python -m pip install --editable "git+https://github.com/inducer/$i#egg=$i"; done

And on macOS:

#.  ``conda install compilers cython git pip pocl islpy pyopencl sympy pyfmmlib pytest``

#.  Type the following command::

        hash -r; for i in pymbolic cgen genpy gmsh_interop modepy pyvisfile loopy boxtree sumpy arraycontext meshmode pytential;do CC=clang python -m pip install --editable "git+https://github.com/inducer/$i#egg=$i"; done

.. note::

    In each case, you may leave out the ``--editable`` flag if you would not like
    a checkout of the source code.

Next time you want to use :mod:`pytential`, just run the following command::

    source /WHERE/YOU/INSTALLED/miniforge3/bin/activate inteq

You may also like to add this to a startup file (like :file:`$HOME/.bashrc`) or create an alias for it.

After this, you should be able to run the `tests <https://github.com/inducer/pytential/tree/master/test>`__
or `examples <https://github.com/inducer/pytential/tree/master/examples>`__.
For example, you should be able to download `this file <https://github.com/inducer/pytential/blob/master/examples/helmholtz-dirichlet.py>`__
and run it as::

    python helmholtz-dirichlet.py

Troubleshooting the Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

/usr/bin/ld: cannot find -lstdc++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try::

    sudo apt-get install libstdc++-6-dev

to install the missing C++ development package.

Assertion 'error == 0'
~~~~~~~~~~~~~~~~~~~~~~~

If you get::

    /opt/conda/conda-bld/home_1484016200338/work/pocl-0.13/lib/CL/devices/common.c:108:
    llvm_codegen: Assertion 'error == 0 ' failed. Aborted (core dumped)

then you're likely out of memory.

Logging
-------

Logging output for scripts that use :mod:`pytential` may be controlled on a
per-module basis through the environment variables ``PYTENTIAL_LOG_`` +
*log_level*. The variable for the desired level should be set to a
colon-separated list of module names. Example usage::

    PYTENTIAL_LOG_DEBUG=pytential:loopy PYTENTIAL_LOG_INFO=boxtree python test.py

This sets the logging level of :mod:`pytential` and :mod:`loopy` to
``logging.DEBUG``, and the logging level of :mod:`boxtree` to
``logging.INFO``.

Note: This feature is incompatible with :func:`logging.basicConfig()`.

User-visible Changes
====================

Version 2013.1
--------------
.. note::

    This version is currently under development. You can get snapshots from
    Pytential's `git repository <https://github.com/inducer/pytential>`_

* Initial release.

.. _license:

License
=======

:mod:`pytential` is licensed to you under the MIT/X Consortium license:

Copyright (c) 2012-13 Andreas Klöckner

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Frequently Asked Questions
==========================

The FAQ is maintained collaboratively on the
`Wiki FAQ page <https://wiki.tiker.net/Pytential/FrequentlyAskedQuestions>`_.

Acknowledgments
===============

Andreas Klöckner's work on :mod:`pytential` was supported in part by

* US Navy ONR grant number N00014-14-1-0117
* the US National Science Foundation under grant numbers DMS-1418961 and CCF-1524433.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.  The
views and opinions expressed herein do not necessarily reflect those of the
funding agencies.

Cross-References to Other Documentation
=======================================

.. currentmodule:: numpy

.. class:: int8

    See :class:`numpy.generic`.
