Installation
============

This command should install :mod:`pytential`::

    pip install pytential

You may need to run this with :command:`sudo`.
If you don't already have `pip <https://pypi.python.org/pypi/pip>`_,
run this beforehand::

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py

For a more manual installation, download the source, unpack it,
and say::

    python setup.py install

In addition, you need to have :mod:`numpy` installed.

Logging
=======

Logging output for scripts that use :mod:`pytential` may be controlled through
the environment variables ``PYTENTIAL_LOG_`` + *log_level*. The variable for the
desired level should be set to a colon-separated list of module names. Example
usage::

    PYTENTIAL_LOG_DEBUG=pytential:loopy PYTENTIAL_LOG_INFO=boxtree python test.py

This sets the logging level of :mod:`pytential` and :mod:`loopy` to
:attr:`logging.DEBUG`, and the logging level of :mod:`boxtree` to
:attr:`logging.INFO`.

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
`Wiki FAQ page <http://wiki.tiker.net/Pytential/FrequentlyAskedQuestions>`_.

Acknowledgments
===============

Andreas Klöckner's work on :mod:`pytential` was supported in part by

* US Navy ONR grant number N00014-14-1-0117
* the US National Science Foundation under grant numbers DMS-1418961 and CCF-1524433.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.  The
views and opinions expressed herein do not necessarily reflect those of the
funding agencies.
