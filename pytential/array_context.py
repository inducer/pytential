__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from meshmode.array_context import (
    PyOpenCLArrayContext as MeshmodePyOpenCLArrayContext)
from sumpy.array_context import (   # noqa: F401
    PyOpenCLArrayContext as SumpyPyOpenCLArrayContext,
    make_loopy_program)
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        register_pytest_array_context_factory)

__doc__ = """
Array Context
=============

.. autoclass:: PyOpenCLArrayContext
"""


# {{{ PyOpenCLArrayContext

class PyOpenCLArrayContext(SumpyPyOpenCLArrayContext):
    def transform_loopy_program(self, t_unit):
        # FIXME: this probably needs some proper logic
        return MeshmodePyOpenCLArrayContext.transform_loopy_program(self, t_unit)

# }}}


# {{{ pytest

def _acf():
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    return PyOpenCLArrayContext(queue, force_device_scalars=True)


class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext

    def __call__(self):
        # NOTE: prevent any cache explosions during testing!
        from sympy.core.cache import clear_cache
        clear_cache()

        return super().__call__()


register_pytest_array_context_factory(
    "pytential.pyopencl",
    PytestPyOpenCLArrayContextFactory)

# }}}
