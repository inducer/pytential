__copyright__ = """
Copyright (C) 2020 Matt Wala
Copyright (C) 2023 University of Illinois Board of Trustees
"""

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

import sys


def sort_arrays_together(*arys, key=None):
    """Sort a sequence of arrays by considering them
    as an array of sequences using the given sorting key

    :param key: a function that takes in a tuple of values
                and returns a value to compare.
    """
    return zip(*sorted(zip(*arys, strict=True), key=key), strict=True)


def pytest_teardown_function():
    from pyopencl.tools import clear_first_arg_caches
    clear_first_arg_caches()

    from sympy.core.cache import clear_cache
    clear_cache()

    import sumpy
    sumpy.code_cache.clear_in_mem_cache()

    from loopy import clear_in_mem_caches
    clear_in_mem_caches()

    import gc
    gc.collect()

    if sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)


# vim: foldmethod=marker
