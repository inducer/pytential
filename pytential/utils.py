__copyright__ = """
Copyright (C) 2020 Matt Wala
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

import numpy as np

from arraycontext import ArrayContext


def flatten_if_needed(actx: ArrayContext, ary: np.ndarray):
    from pytools.obj_array import obj_array_vectorize_n_args
    from meshmode.dof_array import DOFArray, flatten
    from arraycontext import thaw

    if (isinstance(ary, np.ndarray)
            and ary.dtype.char == "O"
            and not isinstance(ary, DOFArray)):
        return obj_array_vectorize_n_args(flatten_if_needed, actx, ary)

    if not isinstance(ary, DOFArray):
        return ary

    if ary.array_context is None:
        ary = thaw(ary, actx)

    return flatten(ary)


def unflatten_from_numpy(actx, discr, ary):
    from pytools.obj_array import obj_array_vectorize
    from meshmode.dof_array import unflatten

    ary = obj_array_vectorize(actx.from_numpy, ary)
    if discr is None:
        return ary
    else:
        return unflatten(actx, discr, ary)


def flatten_to_numpy(actx, ary):
    result = flatten_if_needed(actx, ary)

    from pytools.obj_array import obj_array_vectorize
    return obj_array_vectorize(actx.to_numpy, result)


def sort_arrays_together(*arys, key=None):
    """Sort a sequence of arrays by considering them
    as an array of sequences using the given sorting key

    :param key: a function that takes in a tuple of values
                and returns a value to compare.
    """
    return zip(*sorted([x for x in zip(*arys)], key=key))

# vim: foldmethod=marker
