from __future__ import division
from __future__ import absolute_import

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

import pytential.symbolic.primitives as sym
from pytential.symbolic.execution import bind

from pytools import memoize_on_first_arg


import os

_fmm_logging = os.environ.get("PYTENTIAL_DEBUG_FMM")

if int(_fmm_logging):
    import logging

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    #The background is set with 40 plus the number of the color, and the foreground with 30

    #These are the sequences need to get colored ouput
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    def formatter_message(message, use_color = True):
        if use_color:
            message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
        return message

    COLORS = {
        'WARNING': YELLOW,
        'INFO': CYAN,
        'DEBUG': WHITE,
        'CRITICAL': YELLOW,
        'ERROR': RED
    }

    class ColoredFormatter(logging.Formatter):

        def __init__(self, msg, use_color = True):
            logging.Formatter.__init__(self, msg)
            self.use_color = use_color

        def format(self, record):
            levelname = record.levelname
            if self.use_color and levelname in COLORS:
                levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
                record.levelname = levelname_color
            return logging.Formatter.format(self, record)

    FORMAT = "[$BOLD%(name)s$RESET][%(levelname)s]  %(message)s " \
             "($BOLD%(filename)s$RESET:%(lineno)d)"
    COLOR_FORMAT = formatter_message(FORMAT, True)
    color_formatter = ColoredFormatter(COLOR_FORMAT)

    handler = logging.StreamHandler()
    handler.setFormatter(color_formatter)

    logger = logging.getLogger("pytential")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


@memoize_on_first_arg
def _integral_op(discr):
    from pytential import sym, bind
    return bind(discr,
            sym.integral(
                discr.ambient_dim, discr.dim, sym.var("integrand")))


def integral(discr, queue, x):
    return _integral_op(discr)(queue, integrand=x)


@memoize_on_first_arg
def _norm_op(discr, num_components):
    from pytential import sym, bind
    if num_components is not None:
        from pymbolic.primitives import make_sym_vector
        v = make_sym_vector("integrand", num_components)
        integrand = sym.real(np.dot(sym.conj(v), v))
    else:
        integrand = sym.abs(sym.var("integrand"))**2

    return bind(discr,
            sym.integral(discr.ambient_dim, discr.dim, integrand))


def norm(discr, queue, x):
    from pymbolic.geometric_algebra import MultiVector
    if isinstance(x, MultiVector):
        x = x.as_vector(np.object)

    num_components = None
    if isinstance(x, np.ndarray):
        num_components, = x.shape

    norm_op = _norm_op(discr, num_components)
    from math import sqrt
    return sqrt(norm_op(queue, integrand=x))


__all__ = ["sym", "bind"]
