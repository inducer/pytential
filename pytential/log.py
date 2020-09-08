__copyright__ = "Copyright (C) 2017 Matt Wala"

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

import logging

#
# This file is based heavily on
# http://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
#


# {{{ constants

class colors:  # noqa
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


class term_seq:  # noqa
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"


LEVEL_TO_COLOR = {
    "WARNING": colors.YELLOW,
    "INFO": colors.CYAN,
    "DEBUG": colors.WHITE,
    "CRITICAL": colors.YELLOW,
    "ERROR": colors.RED
}


PYTENTIAL_LOG_FORMAT = (
        "[$BOLD%(name)s$RESET][%(levelname)s]  %(message)s "
        "($BOLD%(filename)s$RESET:%(lineno)d)"
)

# }}}


# {{{ formatting

class ColoredFormatter(logging.Formatter):

    @staticmethod
    def make_formatter_message(message, use_color):
        if use_color:
            message = (message
                    .replace("$RESET", term_seq.RESET_SEQ)
                    .replace("$BOLD", term_seq.BOLD_SEQ))
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
        return message

    def __init__(self, use_color=True):
        logging.Formatter.__init__(
                self, self.make_formatter_message(PYTENTIAL_LOG_FORMAT, use_color))
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in LEVEL_TO_COLOR:
            levelname_color = (
                    (term_seq.COLOR_SEQ % (30 + LEVEL_TO_COLOR[levelname]))
                    + levelname + term_seq.RESET_SEQ)
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

# }}}


def set_up_logging(modules, level, use_color=True):
    """
    :arg modules: A list of modules for which logging output should be enabled
    """

    color_formatter = ColoredFormatter(use_color)

    handler = logging.StreamHandler()
    handler.setFormatter(color_formatter)

    for module in modules:
        logger = logging.getLogger(module)
        logger.addHandler(handler)
        logger.setLevel(level)
