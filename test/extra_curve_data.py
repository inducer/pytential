from __future__ import division, absolute_import, print_function

__copyright__ = """Copyright (C) 2016 Matt Wala"""

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
import numpy.linalg as la


class Curve(object):

    def plot(self, npoints=50):
        import matplotlib.pyplot as plt
        x, y = self(np.linspace(0, 1, npoints))
        plt.plot(x, y, marker=".", lw=0)
        plt.axis("equal")
        plt.show()

    def __add__(self, other):
        return CompositeCurve(self, other)


class CompositeCurve(Curve):
    """
    Parametrization of two or more curves combined.
    """

    def __init__(self, *objs):
        curves = []
        for obj in objs:
            if isinstance(obj, CompositeCurve):
                curves.extend(obj.curves)
            else:
                curves.append(obj)
        self.curves = curves

    def __call__(self, ts):
        ranges = np.linspace(0, 1, len(self.curves) + 1)
        ts_argsort = np.argsort(ts)
        ts_sorted = ts[ts_argsort]
        ts_split_points = np.searchsorted(ts_sorted, ranges)
        # Make sure the last entry = len(ts), otherwise if ts finishes with a
        # trail of 1s, then they won't be forwarded to the last curve.
        ts_split_points[-1] = len(ts)
        result = []
        subranges = [
            slice(*pair) for pair in zip(ts_split_points, ts_split_points[1:])]
        for curve, subrange, (start, end) in zip(
                self.curves, subranges, zip(ranges, ranges[1:])):
            ts_mapped = (ts_sorted[subrange] - start) / (end - start)
            result.append(curve(ts_mapped))
        final = np.concatenate(result, axis=-1)
        assert len(final[0]) == len(ts)
        return final


class Segment(Curve):
    """
    Represents a line segment.
    """

    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)

    def __call__(self, ts):
        return (
            self.start[:, np.newaxis]
            + ts * (self.end - self.start)[:, np.newaxis])


class Arc(Curve):
    """
    Represents an arc of a circle.
    """

    def __init__(self, start, mid, end):
        """
        :arg start: starting point of the arc
        :arg mid: any point along the arc
        :arg end: ending point of the arc
        """
        xs, ys = np.stack((start, mid, end), axis=1)

        # Get center and radius of circle containing the arc.
        # http://math.stackexchange.com/a/1460096
        c = np.array([xs**2 + ys**2, xs, ys, [1, 1, 1]])
        x0 = la.det(np.delete(c, 1, 0)) / (2 * la.det(np.delete(c, 0, 0)))
        y0 = -la.det(np.delete(c, 2, 0)) / (2 * la.det(np.delete(c, 0, 0)))

        self.r = la.norm([start[0] - x0, start[1] - y0])
        self.center = x0 + 1j * y0

        theta_start = np.arctan2(start[1] - y0, start[0] - x0)
        theta_mid = np.arctan2(mid[1] - y0, mid[0] - x0)
        theta_end = np.arctan2(end[1] - y0, end[0] - x0)

        if theta_start <= theta_end:
            crosses_branch = not (theta_start <= theta_mid <= theta_end)
        else:
            crosses_branch = not (theta_start >= theta_mid >= theta_end)

        if crosses_branch:
            # Shift the angles so that branch crossing is not involved.
            if theta_start < 0:
                theta_start += 2 * np.pi
            if theta_mid < 0:
                theta_mid += 2 * np.pi
            if theta_end < 0:
                theta_end += 2 * np.pi

        self.theta_range = np.array(sorted([theta_start, theta_end]))
        self.theta_increasing = theta_start <= theta_end

    def __call__(self, t):
        if self.theta_increasing:
            thetas = (
                self.theta_range[0]
                + t * (self.theta_range[1] - self.theta_range[0]))
        else:
            thetas = (
                self.theta_range[1]
                - t * (self.theta_range[1] - self.theta_range[0]))
        val = (self.r * np.exp(1j * thetas)) + self.center
        return np.array([val.real, val.imag])


# horseshoe curve
#
# To avoid issues with crossing non-smooth regions, make sure the number of
# panels given to this function (for make_curve_mesh) is a multiple of 8.
horseshoe = (
    Segment((0, 0), (-5, 0))
    + Arc((-5, 0), (-5.5, -0.5), (-5, -1))
    + Segment((-5, -1), (0, -1))
    + Arc((0, -1), (1.5, 0.5), (0, 2))
    + Segment((0, 2), (-5, 2))
    + Arc((-5, 2), (-5.5, 1.5), (-5, 1))
    + Segment((-5, 1), (0, 1))
    + Arc((0, 1), (0.5, 0.5), (0, 0))
    )

# unit square
unit_square = (
    Segment((1, -1), (1, 1))
    + Segment((1, 1), (-1, 1))
    + Segment((-1, 1), (-1, -1))
    + Segment((-1, -1), (1, -1)))
