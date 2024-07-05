#!/usr/bin/env python

import sys

from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension


def get_numpy_incpath():
    from importlib.util import find_spec
    from os.path import dirname, exists, join

    origin = find_spec("numpy").origin
    if origin is None:
        raise RuntimeError("origin of numpy package not found")

    pathname = dirname(origin)
    for p in [
        join(pathname, "_core", "include"),  # numpy 2 onward
        join(pathname, "core", "include"),  # numpy prior to 2
    ]:
        if exists(join(p, "numpy", "arrayobject.h")):
            return p

    raise RuntimeError("no valid path for numpy found")


if sys.platform.startswith("linux"):
    openmp_flag = ["-fopenmp"]
else:
    # OpenMP can't be relied upon on MacOS.
    # https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave
    openmp_flag = []


ext_modules = [
    Extension(
        "pytential.qbx.target_specific.impl",
        sources=[
            "pytential/qbx/target_specific/impl.pyx",
            "pytential/qbx/target_specific/helmholtz_utils.c",
        ],
        depends=[
            "pytential/qbx/target_specific/impl.h",
            "pytential/qbx/target_specific/helmholtz_utils.h",
        ],
        extra_compile_args=["-Wall", "-Ofast", *openmp_flag],
        extra_link_args=openmp_flag,
    ),
    Extension(
        "pytential.linalg._decomp_interpolative",
        sources=[
            "pytential/linalg/_decomp_interpolative.pyx",
        ],
        depends=[],
        include_dirs=[get_numpy_incpath()],
        extra_compile_args=["-Wall", "-Ofast", *openmp_flag],
        extra_link_args=["-lm", *openmp_flag],
    )
]

setup(ext_modules=cythonize(ext_modules))
