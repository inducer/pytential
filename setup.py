#!/usr/bin/env python

import sys

from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

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
        extra_compile_args=["-Wall", "-Ofast"] + openmp_flag,
        extra_link_args=openmp_flag,
    ),
]

setup(ext_modules=cythonize(ext_modules))
