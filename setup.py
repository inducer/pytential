#!/usr/bin/env python

import os
import sys

from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension


# {{{ capture git revision at install time

def find_git_revision(tree_root):
    # authoritative version in pytools/__init__.py
    # Keep this routine self-contained so that it can be copy-pasted into
    # setup.py.

    from os.path import abspath, exists, join

    tree_root = abspath(tree_root)

    if not exists(join(tree_root, ".git")):
        return None

    from subprocess import PIPE, STDOUT, Popen

    p = Popen(
        ["git", "rev-parse", "HEAD"],
        shell=False,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        close_fds=True,
        cwd=tree_root,
    )
    (git_rev, _) = p.communicate()

    git_rev = git_rev.decode()
    git_rev = git_rev.rstrip()

    retcode = p.returncode
    assert retcode is not None
    if retcode != 0:
        from warnings import warn

        warn("unable to find git revision")
        return None

    return git_rev


def write_git_revision(package_name):
    from os.path import dirname, join

    dn = dirname(__file__)
    git_rev = find_git_revision(dn)

    with open(join(dn, package_name, "_git_rev.py"), "w") as outf:
        outf.write("GIT_REVISION = %s\n" % repr(git_rev))


write_git_revision("pytential")

# }}}


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


version_dict = {}
init_filename = "pytential/version.py"
os.environ["AKPYTHON_EXEC_FROM_WITHIN_WITHIN_SETUP_PY"] = "1"
exec(compile(open(init_filename).read(), init_filename, "exec"), version_dict)

setup(
    name="pytential",
    version=version_dict["VERSION_TEXT"],
    description="Evaluate layer and volume potentials accurately. "
    "Solve integral equations.",
    long_description=open("README.rst").read(),
    author="Andreas Kloeckner",
    author_email="inform@tiker.net",
    license="MIT",
    url="https://github.com/inducer/pytential",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    python_requires=">=3.8",
    install_requires=[
        "pytools>=2018.2",
        "modepy>=2021.1",
        "pyopencl>=2021.2.6",
        "boxtree>=2019.1",
        "pymbolic>=2022.1",
        "loopy>=2020.2",
        "arraycontext>=2021.1",
        "meshmode>=2021.2",
        "sumpy>=2020.2beta1",

        "scipy>=1.2.0",
        "immutables",
    ],
    extras_require={
        "fmmlib": ["pyfmmlib>=2019.1.1"],
    },
)
