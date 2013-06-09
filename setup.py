#!/usr/bin/env python
# -*- coding: latin1 -*-


def main():
    import ez_setup
    ez_setup.use_setuptools()

    from setuptools import setup

    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

    version_dict = {}
    init_filename = "pytential/version.py"
    exec(compile(open(init_filename, "r").read(), init_filename, "exec"),
            version_dict)

    setup(name="pytential",
          version=version_dict["VERSION_TEXT"],
          description="Evaluate layer and volume potentials accurately. "
          "Solve integral equations.",
          long_description=open("README.rst", "rt").read(),
          author="Andreas Kloeckner",
          author_email="inform@tiker.net",
          license="MIT",
          url="http://wiki.tiker.net/Pytential",
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Natural Language :: English',
              'Programming Language :: Python',
              # We use conditional expressions, so 2.5 is the bare minimum.
              'Programming Language :: Python :: 2.5',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              # 3.x has not yet been tested.
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Visualization',
              'Topic :: Software Development :: Libraries',
              'Topic :: Utilities',
              ],

          packages=[
              "pytential",
              "pytential.mesh",
              "pytential.discretization",
              "pytential.discretization.qbx",
              "pytential.symbolic",
              "pytential.symbolic.pde",
              ],

          install_requires=[
              "pytest>=2.3",
              # FIXME leave out for now
              # https://code.google.com/p/sympy/issues/detail?id=3874
              #"sympy>=0.7.2",

              "modepy>=2013.3",
              "pyopencl>=2013.1",
              "boxtree>=2013.1",
              "pymbolic>=2013.2",
              "loo.py>=2013.1beta",
              "sumpy>=2013.1",
              ],

          # 2to3 invocation
          cmdclass={'build_py': build_py})


if __name__ == '__main__':
    main()
