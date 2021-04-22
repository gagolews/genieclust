"""
genieclust Package
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2021, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import setuptools
from distutils.extension import Extension
from distutils.command.sdist import sdist
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os.path
import glob
import os
import sys


cython_modules = {
    "genieclust.internal": [
        os.path.join("genieclust", "internal.pyx")
    ],
    "genieclust.compare_partitions": [
        os.path.join("genieclust", "compare_partitions.pyx")
    ],
    "genieclust.inequity": [
        os.path.join("genieclust", "inequity.pyx")
    ],
    "genieclust.tools": [
        os.path.join("genieclust", "tools.pyx")
    ]
}


class genieclust_sdist(sdist):
    def run(self):
        for pyx_files in cython_modules.values():
            cythonize(pyx_files)
        sdist.run(self)


class genieclust_build_ext(build_ext):
    def build_extensions(self):

        # This code is adapted from
        # scikit-learn/sklearn/_build_utils/openmp_helpers.py
        # (version last updated on 13 Nov 2019; 9876f74)
        # See https://github.com/scikit-learn/scikit-learn and https://scikit-learn.org/.

        if hasattr(self.compiler, 'compiler'):
            compiler = self.compiler.compiler[0]
        else:
            compiler = self.compiler.__class__.__name__

        if sys.platform == "win32" and ('icc' in compiler or 'icl' in compiler):
            for e in self.extensions:
                e.extra_compile_args += ['/Qopenmp', '/Qstd=c++11']
                e.extra_link_args += ['/Qopenmp']
        elif sys.platform == "win32":
            for e in self.extensions:
                e.extra_compile_args += ['/openmp']
                e.extra_link_args += ['/openmp']
        elif sys.platform == "darwin" and ('icc' in compiler or 'icl' in compiler):
            for e in self.extensions:
                e.extra_compile_args += ['-openmp', '-std=c++11']
                e.extra_link_args += ['-openmp']
        elif sys.platform == "darwin":  # and 'openmp' in os.getenv('CPPFLAGS', ''):
            # -fopenmp can't be passed as compile flag when using Apple-clang.
            # OpenMP support has to be enabled during preprocessing.
            #
            # For example, our macOS wheel build jobs use the following environment
            # variables to build with Apple-clang and the brew installed "libomp":
            #
            # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
            # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
            # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
            # export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib
            #                          -L/usr/local/opt/libomp/lib -lomp"
            for e in self.extensions:
                e.extra_compile_args += ['-std=c++11']
            pass
        elif sys.platform == "linux":
            # Default flag for GCC and clang:
            for e in self.extensions:
                e.extra_compile_args += ['-fopenmp', '-std=c++11']
                e.extra_link_args += ['-fopenmp']
        else:
            pass

        # Old version:
        # c = self.compiler.compiler_type
        # if c == "msvc":
        #     for e in self.extensions:
        #         e.extra_compile_args += "/openmp"
        # elif c == "mingw32":
        #     for e in self.extensions:
        #         e.extra_compile_args += "-fopenmp"
        #         e.extra_link_args += "-fopenmp"
        # elif c == "unix":
        #     # Well... gcc/clang has -fopenmp,
        #     # icc has -openmp, oracle has -xopenmp, etc.
        #     # The user should specify CXXFLAGS and LDFLAGS herself, I think.
        #     pass

        build_ext.build_extensions(self)


ext_kwargs = dict(
    include_dirs=[np.get_include()],
    language="c++",
    depends=glob.glob(os.path.join("src", "c_*.h")) +
            glob.glob(os.path.join("genieclust", "*.pxd")),
    #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)


with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genieclust",
    version="1.0.0",  # see also genieclust/__init__.py; e.g., 0.9.4.dev0
    description="The Genie++ Hierarchical Clustering Algorithm (with Extras)",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Marek Gagolewski",
    author_email="marek@gagolewski.com",
    maintainer="Marek Gagolewski",
    license="GNU Affero General Public License v3",
    install_requires=[
        "numpy",
        "scipy",
        "cython",
        "matplotlib",
        "scikit-learn",
        "nmslib"  # nmslib does not build on 32bit Windows...
      ],
    extras_require={
        "mlpack": ["mlpack"]  # as of 2021-04-22, mlpack is not available for Python 3.9
    },
    download_url="https://github.com/gagolews/genieclust",
    url="https://genieclust.gagolewski.com/",
    project_urls={
        "Bug Tracker":   "https://github.com/gagolews/genieclust/issues",
        "Documentation": "https://genieclust.gagolewski.com/",
        "Source Code":   "https://github.com/gagolews/genieclust",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={
        "sdist": genieclust_sdist,
        "build_ext": genieclust_build_ext
    },
    ext_modules=[
        Extension(module, pyx_files, **ext_kwargs)
        for module, pyx_files in cython_modules.items()
    ]
)
