"""
genieclust Package
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com/>     #
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
from setuptools.command.sdist import sdist
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os.path
import glob
import os
import sys
import re


cython_modules = {
    "genieclust.internal": [
        os.path.join("genieclust", "internal.pyx")
    ],
    "genieclust.oldmst": [
        os.path.join("genieclust", "oldmst.pyx")
    ],
    "genieclust.compare_partitions": [
        os.path.join("genieclust", "compare_partitions.pyx")
    ],
    "genieclust.cluster_validity": [
        os.path.join("genieclust", "cluster_validity.pyx")
    ],
    "genieclust.inequality": [
        os.path.join("genieclust", "inequality.pyx")
    ],
    "genieclust.tools": [
        os.path.join("genieclust", "tools.pyx")
    ]
}


class genieclust_sdist(sdist):
    def run(self):
        for pyx_files in cython_modules.values():
            cythonize(pyx_files, include_path=["genieclust/", "src/", "../src/"])
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
                e.extra_compile_args += ['/Qopenmp', '/Qstd=c++17']
                e.extra_link_args += ['/Qopenmp']
        elif sys.platform == "win32":
            for e in self.extensions:
                e.extra_compile_args += ['/openmp']
                e.extra_link_args += ['/openmp']
        elif sys.platform == "darwin" and ('icc' in compiler or 'icl' in compiler):
            for e in self.extensions:
                e.extra_compile_args += ['-openmp', '-std=c++17']
                e.extra_link_args += ['-openmp']
        elif sys.platform == "darwin":  # and 'openmp' in os.getenv('CPPFLAGS', ''):
            # https://github.com/scikit-learn/scikit-learn/blob/d640b7fc61ce716af9d113a7c92c953c1ec3e36f/sklearn/_build_utils/openmp_helpers.py
            # says: "-fopenmp can't be passed as compile flag when using Apple-clang.
            # OpenMP support has to be enabled during preprocessing.
            # For example, our macOS wheel build jobs use the following environment
            # variables to build with Apple-clang and the brew installed "libomp":"
            #
            # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
            # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
            # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
            # export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib
            #                          -L/usr/local/opt/libomp/lib -lomp"
            for e in self.extensions:
                e.extra_compile_args += ['-std=c++17']
            pass
        elif sys.platform == "linux":
            # Default flags for GCC and clang:
            for e in self.extensions:
                e.extra_compile_args += ['-fopenmp', '-std=c++17']
                e.extra_link_args += ['-fopenmp']
        else:
            pass

        build_ext.build_extensions(self)


ext_kwargs = dict(
    include_dirs=[np.get_include(), "src/", "../src/"],
    language="c++",
    depends=glob.glob(os.path.join("src", "c_*.h")) +
            glob.glob(os.path.join("genieclust", "*.pxd")),
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("GENIECLUST_PYTHON", "1"),
    ]
)


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()


with open("genieclust/__init__.py", "r", encoding="utf8") as fh:
    __version__ = re.search("(?m)^\\s*__version__\\s*=\\s*[\"']([0-9.]+)[\"']", fh.read())
    if __version__ is None:
        raise ValueError("the package version could not be read")
    __version__ = __version__.group(1)


setuptools.setup(
    name="genieclust",
    version=__version__,
    description="Genie: Fast and Robust Hierarchical Clustering with Outlier Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marek Gagolewski",
    author_email="marek@gagolewski.com",
    maintainer="Marek Gagolewski",
    license="GNU Affero General Public License v3",
    install_requires=[
        "Cython",  # not: cython
        "numpy>=2.0.0",
        "matplotlib",
        "scikit-learn",
        "quitefastmst>=0.9.1",
    ],
    python_requires=">=3.9",
    download_url="https://github.com/gagolews/genieclust",
    url="https://genieclust.gagolewski.com/",
    project_urls={
        "Bug Tracker":        "https://github.com/gagolews/genieclust/issues",
        "Documentation":      "https://genieclust.gagolewski.com/",
        "Source Code":        "https://github.com/gagolews/genieclust",
        "Benchmark Datasets": "https://clustering-benchmarks.gagolewski.com/",
        "Author":             "https://www.gagolewski.com/",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        # "Programming Language :: Python :: 3.7",
        # "Programming Language :: Python :: 3.8",
        # "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={
        "sdist": genieclust_sdist,
        "build_ext": genieclust_build_ext
    },
    ext_modules=[
        setuptools.Extension(module, pyx_files, **ext_kwargs)
        for module, pyx_files in cython_modules.items()
    ]
)
