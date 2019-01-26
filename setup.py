"""
Genieclust Python Package
Copyright (C) 2018-2019 Marek.Gagolewski.com
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import setuptools
from distutils.extension import Extension
from distutils.command.sdist import sdist
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os.path
import glob

cython_modules = {
    "genieclust.internal":           [os.path.join("genieclust", "internal.pyx")],
    "genieclust.argfuns":            [os.path.join("genieclust", "argfuns.pyx")],
    "genieclust.disjoint_sets":      [os.path.join("genieclust", "disjoint_sets.pyx")],
    "genieclust.gini_disjoint_sets": [os.path.join("genieclust", "gini_disjoint_sets.pyx")],
    "genieclust.compare_partitions": [os.path.join("genieclust", "compare_partitions.pyx")],
    "genieclust.inequity":           [os.path.join("genieclust", "inequity.pyx")],
    "genieclust.mst":                [os.path.join("genieclust", "mst.pyx")]
}


class genieclust_sdist(sdist):
    def run(self):
        for pyx_files in cython_modules.values():
            cythonize(pyx_files)
        sdist.run(self)


class genieclust_build_ext(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type

        # @TODO once n_jobs params are added
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
    depends=glob.glob(os.path.join("genieclust", "*.h"))+
            glob.glob(os.path.join("genieclust", "*.pxd"))
)



with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genieclust",
    version="0.1a4",
    description="The Genie+ Clustering Algorithm",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Marek Gagolewski",
    author_email="marek@gagolewski.com",
    maintainer="Marek Gagolewski",
    license="BSD-3-Clause",
    install_requires=[
          "numpy",
          "scipy",
          "cython",
          "matplotlib",
          "sklearn"
      ],
    download_url="https://github.com/gagolews/genieclust",
    url="http://www.gagolewski.com/software/genie/",
    project_urls={
        "Bug Tracker":   "https://github.com/gagolews/genieclust/issues",
        "Documentation": "https://github.com/gagolews/genieclust",
        "Source Code":   "https://github.com/gagolews/genieclust",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
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
