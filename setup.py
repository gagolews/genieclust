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
from distutils.command.sdist import sdist as _sdist
from Cython.Distutils import build_ext
import numpy as np


class sdist(_sdist):
    def run(self):
        from Cython.Build import cythonize
        cythonize(["genieclust/internal.pyx"], language="c++")
        cythonize(["genieclust/argfuns.pyx"], language="c++")
        cythonize(["genieclust/disjoint_sets.pyx"], language="c++")
        cythonize(["genieclust/gini_disjoint_sets.pyx"], language="c++")
        cythonize(["genieclust/compare_partitions.pyx"], language="c++")
        cythonize(["genieclust/inequity.pyx"], language="c++")
        cythonize(["genieclust/mst.pyx"], language="c++")
        _sdist.run(self)

cmdclass = {}
cmdclass["sdist"] = sdist

ext_modules = [ ]


ext_kwargs = dict(include_dirs=[np.get_include()], language="c++")
ext_modules += [
    Extension("genieclust.internal",
                ["genieclust/internal.pyx"],
                **ext_kwargs),
    Extension("genieclust.argfuns",
                ["genieclust/argfuns.pyx"],
                **ext_kwargs),
    Extension("genieclust.disjoint_sets",
                ["genieclust/disjoint_sets.pyx"],
                **ext_kwargs),
    Extension("genieclust.gini_disjoint_sets",
                ["genieclust/gini_disjoint_sets.pyx"],
                **ext_kwargs),
    Extension("genieclust.compare_partitions",
                ["genieclust/compare_partitions.pyx"],
                **ext_kwargs),
    Extension("genieclust.inequity",
                ["genieclust/inequity.pyx"],
                **ext_kwargs),
    Extension("genieclust.mst",
                ["genieclust/mst.pyx"],
                **ext_kwargs)
]
cmdclass.update({ 'build_ext': build_ext })



with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genieclust",
    version="0.1a4",
    description="The Genie+ Clustering Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marek Gagolewski",
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
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
