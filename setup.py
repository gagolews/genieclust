"""
Genieclust Python Package
Copyright (C) 2018 Marek.Gagolewski.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import setuptools
from distutils.extension import Extension
from distutils.command.sdist import sdist as _sdist
from Cython.Distutils import build_ext
import numpy as np


class sdist(_sdist):
    def run(self):
        # make sure the distributed .pyx files are up-to-date
        from Cython.Build import cythonize
        cythonize(["genieclust/internal.pyx"])
        cythonize(["genieclust/compare_partitions.pyx"])
        cythonize(["genieclust/inequity.pyx"])
        cythonize(["genieclust/mst.pyx"])
        _sdist.run(self)

cmdclass = {}
cmdclass["sdist"] = sdist

ext_modules = [ ]


ext_modules += [
    Extension("genieclust.internal",
                ["genieclust/internal.pyx"],
                include_dirs=[np.get_include()]),
    Extension("genieclust.compare_partitions",
                ["genieclust/compare_partitions.pyx"],
                include_dirs=[np.get_include()]),
    Extension("genieclust.inequity",
                ["genieclust/inequity.pyx"],
                include_dirs=[np.get_include()]),
    Extension("genieclust.mst",
                ["genieclust/mst.pyx"],
                include_dirs=[np.get_include()])
]
cmdclass.update({ 'build_ext': build_ext })



with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genieclust",
    version="0.1.a2",
    description="The Genie+ Clustering Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marek Gagolewski",
    author_email="marek@gagolewski.com",
    maintainer="Marek Gagolewski",
    maintainer_email="marek@gagolewski.com",
    license="BSD-3-Clause",
    install_requires=[
          "numpy",
          "scipy",
          "cython",
          "matplotlib",
          "sklearn"
      ],
    download_url="https://github.com/gagolews/genieclust",
    url="http://www.gagolewski.com/software/",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
    ),
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
