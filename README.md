genieclust Python Package (**under development**)
=========================

The *Genie*+  Clustering Algorithm
----------------------------------

Author: [Marek Gagolewski](http://www.gagolewski.com)

The time needed to apply a hierarchical clustering algorithm is most
often dominated by the number of computations of a pairwise dissimilarity
measure. Such a constraint, for larger data sets, puts at a disadvantage
the use of all the classical linkage criteria but the single linkage one.
However, it is known that the single linkage clustering algorithm is very
sensitive to outliers, produces highly skewed dendrograms, and therefore
usually does not reflect the true underlying data structure -
unless the clusters are well-separated.

To overcome its limitations, we proposed a new hierarchical clustering linkage
criterion called Genie. Namely, our algorithm links two clusters in such
a way that a chosen economic inequity measure (here, the Gini index)
of the cluster sizes does not increase drastically above a given threshold.

Benchmarks indicate a high practical usefulness of the introduced method:
it most often outperforms the Ward or average linkage, k-means,
spectral clustering, DBSCAN, Birch, and others in terms of the clustering
quality while retaining the single linkage speed. The algorithm is easily
parallelizable and thus may be run on multiple threads to speed up its
execution further on. Its memory overhead is small: there is no need
to precompute the complete distance matrix to perform the computations
in order to obtain a desired clustering.

See: Gagolewski M., Bartoszuk M., Cena A.,
Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
*Information Sciences* **363**, 2016, pp. 8-23.
doi:[10.1016/j.ins.2016.05.003](http://dx.doi.org/10.1016/j.ins.2016.05.003)



This is a new, faster and even more robust implementation
of the original algorithm available on CRAN,
see R package [`genie`](http://www.gagolewski.com/software/genie/).



Package Features
================

* The Genie+ algorithm (using a `scikit-learn`-like interface),
together with a robustified version of HDBSCAN*
* DisjointSets data structure (with extensions)
* Various inequity measures (the Gini index, the Bonferroni index, etc.)
* Functions to compute partition similarity measures
(the Rand, adjusted Rand, Fowlkes-Mallows, and adjusted Fowlkes-Mallows index)
* Various plotting functions


Installation
============

The package requires Python 3.6+ together with `cython`
as well as `numpy`, `scipy`, `matplotlib`, and `sklearn`.


Via `pip` - see [PyPI](https://pypi.org/project/genieclust/):

```bash
conda install pip      # if using conda
pip install genieclust # or pip install genieclust --user
```

The most recent development version:

```bash
git clone https://github.com/gagolews/genieclust.git
cd genieclust
python setup.py build_ext --inplace
python setup.py install --user
```

Examples
========

* [The Genie Algorithm - basic use](https://github.com/gagolews/genieclust/blob/master/example_genie_basic.ipynb)
* [The Genie Algorithm with Noise Points Detection](https://github.com/gagolews/genieclust/blob/master/example_genie_hdbscan.ipynb)


License
=======

This package is licensed under the BSD 3-Clause "New" or "Revised" License.

```
Copyright (C) 2018 Marek.Gagolewski.com
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
```
