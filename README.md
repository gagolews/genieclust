Genieclust Python Package (**under development**)
=========================

The *Genie*  clustering algorithm
---------------------------------

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



This is a new, O(n*sqrt(n)) implementation of the algorithm
(given a pre-computed minimum spanning tree).
For the original version,
see R package [`genie`](https://cran.r-project.org/package=genie).



Package Features
================

* The Genie algorithm (with a scikit-learn-like interface)
* The HDBSCAN* algorithm (with a scikit-learn-like interface)
* DisjointSets data structure (with extensions)
* Various inequity measures (the Gini index, the Bonferroni index, etc.)
* Functions to compute partition similarity measures
(the Rand, adjusted Rand, Fowlkes-Mallows, and adjusted Fowlkes-Mallows index)
* An implementation of the Prim algorithm to compute the minimum spanning tree
(@TODO@ parallelized, requiring O(n**2) time and O(n) memory)
* Various plotting functions


Installation
============

The package requires Python 3.6+ together with
sklearn, numpy, scipy, matplotlib, and cython.


Via `pip`:

```python
# @TODO@ - not yet on PyPI
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

```python
# @TODO@
```


License
=======

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
