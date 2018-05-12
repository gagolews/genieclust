Genieclust Python Package
=========================

Author: [Marek Gagolewski](http://www.gagolewski.com)

The *Genie*  clustering algorithm

See Gagolewski M., Bartoszuk M., Cena A.,
Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
*Information Sciences* **363**, 2016, pp. 8-23.
doi:[10.1016/j.ins.2016.05.003](http://dx.doi.org/10.1016/j.ins.2016.05.003)

This is a new, O(n*sqrt(n)) implementation of the algorithm
(given a pre-computed minimum spanning tree).
For the original version,
see R package [`genie`](https://cran.r-project.org/package=genie).


Package Features
================

* The Genie algorithm (with a scikit-learn-like inferface)
* The HDBSCAN* algorithm (with a scikit-learn-like inferface)
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
@TODO@
```

The most recent development version:

```python
git clone https://github.com/gagolews/genieclust.git
cd genieclust
python setup.py build_ext --inplace
python setup.py install --user
```

Examples
========

```python
@TODO@
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
