*Genie*: Fast and Robust Hierarchical Clustering with Noise Point Detection
===========================================================================

|genieclust for Python| |genieclust for R| |codecov|

   **Genie outputs meaningful clusters and is fast even on large data sets.**

Documentation, tutorials, and benchmarks are available
at https://genieclust.gagolewski.com/.


About
-----

A faster and more powerful version of **Genie** - a robust and outlier
resistant clustering algorithm (see Gagolewski, Bartoszuk, and Cena, 2016),
originally included in the R package
`genie <https://cran.r-project.org/web/packages/genie/>`_.

The idea behind Genie is beautifully simple. First, make each individual
point the only member of its own cluster. Then, keep merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes** a point group of
the smallest size will sometimes be matched with its nearest neighbours.

Genie's appealing simplicity goes hand in hand with its usability;
it **often outperforms other clustering approaches**
such as K-means, BIRCH, or average, Ward, and complete linkage
on `benchmark data <https://github.com/gagolews/clustering_benchmarks_v1/>`_.

Genie is also **very fast** - determining the whole cluster hierarchy
for datasets of millions of points can be completed within a coffee break.
Therefore, it is perfectly suited for solving of **extreme clustering tasks**
(large datasets with any number of clusters to detect) for data (also sparse)
that fit into memory.
Thanks to the use of `nmslib`, sparse or string inputs are also supported.

It also allows clustering with respect to mutual reachability distances
so that it can act as a **noise point detector** or a
robustified version of `HDBSCAN\*`  (see Campello et al., 2015)
that is able to detect a predefined
number of clusters and hence it doesn't dependent on the `DBSCAN`'s somewhat
difficult-to-set `eps` parameter.



Author and Contributors
-----------------------

Author and maintainer: `Marek Gagolewski <https://www.gagolewski.com>`_

Contributors of the code from the original R package `genie`:
`Anna Cena <https://cena.rexamine.com>`_,
`Maciej Bartoszuk <https://bartoszuk.rexamine.com>`_

Computing of some partition similarity scores (namely, the normalised accuracy
and pair sets index) is based on an implementation of the shortest augmenting path
algorithm for the rectangular assignment problem contributed by
`Peter M. Larsen <https://github.com/pmla/>`_\ .




Python and R Package Features
-----------------------------

The implemented algorithms include:

-  Genie++ - a reimplementation of the original Genie algorithm
    with a `scikit-learn`-compatible interface (Gagolewski et al., 2016);
    much faster than the original one; supports approximate disconnected MSTs;

-  Genie+HDBSCAN\* - our robustified (Geniefied) retake on the HDBSCAN\*
   (Campello et al., 2015) method that detects noise points in data and
   outputs clusters of predefined sizes;

-  *(Python only, experimental preview)* Genie+Ic (GIc) - Cena's (2018)
   algorithm to minimise the information theoretic criterion discussed
   by Mueller et al. (2012).

See classes ``genieclust.Genie`` and ``genieclust.GIc`` (Python) or
functions ``gclust()`` and ``genieclust()`` (R).


Other goodies:

-  Inequity measures (the normalised Gini and Bonferroni index);

-  unctions to compare partitions (adjusted&unadjusted Rand,
   adjusted&unadjusted Fowlkes-Mallows (FM),
   adjusted&normalised&unadjusted mutual information (MI) scores,
   normalised accuracy and pair sets index (PSI));

-  *(Python only)* Union-find (disjoint sets) data structures (with
   extensions);

-  *(Python only)* Useful R-like plotting functions.




Examples, Tutorials, and Documentation
--------------------------------------

The Python language version of `genieclust` has a familiar
`scikit-learn`-like look-and-feel:

.. code:: python

   import genieclust
   X = ... # some data
   g = genieclust.Genie(n_clusters=2)
   labels = g.fit_predict(X)

R's interface is compatible with ``hclust()``, but there is more.

.. code:: r

   X <- ... # some data
   h <- gclust(X)
   plot(h) # plot cluster dendrogram
   cutree(h, k=2)
   # or genie(X, k=2)

Check out the tutorials and the package documentation at
https://genieclust.gagolewski.com/.



How to Install
--------------


Python Version
~~~~~~~~~~~~~~

PyPI
^^^^

To install via ``pip`` (see `PyPI <https://pypi.org/project/genieclust/>`_):

.. code:: bash

   pip3 install genieclust


The package requires Python 3.7+ together with ``cython`` as well as
``numpy``, ``scipy``, ``matplotlib``, ``nmslib``, and ``scikit-learn``.
Optional dependency: ``mlpack``.







R Version
~~~~~~~~~

CRAN
^^^^

To install the most recent release, call:

.. code:: r

    install.packages("genieclust")


See the package entry on `CRAN <https://cran.r-project.org/web/packages/genieclust/>`_.




Other
~~~~~

Note that the core functionality is implemented in form of a header-only
C++ library, so it might be relatively easily adapted for use in
other environments.

Any contributions are welcome (e.g., Julia, Matlab, ...).


License
-------

Copyright (C) 2018-2021 Marek Gagolewski (https://www.gagolewski.com)

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License Version 3, 19
November 2007, published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
General Public License Version 3 for more details. You should have
received a copy of the License along with this program. If not, see
(https://www.gnu.org/licenses/).

--------------

The file ``src/c_scipy_rectangular_lsap.h`` is adapted from the
``scipy`` project (https://scipy.org/scipylib/), source:
``/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp``.
Author: Peter M. Larsen. Distributed under the BSD-3-Clause license.




References
----------

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and
outlier-resistant hierarchical clustering algorithm, *Information
Sciences* **363**, 2016, 8-23. doi:10.1016/j.ins.2016.05.003.

Cena A., Gagolewski M., Genie+OWA: Robustifying Hierarchical Clustering
with OWA-based Linkages, *Information Sciences* **520**, 2020, 324-336.
doi:10.1016/j.ins.2020.02.025.

Cena A., *Adaptive hierarchical clustering algorithms based on data
aggregation methods*, PhD Thesis, Systems Research Institute, Polish
Academy of Sciences, 2018.

Campello R., Moulavi D., Zimek A., Sander J., Hierarchical density
estimates for data clustering, visualization, and outlier detection,
*ACM Transactions on Knowledge Discovery from Data* **10**\ (1), 2015,
5:1-5:51. doi:10.1145/2733381.

Mueller A., Nowozin S., Lampert C.H., Information Theoretic Clustering
using Minimum Spanning Trees, *DAGM-OAGM*, 2012.

See https://genieclust.gagolewski.com/ for more.



.. |genieclust for Python| image:: https://github.com/gagolews/genieclust/workflows/genieclust%20for%20Python/badge.svg
   :target: https://pypi.org/project/genieclust/
.. |genieclust for R| image:: https://github.com/gagolews/genieclust/workflows/genieclust%20for%20R/badge.svg
   :target: https://cran.r-project.org/web/packages/genieclust/
.. |codecov| image:: https://codecov.io/gh/gagolews/genieclust/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/gagolews/genieclust
