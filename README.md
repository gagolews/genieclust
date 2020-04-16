`genieclust` Python Package
===========================



The *Genie*++ Hierarchical Clustering Algorithm
-----------------------------------------------

> **Genie outputs meaningful partitions and is fast even for large
> data sets.**



Author: [Marek Gagolewski](https://www.gagolewski.com)

Co-authors/contributors:
[Anna Cena](https://cena.rexamine.com),
[Maciej Bartoszuk](https://bartoszuk.rexamine.com)

The time needed to apply a hierarchical clustering algorithm is most
often dominated by the number of computations of a pairwise
dissimilarity measure. Such a constraint, for larger data sets, puts at
a disadvantage the use of all the classical linkage criteria but the
single linkage one. However, it is known that the single linkage
clustering algorithm is very sensitive to outliers, produces highly
skewed dendrograms and therefore usually does not reflect the true
underlying data structure -- unless the clusters are well-separated.

To overcome its limitations, we proposed a new hierarchical clustering
linkage criterion called Genie. Namely, our algorithm links two clusters
in such a way that a chosen economic inequity measure (here, the Gini
index) of the cluster sizes does not increase drastically above a given
threshold.

The algorithm most often outperforms the Ward or average linkage, k-means,
spectral clustering, DBSCAN, Birch and others in terms of the
clustering quality on benchmark data while retaining the single linkage speed.
The algorithm is easily parallelisable and thus may be run on multiple
threads to speed up its execution further on. Its memory overhead is
small: there is no need to precompute the complete distance matrix to
perform the computations in order to obtain a desired clustering.

This is a new, faster and even more robust implementation of the
original algorithm available on CRAN, see R package
[genie](http://www.gagolewski.com/software/genie/) and the paper:

> Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and
> outlier-resistant hierarchical clustering algorithm, *Information
> Sciences* **363**, 2016, pp. 8-23.
> [doi:10.1016/j.ins.2016.05.003](http://dx.doi.org/10.1016/j.ins.2016.05.003).




Package Features
----------------

-   The Genie++ hierarchical clustering algorithm (with a `scikit-learn`-like
    interface), together with **TODO**: a robustified version of
    HDBSCAN\*, IcA, GC, GIc, ...

-   `DisjointSets` (union-find) data structure (with extensions)

-   Inequity measures (the Gini index, the Bonferroni index
    etc.)

-   Functions to compare partitions (the Rand, adjusted Rand,
    Fowlkes-Mallows and adjusted Fowlkes-Mallows indexes)

-   Useful plotting functions




Examples and Tutorials
----------------------

`genieclust` has a familiar `scikit-learn` look-and-feel:

```python
import genieclust
X = ... # some data
g = genieclust.Genie(n_clusters=2)
g.fit_predict(X)
```

For more illustrations, use cases and details, make sure to check out:

-   [The Genie Algorithm - Basic Use](https://github.com/gagolews/genieclust/blob/master/tutorials/example_genie_basic.ipynb)
-   [The Genie Algorithm with Noise Points Detection](https://github.com/gagolews/genieclust/blob/master/tutorials/example_noisy.ipynb)
-   [Plotting Dendrograms](https://github.com/gagolews/genieclust/blob/master/tutorials/dendrogram.md)
-   [Comparing Different Hierarchical Linkage Methods on Toy Datasets - A `scikit-learn` Example](https://github.com/gagolews/genieclust/blob/master/tutorials/sklearn_toy_example.md)
-   [Auxiliary Plotting Functions](https://github.com/gagolews/genieclust/blob/master/tutorials/plots.md)


Installation
------------

> *This package is in alpha-stage (development is Linux-only).*

The package requires Python 3.6+ together with `cython` as well as
`numpy`, `scipy`, `matplotlib` and `sklearn`.

Optional dependencies: `rpy2`, `faiss` (e.g. `faiss-cpu`).

**TODO**: Windows builds

**TODO**: OS X builds

To install via `pip` (the current version is a little outdated,
see [PyPI](https://pypi.org/project/genieclust/)):

```bash
pip3 install genieclust --user # or sudo pip3 install genieclust
```

To build and install the most recent development version:

```bash
git clone https://github.com/gagolews/genieclust.git
cd genieclust
python3 setup.py install --user
```

To support parallelised computations, build with OpenMP support (for gcc/clang):

```bash
CPPFLAGS="-fopenmp -DNDEBUG" LDFLAGS="-fopenmp" python3 setup.py install --user
```




License
-------

This package is licensed under the BSD 3-Clause "New" or "Revised"
License.

Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
3.  Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS
IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




References
----------

Gagolewski M., Bartoszuk M., Cena A.,
Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
*Information Sciences* **363**, 2016, pp. 8-23.
doi:10.1016/j.ins.2016.05.003

Cena A., Gagolewski M.,
Genie+OWA: Robustifying Hierarchical Clustering with OWA-based Linkages,
*Information Sciences*, 2020,
in press. doi:10.1016/j.ins.2020.02.025

Cena A.,
Adaptive hierarchical clustering algorithms based on data aggregation methods,
PhD Thesis, Systems Research Institute, Polish Academy of Sciences, 2018.

Campello R., Moulavi D., Zimek A., Sander J.,
Hierarchical density estimates for data clustering, visualization, and outlier detection,
*ACM Transactions on Knowledge Discovery from Data* **10**(1), 2015, 5:1–5:51.
doi:10.1145/2733381.

Mueller A., Nowozin S., Lampert C.H.,
Information Theoretic Clustering using Minimum Spanning Trees,
*DAGM-OAGM* 2012.

Jarník V., O jistém problému minimálním,
*Práce Moravské Přírodovědecké Společnosti* **6**, 1930, pp. 57–63.

Olson C.F., Parallel algorithms for hierarchical clustering,
*Parallel Comput.* **21**, 1995, pp. 1313–1325.

Prim R., Shortest connection networks and some generalizations,
*Bell Syst. Tech. J.* **36**, 1957, pp. 1389–1401.

Hubert L., Arabie P., Comparing Partitions,
*Journal of Classification* **2**(1), 1985, pp. 193-218
