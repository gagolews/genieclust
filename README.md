# `genieclust` Python and R Package



## The *Genie*++ Hierarchical Clustering Algorithm (with Extras)

> **Genie outputs meaningful clusters and is fast even for large data sets.**

A reimplementation of **Genie** - a robust and outlier resistant
clustering algorithm (see Gagolewski, Bartoszuk, Cena, 2016),
originally published as an R package
[`genie`](https://cran.r-project.org/web/packages/genie/).

The Genie algorithm is based on a minimum spanning tree (MST) of the
pairwise distance graph of an input point set.
Just like single linkage, it consumes the edges
of the MST in increasing order of weights. However, it prevents
the formation of clusters of highly imbalanced sizes; once the Gini index
of the cluster size distribution raises above an assumed threshold,
a point group of the smallest size is forced to merge with its nearest
neighbouring cluster.

Genie's appealing simplicity goes hand in hand with its usability;
it often outperforms other clustering approaches
such as K-means, BIRCH, average, Ward and complete linkage
on [benchmark data](https://github.com/gagolews/clustering_benchmarks_v1).

Genie is also pretty fast -- determining the whole cluster hierarchy
for datasets of 10M points in low dimensional Euclidean spaces or
100K points in high dimensional ones takes 1-2 minutes.

It allows clustering with respect to mutual reachability distances
so that it can act as a noise point detector or a robustified version of
HDBSCAN* (that is able to detect a predefined number of
clusters and hence it doesn't dependent on the DBSCAN's somehow
difficult-to-set `eps` parameter).





## Author and Contributors


Author: [Marek Gagolewski](https://www.gagolewski.com)

Contributors for the original R package `genie`:
[Anna Cena](https://cena.rexamine.com),
[Maciej Bartoszuk](https://bartoszuk.rexamine.com)




## Python and R Package Features


Implemented algorithms include:

-   Genie++ -- a reimplementation of the original Genie algorithm
    (with a `scikit-learn`-compatible interface; Gagolewski et al., 2016)

-   Genie+HDBSCAN\* -- our robustified (Geniefied) retake on the HDBSCAN\*
    (Campello et al., 2015) method that detects noise points in data
    and outputs clusters of predefined sizes

-   *(Python only, experimental preview)* Genie+Ic (GIc) -- Cena's (2018)
    algorithm to minimise the information theoretic criterion discussed
    by Mueller et al. (2012)

See classes `genieclust.Genie` and `genieclust.GIc` (Python)
or functions `gclust()` and `genieclust()` (R).


Other goodies:

-   Inequity measures (the normalised Gini and Bonferroni index)

-   unctions to compare partitions
    (adjusted&unadjusted Rand,
    adjusted&unadjusted Fowlkes-Mallows (FM),
    adjusted&normalised&unadjusted mutual information (MI) scores,
    normalised accuracy and pair sets index (PSI))

-   *(Python only)* Union-find (disjoint sets) data structures (with extensions)

-   *(Python only)* Useful R-like plotting functions




## Examples and Tutorials


The Python language version of `genieclust` has a familiar `scikit-learn` look-and-feel:

```python
import genieclust
X = ... # some data
g = genieclust.Genie(n_clusters=2)
labels = g.fit_predict(X)
```

R's interface is compatible with `hclust()`, but there is more.

```python
X <- ... # some data
h <- gclust(X)
plot(h) # plot cluster dendrogram
cutree(h, k=2)
# or genie(X, k=2)
```


**TODO**: For more illustrations, use cases and details, make sure to check out:

-   [The Genie Algorithm - Basic Use](https://github.com/gagolews/genieclust/blob/master/tutorials/example_genie_basic.ipynb)
-   [The Genie Algorithm with Noise Points Detection](https://github.com/gagolews/genieclust/blob/master/tutorials/example_noisy.ipynb)
-   [Plotting Dendrograms](https://github.com/gagolews/genieclust/blob/master/tutorials/dendrogram.md)
-   [Comparing Different Hierarchical Linkage Methods on Toy Datasets - A `scikit-learn` Example](https://github.com/gagolews/genieclust/blob/master/tutorials/sklearn_toy_example.md)
-   [Auxiliary Plotting Functions](https://github.com/gagolews/genieclust/blob/master/tutorials/plots.md)





## Installation


> *This package is in beta-stage (development and testing is currently Linux-only).*



### Python Version


#### PyPI

**TODO** To install via `pip` (the current version is a little outdated,
see [PyPI](https://pypi.org/project/genieclust/)):

```bash
pip3 install genieclust --user # or sudo pip3 install genieclust
```



**TODO**: Windows builds

**TODO**: OS X builds






The package requires Python 3.6+ together with `cython` as well as
`numpy`, `scipy`, `matplotlib`, and `sklearn`.
Optional dependencies: `mlpack` and `nmslib`.


#### Development Version

To build and install the most recent development version, call:

```bash
git clone https://github.com/gagolews/genieclust.git
cd genieclust
python3 setup.py install --user
```




### R Version


#### CRAN

**TODO**: CRAN



#### Development Version

To fetch and compile the most recent development version of the package from
github, call (C++11 compiler required;
Windows users see [Rtools](https://cran.r-project.org/bin/windows/Rtools/),
OS X users see [Xcode](https://developer.apple.com/xcode/)):

```bash
devtools::install_github("gagolews/genieclust")
```




## License


Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License
Version 3, 19 November 2007, published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License Version 3 for more details.
You should have received a copy of the License along with this program.
If not, see (https://www.gnu.org/licenses/).


---

The file `src/c_scipy_rectangular_lsap.h` is adapted from the `scipy` project
(https://scipy.org/scipylib/), source:
`/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp`.
Author: PM Larsen. Distributed under the BSD-3-Clause license.





## References


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
Hierarchical density estimates for data clustering, visualization,
and outlier detection,
*ACM Transactions on Knowledge Discovery from Data* **10**(1), 2015, 5:1–5:51.
doi:10.1145/2733381.

Crouse D.F., On implementing 2D rectangular assignment algorithms,
*IEEE Transactions on Aerospace and Electronic Systems* **52**(4), 2016,
pp. 1679-1696, doi:10.1109/TAES.2016.140952.

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
*Journal of Classification* **2**(1), 1985, pp. 193-218.

Rezaei M., Franti P., Set matching measures for external cluster validity,
*IEEE Transactions on Knowledge and Data Mining* **28**(8), 2016, pp. 2173-2186,
doi:10.1109/TKDE.2016.2551240

Vinh N.X., Epps J., Bailey J.,
Information theoretic measures for clusterings comparison:
Variants, properties, normalization and correction for chance,
*Journal of Machine Learning Research* **11**, 2010, pp. 2837-2854.
