# [**genieclust**](https://genieclust.gagolewski.com/) Package for R and Python

## *Genie*: Fast and Robust Hierarchical Clustering with Noise Point Detection


![genieclust for Python](https://github.com/gagolews/genieclust/workflows/genieclust%20for%20Python/badge.svg)
![genieclust for R](https://github.com/gagolews/genieclust/workflows/genieclust%20for%20R/badge.svg)
![codecov](https://codecov.io/gh/gagolews/genieclust/branch/master/graph/badge.svg)


> **Genie finds meaningful clusters quickly – even on large data sets.**

> A comprehensive tutorial, benchmarks, and a reference manual is available
at <https://genieclust.gagolewski.com/>.

When using **genieclust** in research publications, please
cite (Gagolewski, 2021) and (Gagolewski, Bartoszuk, Cena, 2016)
as specified below. Thank you.


## About

A faster and more powerful version of *Genie* – a robust and outlier
resistant clustering algorithm (see Gagolewski, Bartoszuk, Cena, 2016),
originally included in the R package
[**genie**](https://CRAN.R-project.org/package=genie).

The idea behind Genie is beautifully simple. First, make each individual
point the only member of its own cluster. Then, keep merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes** a point group of
the *smallest* size will sometimes be matched with its nearest neighbour.

Genie's appealing simplicity goes hand in hand with its usability;
it **often outperforms other clustering approaches**
such as K-means, BIRCH, or average, Ward, and complete linkage
on [benchmark data](https://github.com/gagolews/clustering-benchmarks/).

Genie is also **very fast** – determining the whole cluster hierarchy
for datasets of millions of points can be completed within minutes.
Therefore, it is nicely suited for solving of **extreme clustering tasks**
(large datasets with any number of clusters to detect) for data (also sparse)
that fit into memory. Thanks to the use of
[**nmslib**](https://github.com/nmslib/nmslib), sparse or string inputs are also supported.

It also allows clustering with respect to mutual reachability distances
so that it can act as a **noise point detector** or a
robustified version of *HDBSCAN\**  (see Campello et al., 2015)
that is able to detect a predefined
number of clusters and hence it doesn't dependent on the *DBSCAN*'s somewhat
difficult-to-set `eps` parameter.



## Author and Contributors

**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com)

**Contributors**:
[Maciej Bartoszuk](http://bartoszuk.rexamine.com),
[Anna Cena](https://cena.rexamine.com) (R packages
[**genie**](https://CRAN.R-project.org/package=genie)
and [**CVI**](https://github.com/gagolews/optim_cvi)),
[Peter M. Larsen](https://github.com/pmla/)
([rectangular_lsap](https://github.com/scipy/scipy/blob/main/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp)).




## Examples, Tutorials, and Documentation

R's interface is compatible with `stats::hclust()`, but there is more.

```r
X <- ... # some data
h <- gclust(X)
plot(h) # plot cluster dendrogram
cutree(h, k=2)
# or genie(X, k=2)
```

The Python language version of **genieclust** has a familiar
**scikit-learn**-like look-and-feel:

```python
import genieclust
X = ... # some data
g = genieclust.Genie(n_clusters=2)
labels = g.fit_predict(X)
```

Tutorials and the package documentation are available
[here](https://genieclust.gagolewski.com/).

*To learn more about Python, check out Marek's recent open-access (free!) textbook*
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/).


## How to Install


### Python Version



To install via `pip` (see [PyPI](https://pypi.org/project/genieclust/)):

```bash
pip3 install genieclust
```

The package requires Python 3.7+ together with **cython** as well as
**numpy**, **scipy**, **matplotlib**, **nmslib**, and **scikit-learn**.
Optional dependency: **mlpack**.







### R Version


To install the most recent release, call:

```r
install.packages("genieclust")
```

See the package entry on
[CRAN](https://CRAN.R-project.org/package=genieclust).




### Other

The core functionality is implemented in the form of a header-only
C++ library. It can thus be easily adapted for use in
other environments.

Any contributions are welcome (e.g., Julia, Matlab, ...).




## License

Copyright (C) 2018–2022 Marek Gagolewski <https://www.gagolewski.com>

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

The file `src/c_scipy_rectangular_lsap.h` is adapted from the
**scipy** project (https://scipy.org/scipylib/), source:
`/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp`.
Author: Peter M. Larsen. Distributed under the BSD-3-Clause license.

The implementation of internal cluster validity measures
were adapted from our previous project (Gagolewski, Bartoszuk, Cena, 2021);
see [optim_cvi](https://github.com/gagolews/optim_cvi).
Originally distributed under the GNU Affero General Public License Version 3.


## References

Gagolewski M., genieclust: Fast and robust hierarchical clustering,
*SoftwareX* **15**, 2021, 100722.
[DOI: 10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and
outlier-resistant hierarchical clustering algorithm, *Information
Sciences* **363**, 2016, 8–23.
[DOI: 10.1016/j.ins.2016.05.003](https://doi.org/10.1016/j.ins.2016.05.003).

Gagolewski M., Bartoszuk M., Cena A., Are cluster validity measures (in)valid?,
*Information Sciences* **581**, 2021, 620–636.
[DOI: 10.1016/j.ins.2021.10.004](https://doi.org/10.1016/j.ins.2021.10.004).

Gagolewski M., *Adjusted asymmetric accuracy: A well-behaving external
cluster validity measure*, 2022, submitted for publication.

Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*.
2022, <https://clustering-benchmarks.gagolewski.com>.

Cena A., *Adaptive hierarchical clustering algorithms based on data
aggregation methods*, PhD Thesis, Systems Research Institute, Polish
Academy of Sciences, 2018.

Campello R., Moulavi D., Zimek A., Sander J., Hierarchical density
estimates for data clustering, visualization, and outlier detection,
*ACM Transactions on Knowledge Discovery from Data* **10**(1), 2015, 5:1–5:51.
[DOI: 10.1145/2733381](https://doi.org/10.1145/2733381).

Mueller A., Nowozin S., Lampert C.H., Information Theoretic Clustering
using Minimum Spanning Trees, *DAGM-OAGM*, 2012.

See the package's [homepage](https://genieclust.gagolewski.com) for more
references.
