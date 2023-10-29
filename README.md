# [**genieclust**](https://genieclust.gagolewski.com/) Package for R and Python

## *Genie*: Fast and Robust Hierarchical Clustering with Noise Point Detection


![genieclust for Python](https://github.com/gagolews/genieclust/workflows/genieclust%20for%20Python/badge.svg)
![genieclust for R](https://github.com/gagolews/genieclust/workflows/genieclust%20for%20R/badge.svg)


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
on [benchmark data](https://github.com/gagolews/clustering-benchmarks).
Of course, there is no, nor will there ever be, a single best
universal clustering approach for every kind of problem, but Genie
is definitely worth a try!

Thanks to its being based on minimal spanning trees of the pairwise distance
graphs, Genie is also **very fast** – determining the whole cluster hierarchy
for datasets of millions of points can be completed within minutes. Therefore,
it is nicely suited for solving **extreme clustering tasks** (large datasets
with any number of clusters to detect) for data (also sparse) that fit into
memory. Thanks to the use of [**nmslib**](https://github.com/nmslib/nmslib)
(if available), sparse or string inputs are also supported.

It also allows clustering with respect to mutual reachability distances
so that it can act as a **noise point detector** or a
robustified version of *HDBSCAN\**  (see Campello et al., 2013)
that is able to detect a predefined
number of clusters and hence it doesn't dependent on the *DBSCAN*'s somewhat
difficult-to-set `eps` parameter.


The package also features an implementation of economic inequality indices
(the Gini, Bonferroni index), external cluster validity measures
(e.g., the normalised clustering accuracy and partition similarity scores
such as the adjusted Rand, Fowlkes-Mallows, adjusted mutual information,
and the pair sets index), and internal cluster validity indices
(e.g., the Calinski-Harabasz, Davies-Bouldin, Ball-Hall, Silhouette,
and generalised Dunn indices).


## Author and Contributors

**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com/)

Contributors:
Maciej Bartoszuk,
[Anna Cena](https://cena.rexamine.com/) (R packages
[**genie**](https://CRAN.R-project.org/package=genie)
and [**CVI**](https://github.com/gagolews/optim_cvi)),
[Peter M. Larsen](https://github.com/pmla)
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


*To learn more about R, check out Marek's open-access (free!) textbook*
[Deep R Programming](https://deepr.gagolewski.com/).



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



To install via `pip` (see [PyPI](https://pypi.org/project/genieclust)):

```bash
pip3 install genieclust
```

The package requires Python 3.7+ together with **cython** as well as
**numpy**, **scipy**, **matplotlib**, and **scikit-learn**.
Optional dependencies: **nmslib** and **mlpack**.







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

Copyright (C) 2018–2023 Marek Gagolewski <https://www.gagolewski.com/>

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
**scipy** project (https://scipy.org/scipylib), source:
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
<https://genieclust.gagolewski.com/>.

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and
outlier-resistant hierarchical clustering algorithm, *Information
Sciences* **363**, 2016, 8–23.
[DOI: 10.1016/j.ins.2016.05.003](https://doi.org/10.1016/j.ins.2016.05.003).

Gagolewski M., Bartoszuk M., Cena A., Are cluster validity measures (in)valid?,
*Information Sciences* **581**, 2021, 620–636.
[DOI: 10.1016/j.ins.2021.10.004](https://doi.org/10.1016/j.ins.2021.10.004).

Gagolewski M., Cena A., Bartoszuk M., Brzozowski L.,
Clustering with minimum spanning trees: How good can it be?, 2023,
under review (preprint),
[DOI: 10.48550/arXiv.2303.05679](https://doi.org/10.48550/arXiv.2303.05679).

Gagolewski M., Normalised clustering accuracy: An asymmetric external
cluster validity measure, 2023, under review (preprint),
[DOI: 10.48550/arXiv.2209.02935](https://doi.org/10.48550/arXiv.2209.02935).

Gagolewski M., A framework for benchmarking clustering algorithms,
*SoftwareX* **20**, 2022, 101270.
[DOI: 10.1016/j.softx.2022.101270](https://doi.org/10.1016/j.softx.2022.101270).
<https://clustering-benchmarks.gagolewski.com/>.

Campello R.J.G.B., Moulavi D., Sander J.,
Density-based clustering based on hierarchical density estimates,
*Lecture Notes in Computer Science* **7819**, 2013, 160–172.
[DOI: 10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14).

Mueller A., Nowozin S., Lampert C.H., Information theoretic clustering
using minimum spanning trees, *DAGM-OAGM*, 2012.

Rezaei M., Fränti P., Set matching measures for external cluster validity,
*IEEE Transactions on Knowledge and Data Engineering* **28**(8), 2016,
2173–2186 [DOI: 10.1109/TKDE.2016.2551240](https://doi.org/10.1109/TKDE.2016.2551240).

See the package's [homepage](https://genieclust.gagolewski.com/) for more
references.
