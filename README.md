<a href="https://genieclust.gagolewski.com/"><img src="https://www.gagolewski.com/_static/img/genieclust.png" align="right" height="128" width="128" /></a>
# [**genieclust**](https://genieclust.gagolewski.com/) Package for R and Python

### *Genie*: Fast and Robust Hierarchical Clustering with Outlier Detection

<!--
![genieclust for Python](https://github.com/gagolews/genieclust/workflows/genieclust%20for%20Python/badge.svg)
![genieclust for R](https://github.com/gagolews/genieclust/workflows/genieclust%20for%20R/badge.svg)
-->

> **Genie finds meaningful clusters. It does so quickly, even in large datasets.**
>
> A comprehensive tutorial, benchmarks, and a reference manual is available
at <https://genieclust.gagolewski.com/>.

When using **genieclust** in research publications, please
cite (Gagolewski, 2021) and (Gagolewski, Bartoszuk, Cena, 2016)
as specified below. Thank you.


## About

*Genie* is a robust and outlier-resistant hierarchical clustering algorithm
(see Gagolewski, Bartoszuk, Cena, 2016). Its original implementation was
included in the R package [**genie**](https://CRAN.R-project.org/package=genie).
This is its faster and more capable variant.

The idea behind *Genie* is beautifully simple. First, make each individual
point the only member of its own cluster. Then, keep merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes**, a point group of
the *smallest* size is sometimes combined with its nearest counterpart.

*Genie*'s appealing simplicity goes hand in hand with its usability.
It **often outperforms other clustering approaches**
such as K-means, BIRCH, or average, complete, and Ward's linkage
on [benchmark data](https://github.com/gagolews/clustering-benchmarks).
Of course, there is no, nor will there ever be, a single best
universal clustering approach for every kind of problem, but Genie
is definitely worth a try.

*Genie* is based on minimal spanning trees of pairwise distance graphs.
Thus, it can also be pretty **fast**: thanks to
[**quitefastmst**](https://quitefastmst.gagolewski.com/),
determining the entire cluster hierarchy for datasets containing millions
of points can be completed in minutes. Therefore, it is well suited to solving
**extreme clustering tasks** (involving large datasets with a high number
of clusters to detect).

**genieclust** allows clustering with respect to mutual reachability distances,
enabling it to act as an **outlier detector** or an alternative to
*HDBSCAN\** (see Campello et al., 2013) that can identify a predefined
number of clusters or their entire hierarchy.  Notably, it
doesn't depend on *DBSCAN*'s somewhat difficult to set `eps` parameter.

The package also features an implementation of:

* economic inequality indices (the Gini, Bonferroni, or de Vergottini index),
* external cluster validity measures (e.g., the normalised clustering accuracy
    and partition similarity indices such as the adjusted Rand, Fowlkes-Mallows,
    or mutual information scores),
* internal cluster validity indices (e.g., the Calinski-Harabasz,
    Davies-Bouldin, Ball-Hall, Silhouette, or generalised Dunn indices).



## Author and Contributors

**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com/)

Contributors:
Maciej Bartoszuk, Anna Cena (R packages
[**genie**](https://CRAN.R-project.org/package=genie) and
[**CVI**](https://github.com/gagolews/optim_cvi)), and
[Peter M. Larsen](https://github.com/pmla)
([`rectangular_lsap`](https://github.com/scipy/scipy/blob/main/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp)).




## Examples, Tutorials, and Documentation

The R interface is compatible with `stats::hclust()`,
but there is more:

```r
X <- ...  # some data
h <- gclust(X)
plot(h)  # plot cluster dendrogram
cutree(h, k=2)
# or simply:  genie(X, k=2)
```


*To learn more about R, check out Marek's open-access textbook*
[Deep R Programming](https://deepr.gagolewski.com/).



The Python language version of **genieclust** has a **scikit-learn**-like API:

```python
import genieclust
X = ...  # some data
g = genieclust.Genie(n_clusters=2)
labels = g.fit_predict(X)
```

Tutorials and the package documentation are available
[here](https://genieclust.gagolewski.com/).

*To learn more about Python, check out Marek's open-access textbook*
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/).



## How to Install

### Python Version

To install from [PyPI](https://pypi.org/project/genieclust), call:

```bash
pip3 install genieclust  # python3 -m pip install genieclust
```

The package requires Python 3.9+ with
**cython**,
**numpy**,
**scikit-learn**,
**matplotlib**,
and
[**quitefastmst**](https://quitefastmst.gagolewski.com/).


### R Version


To install from [CRAN](https://CRAN.R-project.org/package=genieclust), call:

```r
install.packages("genieclust")
```





### Other

The core functionality is implemented in the form of a header-only
C++ library. It can thus be easily adapted for use in other projects.

New contributions are welcome, e.g., Julia, Matlab/GNU Octave wrappers.




## License

Copyright (C) 2018–2026 Marek Gagolewski <https://www.gagolewski.com/>

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
see [`optim_cvi`](https://github.com/gagolews/optim_cvi).
Originally distributed under the GNU Affero General Public License Version 3.


## References

Gagolewski, M., genieclust: Fast and robust hierarchical clustering,
*SoftwareX* **15**, 2021, 100722.
[DOI: 10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).
<https://genieclust.gagolewski.com/>.

Gagolewski, M., Bartoszuk, M., Cena, A., Genie: A new, fast, and
outlier-resistant hierarchical clustering algorithm, *Information
Sciences* **363**, 2016, 8–23.
[DOI: 10.1016/j.ins.2016.05.003](https://doi.org/10.1016/j.ins.2016.05.003).

Gagolewski, M., Bartoszuk, M., Cena, A., Are cluster validity measures (in)valid?,
*Information Sciences* **581**, 2021, 620–636.
[DOI: 10.1016/j.ins.2021.10.004](https://doi.org/10.1016/j.ins.2021.10.004).

Gagolewski, M., Cena, A., Bartoszuk, M., Brzozowski, L.,
Clustering with minimum spanning trees: How good can it be?,
*Journal of Classification* **42**, 2025, 90–112.
[DOI: 10.1007/s00357-024-09483-1](https://doi.org/10.1007/s00357-024-09483-1).

Gagolewski, M., Normalised clustering accuracy: An asymmetric external
cluster validity measure, *Journal of Classification* **42**, 2025, 2–30.
[DOI: 10.1007/s00357-024-09482-2](https://doi.org/10.1007/s00357-024-09482-2).

Gagolewski, M., A framework for benchmarking clustering algorithms,
*SoftwareX* **20**, 2022, 101270.
[DOI: 10.1016/j.softx.2022.101270](https://doi.org/10.1016/j.softx.2022.101270).
<https://clustering-benchmarks.gagolewski.com/>.

Campello, R.J.G.B., Moulavi, D., Sander, J.,
Density-based clustering based on hierarchical density estimates,
*Lecture Notes in Computer Science* **7819**, 2013, 160–172.
[DOI: 10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14).

Mueller, A., Nowozin, S., Lampert, C.H., Information theoretic clustering
using minimum spanning trees, *DAGM-OAGM*, 2012.

See **genieclust**'s [homepage](https://genieclust.gagolewski.com/) for more
references.
