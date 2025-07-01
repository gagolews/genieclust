# *genieclust*: Fast and Robust Hierarchical Clustering with Noise Point Detection

::::{epigraph}
**Genie finds meaningful clusters quickly – even on large data sets.**
::::

::::{image} _static/img/genie_toy_example.png
:class: img-right-align-always
:alt: Genie
:width: 128px
::::


The *genieclust* package {cite}`genieclust` for Python and R implements
a robust and outlier resistant clustering algorithm called *Genie* {cite}`genieins`.

The idea behind *Genie* is beautifully simple. First, make each individual
point the sole member of its own cluster. Then, keep merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes** a point group of the
smallest size will sometimes be combined with its nearest counterpart.

Genie's appealing simplicity goes hand in hand with its usability;
it **often outperforms other clustering approaches**
such as K-means, BIRCH, or average, Ward, and complete linkage
on various kinds of {any}`benchmark dataset <weave/benchmarks_ar>`.
Of course, there is no, nor will there ever be, a single best
universal clustering approach for every kind of problem, but Genie
is definitely worth a try!

Genie is based on minimum spanning trees {cite}`cvimst`
of the pairwise distance graphs. Thus, it can also be pretty **fast**:
determining the whole cluster hierarchy for datasets of millions of points
can be completed within minutes. Therefore, it is nicely suited for solving
**extreme clustering tasks** (large datasets with a high number of clusters
to detect).

Genie also allows clustering with respect to mutual reachability distances
so that it can act as a **noise point detector** or a robustified version
of *HDBSCAN\** {cite}`hdbscan` that is able to identify a predefined
number of clusters (actually, their whole hierarchy. The good news is that it
doesn't dependent on the *DBSCAN*'s somewhat difficult-to-set `eps` parameter.



## Python Version

The **Python version** of *genieclust* is available via
[PyPI](https://pypi.org/project/genieclust/), e.g.,
via a call to:

```bash
pip3 install genieclust
```

from the command line or through your favourite package manager.
Note the *scikit-learn*-like {cite}`sklearn_api` API:

```python
import genieclust
X = ...  # some data
g = genieclust.Genie(n_clusters=2)
labels = g.fit_predict(X)
```

::::{note}
*To learn more about Python, check out Marek's recent open-access (free!) textbook*
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/)
{cite}`datawranglingpy`.
::::



## R Version

The **R version** of *genieclust* can be downloaded from
[CRAN](https://cran.r-project.org/web/packages/genieclust/)
by calling:

```r
install.packages("genieclust")
```

Its interface is compatible with the classic `stats::hclust()`,
but there is more:

```r
X <- ...  # some data
h <- gclust(X)
plot(h)  # plot cluster dendrogram
cutree(h, k=2)
# or simply:  genie(X, k=2)
```

::::{note}
*To learn more about R, check out Marek's recent open-access (free!) textbook*
[Deep R Programming](https://deepr.gagolewski.com/)
{cite}`deepr`.
::::


## Package Features

The implemented algorithms include:

-  *Genie++* – a reimplementation of the original Genie algorithm
    from the R package [*genie*](https://cran.r-project.org/web/packages/genie)
    {cite}`genieins`; much faster than the original one;
    supports arbitrary spanning forests;

-   *Genie+HDBSCAN\** – a robustified (Geniefied) retake on the *HDBSCAN\**
    {cite}`hdbscan` method that detects noise points in data and
    outputs clusters of predefined sizes.

Other features:

-   inequality measures: the normalised Gini, Bonferroni,
    and De Vergottini indices;

-   external cluster validity measures (see {cite}`nca,clustering-benchmarks`
    for discussion): normalised clustering accuracy (NCA) and partition
    similarity scores such as normalised pivoted accuracy (NPA),
    adjusted/unadjusted Rand (AR), adjusted/unadjusted Fowlkes–Mallows (FM),
    adjusted/normalised/unadjusted mutual information (MI) indices;

-   internal cluster validity measures (see {cite}`cvi` for discussion):
    the Caliński–Harabasz, Silhouette, Ball–Hall, Davies–Bouldin,
    generalised Dunn indices, etc.;

-   *(Python only)* union-find (disjoint sets) data structures (with
    extensions);

-   *(Python only)* some R-like plotting functions.



## Contributing

*genieclust* is distributed under the open source GNU AGPL v3 license
and can be downloaded from [GitHub](https://github.com/gagolews/genieclust).
The core functionality is implemented in the form of a header-only C++
library, so it may be adapted to new environments relatively easily:
any valuable contributions are welcome (Julia or Matlab bindings, etc.).


**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com)

Contributors:
[Maciej Bartoszuk](http://bartoszuk.rexamine.com) and
[Anna Cena](https://cena.rexamine.com)
(*genieclust*'s predecessor [*genie*](https://cran.r-project.org/web/packages/genie) {cite}`genieins`
and some internal cluster validity measures [*CVI*](https://github.com/gagolews/optim_cvi)  {cite}`cvi`);
[Peter M. Larsen](https://github.com/pmla/)
(an [implementation](https://github.com/scipy/scipy/blob/main/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp)
of the shortest augmenting path algorithm for the rectangular assignment problem
which we use for computing some external cluster validity measures {cite}`genieins,psi`).



::::{toctree}
:maxdepth: 2
:caption: genieclust
:hidden:

About <self>
Author <https://www.gagolewski.com/>
Source Code (GitHub) <https://github.com/gagolews/genieclust>
Bug Tracker and Feature Suggestions <https://github.com/gagolews/genieclust/issues>
PyPI Entry <https://pypi.org/project/genieclust/>
CRAN Entry <https://CRAN.R-project.org/package=genieclust>
::::


::::{toctree}
:maxdepth: 2
:caption: Examples and Tutorials
:hidden:

weave/basics
weave/sklearn_toy_example
weave/benchmarks_ar
weave/timings
weave/noise
weave/r
::::

<!--
weave/sparse
weave/string
require nmslib!
which cannot be installed currently (hasn't been updated for a while)
-->


::::{toctree}
:maxdepth: 1
:caption: API Documentation
:hidden:

genieclust
rapi
::::


::::{toctree}
:maxdepth: 1
:caption: See Also
:hidden:

Clustering Benchmarks <https://clustering-benchmarks.gagolewski.com>
Minimalist Data Wrangling in Python <https://datawranglingpy.gagolewski.com/>
Deep R Programming <https://deepr.gagolewski.com>
::::


::::{toctree}
:maxdepth: 1
:caption: Appendix
:hidden:

news
weave/benchmarks_details
weave/benchmarks_approx
z_bibliography
::::


<!--
Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
-->
