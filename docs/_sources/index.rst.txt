Python and R Package `genieclust`: Fast and Robust Hierarchical Clustering with Noise Point Detection
=====================================================================================================


    **Genie finds meaningful clusters and is fast even on large data sets.**

    -- by `Marek Gagolewski <https://www.gagolewski.com/>`_


`genieclust` :cite:`genieclust` brings
a faster and more powerful version of **Genie** :cite:`genieins` — a robust
and outlier resistant clustering algorithm, originally published as an R package
`genie <https://cran.r-project.org/web/packages/genie/>`_.

The idea behind Genie is beautifully simple. First, make each individual
point the sole member of its own cluster. Then, keep merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes**
a point group of the smallest size will sometimes be matched with its nearest
neighbours.

Genie's appealing simplicity goes hand in hand with its usability;
it **often outperforms other clustering approaches**
such as K-means, BIRCH, or average, Ward, and complete linkage
on :any:`benchmark data <weave/benchmarks_ar>`\ .

Genie is also **very fast** — determining the whole cluster hierarchy
for datasets of millions of points can be completed within
:any:`a coffee break <weave/timings>`\ .
Therefore, it is perfectly suited for solving of **extreme clustering tasks**
(large datasets with any number of clusters to detect) for data
that fit into memory.
Thanks to the use of `nmslib` :cite:`nmslib`\ ,
sparse or string inputs are also supported.

Genie also allows clustering with respect to mutual reachability distances
so that it can act as a **noise point detector** or a robustified version
of `HDBSCAN\*` :cite:`hdbscan` that is able to detect a predefined
number of clusters and so it doesn't dependent on the `DBSCAN`'s somewhat
difficult-to-set `eps` parameter.



The Python language version of `genieclust` has
a familiar `scikit-learn`-like :cite:`sklearn_api` look-and-feel:

.. code-block:: python

    import genieclust
    X = ... # some data
    g = genieclust.Genie(n_clusters=2)
    labels = g.fit_predict(X)


The R language interface is compatible with ``hclust()``, but there is more.

.. code-block:: r

    X <- ... # some data
    h <- gclust(X)
    plot(h) # plot cluster dendrogram
    cutree(h, k=2)
    # or genie(X, k=2)


The `genieclust` package is available for Python (via
`PyPI <https://pypi.org/project/genieclust/>`_\ )
and R (on `CRAN <https://cran.r-project.org/web/packages/genieclust/>`_\ ).
Its source code is distributed
under the open source GNU AGPL v3 license and can be downloaded from
`GitHub <https://github.com/gagolews/genieclust>`_.
The core functionality is implemented in the form of a header-only C++
library, so it may relatively easily be adapted to new environments —
any contributions are welcome (Julia, Matlab, etc.).


.. toctree::
    :maxdepth: 2
    :caption: Examples and Tutorials

    weave/basics
    weave/sklearn_toy_example
    weave/benchmarks_ar
    weave/timings
    weave/noise
    weave/sparse
    weave/string
    weave/r

.. toctree::
    :maxdepth: 2
    :caption: API Documentation

    genieclust
    rapi
    news

.. toctree::
    :maxdepth: 2
    :caption: External Links

    Source Code (GitHub) <https://github.com/gagolews/genieclust>
    Bug Tracker and Feature Suggestions <https://github.com/gagolews/genieclust/issues>
    PyPI Entry <https://pypi.org/project/genieclust/>
    CRAN Entry <https://cran.r-project.org/web/packages/genieclust/>
    Author's Homepage <https://www.gagolewski.com/>

.. toctree::
    :maxdepth: 2
    :caption: Appendix

    weave/benchmarks_details
    weave/benchmarks_approx
    z_bibliography


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
