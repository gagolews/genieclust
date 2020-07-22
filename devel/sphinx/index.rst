.. genieclust documentation master file, created by
   sphinx-quickstart on Sun Jun 28 11:34:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`genieclust`: Fast and Robust Hierarchical Clustering with Noise Point Detection
================================================================================

    **Genie finds meaningful clusters and is fast even on large data sets.**

    -- by `Marek Gagolewski <https://www.gagolewski.com/>`_


A faster and more powerful version of **Genie** [1]_ — a robust and outlier resistant
clustering algorithm, originally published as an R package
`genie <https://cran.r-project.org/web/packages/genie/>`_.

The idea behind Genie is very simple. First, make each individual
point the only member of its own cluster. Then, keep merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes**
a point group of the smallest size is sometimes matched with its nearest
neighbours.



Genie's appealing simplicity goes hand in hand with its usability;
it **often outperforms other clustering approaches**
such as K-means, BIRCH, or average, Ward, and complete linkage
on `benchmark data <https://github.com/gagolews/clustering_benchmarks_v1/>`_.

Genie is also **very fast** — determining the whole cluster hierarchy
for datasets of millions of points can be completed within :any:`a coffee break <rmd/timings>`\ .
Therefore, it is perfectly suited for solving of **extreme clustering tasks**
(large datasets with any number of clusters to detect) for data (also sparse)
that fit into memory.
Thanks to the use of `nmslib` [3]_, sparse or string inputs are also supported.

It also allows clustering with respect to mutual reachability distances
so that it can act as a **noise point detector** or a
robustified version of `HDBSCAN\*` [2]_ that is able to detect a predefined
number of clusters and hence it doesn't dependent on the `DBSCAN`'s somehow
difficult-to-set `eps` parameter.



The Python language version of `genieclust` has
a familiar `scikit-learn`-like look-and-feel:

.. code-block:: python

    import genieclust
    X = ... # some data
    g = genieclust.Genie(n_clusters=2)
    labels = g.fit_predict(X)


R's interface is compatible with ``hclust()``, but there is more.

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
Note that the core functionality is implemented in the form of a header-only C++
library, hence it might be relatively easily adapted to new environments.


.. toctree::
    :maxdepth: 2
    :caption: Examples and Tutorials

    rmd/basics
    rmd/sklearn_toy_example
    rmd/benchmarks_ar
    rmd/timings
    rmd/noise
    rmd/sparse
    rmd/string
    rmd/r

.. toctree::
    :maxdepth: 2
    :caption: API Documentation

    genieclust
    r

.. toctree::
    :maxdepth: 2
    :caption: External Links

    Source code (GitHub) <https://github.com/gagolews/genieclust>
    Issues and Splendid Ideas Tracker <https://github.com/gagolews/genieclust/issues>

.. toctree::
    :maxdepth: 2
    :caption: Appendix

    rmd/benchmarks_details
    rmd/benchmarks_approx




References
----------

.. [1]
    Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    *Information Sciences* 363, 2016, 8-23.
    doi:10.1016/j.ins.2016.05.003.

.. [2]
    Campello R., Moulavi D., Zimek A., Sander J.,
    Hierarchical density estimates for data clustering, visualization,
    and outlier detection,
    *ACM Transactions on Knowledge Discovery from Data* 10(1), 2015, 5:1-5:51.
    doi:10.1145/2733381.

.. [3]
    Naidan B., Boytsov L., Malkov Y.,  Novak D.,
    *Non-metric space library (NMSLIB) manual*, version 2.0, 2019.
    https://github.com/nmslib/nmslib/blob/master/manual/latex/manual.pdf.



Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
