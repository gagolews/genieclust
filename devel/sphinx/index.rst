.. genieclust documentation master file, created by
   sphinx-quickstart on Sun Jun 28 11:34:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`genieclust`: Fast and Robust Hierarchical Clustering with Noise Point Detection
================================================================================

    **Genie outputs meaningful clusters and is fast even for large data sets.**

    -- by `Marek Gagolewski <https://www.gagolewski.com/>`_


A reimplementation of **Genie** - a robust and outlier resistant
clustering algorithm,
originally published as an R package
`genie <https://cran.r-project.org/web/packages/genie/>`_.

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
on `benchmark data <https://github.com/gagolews/clustering_benchmarks_v1/>`_.

Genie is also pretty fast -- determining the whole cluster hierarchy
for datasets of 10M points in low dimensional Euclidean spaces or
100K points in high dimensional ones takes 1-2 minutes.

It allows clustering with respect to mutual reachability distances
so that it can act as a noise point detector or a robustified version of
HDBSCAN* (that is able to detect a predefined number of
clusters and hence it doesn't dependent on the DBSCAN's somehow
difficult-to-set `eps` parameter).



Source code is available at
`https://github.com/gagolews/genieclust <https://github.com/gagolews/genieclust>`_.






.. toctree::
    :maxdepth: 2
    :caption: Contents:

    genieclust



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
