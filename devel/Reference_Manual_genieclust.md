<!-- toc -->

July 02, 2020

# DESCRIPTION

```
Package: genieclust
Type: Package
Title: The Genie++ Hierarchical Clustering Algorithm with Noise Points
        Detection
Version: 0.9.1
Date: 2020-06-28
Authors@R: c(
    person("Marek", "Gagolewski",
    role = c("aut", "cre"),
    email = "marek@gagolewski.com",
    comment = c(ORCID = "0000-0003-0637-6028"))
    )
Description: A reimplementation of the Genie algorithm - a robust
    hierarchical clustering method
    (Gagolewski, Bartoszuk, Cena, 2016 <DOI:10.1016/j.ins.2016.05.003>).
    Now faster and more memory efficient; determining the whole hierarchy
    for datasets of 10M points in low dimensional Euclidean spaces or
    100K points in high dimensional ones takes 1-2 minutes.
    Allows clustering with respect to mutual reachability distances
    so that it can act as a noise point detector or a robustified version of
    HDBSCAN* (that is able to detect a predefined number of
    clusters and hence it does not dependent on the DBSCAN's somehow
    fragile `eps` parameter).
    The package also features an implementation of economic inequity indices
    (the Gini, Bonferroni index) and external cluster validity measures
    (partition similarity scores; e.g., the adjusted Rand, Fowlkes-Mallows,
    adjusted mutual information, pair sets index).
    See also the Python version of `genieclust` available on PyPI, which
    supports sparse data, more metrics, and larger datasets.
BugReports: https://github.com/gagolews/genieclust/issues
URL: https://genieclust.gagolewski.com/
License: AGPL-3
Imports: Rcpp (>= 1.0.4), stats, utils
Suggests: datasets, emstreeR
LinkingTo: Rcpp
Encoding: UTF-8
SystemRequirements: OpenMP, C++11
RoxygenNote: 7.1.0
Roxygen: list(markdown = TRUE)
Author: Marek Gagolewski [aut, cre] (<https://orcid.org/0000-0003-0637-6028>)
Maintainer: Marek Gagolewski <marek@gagolewski.com>
Built: R 4.0.1; x86_64-pc-linux-gnu; 2020-07-02 05:15:10 UTC; unix```


