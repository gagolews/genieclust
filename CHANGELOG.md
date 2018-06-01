`genieclust` package CHANGELOG
==============================

* to do:

    * make Disjoint sets serializable: implement the
    `__setstate__(self, state)` and `__getstate__(self)` methods.

    * make GiniDisjoint sets serializable: implement the
    `__setstate__(self, state)` and `__getstate__(self)` methods.

    * use `sklearn.neighbors.NearestNeighbors` to determine the M-th NNs
    when computing the mutual reachability distance a.k.a. the "core distance"

    * do not require the full distance matrix when computing MST_pair;
    compute the distances on the fly (perhaps taking into account the core dist)

        > [how to handle different distances, say, we can support
        “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or ‘precomputed’
        just like sklearn.cluster.AgglomerativeClustering does]?
        > [add support for the "connectivity" matrix]

    * output full cluster hierarchy - see the Z matrix in
    `scipy.cluster.hierarchy.linkage`

    * Implement Ania's and Adreas' linkage criteria

    * post-process leaves which are border pts = noncore objects within
    eps-neighborhoods of core objects


* genieclust 0.XX (under development)

    * ...


* genieclust 0.1a2 (2018-05-23)

    * initial PyPI release
