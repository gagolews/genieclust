`genieclust` package CHANGELOG
==============================

* to do:

    * [INTERNAL] [VERY LOW PRIORITY] make DisjointSets and GiniDisjointSets
    serializable: implement the `__setstate__(self, state)` and
    `__getstate__(self)` methods.

    * use `sklearn.neighbors.NearestNeighbors` to determine the M-th NNs
    when computing the mutual reachability distance a.k.a. the "core distance",
    see core_distance()

    * allow approximate nearest neighbors, e.g.,
    http://www.cs.ubc.ca/research/flann/
    https://github.com/facebookresearch/faiss or
    https://github.com/spotify/annoy
    https://github.com/nmslib/nmslib  [NN-descent...]
    https://github.com/nmslib/hnswlib ;
    some will also enable sparse input data (e.g., for text mining applications)

    * do not require the full distance matrix when computing MST_pair;
    compute the distances on the fly (perhaps taking into account the core dist)

        > [how to handle different distances, say, we can support
        > “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or ‘precomputed’
        > just like `sklearn.cluster.AgglomerativeClustering` does]?
        > [add support for the "connectivity" matrix]

    * output full cluster hierarchy - see the Z matrix in
    `scipy.cluster.hierarchy.linkage`

    * implement Ania's and Adreas' linkage criteria


* genieclust 0.XX (under development)

    * [INTERNAL] refactored DisjointSets and GiniDisjointSets as C++ classes.

    * [BUGFIX] MST_pair() did not return all weights of the MST edges.



* genieclust 0.1a3 (unreleased)

    * The `Genie` class constructor has a new parameter: `postprocess`,
    which allows for merging boundary points with their nearest core points
    (in effect for smoothing parameters M>1).


* genieclust 0.1a2 (2018-05-23)

    * Initial PyPI release.
