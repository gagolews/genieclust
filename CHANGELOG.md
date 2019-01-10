`genieclust` package CHANGELOG
==============================

* to do:

  * allow approximate nearest neighbors, e.g.,
  http://www.cs.ubc.ca/research/flann/
  https://github.com/facebookresearch/faiss or
  https://github.com/spotify/annoy
  https://github.com/nmslib/nmslib  [NN-descent...]
  https://github.com/nmslib/hnswlib ;
  some will also enable sparse input data (e.g., for text mining applications)

  * Rewrite `genie_from_mst` in C++.

  * Add support for other `scipy.spatial` distances when computing
  an exact MST, in particular, the weighted Euclidean metric.

  * Add support for sparse input matrices

  * Add support for connectivity matrices

  * Output full cluster hierarchy - see the Z matrix in
  `scipy.cluster.hierarchy.linkage`

  * Implement Ania's and Adreas' linkage criteria

  * [INTERNAL] [VERY LOW PRIORITY] make DisjointSets and GiniDisjointSets
  serializable: implement the `__setstate__(self, state)` and
  `__getstate__(self)` methods.

  * Add genieclust.plots.abline()

* genieclust 0.XX (under development)

  * The full distance matrix is not required now for computing an exact MST -
  the distances are computed on the fly; this is currently supported
  for `"euclidean"`, `"cityblock"`, and `"cosine"` distances.
  This saves a lot of memory ($O(n)$ instead of $O(n^2)$) and so genieclust
  can solve much larger problems now.

  * `sklearn.neighbors.NearestNeighbors` are used to determine few first NNs
  when computing the mutual reachability distance a.k.a. the "core distance",
  see core_distance()

  * [INTERNAL] Most of the code was rewritten in C++, in particular
  the `DisjointSets` and `GiniDisjointSets` classes, so that:
  a) they can be used in other projects,
  b) genieclust can be easily made available in other
  environments in the future.

  * [INTERNAL] Use OpenMP for distance computations.

  * [BUGFIX] Internal function MST_pair() did not return all weights
  of the MST edges.

* genieclust 0.1a3 (unreleased)

  * The `Genie` class constructor has a new parameter: `postprocess`,
  which allows for merging boundary points with their nearest core points
  (in effect for smoothing parameters M>1).

* genieclust 0.1a2 (2018-05-23)

  * Initial PyPI release.
