# Changelog

## To Do

*   Check for NA/NaN/Inf in input matrices.

*   Bring back support for non-numeric data (needs updates in `deadwood`).

*   Add support for non-square confusion matrices in
    `normalized_pivoted_accuracy` and `normalized_clustering_accuracy`.


## (under development), to-be 1.3.0

*   The package was heavily refactored; common MST-related functions and classes
    as well as functions from the `tools` and `plots` modules were moved to
    the new [`deadwood`](https://deadwood.gagolewski.com/) package, which is
    now required.

*   [BACKWARD INCOMPATIBILITY]  Outlier detection based solely on whether
    a node is a leaf of a minimum spanning tree w.r.t. some mutual reachability
    distance turned out to be subpar in more detailed experiments,
    especially for smaller smoothing factors.  Note that in the previous
    versions of the package, this feature was deemed merely experimental;
    Hence, `detect_noise` in `genie.default` and `skip_leaves`, `preprocess`,
    and `postprocess` elsewhere are no longer available.  Instead, use the more
    universal `deadwood` package now.

*   [BACKWARD INCOMPATIBILITY]  `quitefastmst` version >= 0.9.1 is now required;
    the introduced backward-incompatible changes have been addressed.
    In particular, the definition of mutual reachability distances has changed.
    Unlike in Campello et al.'s 2013 paper, now the core distance is the
    distance to the *M*-th nearest neighbour, not the *(M-1)*-th one
    (not including self).

*   [Python] [BACKWARD INCOMPATIBILITY]  `internal` module was renamed `core`.

*   [BACKWARD INCOMPATIBILITY]  Deprecated functions such as `mst_from_nn`
    have been removed.

*   [Python] [BACKWARD INCOMPATIBILITY]  `compute_full_tree` is now always True.

*   [BUGFIX]  #92: Passing a non-square confusion matrix to
    `normalized_pivoted_accuracy` and `normalized_clustering_accuracy`
    yields an error as such objects are yet to be supported.

*   [R]  `gclust` and `genie` now return the computed MST via the `mst`
    object attribute.  `genie` returns an object of the class `mstclust`.
    This makes it operable with `deadwood`.

*   [Python] [BUGFIX]  Modifying `quitefastmst_params` via `set_state`
    now invalidates the cached MST.

*   [Python] [NEW FEATURE]  `plots.plot_scatter` has new arguments:
    `asp`, `markers`, and `colours`.  The module globals `mrk` and `col` were
    renamed accordingly.  However, as mentioned above, `plots` was
    moved to `deadwood`.

*   [Python] [BACKWARD INCOMPATIBILITY]  `compute_all_cuts` in `Genie` was
    renamed `coarser`.  If `True`, `labels_` is still a vector representing
    the requested `n_clusters`.  The coarser-grained labels are now stored
    in `labels_matrix_` whose `i`-th row represents an `(i+1)`-partition.


## 1.2.0 (2025-07-24)

*   [Python and R] Using the new implementation of Euclidean and mutual
    reachability minimum spanning trees (pretty fast in low dimensional spaces)
    from the [`quitefastmst`](https://quitefastmst.gagolewski.com/) package.

*   [BACKWARD INCOMPATIBILITY] [Python] Seeking approximate near-neighbours
    with `nmslib` is no longer supported directly; unfortunately, the package
    has not been updated for a while.

*   [BACKWARD INCOMPATIBILITY] `mlpack` is not used anymore.

*   [Python] `MSTClusterMixin`: A base class for Genie, GIc, and other MST-based
    clustering algorithms.  [later moved to `deadwood`]

*   [BACKWARD INCOMPATIBILITY] [Python] `Genie` and `GIc`: `affinity` was
    renamed `metric`.


## 1.1.6 (2024-08-22)

*   [Python] The package now works with `numpy` 2.0.


## 1.1.5 (2023-10-18)

*   [BACKWARD INCOMPATIBILITY] Inequality measures are no longer referred
    to as inequity measures.

*   [BACKWARD INCOMPATIBILITY] Some external cluster validity measures were
    renamed: `adjusted_asymmetric_accuracy` → `normalized_clustering_accuracy`,
    `normalized_accuracy` → `normalized_pivoted_accuracy`.

*   [BACKWARD INCOMPATIBILITY] [Python] `compare_partitions2` has been removed,
    as `compare_partitions` and other partition similarity scores
    now support both pairs of label vectors `(x, y)` and confusion matrices
    `(x=C, y=None)`.

*   [Python and R] New parameter to `pair_sets_index`: `clipped`.

*   In `normalizing_permutation` and external cluster validity measures,
    the input matrices can now be of the type `double`.

*   [BUGFIX] [Python] #80: Fixed adjustment for `nmslib_n_neighbors`
    in small samples.

*   [BUGFIX] [Python] #82: `cluster_validity` submodule not imported.

*   [BUGFIX] Some external cluster validity measures
    now handle NaNs better and are slightly less prone to round-off errors.


## 1.1.4 (2023-03-31)

*   [Python] The GIc algorithm is no longer marked as experimental;
    its description is provided in <https://doi.org/10.1007/s00357-024-09483-1>.


## 1.1.3 (2023-01-17)

*   [R] `mst.default` now throws an error if any element in the input matrix
    is missing/infinite.

*   [Python] The call to `mlpack.emst` that stopped working
    with the new version of `mlpack` has been fixed.


## 1.1.2 (2022-09-17)

*   [Python and R] `adjusted_asymmetric_accuracy`
    now accepts confusion matrices with fewer columns than rows.
    Such "missing" columns are now treated as if they were filled with 0s.

*   [Python and R] `pair_sets_index`, and `normalized_accuracy` return
    the same results for non-symmetric confusion matrices and transposes thereof.


## 1.1.1 (2022-09-15)

*   [Python] #75: `nmslib` is now optional.

*   [BUILD TIME] The use of `ssize_t` was not portable.


## 1.1.0 (2022-09-05)

*   [Python and R] New function: `adjusted_asymmetric_accuracy`.

*   [Python and R] Implementations of the so-called internal cluster
    validity measures discussed in
    DOI: [10.1016/j.ins.2021.10.004](https://doi.org/10.1016/j.ins.2021.10.004);
    see our (GitHub-only) [CVI](https://github.com/gagolews/optim_cvi) package
    for R. In particular, the generalised Dunn indices are based on the code
    originally authored by Maciej Bartoszuk. Thanks.

    Functions added (`cluster_validity` module):
    `calinski_harabasz_index`,
    `dunnowa_index`,
    `generalised_dunn_index`,
    `negated_ball_hall_index`,
    `negated_davies_bouldin_index`,
    `negated_wcss_index`,
    `silhouette_index`,
    `silhouette_w_index`,
    `wcnn_index`.

    These cluster validity measures are discussed
    in more detail at <https://clustering-benchmarks.gagolewski.com/>.

*   [BACKWARD INCOMPATIBILITY] `normalized_confusion_matrix`
    now solves the maximal assignment problem instead of applying
    the somewhat primitive partial pivoting.

*   [Python and R] New function: `normalizing_permutation`

*   [R] New function: `normalized_confusion_matrix`.

*   [Python and R] New parameter to `pair_sets_index`: `simplified`.

*   [Python] New parameters to `plots.plot_scatter`:
    `axis`, `title`, `xlabel`, `ylabel`, `xlim`, `ylim`.


## 1.0.1 (2022-08-08)

*   A paper on the *genieclust* package is now available:
    M. Gagolewski, genieclust: Fast and robust hierarchical clustering,
    SoftwareX 15, 100722, 2021, DOI:
    [10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).

*   [Python] `plots.plot_scatter` now uses a more accessible default palette
    (from R 4.0.0).

*   [Python and R] New function: `devergottini_index`.


## 1.0.0 (2021-04-22)

*   [R] Using `mlpack` instead of `RcppMLPACK` (#72).
    This package is merely suggested, not dependent upon.


## 0.9.8 (2021-01-08)

*   [Python] Require Python >= 3.7 (implied by `numpy`).

*   [Python] Require `nmslib`.

*   [R] Use `RcppMLPACK` directly; remove dependency on `emstreeR`.

*   [R] Use `tinytest` for unit testing instead of `testthat`.


## 0.9.4 (2020-07-31)

*   [BUGFIX] [R] Fixed build errors on Solaris.


## 0.9.3 (2020-07-25)

*   [BUGFIX] [Python] Added code coverage CI. Fixed some minor inconsistencies.
    Automated the `bdist` build chain.

*   [R] Updated DESCRIPTION to meet the CRAN policies.


## 0.9.2 (2020-07-22)

*   [BUGFIX] [Python] Fix broken build script for OS X with no OpenMP.


## 0.9.1 (2020-07-18)

*   The package has been completely rewritten.
    The core functionality is now implemented in C++ (with OpenMP).

*   [R] R version is now available.

*   [EXPERIMENTAL] A preliminary version of clustering with respect to
    DBSCAN*-like mutual reachability distances is now supported.

*   The parallelised Jarník-Prim algorithm now supports on-the-fly
    distance computations.  Euclidean minimum spanning tree can be determined
    with `mlpack`, which is much faster in low-dimensional spaces.

*   [EXPERIMENTAL] [Python] The GIc algorithm proposed by Anna Cena
    in her 2018 PhD thesis wad added.

*   [Python] Approximate version based on nearest neighbour graphs produced
    by `nmslib` was added.


## 0.1a2 (2018-05-23)

*   [Python] Initial PyPI release.
