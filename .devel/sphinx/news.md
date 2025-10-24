# Changelog

## To Do

*   Check for NA/NaN/Inf in the input matrices.

*   For M>1 and non-Euclidean distances, the output MST edge weights
    are slightly perturbed because of the applied continuity correction.
    These values should be manually corrected at the postprocessing stage
    after/in a call to `Cmst_from_complete`.

*   For M>1, add an option to "preprocess" that marks nodes incident to
    cut edges as midliers/outliers.


## (under development), to-be 1.3.0

*   `quitefastmst` version >= 0.9.1 is now required; the introduced
    backward-incompatible changes have been addressed (see below).

*   [NEW FEATURE]  Outlier detection based on mutual reachability distances
    is no longer marked as experimental.  Note that we now rely on mutual
    reachability distances adjusted for the presence of ties such that
    neighbours with smaller core distances are preferred;
    moreover, some leaves of the MST are reconnected so that they
    are adjacent to vertices that have them amongst their *M* nearest
    neighbours; see (in preparation: Gagolewski, 2025, TODO) for discussion.

*   [BACKWARD INCOMPATIBILITY]  The definition of the mutual reachability
    distance has changed.  Unlike in Campello et al.'s 2013 paper,
    now the core distance is the distance to the *M*-th nearest neighbour,
    not the *(M-1)*-th one (not including self).

*   [BACKWARD INCOMPATIBILITY]  `detect_noise` in `genie.default`
    was renamed `skip_leaves`.

*   [BACKWARD INCOMPATIBILITY]  `postprocess` can now be one of
    `"midliers"`, `"none"`, and `"all"`.

*   [Python] [BUGFIX] Modifying `quitefastmst_params` via `set_state`
    now invalidates the cached MST.


## 1.2.0 (2025-07-24)

*   [Python and R] Using the new implementation of Euclidean and mutual
    reachability minimum spanning trees (quite fast in low dimensional spaces)
    from the [`quitefastmst`](https://quitefastmst.gagolewski.com/) package.

*   [BACKWARD INCOMPATIBILITY] `mlpack` is not used anymore.

*   [BACKWARD INCOMPATIBILITY] [Python] Seeking approximate near-neighbours
    with `nmslib` is no longer supported directly; unfortunately, the package
    has not been updated for a while.

*   [Python] `MSTClusterMixin(BaseEstimator, ClusterMixin)`: A base class for
    Genie, GIc, and other MST-based clustering algorithms.

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
