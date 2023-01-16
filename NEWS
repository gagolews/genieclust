# What Is New in *genieclust*


## 1.1.3 (2023-01-17)

*  [R] `mst.default` now throws an error if any element in the input matrix
   is missing/infinite.

*  [Python] Fixed call to `mlpack.emst` that no longer worked
   with the new version of `mlpack`.


## 1.1.2 (2022-09-17)

*  [Python and R] `adjusted_asymmetric_accuracy`
   now accepts confusion matrices with fewer columns than rows.
   Such "missing" columns are now treated as if they were filled with 0s.

*  [Python and R] `pair_sets_index`, and `normalized_accuracy` return
   the same results for non-symmetric confusion matrices and transposes thereof.


## 1.1.1 (2022-09-15)

*  [Python] #75: `nmslib` is now optional.

*  [BUILD TIME]: The use of `ssize_t` was not portable.


## 1.1.0 (2022-09-05)

*  [GENERAL] The below-mentioned cluster validity measures are discussed
   in more detail at <https://clustering-benchmarks.gagolewski.com>.

*  [Python and R] New function:
   `compare_partitions.adjusted_asymmetric_accuracy`.

*  [Python and R] Implementations of the so-called internal cluster
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

*  [BACKWARD INCOMPATIBILITY] `compare_partitions.normalized_confusion_matrix`
   now solves the maximal assignment problem instead of applying
   a primitive partial pivoting.

*  [Python and R] New function: `compare_partitions.normalizing_permutation`

*  [R] New function: `normalized_confusion_matrix`.

*  [Python and R] New parameter to `compare_partitions.pair_sets_index`:
      `simplified`.

*  [Python] New parameters to `plots.plot_scatter`:
   `axis`, `title`, `xlabel`, `ylabel`, `xlim`, `ylim`.


## 1.0.1 (2022-08-08)

*  [GENERAL] A paper on the `genieclust` package is now available:
   M. Gagolewski, genieclust: Fast and robust hierarchical clustering,
   SoftwareX 15, 100722, 2021, DOI:
   [10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).

*  [Python] `plots.plot_scatter` now uses a more accessible default palette
   (from R 4.0.0).

*  [Python] New function: `inequity.devergottini_index`.

*  [R] New function: `devergottini_index`.


## 1.0.0 (2021-04-22)

*  [R] Use `mlpack` instead of `RcppMLPACK` (#72).
   This package is merely suggested, not dependent upon.


## 0.9.8 (2021-01-08)

*  [Python] Require Python >= 3.7 (implied by `numpy`).

*  [Python] Require `nmslib`.

*  [R] Use `RcppMLPACK` directly; remove dependency on `emstreeR`.

*  [R] Use `tinytest` for unit testing instead of `testthat`.


## 0.9.4 (2020-07-31)

*  [BUGFIX] [R] Fix build errors on Solaris.


## 0.9.3 (2020-07-25)

*  [BUGFIX] [Python] Add code coverage CI. Fix some minor inconsistencies.
   Automate the `bdist` build chain.

*  [R] Update DESCRIPTION to meet the CRAN policies.


## 0.9.2 (2020-07-22)

*  [BUGFIX] [Python] Fix broken build script for OS X with no OpenMP.


## 0.9.1 (2020-07-18)

*  [GENERAL] The package has been completely rewritten.
   The core functionality is now implemented in C++ (with OpenMP).

*  [GENERAL] Clustering with respect to HDBSCAN*-like
   mutual reachability distances is supported.

*  [GENERAL] The parallelised Jarnik-Prim algorithm now supports on-the-fly
   distance computations. Euclidean minimum spanning tree can be
   determined with `mlpack`, which is much faster in low-dimensional spaces.

*  [R] R version is now available.

*  [Python] [Experimental] The GIc algorithm proposed by Anna Cena
   in her 2018 PhD thesis is added.

*  [Python] Approximate version based on nearest neighbour graphs produced
   by `nmslib` is added.


## 0.1a2 (2018-05-23)

*  [Python] Initial PyPI release.
