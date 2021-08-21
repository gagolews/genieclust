# mst: Minimum Spanning Tree of the Pairwise Distance Graph

## Description

An parallelised implementation of a Jarnik (Prim/Dijkstra)-like algorithm for determining a(\*) minimum spanning tree (MST) of a complete undirected graph representing a set of n points with weights given by a pairwise distance matrix.

(\*) Note that there might be multiple minimum trees spanning a given graph.

## Usage

```r
mst(d, ...)

## Default S3 method:
mst(
  d,
  distance = c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
  M = 1L,
  cast_float32 = TRUE,
  verbose = FALSE,
  ...
)

## S3 method for class 'dist'
mst(d, M = 1L, verbose = FALSE, ...)
```

## Arguments

|                |                                                                                                                                                                                                                       |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `d`            | either a numeric matrix (or an object coercible to one, e.g., a data frame with numeric-like columns) or an object of class `dist`, see [`dist`](https://stat.ethz.ch/R-manual/R-devel/library/stats/help/dist.html). |
| `...`          | further arguments passed to or from other methods.                                                                                                                                                                    |
| `distance`     | metric used to compute the linkage, one of: `"euclidean"` (synonym: `"l2"`), `"manhattan"` (a.k.a. `"l1"` and `"cityblock"`), `"cosine"`.                                                                             |
| `M`            | smoothing factor; `M` = 1 gives the selected `distance`; otherwise, the mutual reachability distance is used.                                                                                                         |
| `cast_float32` | logical; whether to compute the distances using 32-bit instead of 64-bit precision floating-point arithmetic (up to 2x faster).                                                                                       |
| `verbose`      | logical; whether to print diagnostic messages and progress information.                                                                                                                                               |

## Details

If `d` is a numeric matrix of size *n p*, the *n (n-1)/2* distances are computed on the fly, so that *O(n M)* memory is used.

The algorithm is parallelised; set the `OMP_NUM_THREADS` environment variable [`Sys.setenv`](https://stat.ethz.ch/R-manual/R-devel/library/base/help/Sys.setenv.html) to control the number of threads used.

Time complexity is *O(n\^2)* for the method accepting an object of class `dist` and *O(p n\^2)* otherwise.

If `M` \>= 2, then the mutual reachability distance *m(i,j)* with smoothing factor `M` (see Campello et al. 2015) is used instead of the chosen \"raw\" distance *d(i,j)*. It holds *m(i, j)=\\max(d(i,j), c(i), c(j))*, where *c(i)* is *d(i, k)* with *k* being the (`M`-1)-th nearest neighbour of *i*. This makes \"noise\" and \"boundary\" points being \"pulled away\" from each other. Genie++ clustering algorithm (see [`gclust`](gclust.md)) with respect to the mutual reachability distance gains the ability to identify some observations are noise points.

Note that the case `M` = 2 corresponds to the original distance, but we are determining the 1-nearest neighbours separately as well, which is a bit suboptimal; you can file a feature request if this makes your data analysis tasks too slow.

## Value

Matrix of class `mst` with n-1 rows and 3 columns: `from`, `to` and `dist`. It holds `from` \< `to`. Moreover, `dist` is sorted nondecreasingly. The i-th row gives the i-th edge of the MST. `(from[i], to[i])` defines the vertices (in 1,\...,n) and `dist[i]` gives the weight, i.e., the distance between the corresponding points.

The `method` attribute gives the name of the distance used. The `Labels` attribute gives the labels of all the input points.

If `M` \> 1, the `nn` attribute gives the indices of the `M`-1 nearest neighbours of each point.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## References

V. Jarnik, O jistem problemu minimalnim, Prace Moravske Prirodovedecke Spolecnosti 6 (1930) 57-63.

Olson C.F., Parallel algorithms for hierarchical clustering, Parallel Comput. 21 (1995) 1313-1325.

Prim R., Shortest connection networks and some generalisations, Bell Syst. Tech. J. 36 (1957) 1389-1401.

Campello R., Moulavi D., Zimek A., Sander J., Hierarchical density estimates for data clustering, visualization, and outlier detection, ACM Transactions on Knowledge Discovery from Data 10(1) (2015) 5:1-5:51.

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

[`emst_mlpack()`](emst_mlpack.md) for a very fast alternative in case of (very) low-dimensional Euclidean spaces (and `M` = 1).

## Examples




```r
library("datasets")
data("iris")
X <- iris[1:4]
tree <- mst(X)
```
