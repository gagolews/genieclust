# mst: Minimum Spanning Tree of the Pairwise Distance Graph

## Description

Determine a(\*) minimum spanning tree (MST) of the complete undirected graph representing a set of $n$ points whose weights correspond to the pairwise distances between the points.

## Usage

``` r
mst(d, ...)

## Default S3 method:
mst(
  d,
  distance = c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
  M = 1L,
  algorithm = c("auto", "jarnik", "mlpack"),
  leaf_size = 1L,
  cast_float32 = FALSE,
  verbose = FALSE,
  ...
)

## S3 method for class 'dist'
mst(d, M = 1L, verbose = FALSE, ...)
```

## Arguments

|  |  |
|----|----|
| `d` | either a numeric matrix (or an object coercible to one, e.g., a data frame with numeric-like columns) or an object of class `dist`; see [`dist`](https://stat.ethz.ch/R-manual/R-devel/library/stats/help/dist.html) |
| `...` | further arguments passed to or from other methods |
| `distance` | metric used in the case where `d` is a matrix; one of: `"euclidean"` (synonym: `"l2"`), `"manhattan"` (a.k.a. `"l1"` and `"cityblock"`), `"cosine"` |
| `M` | smoothing factor; `M` = 1 gives the selected `distance`; otherwise, the mutual reachability distance is used |
| `algorithm` | MST algorithm to use: `"auto"` (default), `"jarnik"`, or `"mlpack"`; if `"auto"`, select `"mlpack"` for low-dimensional Euclidean spaces and `"jarnik"` otherwise |
| `leaf_size` | size of leaves in the K-d tree (`"mlpack"`); controls the trade-off between speed and memory consumption |
| `cast_float32` | logical; whether to compute the distances using 32-bit instead of 64-bit precision floating-point arithmetic (up to 2x faster) |
| `verbose` | logical; whether to print diagnostic messages and progress information |

## Details

(\*) Note that if the distances are non unique, there might be multiple minimum trees spanning a given graph.

Two MST algorithms are available. First, our implementation of the Jarnik (Prim/Dijkstra)-like method requires $O(n^2)$ time. The algorithm is parallelised; the number of threads is determined by the `OMP_NUM_THREADS` environment variable; see [`Sys.setenv`](https://stat.ethz.ch/R-manual/R-devel/library/base/help/Sys.setenv.html). This method is recommended for high-dimensional spaces. As a rule of thumb, datasets up to 100000 points should be processed quite quickly. For 1M points, give it an hour or so.

Second, we give access to the implementation of the Dual-Tree Boruvka algorithm from the `mlpack` library. The algorithm is based on K-d trees and is very fast but only for low-dimensional Euclidean spaces (due to the curse of dimensionality). The Jarnik algorithm should be used if there are more than 5-10 features.

If `d` is a numeric matrix of size $n$ by $p$, representing $n$ points in a $p$-dimensional space, the $n (n-1)/2$ distances are computed on the fly: the algorithms requires $O(n)$ memory.

If `M` \>= 2, then the mutual reachability distance $m(i,j)$ with the smoothing factor `M` (see Campello et al. 2013) is used instead of the chosen \"raw\" distance $d(i,j)$. It holds $m(i, j)=\max(d(i,j), c(i), c(j))$, where $c(i)$ is $d(i, k)$ with $k$ being the (`M`-1)-th nearest neighbour of $i$. This makes \"noise\" and \"boundary\" points being \"pulled away\" from each other. The Genie++ clustering algorithm (see [`gclust`](gclust.md)) with respect to the mutual reachability distance can mark some observations are noise points.

Note that the case `M` = 2 corresponds to the original distance, but we return the (1-)nearest neighbours as well.

## Value

Returns a numeric matrix of class `mst` with n-1 rows and 3 columns: `from`, `to`, and `dist` sorted nondecreasingly. Its i-th row specifies the i-th edge of the MST which is incident to the vertices `from[i]` and `to[i]` `from[i] < to[i]` (in 1,\...,n) and `dist[i]` gives the corresponding weight, i.e., the distance between the point pair.

The `Size` attribute specifies the number of points, $n$. The `Labels` attribute gives the labels of the input points (optionally). The `method` attribute gives the name of the distance used.

If `M` \> 1, the `nn` attribute gives the indices of the `M`-1 nearest neighbours of each point.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## References

Jarnik V., O jistem problemu minimalnim, *Prace Moravske Prirodovedecke Spolecnosti* 6, 1930, 57-63.

Olson C.F., Parallel algorithms for hierarchical clustering, *Parallel Comput.* 21, 1995, 1313-1325.

Prim R., Shortest connection networks and some generalisations, *Bell Syst. Tech. J.* 36, 1957, 1389-1401.

March W.B., Ram P., Gray A.G., Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications, *Proc. ACM SIGKDD\'10*, 2010, 603-611, <https://mlpack.org/papers/emst.pdf>.

Curtin R.R., Edel M., Lozhnikov M., Mentekidis Y., Ghaisas S., Zhang S., mlpack 3: A fast, flexible machine learning library, *Journal of Open Source Software* 3(26), 2018, 726.

Campello R.J.G.B., Moulavi D., Sander J., Density-based clustering based on hierarchical density estimates, *Lecture Notes in Computer Science* 7819, 2013, 160-172, [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14).

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

Gagolewski M., <span class="pkg">genieclust</span>: Fast and robust hierarchical clustering, *SoftwareX* 15:100722, 2021, [doi:10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).

[`emst_mlpack()`](emst_mlpack.md) for a very fast alternative in the case of (very) low-dimensional Euclidean spaces (and `M` = 1).

## Examples




``` r
library("datasets")
data("iris")
X <- iris[1:4]
tree <- mst(X)
```
