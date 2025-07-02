# omp: Get or Set the Number of Threads

## Description

These functions get or set the maximal number of OpenMP threads that can be used by [`knn_euclid`](fastknn.md) and [`mst_euclid`](fastmst.md), amongst others.

## Usage

``` r
omp_set_num_threads(n_threads)

omp_get_max_threads()
```

## Arguments

|             |                                   |
|-------------|-----------------------------------|
| `n_threads` | maximum number of threads to use; |

## Value

`omp_get_max_threads` returns the maximal number of threads that will be used during the next call to a parallelised function, not the maximal number of threads possibly available. It there is no built-in support for OpenMP, 1 is always returned.

For `omp_set_num_threads`, the previous value of `max_threads` is output.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

Gagolewski M., <span class="pkg">genieclust</span>: Fast and robust hierarchical clustering, *SoftwareX* 15:100722, 2021, [doi:10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).
