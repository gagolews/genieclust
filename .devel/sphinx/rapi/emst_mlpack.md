# emst_mlpack: Euclidean Minimum Spanning Tree \[DEPRECATED\]

## Description

This function is deprecated. Use [`mst()`](mst.md) instead.

## Usage

``` r
emst_mlpack(d, leaf_size = 1, verbose = FALSE)
```

## Arguments

|  |  |
|----|----|
| `d` | a numeric matrix (or an object coercible to one, e.g., a data frame with numeric-like columns) |
| `leaf_size` | size of leaves in the K-d tree, controls the trade-off between speed and memory consumption |
| `verbose` | logical; whether to print diagnostic messages |

## Value

An object of class `mst`, see [`mst()`](mst.md) for details.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

Gagolewski M., <span class="pkg">genieclust</span>: Fast and robust hierarchical clustering, *SoftwareX* 15:100722, 2021, [doi:10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).
