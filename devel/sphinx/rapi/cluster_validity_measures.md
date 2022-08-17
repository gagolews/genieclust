# cluster_validity_measures: Internal Cluster Validity Measures

## Description

Implementation of a number of so-called cluster validity indices critically reviewed in (Gagolewski, Bartoszuk, Cena, 2021). See Section 2 therein and (Gagolewski, 2022) for the respective definitions.

The greater the index value, the more *valid* (whatever that means) the assessed partition. For consistency, the Ball-Hall and Davies-Bouldin indexes as well as the within-cluster sum of squares (WCSS) take negative values.

## Usage

``` r
calinski_harabasz_index(X, y)

dunnowa_index(X, y, M = 10L, owa_numerator = "Min", owa_denominator = "Max")

generalised_dunn_index(X, y, lowercase_delta, uppercase_delta)

negated_ball_hall_index(X, y)

negated_davies_bouldin_index(X, y)

negated_wcss_index(X, y)

silhouette_index(X, y)

silhouette_w_index(X, y)

wcnn_index(X, y, M = 10L)
```

## Arguments

|                                  |                                                                                                                                                                                                                              |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `X`                              | numeric matrix with `n` rows and `d` columns, representing `n` points in a `d`-dimensional space                                                                                                                             |
| `y`                              | vector of `n` integer labels, representing a partition whose *quality* is to be assessed; `y[i]` is the cluster ID of the `i`-th point, `X[i, ]`; `1 <= y[i] <= K`, where `K` is the number or clusters                      |
| `M`                              | number of nearest neighbours                                                                                                                                                                                                 |
| `owa_numerator, owa_denominator` | single string defining the OWA operator to use in the definition of the DuNN index; one of: `"Mean"`, `"Min"`, `"Max"`, `"Const"`, `"SMin:M"`, `"SMax:M"`, where `M` is an integer defining the number of nearest neighbours |
| `lowercase_delta`                | an integer between 1 and 6, denoting $d_1$, \..., $d_6$ in the definition of the generalised Dunn index (numerator)                                                                                                          |
| `uppercase_delta`                | an integer between 1 and 3, denoting $D_1$, \..., $D_3$ in the definition of the generalised Dunn index (denominator)                                                                                                        |

## Value

A single numeric value (the more, the *better*).

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## References

Ball G.H., Hall D.J., *ISODATA: A novel method of data analysis and pattern classification*, Technical report No. AD699616, Stanford Research Institute, 1965.

Bezdek J., Pal N., Some new indexes of cluster validity, *IEEE Transactions on Systems, Man, and Cybernetics, Part B* 28, 1998, 301-315, [doi:10.1109/3477.678624](https://doi.org/10.1109/3477.678624).

Calinski T., Harabasz J., A dendrite method for cluster analysis, *Communications in Statistics* 3(1), 1974, 1-27, [doi:10.1080/03610927408827101](https://doi.org/10.1080/03610927408827101).

Davies D.L., Bouldin D.W., A Cluster Separation Measure, *IEEE Transactions on Pattern Analysis and Machine Intelligence* PAMI-1 (2), 1979, 224-227, [doi:10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909).

Dunn J.C., A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters, *Journal of Cybernetics* 3(3), 1973, 32-57, [doi:10.1080/01969727308546046](https://doi.org/10.1080/01969727308546046).

Gagolewski M., Bartoszuk M., Cena A., Are cluster validity measures (in)valid?, *Information Sciences* 581, 620-636, 2021, [doi:10.1016/j.ins.2021.10.004](https://doi.org/10.1016/j.ins.2021.10.004); preprint: <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>.

Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*, 2022, <https://clustering-benchmarks.gagolewski.com>.

Rousseeuw P.J., Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis, *Computational and Applied Mathematics* 20, 1987, 53-65, [doi:10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427%2887%2990125-7).

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

Gagolewski M., <span class="pkg">genieclust</span>: Fast and robust hierarchical clustering, *SoftwareX* 15:100722, 2021, [doi:10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).

## Examples




```r
X <- as.matrix(iris[,1:4])
X[,] <- jitter(X)  # otherwise we get a non-unique solution
y <- as.integer(iris[[5]])
calinski_harabasz_index(X, y)  # good
## [1] 486.6681
calinski_harabasz_index(X, sample(1:3, nrow(X), replace=TRUE))  # bad
## [1] 2.836713
```
