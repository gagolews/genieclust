# comparing_partitions: Pairwise Partition Similarity Scores

## Description

The functions described in this section quantify the similarity between two label vectors `x` and `y` which represent two partitions of a set of $n$ elements into, respectively, $K$ and $L$ nonempty and pairwise disjoint subsets.

For instance, `x` and `y` can be two clusterings of a dataset with $n$ observations specified by two vectors of labels. Hence, these functions can be used as external cluster validity measures, i.e., in the presence of reference (ground-truth) partitions (compare Gagolewski, 2022).

## Usage

``` r
adjusted_rand_score(x, y = NULL)

rand_score(x, y = NULL)

adjusted_fm_score(x, y = NULL)

fm_score(x, y = NULL)

mi_score(x, y = NULL)

normalized_mi_score(x, y = NULL)

adjusted_mi_score(x, y = NULL)

normalized_accuracy(x, y = NULL)

pair_sets_index(x, y = NULL, simplified = FALSE)

normalized_confusion_matrix(x, y = NULL)

normalizing_permutation(x, y = NULL)
```

## Arguments

|           |                                                                                                                                                                                                                                             |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `x`       | an integer vector of length n (or an object coercible to) representing a K-partition of an n-set, or a confusion matrix with K rows and L columns (see [`table(x, y)`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/table.html)) |
| `y`       | an integer vector of length n (or an object coercible to) representing an L-partition of the same set), or NULL (if x is an K\*L confusion matrix)                                                                                          |
| `Whether` | to assume E=1 in the definition of the pair sets index index, i.e., use Eq. (20) instead of (18), see (Rezaei, Franti, 2016).                                                                                                               |

## Details

Every index except `mi_score()` (which computes the mutual information score) outputs 1 given two identical partitions. Note that partitions are always defined up to a bijection of the set of possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4) represent the same 2-partition.

`rand_score()` gives the Rand score (the \"probability\" of agreement between the two partitions) and `adjusted_rand_score()` is its version corrected for chance, see (Hubert, Arabie, 1985), its expected value is 0.0 given two independent partitions. Due to the adjustment, the resulting index might also be negative for some inputs.

Similarly, `fm_score()` gives the Fowlkes-Mallows (FM) score and `adjusted_fm_score()` is its adjusted-for-chance version, see (Hubert, Arabie, 1985).

Note that both the (unadjusted) Rand and FM scores are bounded from below by $1/(K+1)$ if $K=L$, hence their adjusted versions are preferred.

`mi_score()`, `adjusted_mi_score()` and `normalized_mi_score()` are information-theoretic scores, based on mutual information, see the definition of $AMI_{sum}$ and $NMI_{sum}$ in (Vinh et al., 2010).

`normalized_accuracy()` is defined as $(Accuracy(C_\sigma)-1/L)/(1-1/L)$, where $C_\sigma$ is a version of the confusion matrix for given `x` and `y`, $K \leq L$, with columns permuted based on the solution to the Maximal Linear Sum Assignment Problem; see [`normalized_confusion_matrix`](comparing_partitions.md). The $Accuracy(C_\sigma)$ part is sometimes referred to as set-matching classification rate.

`pair_sets_index()` gives the Pair Sets Index (PSI) adjusted for chance (Rezaei, Franti, 2016), $K \leq L$. Pairing is based on the solution to the Linear Sum Assignment Problem of a transformed version of the confusion matrix. Its simplified version assumes E=1 in the definition of the index, i.e., uses Eq. (20) instead of (18).

`normalized_confusion_matrix()` computes the confusion matrix and permutes its rows and columns so that the sum of the elements of the main diagonal is the largest possible (by solving the maximal assignment problem). The function only accepts $K \leq L$. The sole reordering of the columns of a confusion matrix can be determined by calling `normalizing_permutation()`.

Also note that the built-in [`table()`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/table.html) determines the standard confusion matrix.

## Value

Each partition similarity score is a single numeric value.

`normalized_confusion_matrix()` returns an integer matrix.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## References

Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*, 2022, <https://clustering-benchmarks.gagolewski.com>.

Hubert L., Arabie P., Comparing partitions, *Journal of Classification* 2(1), 1985, 193-218, esp. Eqs. (2) and (4).

Rendon E., Abundez I., Arizmendi A., Quiroz E.M., Internal versus external cluster validation indexes, *International Journal of Computers and Communications* 5(1), 2011, 27-34.

Rezaei M., Franti P., Set matching measures for external cluster validity, *IEEE Transactions on Knowledge and Data Mining* 28(8), 2016, 2173-2186.

Vinh N.X., Epps J., Bailey J., Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance, *Journal of Machine Learning Research* 11, 2010, 2837-2854.

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

Gagolewski M., <span class="pkg">genieclust</span>: Fast and robust hierarchical clustering, *SoftwareX* 15:100722, 2021, [doi:10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).

## Examples




```r
y_true <- iris[[5]]
y_pred <- kmeans(as.matrix(iris[1:4]), 3)$cluster
adjusted_rand_score(y_true, y_pred)
## [1] 0.7302383
rand_score(table(y_true, y_pred)) # the same
## [1] 0.8797315
adjusted_fm_score(y_true, y_pred)
## [1] 0.7304411
fm_score(y_true, y_pred)
## [1] 0.8208081
mi_score(y_true, y_pred)
## [1] 0.8255911
normalized_mi_score(y_true, y_pred)
## [1] 0.7581757
adjusted_mi_score(y_true, y_pred)
## [1] 0.7551192
normalized_accuracy(y_true, y_pred)
## [1] 0.84
pair_sets_index(y_true, y_pred)
## [1] 0.7568238
normalized_confusion_matrix(y_true, y_pred)
##      [,1] [,2] [,3]
## [1,]   50    0    0
## [2,]    0   48    2
## [3,]    0   14   36
normalizing_permutation(y_true, y_pred)
## [1] 1 2 3
```
