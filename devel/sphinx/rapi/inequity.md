# inequity: Inequity (Inequality) Measures

## Description

`gini_index()` gives the normalised Gini index, `bonferroni_index()` implements the Bonferroni index, and `devergottini_index()` implements the De Vergottini index.

## Usage

```r
gini_index(x)

bonferroni_index(x)

devergottini_index(x)
```

## Arguments

|     |                                       |
|-----|---------------------------------------|
| `x` | numeric vector of non-negative values |

## Details

These indices can be used to quantify the \"inequity\" of a numeric sample. They can be perceived as measures of data dispersion. For constant vectors (perfect equity), the indices yield values of 0. Vectors with all elements but one equal to 0 (perfect inequity), are assigned scores of 1. They follow the Pigou-Dalton principle (are Schur-convex): setting *x\_i = x\_i - h* and *x\_j = x\_j + h* with *h \> 0* and *x\_i - h ≥q x\_j + h* (taking from the \"rich\" and giving to the \"poor\") decreases the inequity.

These indices have applications in economics, amongst others. The Genie clustering algorithm uses the Gini index as a measure of the inequality of cluster sizes.

The normalised Gini index is given by:

*G(x\_1,...,x\_n) = \\frac{ ∑\_{i=1}\^{n} (n-2i+1) x\_{σ(n-i+1)} }{ (n-1) ∑\_{i=1}\^n x\_i },*

The normalised Bonferroni index is given by:

*B(x\_1,...,x\_n) = \\frac{ ∑\_{i=1}\^{n} (n-∑\_{j=1}\^i \\frac{n}{n-j+1}) x\_{σ(n-i+1)} }{ (n-1) ∑\_{i=1}\^n x\_i }.*

The normalised De Vergottini index is given by:

*V(x\_1,...,x\_n) = \\frac{1}{∑\_{i=2}\^n \\frac{1}{i}} ≤ft( \\frac{ ∑\_{i=1}\^n ≤ft( ∑\_{j=i}\^{n} \\frac{1}{j}\\right) x\_{σ(n-i+1)} }{∑\_{i=1}\^{n} x\_i} - 1 \\right).*

Here, *σ* is an ordering permutation of *(x\_1,...,x\_n)*.

Time complexity: *O(n)* for sorted (increasingly) data. Otherwise, the vector will be sorted.

## Value

The value of the inequity index, a number in *\[0, 1\]*.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## References

Bonferroni C., Elementi di Statistica Generale, Libreria Seber, Firenze, 1930.

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm, Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

Gini C., Variabilita e Mutabilita, Tipografia di Paolo Cuppini, Bologna, 1912.

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

## Examples




```r
gini_index(c(2, 2, 2, 2, 2))  # no inequality
## [1] 0
gini_index(c(0, 0, 10, 0, 0)) # one has it all
## [1] 1
gini_index(c(7, 0, 3, 0, 0))  # give to the poor, take away from the rich
## [1] 0.85
gini_index(c(6, 0, 3, 1, 0))  # (a.k.a. Pigou-Dalton principle)
## [1] 0.75
bonferroni_index(c(2, 2, 2, 2, 2))
## [1] 0
bonferroni_index(c(0, 0, 10, 0, 0))
## [1] 1
bonferroni_index(c(7, 0, 3, 0, 0))
## [1] 0.90625
bonferroni_index(c(6, 0, 3, 1, 0))
## [1] 0.8333333
devergottini_index(c(2, 2, 2, 2, 2))
## [1] 0
devergottini_index(c(0, 0, 10, 0, 0))
## [1] 1
devergottini_index(c(7, 0, 3, 0, 0))
## [1] 0.7662338
devergottini_index(c(6, 0, 3, 1, 0))
## [1] 0.6493506
```
