# inequality: Inequality Measures

## Description

`gini_index()` gives the normalised Gini index, `bonferroni_index()` implements the Bonferroni index, and `devergottini_index()` implements the De Vergottini index.

## Usage

``` r
gini_index(x)

bonferroni_index(x)

devergottini_index(x)
```

## Arguments

|     |                                       |
|-----|---------------------------------------|
| `x` | numeric vector of non-negative values |

## Details

These indices can be used to quantify the \"inequality\" of a numeric sample. They can be conceived as normalised measures of data dispersion. For constant vectors (perfect equity), the indices yield values of 0. Vectors with all elements but one equal to 0 (perfect inequality), are assigned scores of 1. They follow the Pigou-Dalton principle (are Schur-convex): setting $x_i = x_i - h$ and $x_j = x_j + h$ with $h > 0$ and $x_i - h \geq  x_j + h$ (taking from the \"rich\" and giving to the \"poor\") decreases the inequality

These indices have applications in economics, amongst others. The Genie clustering algorithm uses the Gini index as a measure of the inequality of cluster sizes.

The normalised Gini index is given by:

$$
    G(x_1,\dots,x_n) = \frac{
    \sum_{i=1}^{n} (n-2i+1) x_{\sigma(n-i+1)}
    }{
    (n-1) \sum_{i=1}^n x_i
    },$$

The normalised Bonferroni index is given by:

$$
    B(x_1,\dots,x_n) = \frac{
    \sum_{i=1}^{n}  (n-\sum_{j=1}^i \frac{n}{n-j+1})
         x_{\sigma(n-i+1)}
    }{
    (n-1) \sum_{i=1}^n x_i
    }.$$

The normalised De Vergottini index is given by:

$$
    V(x_1,\dots,x_n) =
    \frac{1}{\sum_{i=2}^n \frac{1}{i}} \left(
       \frac{ \sum_{i=1}^n \left( \sum_{j=i}^{n} \frac{1}{j}\right)
       x_{\sigma(n-i+1)} }{\sum_{i=1}^{n} x_i} - 1
    \right).$$

Here, $\sigma$ is an ordering permutation of $(x_1,\dots,x_n)$.

Time complexity: $O(n)$ for sorted (increasingly) data. Otherwise, the vector will be sorted.

## Value

The value of the inequality index, a number in $[0, 1]$.

## Author(s)

[Marek Gagolewski](https://www.gagolewski.com/) and other contributors

## References

Bonferroni C., *Elementi di Statistica Generale*, Libreria Seber, Firenze, 1930.

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm, *Information Sciences* 363, 2016, pp. 8-23. [doi:10.1016/j.ins.2016.05.003](https://doi.org/10.1016/j.ins.2016.05.003)

Gini C., *Variabilita e Mutabilita*, Tipografia di Paolo Cuppini, Bologna, 1912.

## See Also

The official online manual of <span class="pkg">genieclust</span> at <https://genieclust.gagolewski.com/>

Gagolewski M., <span class="pkg">genieclust</span>: Fast and robust hierarchical clustering, *SoftwareX* 15:100722, 2021, [doi:10.1016/j.softx.2021.100722](https://doi.org/10.1016/j.softx.2021.100722).

## Examples

``` r
gini_index(c(2, 2, 2, 2, 2))   # no inequality
gini_index(c(0, 0, 10, 0, 0))  # one has it all
gini_index(c(7, 0, 3, 0, 0))   # give to the poor, take away from the rich
gini_index(c(6, 0, 3, 1, 0))   # (a.k.a. Pigou-Dalton principle)
bonferroni_index(c(2, 2, 2, 2, 2))
bonferroni_index(c(0, 0, 10, 0, 0))
bonferroni_index(c(7, 0, 3, 0, 0))
bonferroni_index(c(6, 0, 3, 1, 0))
devergottini_index(c(2, 2, 2, 2, 2))
devergottini_index(c(0, 0, 10, 0, 0))
devergottini_index(c(7, 0, 3, 0, 0))
devergottini_index(c(6, 0, 3, 1, 0))
```
