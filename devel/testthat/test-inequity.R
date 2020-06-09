library("testthat")
library("genieclust")
context("inequity")

# More thorough tests are performed by pytest

for (n in c(2, 5, 100)) {
    expect_equal(gini(sample(c(n, rep(0, n)))), 1.0)
    expect_equal(gini(rep(n, n)), 0.0)
    expect_equal(bonferroni(sample(c(n, rep(0, n)))), 1.0)
    expect_equal(bonferroni(rep(n, n)), 0.0)
}
