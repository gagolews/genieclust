library("tinytest")
library("genieclust")


# More thorough tests are performed by pytest

for (n in c(2, 5, 100)) {
    expect_equal(gini_index(sample(c(n, rep(0, n)))), 1.0)
    expect_equal(gini_index(rep(n, n)), 0.0)
    expect_error(gini_index(c(0, -1, 2)))
    expect_error(gini_index(c(0, 0, 0)))
    expect_equal(bonferroni_index(sample(c(n, rep(0, n)))), 1.0)
    expect_equal(bonferroni_index(rep(n, n)), 0.0)
    expect_error(bonferroni_index(c(0, -1, 2)))
    expect_error(bonferroni_index(c(0, 0, 0)))
}
