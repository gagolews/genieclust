library("testthat")
library("genieclust")
context("compare-partitions")

# More thorough tests are performed by pytest

x <- c(1, 1, 1, 3, 3, 2, 3)
expect_equal(adjusted_rand_index(x, x), 1.0)
expect_equal(adjusted_rand_index(x, x), adjusted_rand_index(table(x, x)))
expect_equal(adjusted_rand_index(x, x), adjusted_rand_index(x, 3-x+1))


x <- c(1, 1, 1, 2, 2, 2, 3, 4, 4)
y <- c(1, 1, 1, 2, 2, 2, 3, 2, 1)
expect_equal(adjusted_rand_index(x, x), 1.0)
expect_equal(adjusted_rand_index(y, y), 1.0)
expect_equal(adjusted_rand_index(x, y), adjusted_rand_index(table(x, y)))
expect_equal(adjusted_rand_index(x, y), adjusted_rand_index(x, 40-y))
