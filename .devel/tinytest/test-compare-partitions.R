library("tinytest")
library("genieclust")

# More thorough tests are performed by pytest

scores <- list(adjusted_rand_score, rand_score, adjusted_fm_score, fm_score,
    adjusted_mi_score, normalized_mi_score, normalized_accuracy, pair_sets_index)

for (score in scores) {
    x <- c(1, 1, 1, 3, 3, 2, 3)
    expect_equal(score(x, x), 1.0)
    expect_equal(score(x, x), score(table(x, x)))
    expect_equal(score(x, x), score(x, 3-x+1))

    expect_error(score(c("a", "b"), 1:2))

    expect_equal(score(c(1L, 2L, 2L), c(1L, 2L, 2L)), 1.0)
    expect_equal(score(c(1.0, 2.0, 2.0), c(1.0, 2.0, 2.0)), 1.0)
    expect_equal(score(factor(c("a", "b", "b")), factor(c("a", "b", "b"))), 1.0)
    expect_equal(score(c(TRUE, FALSE, FALSE), c(FALSE, TRUE, TRUE)), 1.0)

    x <- c(1, 1, 1, 2, 2, 2, 3, 2, 1)
    y <- c(1, 1, 1, 2, 2, 2, 3, 4, 4)
    expect_equal(score(x, x), 1.0)
    expect_equal(score(y, y), 1.0)
    expect_equal(score(x, y), score(table(x, y)))
    expect_equal(score(x, y), score(x, 40-y))

    for (n in c(10, 100, 1000)) {
        for (K in 2:9) {
            x <- sample(c(1:K, sample(K, n-K, replace=TRUE)))
            y <- sample(c(1:K, sample(K, n-K, replace=TRUE)))
            s <- score(x, y)
            expect_true(s < 1.0+1e-9)

            C <- table(x, y)
            expect_equal(
                sum(C), sum(normalized_confusion_matrix(x, y))
            )

            expect_equivalent(
                normalized_confusion_matrix(x, y),
                unclass(C[, normalizing_permutation(C)])
            )
        }
    }
}

x <- c(1, 1, 1, 2, 2, 2, 3, 2, 1)
y <- c(1, 1, 1, 2, 2, 2, 3, 4, 4)
expect_error(normalized_accuracy(y, x))
expect_error(pair_sets_index(y, x))

expect_true(mi_score(x, y) >= 0)
