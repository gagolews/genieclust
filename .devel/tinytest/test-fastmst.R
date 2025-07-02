library("tinytest")
library("genieclust")

# more tests in the Python version

#set.seed(123)
n <- 1000

for (d in c(2, 20)) {
    X <- matrix(rnorm(n*d), nrow=n)

    t0 <- mst_euclid(X, algorithm="auto")

    ts <- list(
        mst_euclid(X, algorithm="brute"),
        mst_euclid(X, algorithm="kd_tree_single"),
        mst_euclid(X, algorithm="kd_tree_dual")
    )

    for (t1 in ts) {
        expect_true(all(t1$mst.index[,1] < t1$mst.index[,2]))
        expect_true(!is.unsorted(t1$mst.dist))
        expect_true(abs(sum(t0$mst.dist)-sum(t1$mst.dist)) < 1e-9)
    }


    for (M in c(2, 5, 10)) {
        t0 <- mst_euclid(X, M=M, algorithm="auto")
        nna <- knn_euclid(X, M-1)
        expect_equal(nna$nn.index, t0$nn.index)
        expect_equal(nna$nn.dist, t0$nn.dist)

        ts <- list(
            mst_euclid(X, M=M, algorithm="brute"),
            mst_euclid(X, M=M, algorithm="kd_tree_single"),
            mst_euclid(X, M=M, algorithm="kd_tree_dual")
        )

        for (t1 in ts) {
            expect_true(all(t1$mst.index[,1] < t1$mst.index[,2]))
            expect_true(!is.unsorted(t1$mst.dist))
            expect_true(abs(sum(t0$mst.dist)-sum(t1$mst.dist)) < 1e-9)
            expect_equal(t1$nn.index, t0$nn.index)
            expect_equal(t1$nn.dist, t0$nn.dist)
            expect_equal(dim(t1$nn.dist), dim(t1$nn.index))
            expect_equal(dim(t1$nn.dist), c(n, M-1))

        }
    }
}
