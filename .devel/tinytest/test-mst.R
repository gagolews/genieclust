library("tinytest")
library("genieclust")
library("ade4")

ade4_mstree <- function(D)
{
    t0 <- unclass(ade4::mstree(D))[,]
    t0[] <- t(apply(t0, 1, sort))
    t0 <- cbind(t0, as.matrix(D)[t0])
    t0 <- t0[order(t0[, 3]), ]
    list(
        mst.index=t0[,-3],
        mst.dist=t0[,3]
    )
}

# more tests in the Python version

#set.seed(123)
n <- 1000

for (d in c(2, 20)) {
    X <- matrix(rnorm(n*d), nrow=n)

    D <- dist(X)
    t0 <- ade4_mstree(D)

    ts <- list(
        mst_euclid(X, algorithm="auto"),
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

        ts <- list(
            mst_euclid(X, M=M, algorithm="brute"),
            mst_euclid(X, M=M, algorithm="kd_tree_single"),
            mst_euclid(X, M=M, algorithm="kd_tree_dual")
        )

        for (t1 in ts) {
            expect_true(all(t1$mst.index[,1] < t1$mst.index[,2]))
            expect_true(!is.unsorted(t1$mst.dist))
            expect_true(abs(sum(t0$mst.dist)-sum(t1$mst.dist)) < 1e-9)
        }
    }
}
