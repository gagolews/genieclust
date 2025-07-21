library("tinytest")
library("genieclust")
library("ade4")
library("quitefastmst")


ade4_mstree <- function(D)
{
    t0 <- unclass(ade4::mstree(D))[,]
    t0[] <- t(apply(t0, 1, sort))
    t0 <- cbind(t0, as.matrix(D)[t0])
    t0 <- t0[order(t0[, 3]), ]
    t0
}

quitefastmst_euclid <- function(X, M=1, ...)
{
    .res <- mst_euclid(X, M, ...)
    result <- cbind(.res[["mst.index"]], .res[["mst.dist"]])
    attr(result, "nn.index")  <- .res[["nn.index"]]
    attr(result, "nn.dist")   <- .res[["nn.dist"]]
    result
}


n <- 1000

for (d in c(2, 20)) {
    X <- matrix(rnorm(n*d), nrow=n)

    D <- dist(X)
    t0 <- ade4_mstree(D)

    ts <- list(
        quitefastmst_euclid(X, algorithm="auto"),
        quitefastmst_euclid(X, algorithm="brute"),
        quitefastmst_euclid(X, algorithm="single_kd_tree"),
        quitefastmst_euclid(X, algorithm="sesqui_kd_tree"),
        quitefastmst_euclid(X, algorithm="dual_kd_tree"),
        genieclust:::.oldmst.matrix(X, "euclidean"),
        genieclust::mst(X, distance="euclidean"),
        genieclust::mst(D)
    )

    for (t1 in ts) {
        expect_true(all(t1[,1] < t1[,2]))
        expect_true(!is.unsorted(t1[,3]))
        expect_equal(sum(t0[,3]), sum(t1[,3]))
    }


    for (M in c(2, 5, 10)) {
        t0 <- genieclust:::.oldmst.matrix(X, M=M, distance="euclidean")

        ts <- list(
            quitefastmst_euclid(X, M=M, algorithm="auto"),
            quitefastmst_euclid(X, M=M, algorithm="brute"),
            quitefastmst_euclid(X, M=M, algorithm="single_kd_tree"),
            quitefastmst_euclid(X, M=M, algorithm="sesqui_kd_tree"),
            quitefastmst_euclid(X, M=M, algorithm="dual_kd_tree"),
            genieclust::mst(X, M=M, distance="euclidean"),
            genieclust::mst(D, M=M)
        )

        for (t1 in ts) {
            expect_true(all(t1[,1] < t1[,2]))
            expect_true(!is.unsorted(t1[,3]))
            expect_equal(sum(t0[,3]), sum(t1[,3]))
            expect_equal(attr(t1, "nn.index"), attr(t0, "nn.index"))
            expect_equal(attr(t1, "nn.dist"), attr(t0, "nn.dist"))
        }
    }
}
