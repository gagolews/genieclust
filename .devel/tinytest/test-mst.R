library("tinytest")
library("genieclust")


ade4_mstree <- function(D)
{
    t0 <- unclass(ade4::mstree(D))[,]
    t0[] <- t(apply(t0, 1, sort))
    t0 <- cbind(t0, as.matrix(D)[t0])
    t0 <- t0[order(t0[, 3]), ]
    t0
}

set.seed(123)
n <- 10
d <- 2
X <- matrix(rnorm(n*d), nrow=n)

for (distance in c("euclidean", "manhattan")) {

    D <- dist(X, method=distance)
    t0 <- ade4_mstree(D)

    ts <- list(
        mst(X, cast_float32=FALSE, distance=distance),
        mst(D),
        if (distance == "euclidean") emst_mlpack(X)
    )

    for (t1 in ts) {
        if (is.null(t1)) next
        expect_true(all(t1[,1] < t1[,2]))
        expect_true(!is.unsorted(t1[,3]))
        expect_true(abs(sum(t0[,3])-sum(t1[,3])) < 1e-16)
    }

}


X <- jitter(as.matrix(iris[1:4]))
for (M in c(1, 5, 10)) {
    for (distance in c("euclidean", "manhattan")) {
        t1 <- mst(X, distance=distance, M=M, cast_float32=FALSE)
#         print(t1)
        expect_true(all(t1[,1] < t1[,2]))
        expect_true(all(diff(t1[,3])>=0))

        t2 <- mst(dist(X, method=distance), M=M)
        if (M == 1) {

         expect_equal(t1[,1], t2[,1])
            expect_equal(t1[,2], t2[,2])
        }
#         print(abs(sum(t1[,3])-sum(t2[,3])))
        expect_true(abs(sum(t1[,3])-sum(t2[,3]))<1e-12)

        if (distance == "euclidean" && M == 1) {
            t2 <- emst_mlpack(X)
            expect_equal(t1[,1], t2[,1])
            expect_equal(t1[,2], t2[,2])
#             print(abs(sum(t1[,3])-sum(t2[,3])))
            expect_true(abs(sum(t1[,3])-sum(t2[,3]))<1e-12)
        }
    }
}
