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


#set.seed(123)
n <- 1000

for (d in c(2, 10)) {
    X <- matrix(rnorm(n*d), nrow=n)
    for (distance in c("euclidean", "manhattan")) {

        D <- dist(X, method=distance)
        t0 <- ade4_mstree(D)

        ts <- list(
            mst(X, distance=distance),
            mst(X, distance=distance, algorithm="jarnik"),
            mst(D),
            if (distance == "euclidean") mst(X, distance=distance, algorithm="mlpack")
        )

        for (t1 in ts) {
            if (is.null(t1)) next
            expect_true(all(t1[,1] < t1[,2]))
            expect_true(!is.unsorted(t1[,3]))
            expect_true(abs(sum(t0[,3])-sum(t1[,3])) < 1e-16)
            expect_true(attr(t1, "method") == distance)
            expect_true(attr(t1, "Size") == n)
        }
    }
}


X <- jitter(as.matrix(iris[1:4]))
for (M in c(1, 5, 10)) {
    for (distance in c("euclidean", "manhattan")) {
        t1 <- mst(X, distance=distance, M=M)
#         print(t1)
        expect_true(all(t1[,1] < t1[,2]))
        expect_true(all(diff(t1[,3])>=0))

        t2 <- mst(dist(X, method=distance), M=M)
        if (M == 1) {
            expect_equal(t1[,1], t2[,1])
            expect_equal(t1[,2], t2[,2])
        }
#         print(abs(sum(t1[,3])-sum(t2[,3])))
        expect_true(abs(sum(t1[,3])-sum(t2[,3]))<1e-6)

        if (distance == "euclidean" && M == 1) {
            t2 <- mst(X, algorithm="mlpack")
            expect_equal(t1[,1], t2[,1])
            expect_equal(t1[,2], t2[,2])
#             print(abs(sum(t1[,3])-sum(t2[,3])))
            expect_true(abs(sum(t1[,3])-sum(t2[,3]))<1e-9)
        }
    }
}
