library("tinytest")
library("genieclust")


#set.seed(123)
n <- 100

for (d in c(2, 10, 50)) {
    X <- matrix(rnorm(n*d), nrow=n)

    for (k in c(1, 2, 5, n-1)) {
        nna <- knn_euclid(X, k)
        nnb <- knn_euclid(X, k, algorithm="brute")
        expect_equal(nna$nn.index, nnb$nn.index)
        expect_equal(nna$nn.dist, nnb$nn.dist)
        if (d <= 20) {
            nnk <- knn_euclid(X, k, algorithm="kd_tree")
            expect_equal(nna$nn.index, nnk$nn.index)
            expect_equal(nna$nn.dist, nnk$nn.dist)
        }
    }
}



X <- as.matrix(iris[, -5])
n <- nrow(X)
D <- as.matrix(dist(X))
D <- `dimnames<-`(t(apply(D, 2, sort)), NULL)


for (k in c(1, 2, 5, n-1)) {
    ddr <- D[, 2:(k+1), drop=FALSE]
    dda <- knn_euclid(X, k)$nn.dist
    ddb <- knn_euclid(X, k, algorithm="brute")$nn.dist
    ddk <- knn_euclid(X, k, algorithm="kd_tree")$nn.dist
    expect_equal(ddr, dda)
    expect_equal(ddr, ddb)
    expect_equal(ddr, ddk)
}
