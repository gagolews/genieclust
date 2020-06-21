library("testthat")
library("genieclust")
context("gclust")


verbose <- FALSE
set.seed(123)
n <- 1000
d <- 2
X <- matrix(rnorm(n*d), nrow=n)
cat(sprintf("n=%d, d=%d\n", n, d))

print(system.time(h1 <- gclust(X, verbose=verbose)))

print(system.time(h2 <- gclust(X, verbose=verbose, cast_float32=FALSE)))
#for (k in 2:20) expect_equal(adjusted_rand_score(cutree(h1, k), cutree(h2, k)), 1.0)

if (require("genie")) {
    print(system.time(h2 <- hclust2(objects=X)))
    #for (k in 2:20) expect_equal(adjusted_rand_score(cutree(h1, k), cutree(h2, k)), 1.0)
}

if (require("emstreeR")) {
    print(system.time(h2 <- gclust(emst_mlpack(X, verbose=verbose), verbose=verbose)))
    #for (k in 2:20) expect_equal(adjusted_rand_score(cutree(h1, k), cutree(h2, k)), 1.0)
}


# 2020-06-21
# n=100000, d=3
#    user  system elapsed
#  33.520   0.083   8.436!!
#    user  system elapsed
#  40.310   0.088  10.137!!
#    user  system elapsed
#  50.101   0.044  24.091!!
#    user  system elapsed
#   1.353   0.000   1.354


# 2020-06-21
# gclustn=100000, d=100
#    user  system elapsed
# 651.079     0.176  162.907 !!
#     user   system  elapsed
# 1184.137    0.196  296.280 !!
#     user   system  elapsed
# 1121.105    0.220  290.941 !!
#     user   system  elapsed
# 4865.173    0.136 4865.264




X <- jitter(as.matrix(iris[1:4]))

for (g in c(0.1, 0.3, 0.5, 0.7, 1.0)) {
    for (distance in c("euclidean", "manhattan")) {
        h1 <- gclust(X, gini_threshold=g, distance=distance)
        h4 <- gclust(dist(X, method=distance), gini_threshold=g)
        expect_equal(adjusted_rand_score(cutree(h1, 3), cutree(h4, 3)), 1.0)


        if (require("genie")) {
            h5 <- hclust2(dist(X, method=distance), thresholdGini=g)
            expect_equal(adjusted_rand_score(cutree(h1, 3), cutree(h5, 3)), 1.0)
        }
    }
}

#print(genieclust::gclust(iris[1:4], M=5))
#print(genieclust::gclust(dist(iris[1:4]), M=5))


