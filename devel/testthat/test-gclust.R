library("testthat")
library("genieclust")
context("gclust")


# TODO:



if (require("genie")) {

    set.seed(123)
    n <- 10000
    d <- 2
    X <- matrix(rnorm(n*d), nrow=n)

    cat(sprintf("n=%d, d=%d\n", n, d))
    print(system.time(gclust(X)))
    print(system.time(gclust(X, cast_float32=FALSE)))
    print(system.time(hclust2(objects=X)))
        #gclust(dist(X)),
        #hclust2(dist(X)),

}

#  2020-06-16 15:26
# n=100000, d=100
#    user  system elapsed
# 656.823   0.212 164.319
#     user   system  elapsed
# 1161.052    0.248  290.456
#     user   system  elapsed
# 1032.423    0.140  268.770


n <- 10000
d <- 100
X <- matrix(rnorm(n*d), nrow=n)
print(system.time(gclust(X, verbose=TRUE)))


print(genieclust::gclust(iris[1:4]))
print(genieclust::gclust(dist(iris[1:4])))
print(genieclust::gclust(iris[1:4], gini_threshold=0.5, distance="manhattan"))
print(genieclust::gclust(dist(iris[1:4], method="manhattan"), gini_threshold=0.5))



if (require("genie")) {
    h1 <- gclust(iris[1:4])
    h2 <- gclust(dist(iris[1:4]))
    h3 <- hclust2(dist(iris[1:4]))

    expect_equal(adjusted_rand_score(cutree(h1, 3), cutree(h3, 3)), 1.0)
    expect_equal(adjusted_rand_score(cutree(h2, 3), cutree(h3, 3)), 1.0)
}


#print(genieclust::gclust(iris[1:4], M=5))
#print(genieclust::gclust(dist(iris[1:4]), M=5))


