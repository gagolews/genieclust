library("testthat")
library("genieclust")
context("gclust")


# TODO:




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


if (require("genie")) {

    n <- 10000
    d <- 100
    X <- matrix(rnorm(n*d), nrow=n)

    print(system.time(hclust2(objects=X)))
    print(system.time(gclust(X)))
        #gclust(dist(X)),
        #hclust2(dist(X)),

}


