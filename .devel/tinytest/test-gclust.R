library("tinytest")
library("genieclust")

verbose <- FALSE
#set.seed(123)
n <- 1000
d <- 2
X <- rbind(
    matrix(rnorm(n*d, -5), nrow=n),
    matrix(rnorm(n*d, 5), nrow=n)
)
y_true <- rep(c(1, 2), c(n, n))

#cat(sprintf("n=%d, d=%d\n", n, d))


# X <- jitter(as.matrix(iris[1:4]))

for (g in c(0.1, 0.3, 0.5, 0.7, 1.0)) {
    for (distance in c("euclidean", "manhattan")) {
        h1 <- gclust(X, gini_threshold=g, distance=distance)
        h4 <- gclust(dist(X, method=distance), gini_threshold=g)
        expect_equal(adjusted_rand_score(cutree(h1, 3), cutree(h4, 3)), 1.0)
        expect_equal(adjusted_rand_score(cutree(h1, 2), y_true), 1.0)

        c3 <- genie(dist(X, method=distance), 3, gini_threshold=g)
        expect_equal(adjusted_rand_score(cutree(h1, 3), c3), 1.0)

        c3 <- genie(X, 3, gini_threshold=g, distance=distance)
        expect_equal(adjusted_rand_score(cutree(h1, 3), c3), 1.0)

        if (require("genie", quietly=TRUE)) {
            h5 <- hclust2(dist(X, method=distance), thresholdGini=g)
            expect_equal(adjusted_rand_score(cutree(h1, 3), cutree(h5, 3)), 1.0)
        }
    }
}




for (M in c(0, 1, 5)) {
    for (g in c(0.1, 0.3, 0.5, 0.7, 1.0)) {
        for (distance in c("euclidean", "manhattan")) {

            c3a <- genie(dist(X, method=distance), 2, gini_threshold=g, M=M)
            c3b <- genie(X, 2, gini_threshold=g, M=M, distance=distance)

            expect_equal(adjusted_rand_score(c3a, y_true), 1.0)
            expect_equal(adjusted_rand_score(c3b, y_true), 1.0)

        }
    }
}

#print(genieclust::gclust(iris[1:4], M=5))
#print(genieclust::gclust(dist(iris[1:4]), M=5))
