library("tinytest")
library("genieclust")
library("deadwood")

set.seed(123)
n1 <- 1003
n2 <- 235

X <- t(cbind(
    `dim<-`(runif(n1*2), c(2, n1)),
    `dim<-`(runif(n2*2), c(2, n2))+c(1.2, 0)
))
y_true <- rep(1:2, c(n1, n2))

y <- deadwood(X, M=10)
# plot(X, col=y+1, asp=1)
expect_equal(mean(y[y_true==2]), 1)

cl <- genie(X, 2, M=10, gini=0.5)
stopifnot(all(cl==y_true) || all(3-cl==y_true))
plot(X, col=cl, asp=1)

y <- deadwood(cl)
plot(X, col=y+1, asp=1)

expect_true(all(attr(y, "contamination")>0.1))
expect_true(all(sapply(split(y, y_true), mean)>0.1))
