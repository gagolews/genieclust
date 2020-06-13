library("testthat")
library("genieclust")
context("gclust")


# TODO:

print(genieclust::gclust(iris[1:4]))
print(genieclust::gclust(dist(iris[1:4])))
print(genieclust::gclust(iris[1:4], gini_threshold=0.5, distance="manhattan"))
print(genieclust::gclust(dist(iris[1:4], method="manhattan"), gini_threshold=0.5))
print(genieclust::gclust(iris[1:4], M=5))
print(genieclust::gclust(dist(iris[1:4]), M=5))
