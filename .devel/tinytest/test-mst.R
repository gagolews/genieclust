library("tinytest")
library("genieclust")



set.seed(123)
n <- 10000
d <- 2
X <- matrix(rnorm(n*d), nrow=n)

cat(sprintf("n=%d, d=%d\n", n, d))
print(system.time(t1 <- mst(X, cast_float32=FALSE)))
expect_true(all(t1[,1] < t1[,2]))
expect_true(all(diff(t1[,3])>=0))


print(system.time(t2 <- mst(X)))
# expect_equal(t1[,1], t2[,1])
# expect_equal(t1[,2], t2[,2])
# print(abs(sum(t1[,3])-sum(t2[,3])))
expect_true(abs(sum(t1[,3])-sum(t2[,3]))<1e-5)
expect_true(all(t2[,1] < t2[,2]))
expect_true(all(diff(t2[,3])>=0))


print(system.time(t2 <- emst_mlpack(X)))
expect_equal(t1[,1], t2[,1])
expect_equal(t1[,2], t2[,2])
#     print(abs(sum(t1[,3])-sum(t2[,3])))
expect_true(abs(sum(t1[,3])-sum(t2[,3]))<1e-12)



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
