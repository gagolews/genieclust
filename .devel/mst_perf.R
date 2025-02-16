env <- "-O3 -march=native"

system2("R", c("CMD", "INSTALL", ".", "--preclean"), env=sprintf("CXX_DEFS='%s'", env))
library("genieclust")

set.seed(123)
n <- 100000
d <- 2
X <- matrix(rnorm(n*d), nrow=n)

t1 <- system.time(mst1 <- mst(X, cast_float32=FALSE))
t2 <- system.time(mst2 <- emst_mlpack(X))

stopifnot(abs(sum(mst1[,3])-sum(mst2[,3])) < 1e-16)

print(t1)
print(t2)
