n <- 10000
env <- "-O3 -march=native"
#Sys.setenv("OMP_NUM_THREADS"=1)

if (FALSE) {
    system2("R", c("CMD", "INSTALL", "~/Python/genieclust", "--preclean"),
        env=sprintf("CXX_DEFS='%s'", env))
}

library("genieclust")

res <- list()

for (d in c(2, 7)) {
    set.seed(123)
    X <- matrix(rnorm(n*d), nrow=n)

    cat(sprintf("env='%s' n=%d d=%d threads=%d\n", env, n, d, as.integer(Sys.getenv("OMP_NUM_THREADS"))))

    t1 <- system.time(mst1 <- genieclust::mst(X, algorithm="jarnik"))

    t2 <- system.time(mst2 <- genieclust::mst(X, algorithm="mlpack"))

    stopifnot(abs(sum(mst1[,3])-sum(mst2[,3])) < 1e-16)

    .res <- rbind(genieclust=t1, mlpack=t2, geniePrim=t3, genieVp=t4)[, 1:3]
    .res <- as.data.frame(.res)
    .res$n <- n
    .res$d <- d
    .res$env <- env
    .res$host <- Sys.info()["nodename"]

    res[[length(res)+1]] <- .res
    print(.res)
}

res <- do.call(rbind.data.frame, res)
print(res)
