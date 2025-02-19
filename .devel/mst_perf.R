n <- 100000
env <- "-O3 -march=native"

system2("R", c("CMD", "INSTALL", ".", "--preclean"),
    env=sprintf("CXX_DEFS='%s'", env))

library("genieclust")

res <- list()

for (d in c(2, 7)) {
    set.seed(123)
    X <- matrix(rnorm(n*d), nrow=n)

    cat(sprintf("env='%s' n=%d d=%d threads=%d\n", env, n, d, as.integer(Sys.getenv("OMP_NUM_THREADS"))))

    t1 <- system.time(mst1 <- mst(X, algorithm="jarnik"))

    if (d > 7)
        t2 <- system.time(mst2 <- NULL)
    else
        t2 <- system.time(mst2 <- mst(X, algorithm="mlpack"))

    if (!is.null(mst2))
        stopifnot(abs(sum(mst1[,3])-sum(mst2[,3])) < 1e-16)

    .res <- rbind(genieclust=t1, mlpack=t2)[, 1:3]
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
