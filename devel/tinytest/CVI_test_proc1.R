CVI_test_proc1 <- function(CVI_name, CVI_fun, reference_fun)
{
    #cat(sprintf("%s %20s %s\n", stri_dup("*", 30), CVI_name, stri_dup("*", 30)))

    #context(CVI_name)
    #test_that(CVI_name, {

    library("datasets")
    data("iris")
    set.seed(123)

    expect_error(CVI_fun(cbind(1:4, 1:4), c(4, 2, 2, 1), 4))

    X1 <- as.matrix(iris[,1:4])
    X1[,] <- jitter(X1)  # otherwise we get a non-unique solution
    y1 <- as.integer(iris[[5]])

    n <- 1000
    d <- 4
    y2 <- sample(1:5, n, replace=TRUE, prob=c(0.4, 0.2, 0.1, 0.29, 0.01))
    X2 <- do.call(cbind, lapply(1:d, function(i) rnorm(n, y2)))

    X3 <- X1
    y3 <- sample(rep(sample(1:3), times=c(50, 1, nrow(X3)-50-1)))

    X4 <- X1
    y4 <- sample(rep(sample(1:3), times=c(2, 50, nrow(X4)-50-2)))

    Xs <- list(X1, X2, X3, X4)
    ys <- list(y1, y2, y3, y4)

    for (u in seq_along(Xs)) {
        X <- Xs[[u]]
        y <- ys[[u]]
        K <- max(u)
        n <- nrow(X)

        K <- max(y)

        if (!is.null(reference_fun)) {
#             print(microbenchmark::microbenchmark(
#                 direct={i1 <- CVI_fun(X, y, K)},
#                 reference={i2 <- reference_fun(X, y)},
#                 times=1
#             ), unit="ms")

            i1 <- CVI_fun(X, y)
            i2 <- reference_fun(X, y)

            if (is.finite(i2)) expect_equivalent(i2, i1, tolerance=1e-7)
        }
    }
    #})
}
