source("CVI_test_proc1.R", local=TRUE)

for (M in c(5)) {

    CVI_fun <- function(X, y, K) wcnn_index(X, y, K, M)
    CVI_name <- sprintf("WCNN_%d", M)

    reference_fun <- function(X, y) {
        if (min(tabulate(y)) <= M) return(-Inf)

        D <- as.matrix(dist(X))
        n <- nrow(X)
        num <- 0
        for (i in 1:n) {
            o <- order(D[i,])
            num <- num + sum(y[o[2:(M+1)]] == y[i])
        }
        num/(M*n)
    }

    CVI_test_proc1(CVI_name, CVI_fun, reference_fun)
}
