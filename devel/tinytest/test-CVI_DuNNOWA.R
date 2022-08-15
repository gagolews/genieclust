source("CVI_test_proc1.R", local=TRUE)

owas_numerator <- c("Min", "Mean", "Max", "SMin:2", "SMax:5")
owas_denominator <- c("Min", "Const")#c("Min", "Mean", "Max", "SMin:2", "SMin:5", "SMax:2", "SMax:5", "Const")


smin <- function(x, delta) {
    n <- min(length(x), 3*delta)
    x <- sort(x)[1:n]
    w <- dnorm(1:n, 1, delta)
    sum(x*w)/sum(w)
}


smax <- function(x, delta) {
    n <- min(length(x), 3*delta)
    x <- sort(x, decreasing=TRUE)[1:n]
    w <- dnorm(1:n, 1, delta)
    sum(x*w)/sum(w)
}


for (owa_numerator in owas_numerator) {
    for (owa_denominator in owas_denominator) {
        for (M in c(5)) {

            CVI_fun <- function(X, y, K) dunnowa_index(X, y, K, M, owa_numerator, owa_denominator)
            CVI_name <- sprintf("DuNN_%d_%s_%s", M, owa_numerator, owa_denominator)

            reference_fun <- function(X, y) {
                if (min(tabulate(y)) <= M) return(-Inf)

                D <- as.matrix(dist(X))
                n <- nrow(X)

                od <- t(apply(D, 1, order))
                stopifnot(od[,1] == 1:n)

                nn_i <- od[, 2:(M+1)]
                nn_d <- matrix(D[cbind(rep(1:n, M), as.numeric(nn_i))], nrow=n)
                nn_w <- matrix(y[as.numeric(nn_i)], nrow=n)

                nn_same <- matrix(y[rep(1:n, M)] == as.numeric(nn_w), nrow=n)

                all_d_numerator <- sort(nn_d[!nn_same])
                all_d_denominator <- sort(nn_d[nn_same])

                numerator <- switch(owa_numerator,
                    Min=min(all_d_numerator),
                    Max=max(all_d_numerator),
                    Mean=mean(all_d_numerator),
                    "SMin:2"=smin(all_d_numerator, 2),
                    "SMin:5"=smin(all_d_numerator, 5),
                    "SMax:2"=smax(all_d_numerator, 2),
                    "SMax:5"=smax(all_d_numerator, 5),
                    Const=1.0)
                denominator <- switch(owa_denominator,
                    Min=min(all_d_denominator),
                    Max=max(all_d_denominator),
                    Mean=mean(all_d_denominator),
                    "SMin:2"=smin(all_d_denominator, 2),
                    "SMin:5"=smin(all_d_denominator, 5),
                    "SMax:2"=smax(all_d_denominator, 2),
                    "SMax:5"=smax(all_d_denominator, 5),
                    Const=1.0)

                numerator / denominator
            }

            CVI_test_proc1(CVI_name, CVI_fun, reference_fun)
        }
    }
}
