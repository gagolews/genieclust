if (require("clusterCrit", quietly=TRUE)) {

    source("CVI_test_proc1.R", local=TRUE)

    CVI_GDunn_Factory <- function(lower, upper) {
        CVI_GDunn_specialized <- function(X, y) {
            return(generalised_dunn_index(X, y, lower, upper))
        }
        CVI_GDunn_specialized
    }

    CVI_funs <- list(list(CVI_GDunn_Factory(1, 1), CVI_GDunn_Factory(1, 2), CVI_GDunn_Factory(1, 3)),
                    list(CVI_GDunn_Factory(2, 1), CVI_GDunn_Factory(2, 2), CVI_GDunn_Factory(2, 3)),
                    list(CVI_GDunn_Factory(3, 1), CVI_GDunn_Factory(3, 2), CVI_GDunn_Factory(3, 3)),
                    list(CVI_GDunn_Factory(4, 1), CVI_GDunn_Factory(4, 2), CVI_GDunn_Factory(4, 3)),
                    list(CVI_GDunn_Factory(5, 1), CVI_GDunn_Factory(5, 2), CVI_GDunn_Factory(5, 3)))

    for (lowercase_delta in c(1, 2, 3, 4, 5)) {
        for (uppercase_delta in c(1, 2, 3)) {
            CVI_fun <- CVI_funs[[lowercase_delta]][[uppercase_delta]]
            CVI_name <- sprintf("GDunn_d%d_D%d", lowercase_delta, uppercase_delta)

            reference_fun <- function(X, y) {
                clusterCrit::intCriteria(X, y, sprintf("GDI%d%d", lowercase_delta, uppercase_delta))[[1]]
            }

            CVI_test_proc1(CVI_name, CVI_fun, reference_fun)
        }
    }


    reference_fun <- function(X, y) {
        # print(X)
        # print(dim(X))
        # print(y)

        D <- as.matrix(dist(X))
        n <- nrow(X)
        K <- max(y)

        # print("---")
        # print(y)
        # print(K)
        # print(tabulate(y))
        # print(length(tabulate(y)))

        if (length(tabulate(y)) != K) return(-Inf)


        delta6 <- function(k, l) {

            delta_pom <- function(k, l) {
                D_kl <- D[y==k, y==l]
                # if (!is.matrix(D_kl))
                    # print(D_kl)

                if(!is.matrix(D_kl))
                {
                    if(sum(y==k) == 1)
                    return(min(D_kl))
                    if(sum(y==l) == 1)
                    return(max(D_kl))
                }

                minimums <- apply(D_kl, 1, min)
                max(minimums)
            }

            max(delta_pom(k, l), delta_pom(l, k))
        }

        Delta1 <- function(k) {
            #X_k = X[y==k,]
            #D_k <- as.matrix(dist(X_k))
            D_k <- D[y==k, y==k]
            # if (!is.matrix(D_k))
            #   print(D_k)
            max(D_k)
        }

        numerator = Inf

        for (k in seq_len(K-1)) {
            for (l in seq(k+1, K)) {
                d <- delta6(k, l)
                if (d < numerator) {
                    numerator <- d
                }
            }
        }

        denominator <- -Inf

        for (k in seq_len(K)) {
            d <- Delta1(k)
            if (d > denominator) {
                denominator <- d
            }
        }

        numerator / denominator
    }

    lowercase_delta <- 6
    uppercase_delta <- 1

    CVI_fun <- CVI_GDunn_Factory(6, 1)
    CVI_name <- sprintf("GDunn_d%d_D%d", lowercase_delta, uppercase_delta)

    CVI_test_proc1(CVI_name, CVI_fun, reference_fun)
}
