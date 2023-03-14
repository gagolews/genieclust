if (require("clusterCrit", quietly=TRUE)) {

    source("CVI_test_proc1.R", local=TRUE)

    CVI_fun <- negated_davies_bouldin_index
    CVI_name <- "DaviesBouldin"

    reference_fun <- function(X, y) {
        if (min(tabulate(y))<=1) -Inf
        else -clusterCrit::intCriteria(X, y, "Davies_Bouldin")[[1]]
    #         else -index.DB(X, y, q=1)$DB
        }

    CVI_test_proc1(CVI_name, CVI_fun, reference_fun)
}
