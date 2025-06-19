/*
 *  An implementation of kd-trees and MSTs based upon them /testing/
 *
 *  Copyleft (C) 2025, Marek Gagolewski <https://www.gagolewski.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If this is not the case, refer to <https://www.gnu.org/licenses/>.
 */


#define GENIECLUST_R
#include "../src/c_fastmst.h"



// [[Rcpp::export]]
Rcpp::RObject test_knn(Rcpp::NumericMatrix X, int k=1, bool use_kdtree=true, int max_leaf_size=32)
{
    using FLOAT = double;  // float is not faster..

    Py_ssize_t n = (Py_ssize_t)X.nrow();
    Py_ssize_t d = (Py_ssize_t)X.ncol();
    if (n < 1 || d < 1) return R_NilValue;
    if (k < 1) return R_NilValue;

    std::vector<FLOAT> XC(n*d);
    Py_ssize_t j = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        for (Py_ssize_t u=0; u<d; ++u)
            XC[j++] = (FLOAT)X(i, u);  // row-major

    std::vector<FLOAT>  nn_dist(n*k);
    std::vector<Py_ssize_t> nn_ind(n*k);

    if (use_kdtree && d >= 2 && d <= 20) {
        Cknn_sqeuclid_kdtree<FLOAT>(XC.data(), n, d, k, nn_dist.data(), nn_ind.data(), max_leaf_size);
    }
    else {
        Cknn_sqeuclid_brute<FLOAT>(XC.data(), n, d, k, nn_dist.data(), nn_ind.data());
    }

    Rcpp::IntegerMatrix out_ind(n, k);
    Rcpp::NumericMatrix out_dist(n, k);
    Py_ssize_t u = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        for (Py_ssize_t j=0; j<k; ++j) {
            out_ind(i, j)  = nn_ind[u]+1.0;  // R-based indexing
            out_dist(i, j) = sqrt(nn_dist[u]);
            u++;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("nn.index")=out_ind,
        Rcpp::Named("nn.dist")=out_dist
    );
}



// [[Rcpp::export]]
Rcpp::RObject test_mst(Rcpp::NumericMatrix X, int M=1, bool use_kdtree=true,
    int max_leaf_size=16, int first_pass_max_brute_size=0, bool use_dtb=false)
{
    using FLOAT = double;  // float is not faster..

    Py_ssize_t n = (Py_ssize_t)X.nrow();
    Py_ssize_t d = (Py_ssize_t)X.ncol();
    if (n < 1 || d < 1) return R_NilValue;
    if (M < 1) return R_NilValue;

    std::vector<FLOAT> XC(n*d);
    Py_ssize_t j = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        for (Py_ssize_t u=0; u<d; ++u)
            XC[j++] = (FLOAT)X(i, u);  // row-major

    std::vector<FLOAT>  tree_dist(n-1);
    std::vector<Py_ssize_t> tree_ind(2*(n-1));
    std::vector<FLOAT>  d_core;

    FLOAT* d_core_ptr;
    if (M > 1) {
        d_core.resize(n);
        d_core_ptr = d_core.data();
    }
    else
        d_core_ptr = NULL;

    if (use_kdtree && d >= 2 && d <= 20) {
        Cmst_euclid_kdtree<FLOAT>(XC.data(), n, d, M, tree_dist.data(), tree_ind.data(), d_core_ptr, max_leaf_size, first_pass_max_brute_size, use_dtb);
    }
    else {
        Cmst_euclid_brute<FLOAT>(XC.data(), n, d, M, tree_dist.data(), tree_ind.data(), d_core_ptr);
    }

    Rcpp::NumericMatrix out(n-1, 3);
    for (Py_ssize_t i=0; i<n-1; ++i) {
        out(i, 0)  = tree_ind[2*i+0] + 1.0;  // R-based indexing // i1 < i2
        out(i, 1)  = tree_ind[2*i+1] + 1.0;  // R-based indexing
        out(i, 2)  = tree_dist[i];
    }

    if (M > 1) {
        Rcpp::NumericVector d_core_out(n);
        for (Py_ssize_t i=0; i<n; ++i) {
            d_core_out[i] = d_core[i];
        }
        out.attr("d_core") = d_core_out;
    }

    return out;
}


// CXX_DEFS="-O3 -march=native" R CMD INSTALL ~/Python/genieclust --preclean
// OMP_NUM_THREADS=1 CXX_DEFS="-O3 -march=native" Rscript -e 'Rcpp::sourceCpp("/home/gagolews/Python/genieclust/.devel/kdtree_test_rcpp.cpp", echo=FALSE)'


/*** R

options(width=200, echo=FALSE)

knn_rann <- function(X, k) {
    res_rann <- RANN::nn2(X, k=k+1)
    res_rann[[1]] <- res_rann[[1]][,-1]
    res_rann[[2]] <- res_rann[[2]][,-1]
    res_rann
}

funs_knn <- list(
#genieclust_brute=function(X, k) test_knn(X, k, use_kdtree=FALSE),
    rann=knn_rann,
    new_kdtree=function(X, k) test_knn(X, k, use_kdtree=TRUE)
)

funs_mst <- list(
    #genieclust_brute=function(X) genieclust:::.mst.default(X, "l2", 1L, cast_float32=FALSE, verbose=FALSE),
    mlpack_1=function(X) genieclust:::.emst_mlpack(X, 1L, FALSE),
    # mlpack_2=function(X) genieclust:::.emst_mlpack(X, 2L, FALSE),
    mlpack_4=function(X) genieclust:::.emst_mlpack(X, 4L, FALSE),
    # mlpack_8=function(X) genieclust:::.emst_mlpack(X, 8L, FALSE),
    # new_1_00=function(X) test_mst(X, 1L, 1L, 0L),
    # new_1_16=function(X) test_mst(X, 1L, 1L, 16L),
    # new_1_32=function(X) test_mst(X, 1L, 1L, 32L),
    # new_2_00=function(X) test_mst(X, 1L, 2L, 0L),
    #new_2_16=function(X) test_mst(X, 1L, 2L, 16L),
    #new_2_32=function(X) test_mst(X, 1L, 2L, 32L),
    new_4_00=function(X) test_mst(X, 1L, 4L, 0L),
    new_4_16=function(X) test_mst(X, 1L, 4L, 16L),
    # new_4_32=function(X) test_mst(X, 1L, 4L, 32L),
    new_4_64=function(X) test_mst(X, 1L, 4L, 64L)
)

funs_mst_mutreach <- list(
#   new_4_0=function(X, M) test_mst(X, M, TRUE, 4L, 0L),
#   new_8_0=function(X, M) test_mst(X, M, TRUE, 8L, 0L),
#   new_4_16=function(X, M) test_mst(X, M, TRUE, 4L, 16L),
    new_16_0=function(X, M) test_mst(X, M, TRUE, 16L, 0L),
    #new_32_0=function(X, M) test_mst(X, M, TRUE, 32L, 0L),
    dtb_4_16=function(X, M) test_mst(X, M, TRUE, 4L, 16L, use_dtb=TRUE),
    genieclust_brute=function(X, M) {
        if (nrow(X) <= 100000) test_mst(X, M, FALSE)
        else matrix(NA, nrow=1, ncol=3)
    }
)

# n <- 1000000
# for (d in c()) {
#     k <- 1L
#     set.seed(123)
#     X <- matrix(rnorm(n*d), ncol=d)
#     cat(sprintf("n=%d, d=%d, k=%d, OMP_NUM_THREADS=%s\n", n, d, Sys.getenv("OMP_NUM_THREADS")))
#
#     res <- lapply(`names<-`(seq_along(funs_knn), names(funs_knn)), function(i) {
#         f <- funs_knn[[i]]
#         t <- system.time(y <- f(X, k))
#         list(time=t, index=y[[1]], dist=y[[2]])
#     })
#
#     print(cbind(
#         as.data.frame(t(sapply(res, `[[`, 1)))[, 1:3],
#         Δdist=sapply(res, function(e) sum(e[[2]])-sum(res[[1]][[2]])),
#         Δidx=sapply(res, function(e) sum(e[[1]] != res[[1]][[1]]))
#     ))
# }


for (n in c(1000000)) for (d in c(2, 5)) {
    set.seed(123)
    X <- matrix(rnorm(n*d), ncol=d)

    for (M in c(1, 10)) {
        cat(sprintf("n=%d, d=%d, M=%d, OMP_NUM_THREADS=%s\n", n, d, M, Sys.getenv("OMP_NUM_THREADS")))

        res <- lapply(`names<-`(seq_along(funs_mst_mutreach), names(funs_mst_mutreach)), function(i) {
            f <- funs_mst_mutreach[[i]]
            t <- system.time(y <- f(X, M))
            list(time=t, y)
        })

        print(cbind(
            as.data.frame(t(sapply(res, `[[`, 1)))[, 1:3],
            Δdist=sapply(res, function(e) sum(e[[2]][,3])-sum(res[[1]][[2]][, 3])),
            Σdist=sapply(res, function(e) sum(e[[2]][,3])),
            Δidx=sapply(res, function(e) sum(res[[1]][[2]][,-3] != e[[2]][, -3]))
        ))
    }
}

*/




/*

apollo @ 2025-06-19 10:00
n=1000000, d=2, M=1, OMP_NUM_THREADS=1                          n=1000000, d=2, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0             1.591    0.020   1.613     0 3227.846    0          new_16_0             0.792     0 3227.846    0
dtb_4_16             1.199    0.028   1.229     0 3227.846    0          dtb_4_16             1.101     0 3227.846    0
genieclust_brute     0.000    0.000   0.000    NA       NA   NA          genieclust_brute     0.000    NA       NA   NA
n=1000000, d=2, M=10, OMP_NUM_THREADS=1                         n=1000000, d=2, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0             1.472    0.020   1.493     0 8381.135    0          new_16_0             0.703     0 8381.135    0
dtb_4_16             1.987    0.035   2.022     0 8381.135    0          dtb_4_16             1.665     0 8381.135    0
genieclust_brute     0.002    0.000   0.001    NA       NA   NA          genieclust_brute     0.002    NA       NA   NA
n=1000000, d=5, M=1, OMP_NUM_THREADS=1                          n=1000000, d=5, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0            13.556    0.001  13.559     0 195160.6    0          new_16_0             3.889     0 195160.6    0
dtb_4_16            20.947    0.052  21.007     0 195160.6    0          dtb_4_16            19.424     0 195160.6    0
genieclust_brute     0.000    0.000   0.000    NA       NA   NA          genieclust_brute     0.000    NA       NA   NA
n=1000000, d=5, M=10, OMP_NUM_THREADS=1                         n=1000000, d=5, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0             7.833    0.054   7.891     0 302141.8    0          new_16_0             2.315     0 302141.8    0
dtb_4_16            16.725    0.025  16.755     0 302141.8    0          dtb_4_16            12.205     0 302141.8    0
genieclust_brute     0.000    0.000   0.000    NA       NA   NA          genieclust_brute     0.000    NA       NA   NA




hades @ 2025-06-17 16:12 1 thread
n=100000, d=2, M=1
                 user.self sys.self elapsed Δdist Δidx
new_16_0             0.137    0.003   0.140     0    0
dtb_4_16             0.113    0.001   0.114     0    0
genieclust_brute     8.892    0.009   8.903     0    0
n=100000, d=2, M=10
                 user.self sys.self elapsed Δdist Δidx
new_16_0             0.126    0.004   0.130     0    0
dtb_4_16             0.161    0.001   0.162     0    0
genieclust_brute    19.906    0.024  19.934     0    0
n=100000, d=5, M=1
                 user.self sys.self elapsed Δdist Δidx
new_16_0             1.171    0.000   1.170     0    0
dtb_4_16             1.731    0.000   1.731     0    0
genieclust_brute    13.049    0.004  13.054     0    0
n=100000, d=5, M=10
                 user.self sys.self elapsed Δdist Δidx
new_16_0             0.688    0.000   0.688     0    0
dtb_4_16             1.256    0.000   1.256     0    0
genieclust_brute    28.448    0.019  28.470     0   19
n=1000000, d=2, M=1
                 user.self sys.self elapsed Δdist Δidx
new_16_0             1.608    0.009   1.618     0    0
dtb_4_16             1.183    0.024   1.207     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA
n=1000000, d=2, M=10
                 user.self sys.self elapsed Δdist Δidx
new_16_0             1.438    0.014   1.453     0    0
dtb_4_16             1.959    0.034   1.993     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA
n=1000000, d=5, M=1
                 user.self sys.self elapsed Δdist Δidx
new_16_0            13.813    0.009  13.823     0    0
dtb_4_16            20.854    0.034  20.889     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA
n=1000000, d=5, M=10
                 user.self sys.self elapsed Δdist Δidx
new_16_0             7.875    0.011   7.887     0    0
dtb_4_16            16.639    0.023  16.663     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA


2025-06-17 17:23
n=100000, d=2, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0             0.209    0.012   0.071     0    0
dtb_4_16             0.120    0.003   0.102     0    0
genieclust_brute    15.798    1.314   5.857     0    0
n=100000, d=2, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0             0.175    0.004   0.066     0    0
dtb_4_16             0.180    0.002   0.128     0    0
genieclust_brute    38.848    2.773  13.450     0    0
n=100000, d=5, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0             1.609    0.000   0.351     0    0
dtb_4_16             1.801    0.000   1.592     0    0
genieclust_brute    26.121    1.295   8.272     0    0
n=100000, d=5, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0             0.900    0.000   0.201     0    0
dtb_4_16             1.438    0.000   0.842     0    0
genieclust_brute    59.741    3.053  18.451     0   19
n=1000000, d=2, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0             2.422    0.061   0.786     0    0
dtb_4_16             1.271    0.020   1.097     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA
n=1000000, d=2, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0             1.917    0.017   0.680     0    0
dtb_4_16             2.136    0.036   1.651     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA
n=1000000, d=5, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0            21.357    0.000   4.539     0    0
dtb_4_16            22.087    0.011  19.464     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA
n=1000000, d=5, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist Δidx
new_16_0            10.206    0.010   2.299     0    0
dtb_4_16            18.608    0.019  12.105     0    0
genieclust_brute     0.000    0.000   0.000    NA   NA
*/
