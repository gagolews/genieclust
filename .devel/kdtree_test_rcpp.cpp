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


#ifndef GENIECLUST_ASSERT
#define __GENIECLUST_STR(x) #x
#define GENIECLUST_STR(x) __GENIECLUST_STR(x)

#define GENIECLUST_ASSERT(EXPR) { if (!(EXPR)) \
    throw std::runtime_error( "genieclust: Assertion " #EXPR " failed in "\
        __FILE__ ":" GENIECLUST_STR(__LINE__) ); }
#endif

#include <sys/types.h>
typedef ssize_t         Py_ssize_t;

#include "../src/c_fastmst.h"


#include <Rcpp.h>


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
Rcpp::RObject test_mst(Rcpp::NumericMatrix X, int M=1, bool use_kdtree=true, int max_leaf_size=4, int first_pass_max_brute_size=16)
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
        Cmst_euclid_kdtree<FLOAT>(XC.data(), n, d, M, tree_dist.data(), tree_ind.data(), d_core_ptr, max_leaf_size, first_pass_max_brute_size);
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
    genieclust_brute=function(X, k) test_knn(X, k, use_kdtree=FALSE),
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
    genieclust_brute=function(X, M) test_mst(X, M, FALSE),
    new_4_16=function(X, M) test_mst(X, M, TRUE, 4L, 16L)
)

for (d in c(2, 5)) {
    k <- 10L
    set.seed(123)
    n <- 100000
    X <- matrix(rnorm(n*d), ncol=d)
    cat(sprintf("n=%d, d=%d, k=%d\n", n, d, k))

    res <- lapply(`names<-`(seq_along(funs_knn), names(funs_knn)), function(i) {
        f <- funs_knn[[i]]
        t <- system.time(y <- f(X, k))
        list(time=t, index=y[[1]], dist=y[[2]])
    })

    print(cbind(
        as.data.frame(t(sapply(res, `[[`, 1)))[,1:3],
        Δdist=sapply(res, function(e) sum(e[[2]])-sum(res[[1]][[2]])),
        Δidx=sapply(res, function(e) sum(e[[1]] != res[[1]][[1]]))
    ))
}


for (d in c(2, 5)) {
    set.seed(123)
    n <- 100000
    X <- matrix(rnorm(n*d), ncol=d)
    M <- 10L
    cat(sprintf("n=%d, d=%d, M=%d\n", n, d, M))

    res <- lapply(`names<-`(seq_along(funs_mst_mutreach), names(funs_mst_mutreach)), function(i) {
        f <- funs_mst_mutreach[[i]]
        t <- system.time(y <- f(X, M))
        list(time=t, y)
    })

    print(cbind(
        as.data.frame(t(sapply(res, `[[`, 1)))[,1:3],
        Δdist=sapply(res, function(e) sum(e[[2]][,3])-sum(res[[1]][[2]][,3])),
        Δidx=sapply(res, function(e) sum(res[[1]][[2]][,-3] != e[[2]][,-3]))
    ))
}

*/




/*


apollo @ 2025-06-13 9:11
n=100000, d=2, k=1
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute     9.626    0.012   9.637     0    0
rann                 0.102    0.004   0.106     0    3
new_kdtree           0.026    0.000   0.026     0    3
n=100000, d=5, k=1
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute    13.328    0.006  13.335     0    0
rann                 0.654    0.007   0.661     0    3
new_kdtree           0.175    0.000   0.174     0    3
n=100000, d=2, M=1
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute     9.618    0.008   9.627     0    0
new_4_16             0.110    0.000   0.110     0    0
n=100000, d=5, M=1
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute    13.175    0.008  13.183     0    0
new_4_16             1.692    0.000   1.692     0    0

n=100000, d=2, k=10
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute    11.781    0.015  11.800     0    0
rann                 0.219    0.005   0.224     0    3
new_kdtree           0.074    0.004   0.078     0    3
n=100000, d=5, k=10
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute    15.854    0.023  15.878     0    0
rann                 1.979    0.007   1.986     0    3
new_kdtree           0.500    0.000   0.500     0    3
n=100000, d=2, M=10
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute    19.940    0.025  19.966     0    0
new_4_16             0.135    0.000   0.136     0   40
n=100000, d=5, M=10
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute    27.824    0.033  27.857     0    0
new_4_16             1.681    0.000   1.681     0   22





hades @ 2025-06-11 13:45
n=250000, d=2
         user.self sys.self elapsed Δdist Δidx
mlpack_1     0.982    0.026   1.008     0    0
mlpack_4     0.721    0.000   0.721     0    0
new_4_00     0.290    0.000   0.291     0    0
new_4_16     0.286    0.000   0.285     0    0
new_4_64     0.291    0.000   0.291     0    0
n=250000, d=5
         user.self sys.self elapsed Δdist Δidx
mlpack_1     9.818    0.019   9.842     0    0
mlpack_4    10.819    0.000  10.823     0    0
new_4_00     4.731    0.000   4.732     0    0
new_4_16     4.517    0.000   4.518     0    0
new_4_64     4.421    0.000   4.423     0    0

n=100000, d=2, M=10
                 user.self sys.self elapsed         Δdist  Δidx
genieclust_brute    33.091    0.066  33.164  0.000000e+00     0
new_4_16             0.125    0.002   0.127 -3.687555e-07 81824
n=100000, d=5, M=10
                 user.self sys.self elapsed         Δdist  Δidx
genieclust_brute    49.109    0.067  49.183  0.000000e+00     0
new_4_16             1.683    0.001   1.685 -2.519664e-08 86759




*/
