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

#include "../src/c_kdtree.h"
#include "../src/c_dtb.h"


#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::RObject test_kdtree(Rcpp::NumericMatrix X, int k, int max_leaf_size=32)
{
    size_t n = X.nrow();
    size_t d = X.ncol();
    if (k < 1 || (size_t)k >= n) return R_NilValue;
    if (n < 1) return R_NilValue;

    std::vector<float> XC(n*d);
    size_t j = 0;
    for (size_t i=0; i<n; ++i)
        for (size_t u=0; u<d; ++u)
            XC[j++] = X(i, u);  // row-major

    std::vector<size_t> knn_ind(n*k);
    std::vector<float> knn_dist(n*k);

    // omfg; templates...
    if (d == 2) {
        mgtree::kdtree<float, 2> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 2>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 3) {
        mgtree::kdtree<float, 3> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 3>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 4) {
        mgtree::kdtree<float, 4> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 4>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 5) {
        mgtree::kdtree<float, 5> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 5>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 6) {
        mgtree::kdtree<float, 6> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 6>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 7) {
        mgtree::kdtree<float, 7> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 7>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 8) {
        mgtree::kdtree<float, 8> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 8>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 9) {
        mgtree::kdtree<float, 9> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 9>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 10) {
        mgtree::kdtree<float, 10> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours<float, 10>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else
        return R_NilValue;  // TODO


    Rcpp::IntegerMatrix out_ind(n, k);
    Rcpp::NumericMatrix out_dist(n, k);
    size_t u = 0;
    for (size_t i=0; i<n; ++i) {
        for (int j=0; j<k; ++j) {
            out_ind(i, j)  = knn_ind[u]+1;  // R-based indexing
            out_dist(i, j) = knn_dist[u];
            u++;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("nn.index")=out_ind,
        Rcpp::Named("nn.dist")=out_dist
    );
}



// [[Rcpp::export]]
Rcpp::RObject test_mst(Rcpp::NumericMatrix X, int max_leaf_size=2, int first_pass_max_brute_size=32)
{
    size_t n = X.nrow();
    size_t d = X.ncol();
    if (n < 1) return R_NilValue;


    std::vector<float> XC(n*d);
    size_t j = 0;
    for (size_t i=0; i<n; ++i)
        for (size_t u=0; u<d; ++u)
            XC[j++] = X(i, u);  // row-major

    std::vector<size_t> tree_ind(2*(n-1));
    std::vector<float> tree_dist(n-1);

    // omfg; templates...
    if (d == 2) {
        mgtree::dtb<float, 2> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 2>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 3) {
        mgtree::dtb<float, 3> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 3>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 4) {
        mgtree::dtb<float, 4> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 4>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 5) {
        mgtree::dtb<float, 5> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 5>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 6) {
        mgtree::dtb<float, 6> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 6>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 7) {
        mgtree::dtb<float, 7> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 7>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 8) {
        mgtree::dtb<float, 8> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 8>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 9) {
        mgtree::dtb<float, 9> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 9>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 10) {
        mgtree::dtb<float, 10> tree(XC.data(), n, max_leaf_size, first_pass_max_brute_size);
        mgtree::mst<float, 10>(tree, tree_dist.data(), tree_ind.data());
    }
    else
        return R_NilValue;  // TODO


    Rcpp::NumericMatrix out(n-1, 3);
    for (size_t i=0; i<n-1; ++i) {
        out(i, 0)  = tree_ind[i*2+0]+1;  // R-based indexing
        out(i, 1)  = tree_ind[i*2+1]+1;  // R-based indexing
        out(i, 2)  = tree_dist[i];
    }

    return out;
}


// OMP_NUM_THREADS=1 CXX_DEFS="-O3 -march=native" Rscript -e 'Rcpp::sourceCpp("/home/gagolews/Python/genieclust/.devel/kdtree_test_rcpp.cpp")'


/*** R

options(width=200)

knn_rann <- function(X, k) {
    res_rann <- RANN::nn2(X, k=k+1)
    res_rann[[1]] <- res_rann[[1]][,-1]
    res_rann[[2]] <- res_rann[[2]][,-1]**2
    res_rann
}

funs_knn <- list(
    genieclust_brute=genieclust:::knn_sqeuclid,
#        rann=knn_rann,
    new_kdtree=test_kdtree
)

funs_mst <- list(
    genieclust_brute=function(X) genieclust:::.mst.default(X, "l2", 1L, TRUE, FALSE),
    mlpack_1=function(X) genieclust:::.emst_mlpack(X, 1L, FALSE),
    # mlpack_2=function(X) genieclust:::.emst_mlpack(X, 2L, FALSE),
    mlpack_4=function(X) genieclust:::.emst_mlpack(X, 4L, FALSE),
    # mlpack_8=function(X) genieclust:::.emst_mlpack(X, 8L, FALSE),
    # new_1_00=function(X) test_mst(X, 1L, 0L),
    # new_1_16=function(X) test_mst(X, 1L, 16L),
    # new_1_32=function(X) test_mst(X, 1L, 32L),
    # new_2_00=function(X) test_mst(X, 2L, 0L),
    new_2_16=function(X) test_mst(X, 2L, 16L),
    new_2_32=function(X) test_mst(X, 2L, 32L),
    new_4_00=function(X) test_mst(X, 4L, 0L),
    new_4_16=function(X) test_mst(X, 4L, 16L),
    # new_4_32=function(X) test_mst(X, 4L, 32L),
    new_4_64=function(X) test_mst(X, 4L, 64L)
)



for (d in c(2, 5)) {
    set.seed(123)
    n <- 100000
    X <- matrix(rnorm(n*d), ncol=d)

    cat(sprintf("n=%d, d=%d\n", n, d))

    k <- 10

    if (FALSE) {
        res <- lapply(`names<-`(seq_along(funs_knn), names(funs_knn)), function(i) {
            f <- funs_knn[[i]]
            t <- system.time(y <- f(X, k))
            list(time=t, index=y[[1]], dist=y[[2]])
        })

        print(rbind(
            sapply(res, `[[`, 1),
            sum_dist=sapply(res, function(e) sum(e$dist)),
            idx_different=sapply(res, function(e) sum(res[[1]]$index != e$index))
        ))
    }
    else {
        res <- lapply(`names<-`(seq_along(funs_mst), names(funs_mst)), function(i) {
            f <- funs_mst[[i]]
            t <- system.time(y <- f(X))
            list(time=t, y)
        })

        print(rbind(
            sapply(res, `[[`, 1),
            sum_dist=sapply(res, function(e) sum(e[[2]][,3])),
            idx_different=sapply(res, function(e) sum(res[[1]][[2]][,-3] != e[[2]][,-3]))
        ))
    }
}

*/



/*
apollo @ 2025-06-08 9:33
n=50000, d=2
              genieclust_brute  mlpack_1 mlpack_16 mlpack_32 new_mst_1 new_mst_2 new_mst_16 new_mst_32
user.self               2.1850    0.1790    0.1760    0.2390    0.0960    0.0940     0.1090     0.1630
sys.self                0.0010    0.0030    0.0000    0.0000    0.0000    0.0000     0.0000     0.0000
elapsed                 2.1870    0.1820    0.1760    0.2390    0.0960    0.0940     0.1100     0.1630
user.child              0.0000    0.0000    0.0000    0.0000    0.0000    0.0000     0.0000     0.0000
sys.child               0.0000    0.0000    0.0000    0.0000    0.0000    0.0000     0.0000     0.0000
sum_dist              713.4248  713.4248  713.4248  713.4248  713.4248  713.4248   713.4248   713.4248
idx_different           0.0000 6218.0000 6218.0000 6218.0000   40.0000   40.0000    40.0000    40.0000
n=50000, d=5
              genieclust_brute  mlpack_1 mlpack_16 mlpack_32 new_mst_1 new_mst_2 new_mst_16 new_mst_32
user.self                3.253     1.589     2.174     2.402     1.688     1.797      9.352     14.573
sys.self                 0.006     0.000     0.000     0.000     0.000     0.000      0.000      0.000
elapsed                  3.260     1.589     2.174     2.401     1.688     1.797      9.352     14.575
user.child               0.000     0.000     0.000     0.000     0.000     0.000      0.000      0.000
sys.child                0.000     0.000     0.000     0.000     0.000     0.000      0.000      0.000
sum_dist             17593.032 17593.032 17593.032 17593.032 17593.032 17593.032  17593.032  17593.032
idx_different            0.000   412.000   412.000   412.000    60.000    60.000     60.000     60.000


12:49
n=50000, d=2
              genieclust_brute  mlpack_1  mlpack_2  mlpack_4  mlpack_8 new_1_00 new_1_16 new_1_32 new_2_00 new_2_16 new_2_32 new_4_00 new_4_16 new_4_32 new_4_64
user.self               2.1750    0.1840    0.1480    0.1350    0.1460   0.0700   0.0680   0.0670   0.0660   0.0640   0.0660   0.0670   0.0650   0.0660   0.0660
sys.self                0.0030    0.0040    0.0000    0.0000    0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0010
elapsed                 2.1780    0.1870    0.1490    0.1350    0.1460   0.0710   0.0680   0.0670   0.0660   0.0640   0.0660   0.0670   0.0650   0.0660   0.0670
user.child              0.0000    0.0000    0.0000    0.0000    0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
sys.child               0.0000    0.0000    0.0000    0.0000    0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
sum_dist              713.4248  713.4248  713.4248  713.4248  713.4248 713.4248 713.4248 713.4248 713.4248 713.4248 713.4248 713.4248 713.4248 713.4248 713.4248
idx_different           0.0000 6218.0000 6218.0000 6218.0000 6218.0000  40.0000  40.0000  40.0000  40.0000  40.0000  40.0000  40.0000  40.0000  40.0000  40.0000
n=50000, d=5
              genieclust_brute  mlpack_1  mlpack_2  mlpack_4  mlpack_8  new_1_00  new_1_16  new_1_32  new_2_00  new_2_16  new_2_32  new_4_00  new_4_16  new_4_32  new_4_64
user.self                3.264     1.587     1.643     1.813     1.995     0.977     0.908     0.893     0.900     0.839     0.826     0.866     0.827     0.812     0.806
sys.self                 0.002     0.006     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
elapsed                  3.267     1.595     1.644     1.813     1.995     0.978     0.908     0.893     0.899     0.840     0.827     0.866     0.826     0.811     0.807
user.child               0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sys.child                0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sum_dist             17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032 17593.032
idx_different            0.000   412.000   412.000   412.000   412.000    60.000    60.000    60.000    60.000    60.000    60.000    60.000    60.000    60.000    60.000
n=50000, d=10
              genieclust_brute  mlpack_1  mlpack_2  mlpack_4  mlpack_8  new_1_00  new_1_16  new_1_32  new_2_00  new_2_16  new_2_32  new_4_00  new_4_16  new_4_32  new_4_64
user.self                5.465    35.203    38.249    43.036    43.202    25.729    24.351    23.783    24.207    22.978    22.423    23.928    23.000    22.470    21.906
sys.self                 0.005     0.001     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.001     0.000     0.000     0.000
elapsed                  5.471    35.210    38.252    43.041    43.209    25.730    24.355    23.785    24.209    22.978    22.428    23.934    23.003    22.474    21.911
user.child               0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sys.child                0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sum_dist             63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167 63050.167
idx_different            0.000   364.000   364.000   364.000   364.000   152.000   152.000   152.000   152.000   152.000   152.000   152.000   152.000   152.000   152.000


13:00
n=100000, d=2
              genieclust_brute  mlpack_1  mlpack_4 new_2_16 new_2_32 new_4_00 new_4_16 new_4_64
user.self                8.482     0.366     0.289    0.130    0.132    0.135    0.134    0.145
sys.self                 0.027     0.010     0.000    0.000    0.000    0.000    0.000    0.000
elapsed                  8.509     0.376     0.290    0.131    0.131    0.135    0.134    0.145
user.child               0.000     0.000     0.000    0.000    0.000    0.000    0.000    0.000
sys.child                0.000     0.000     0.000    0.000    0.000    0.000    0.000    0.000
sum_dist              1013.976  1013.976  1013.976 1013.976 1013.976 1013.976 1013.976 1013.976
idx_different            0.000 31929.000 31929.000   84.000   84.000   84.000   84.000   84.000
n=100000, d=5
              genieclust_brute  mlpack_1  mlpack_4  new_2_16  new_2_32  new_4_00  new_4_16  new_4_64
user.self               12.989     3.403     3.833     1.993     1.967     2.037     1.953     1.913
sys.self                 0.029     0.005     0.000     0.000     0.000     0.000     0.000     0.000
elapsed                 13.022     3.408     3.833     1.993     1.967     2.038     1.952     1.914
user.child               0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sys.child                0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sum_dist             30703.016 30703.016 30703.016 30703.016 30703.016 30703.016 30703.016 30703.016
idx_different            0.000  2176.000  2176.000   332.000   332.000   332.000   332.000   332.000


*/
