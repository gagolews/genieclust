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

/*
// [[Rcpp::export]]
Rcpp::RObject test_kdtree(Rcpp::NumericMatrix X, int k, int max_leaf_size=32)
{
    using FLOAT = float;
    using DISTANCE=,......... TODO

    size_t n = X.nrow();
    size_t d = X.ncol();
    if (k < 1 || (size_t)k >= n) return R_NilValue;
    if (n < 1) return R_NilValue;


    std::vector<FLOAT> XC(n*d);
    size_t j = 0;
    for (size_t i=0; i<n; ++i)
        for (size_t u=0; u<d; ++u)
            XC[j++] = X(i, u);  // row-major

    std::vector<size_t> knn_ind(n*k);
    std::vector<FLOAT> knn_dist(n*k);

    // omfg; templates...
    if (d == 2) {
        mgtree::kdtree_sqeuclid<FLOAT, 2> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 2>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 3) {
        mgtree::kdtree_sqeuclid<FLOAT, 3> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 3>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 4) {
        mgtree::kdtree_sqeuclid<FLOAT, 4> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 4>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 5) {
        mgtree::kdtree_sqeuclid<FLOAT, 5> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 5>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 6) {
        mgtree::kdtree_sqeuclid<FLOAT, 6> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 6>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 7) {
        mgtree::kdtree_sqeuclid<FLOAT, 7> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 7>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 8) {
        mgtree::kdtree_sqeuclid<FLOAT, 8> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 8>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 9) {
        mgtree::kdtree_sqeuclid<FLOAT, 9> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 9>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 10) {
        mgtree::kdtree_sqeuclid<FLOAT, 10> tree(XC.data(), n, max_leaf_size);
        mgtree::kneighbours_sqeuclid<FLOAT, 10>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else
        return R_NilValue;  // TODO


    // TODO: recompute distances with full precision
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
*/


template <class FLOAT, Py_ssize_t D, class DISTANCE>
void _test_mst(
    FLOAT* XC, size_t n, size_t max_leaf_size, size_t first_pass_max_brute_size,
    FLOAT* tree_dist, size_t* tree_ind
) {
    mgtree::dtb<FLOAT, D, DISTANCE> tree(XC, n, max_leaf_size, first_pass_max_brute_size);
    mgtree::mst<FLOAT, D>(tree, tree_dist, tree_ind, false);
}


template <class FLOAT, Py_ssize_t D>
void _test_mst_sqeuclid(
    FLOAT* XC, size_t n, size_t max_leaf_size, size_t first_pass_max_brute_size,
    FLOAT* tree_dist, size_t* tree_ind
) {
    using DISTANCE=mgtree::kdtree_distance_sqeuclid<FLOAT, D>;
    _test_mst<FLOAT, D, DISTANCE>(XC, n, max_leaf_size, first_pass_max_brute_size, tree_dist, tree_ind);
}


// [[Rcpp::export]]
Rcpp::RObject test_mst(Rcpp::NumericMatrix X, int max_leaf_size=4, int first_pass_max_brute_size=16)
{
    using FLOAT = float;

    size_t n = X.nrow();
    size_t d = X.ncol();
    if (n < 1) return R_NilValue;

    std::vector<FLOAT> XC(n*d);
    size_t j = 0;
    for (size_t i=0; i<n; ++i)
        for (size_t u=0; u<d; ++u)
            XC[j++] = (FLOAT)X(i, u);  // row-major

    std::vector<FLOAT>  tree_dist(n-1);
    std::vector<size_t> tree_ind(2*(n-1));

    // LMAO! Templates...
    if (d == 2) {
        _test_mst_sqeuclid<FLOAT, 2>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 3) {
        _test_mst_sqeuclid<FLOAT, 3>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 4) {
        _test_mst_sqeuclid<FLOAT, 4>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 5) {
        _test_mst_sqeuclid<FLOAT, 5>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 6) {
        _test_mst_sqeuclid<FLOAT, 6>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 7) {
        _test_mst_sqeuclid<FLOAT, 7>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 8) {
        _test_mst_sqeuclid<FLOAT, 8>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 9) {
        _test_mst_sqeuclid<FLOAT, 9>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else if (d == 10) {
        _test_mst_sqeuclid<FLOAT, 10>(XC.data(), n, max_leaf_size, first_pass_max_brute_size, tree_dist.data(), tree_ind.data());
    }
    else
        return R_NilValue;  // TODO


    // TODO: ----> a separate function, use it in the brute-force algo too
    // TODO !!!!!!!!!!!!!!!!!!!!!
    // recompute distances with full precision,
    // use L2 distance,
    // sort MST edges wrt dist
    std::vector< CMstTriple<double> > mst(n-1);

    for (size_t i=0; i<n-1; ++i) {
        GENIECLUST_ASSERT(tree_ind[2*i+0] != tree_ind[2*i+1]);
        GENIECLUST_ASSERT(tree_ind[2*i+0] < n);
        GENIECLUST_ASSERT(tree_ind[2*i+1] < n);

        double _dist = 0.0;
        for (size_t j=0; j<d; ++j)
            _dist += square(X(tree_ind[2*i+0], j)-X(tree_ind[2*i+1], j));
        _dist = sqrt(_dist);

        mst[i] = CMstTriple<double>(tree_ind[2*i+0], tree_ind[2*i+1], _dist);
    }

    std::sort(mst.begin(), mst.end());

    Rcpp::NumericMatrix out(n-1, 3);
    for (size_t i=0; i<n-1; ++i) {
        out(i, 0)  = mst[i].i1 + 1.0;  // R-based indexing // i1 < i2
        out(i, 1)  = mst[i].i2 + 1.0;  // R-based indexing
        out(i, 2)  = mst[i].d;
    }

    return out;
}


// CXX_DEFS="-O3 -march=native" R CMD INSTALL ~/Python/genieclust --preclean
// OMP_NUM_THREADS=1 CXX_DEFS="-O3 -march=native" Rscript -e 'Rcpp::sourceCpp("/home/gagolews/Python/genieclust/.devel/kdtree_test_rcpp.cpp")'


/*** R

options(width=200)

knn_rann <- function(X, k) {
    res_rann <- RANN::nn2(X, k=k+1)
    res_rann[[1]] <- res_rann[[1]][,-1]
    res_rann[[2]] <- res_rann[[2]][,-1]**2
    res_rann
}

#funs_knn <- list(
#    genieclust_brute=genieclust:::knn_sqeuclid,
#        rann=knn_rann,
#    new_kdtree=test_kdtree
#)

funs_mst <- list(
    genieclust_brute=function(X) genieclust:::.mst.default(X, "l2", 1L, cast_float32=FALSE, verbose=FALSE),
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

        print(cbind(
            as.data.frame(t(sapply(res, `[[`, 1)))[,1:3],
            Δdist=sapply(res, function(e) sum(e$dist)-sum(res[[1]]$dist)),
            Δidx=sapply(res, function(e) sum(res[[1]]$index != e$index))
        ))
    }
    else {
        res <- lapply(`names<-`(seq_along(funs_mst), names(funs_mst)), function(i) {
            f <- funs_mst[[i]]
            t <- system.time(y <- f(X))
            list(time=t, y)
        })

        print(cbind(
            as.data.frame(t(sapply(res, `[[`, 1)))[,1:3],
            Δdist=sapply(res, function(e) sum(e[[2]][,3])-sum(res[[1]][[2]][,3])),
            Δidx=sapply(res, function(e) sum(res[[1]][[2]][,-3] != e[[2]][,-3]))
        ))
    }
}

*/




/*
hades @ 2025-06-09 10:30

              =============== -O3 -march=native == =============== -O2 ===============
n=100000, d=2
              genieclust_brute  mlpack_1  mlpack_4 genieclust_brute  mlpack_1  mlpack_4 new_2_16 new_2_32 new_4_00 new_4_16 new_4_64
user.self               11.055     0.392     0.290            8.699     0.314     0.250    0.117    0.119    0.109    0.107    0.110
sys.self                 0.007     0.010     0.000            0.015     0.009     0.000    0.000    0.000    0.000    0.000    0.000
elapsed                 11.063     0.402     0.290            8.715     0.323     0.250    0.118    0.119    0.110    0.107    0.111
user.child               0.000     0.000     0.000            0.000     0.000     0.000    0.000    0.000    0.000    0.000    0.000
sys.child                0.000     0.000     0.000            0.000     0.000     0.000    0.000    0.000    0.000    0.000    0.000
sum_dist              1013.976  1013.976  1013.976         1013.976  1013.976  1013.976 1013.976 1013.976 1013.976 1013.976 1013.976
idx_different            0.000 31947.000 31947.000            0.000 31929.000 31929.000   84.000   84.000   84.000   84.000   84.000
n=100000, d=5
              genieclust_brute  mlpack_1  mlpack_4 genieclust_brute  mlpack_1  mlpack_4  new_2_16  new_2_32 new_4_00  new_4_16  new_4_64
user.self               12.450     3.515     3.934           13.111     2.972     3.460     1.909     1.876     1.76     1.676     1.634
sys.self                 0.006     0.006     0.000            0.008     0.006     0.001     0.000     0.000     0.00     0.000     0.000
elapsed                 12.457     3.520     3.934           13.120     2.979     3.460     1.908     1.876     1.76     1.676     1.635
user.child               0.000     0.000     0.000            0.000     0.000     0.000     0.000     0.000     0.00     0.000     0.000
sys.child                0.000     0.000     0.000            0.000     0.000     0.000     0.000     0.000     0.00     0.000     0.000
sum_dist             30703.016 30703.016 30703.016        30703.016 30703.016 30703.016 30703.016 30703.016 30703.02 30703.016 30703.016
idx_different            0.000  2188.000  2188.000            0.000  2176.000  2176.000   332.000   332.000   332.00   332.000   332.000

n=100000, d=2
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute     8.443    0.011   8.455     0    0  float64
mlpack_1             0.319    0.005   0.325     0    0
mlpack_4             0.249    0.000   0.249     0    0
new_2_16             0.119    0.000   0.119     0    0
new_2_32             0.119    0.000   0.118     0    0
new_4_00             0.110    0.000   0.110     0    0
new_4_16             0.106    0.000   0.107     0    0
new_4_64             0.109    0.000   0.110     0    0
n=100000, d=5
                 user.self sys.self elapsed Δdist Δidx
genieclust_brute    13.255    0.009  13.266     0    0
mlpack_1             2.984    0.007   2.991     0    0
mlpack_4             3.462    0.000   3.462     0    0
new_2_16             1.907    0.000   1.907     0    0
new_2_32             1.882    0.000   1.881     0    0
new_4_00             1.769    0.000   1.768     0    0
new_4_16             1.678    0.000   1.679     0    0
new_4_64             1.640    0.000   1.640     0    0



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

13:09
n=100000, d=2
              genieclust_brute  mlpack_1  mlpack_4 new_2_16 new_2_32 new_4_00 new_4_16 new_4_64
user.self                8.476     0.376     0.278    0.118    0.119    0.111    0.107    0.112
sys.self                 0.010     0.007     0.000    0.000    0.000    0.000    0.000    0.000
elapsed                  8.488     0.383     0.278    0.118    0.119    0.111    0.107    0.112
user.child               0.000     0.000     0.000    0.000    0.000    0.000    0.000    0.000
sys.child                0.000     0.000     0.000    0.000    0.000    0.000    0.000    0.000
sum_dist              1013.976  1013.976  1013.976 1013.976 1013.976 1013.976 1013.976 1013.976
idx_different            0.000 31929.000 31929.000   84.000   84.000   84.000   84.000   84.000
n=100000, d=5
              genieclust_brute  mlpack_1  mlpack_4  new_2_16  new_2_32  new_4_00  new_4_16  new_4_64
user.self               12.957     3.403     3.832     1.907     1.879     1.763     1.674     1.637
sys.self                 0.008     0.007     0.000     0.000     0.000     0.000     0.000     0.000
elapsed                 12.967     3.410     3.831     1.907     1.878     1.764     1.675     1.636
user.child               0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sys.child                0.000     0.000     0.000     0.000     0.000     0.000     0.000     0.000
sum_dist             30703.016 30703.016 30703.016 30703.016 30703.016 30703.016 30703.016 30703.016
idx_different            0.000  2176.000  2176.000   332.000   332.000   332.000   332.000   332.000

n=1000000, d=2
              new_2_16 new_2_32 new_4_00 new_4_16 new_4_64
user.self        1.280    1.301    1.197    1.183    1.210
sys.self         0.038    0.027    0.014    0.010    0.011
elapsed          1.319    1.329    1.217    1.193    1.222
user.child       0.000    0.000    0.000    0.000    0.000
sys.child        0.000    0.000    0.000    0.000    0.000
sum_dist      3227.846 3227.846 3227.846 3227.846 3227.846
idx_different    0.000    0.000    0.000    0.000    0.000
n=1000000, d=5
                new_2_16   new_2_32   new_4_00   new_4_16   new_4_64
user.self         22.766     22.525     21.037     20.162     19.745
sys.self           0.062      0.033      0.025      0.018      0.026
elapsed           22.831     22.559     21.064     20.183     19.775
user.child         0.000      0.000      0.000      0.000      0.000
sys.child          0.000      0.000      0.000      0.000      0.000
sum_dist      195160.645 195160.645 195160.645 195160.645 195160.645
idx_different      0.000      0.000      0.000      0.000      0.000

*/
