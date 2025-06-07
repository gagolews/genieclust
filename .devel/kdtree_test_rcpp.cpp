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
Rcpp::RObject test_mst(Rcpp::NumericMatrix X, int max_leaf_size=2)
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
        mgtree::kdtree<float, 2> tree(XC.data(), n, max_leaf_size);
        mgtree::mst<float, 2>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 3) {
        mgtree::kdtree<float, 3> tree(XC.data(), n, max_leaf_size);
        mgtree::mst<float, 3>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 4) {
        mgtree::kdtree<float, 4> tree(XC.data(), n, max_leaf_size);
        mgtree::mst<float, 4>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 5) {
        mgtree::kdtree<float, 5> tree(XC.data(), n, max_leaf_size);
        mgtree::mst<float, 5>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 6) {
        mgtree::kdtree<float, 6> tree(XC.data(), n, max_leaf_size);
        mgtree::mst<float, 6>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 7) {
        mgtree::kdtree<float, 7> tree(XC.data(), n);
        mgtree::mst<float, 7>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 8) {
        mgtree::kdtree<float, 8> tree(XC.data(), n, max_leaf_size);
        mgtree::mst<float, 8>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 9) {
        mgtree::kdtree<float, 9> tree(XC.data(), n, max_leaf_size);
        mgtree::mst<float, 9>(tree, tree_dist.data(), tree_ind.data());
    }
    else if (d == 10) {
        mgtree::kdtree<float, 10> tree(XC.data(), n, max_leaf_size);
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
    mlpack_16=function(X) genieclust:::.emst_mlpack(X, 16L, FALSE),
    mlpack_32=function(X) genieclust:::.emst_mlpack(X, 32L, FALSE),
    new_mst_1=function(X) test_mst(X, 1L),
    new_mst_16=function(X) test_mst(X, 16L),
    new_mst_32=function(X) test_mst(X, 32L)
)



for (d in c(2, 5, 10)) {
    set.seed(123)
    n <- 10000
    X <- matrix(rnorm(n*d), ncol=d)

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
