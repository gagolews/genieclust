/*
 *  An implementation of kd-trees
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

typedef ssize_t         Py_ssize_t;

#include "../src/c_kdtree.h"

#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::RObject test_kdtree(Rcpp::NumericMatrix X, int k)
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
        kdtree<float, 2> tree(XC.data(), n);
        kneighbours<float, 2>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 3) {
        kdtree<float, 3> tree(XC.data(), n);
        kneighbours<float, 3>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 4) {
        kdtree<float, 4> tree(XC.data(), n);
        kneighbours<float, 4>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 5) {
        kdtree<float, 5> tree(XC.data(), n);
        kneighbours<float, 5>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 6) {
        kdtree<float, 6> tree(XC.data(), n);
        kneighbours<float, 6>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 7) {
        kdtree<float, 7> tree(XC.data(), n);
        kneighbours<float, 7>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 8) {
        kdtree<float, 8> tree(XC.data(), n);
        kneighbours<float, 8>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 9) {
        kdtree<float, 9> tree(XC.data(), n);
        kneighbours<float, 9>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 10) {
        kdtree<float, 10> tree(XC.data(), n);
        kneighbours<float, 10>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 11) {
        kdtree<float, 11> tree(XC.data(), n);
        kneighbours<float, 11>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 12) {
        kdtree<float, 12> tree(XC.data(), n);
        kneighbours<float, 12>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 13) {
        kdtree<float, 13> tree(XC.data(), n);
        kneighbours<float, 13>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 14) {
        kdtree<float, 14> tree(XC.data(), n);
        kneighbours<float, 14>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 15) {
        kdtree<float, 15> tree(XC.data(), n);
        kneighbours<float, 15>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 16) {
        kdtree<float, 16> tree(XC.data(), n);
        kneighbours<float, 16>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 17) {
        kdtree<float, 17> tree(XC.data(), n);
        kneighbours<float, 17>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 18) {
        kdtree<float, 18> tree(XC.data(), n);
        kneighbours<float, 18>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 19) {
        kdtree<float, 19> tree(XC.data(), n);
        kneighbours<float, 19>(tree, knn_dist.data(), knn_ind.data(), k);
    }
    else if (d == 20) {
        kdtree<float, 20> tree(XC.data(), n);
        kneighbours<float, 20>(tree, knn_dist.data(), knn_ind.data(), k);
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


//CXX_DEFS="-O3 -march=native" Rscript -e 'Rcpp::sourceCpp("/home/gagolews/Python/genieclust/.devel/kdtree_test_rcpp.cpp")'


/*** R

set.seed(123)
d <- 10
n <- 100000
X <- matrix(rnorm(n*d), ncol=d)

k <- 10

knn_rann <- function(X, k) {
    res_rann <- RANN::nn2(X, k=k+1)
    res_rann[[1]] <- res_rann[[1]][,-1]
    res_rann[[2]] <- res_rann[[2]][,-1]**2
    res_rann
}

funs <- list(
    genieclust_brute=genieclust:::knn_sqeuclid,
    new_kdtree=test_kdtree,
    rann=knn_rann
)


res <- lapply(`names<-`(seq_along(funs), names(funs)), function(i) {
    f <- funs[[i]]
    t <- system.time(y <- f(X, k))
    list(time=t, index=y[[1]], dist=y[[2]])
})

print(rbind(
    sapply(res, `[[`, 1),
    sum_dist=sapply(res, function(e) sum(e$dist)),
    idx_different=sapply(res, function(e) sum(res[[1]]$index != e$index))
))

*/
