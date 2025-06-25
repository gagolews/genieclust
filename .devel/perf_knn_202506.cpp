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




// CXX_DEFS="-O3 -march=native" R CMD INSTALL ~/Python/genieclust --preclean
// OMP_NUM_THREADS=10 CXX_DEFS="-std=c++17 -O3 -march=native" Rscript -e 'Rcpp::sourceCpp("~/Python/genieclust/.devel/perf_knn_202506.cpp", echo=FALSE)'

// CXX_DEFS="-O3 -march=native" Rscript -e 'install.packages(c("RANN", "Rnanoflann", "dbscan", "nabor", "reticulate", "mlpack"))'
// CPPFLAGS="-O3 -march=native" pip3 install pykdtree --force --no-binary="pykdtree" --verbose


#define GENIECLUST_R
#include "../src/c_fastmst.h"
#include "perf_knn_202506-pico_tree.h"

/*
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
        Cknn1_euclid_kdtree<FLOAT>(XC.data(), n, d, k, nn_dist.data(), nn_ind.data(), max_leaf_size);
    }
    else {
        Cknn1_euclid_brute<FLOAT>(XC.data(), n, d, k, nn_dist.data(), nn_ind.data());
    }

    Rcpp::IntegerMatrix out_ind(n, k);
    Rcpp::NumericMatrix out_dist(n, k);
    Py_ssize_t u = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        for (Py_ssize_t j=0; j<k; ++j) {
            out_ind(i, j)  = nn_ind[u]+1.0;  // R-based indexing
            out_dist(i, j) = nn_dist[u];
            u++;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("nn.index")=out_ind,
        Rcpp::Named("nn.dist")=out_dist
    );
}
*/



/*** R

options(width=200, echo=FALSE)
host <- Sys.info()[["nodename"]]
nthreads <- as.integer(Sys.getenv("OMP_NUM_THREADS", 1))

ntries <- 3L
ns <- as.integer(c(2**(17:18)))  # from 2**13
ds <- as.integer(c(2:10))   # 2:10
ks <- as.integer(c(1, 10))


pkgs <- c("genieclust", "RANN", "nabor", "dbscan", "mlpack", "reticulate")
for (pkg in pkgs)
    suppressPackageStartupMessages(library(pkg, character.only=TRUE))

use_virtualenv("/home/gagolews/.virtualenvs/python3-default/")
py_numpy      <- import("numpy", convert=FALSE)
py_genieclust <- import("genieclust", convert=FALSE)
py_pykdtree   <- import("pykdtree", convert=FALSE)
py_sklearn    <- import("sklearn", convert=FALSE)


# print(as.data.frame(installed.packages()[pkgs, c("Version"), drop=FALSE]))
#
# import importlib
# modules = ['numba', 'cython', 'numpy', 'scipy', 'sklearn', 'pykdtree', 'genieclust']
# for m in modules:
#     try:
#         print("%20s %s" % (m, importlib.import_module(m).__version__))
#     except:
#         print("%20s ?" %  (m, ))



knn_genieclust_brute <- function(X, k) {
    if (nrow(X) > 300000) return(NULL)
    res <- genieclust::knn_euclid(X, k, algorithm="brute")
    `names<-`(res, c("index", "dist"))
}

knn_genieclust_kdtree <- function(X, k) {
    res <- genieclust::knn_euclid(X, k, algorithm="kd_tree")
    `names<-`(res, c("index", "dist"))
}

knn_py_pykdtree <- function(X, k) {
    k <- as.integer(k+1)
    res <- py_pykdtree$kdtree$KDTree(X)$query(X, k)
    res <- py_to_r(res)[c(2, 1)]
    res[[1]] <- res[[1]][,-1,drop=FALSE]
    res[[2]] <- res[[2]][,-1,drop=FALSE]
    res[[1]] <- res[[1]]+1
    `names<-`(res, c("index", "dist"))
}

knn_py_genieclust_fastmst <- function(X, k) {
    k <- as.integer(k)
    res <- py_genieclust$fastmst$knn_euclid(X, k)
    res <- py_to_r(res)[c(2, 1)]
    res[[1]] <- res[[1]]+1
    `names<-`(res, c("index", "dist"))
}

knn_py_sklearn_neighbours <- function(X, k) {
    k <- as.integer(k)
    res <- py_sklearn$neighbors$NearestNeighbors(n_neighbors=k, n_jobs=nthreads, algorithm="kd_tree")$fit(X)$kneighbors()
    res <- py_to_r(res)[c(2, 1)]
    res[[1]] <- res[[1]]+1
    `names<-`(res, c("index", "dist"))
}


knn_mlpack_kd_singletree <- function(X, k) {
    k <- as.integer(k)
    res <- mlpack::knn(k=k, epsilon=0, reference=X, algorithm='single_tree')[c(2,1)]
    res[[1]] <- 1+res[[1]]#[,-1,drop=FALSE]
    res[[2]] <- res[[2]]#[,-1,drop=FALSE]
    `names<-`(res, c("index", "dist"))
}

knn_mlpack_kd_dualtree <- function(X, k) {
    k <- as.integer(k)
    res <- mlpack::knn(k=k, epsilon=0, reference=X, algorithm='dual_tree')[c(2,1)]
    res[[1]] <- 1+res[[1]]#[,-1,drop=FALSE]
    res[[2]] <- res[[2]]#[,-1,drop=FALSE]
    `names<-`(res, c("index", "dist"))
}


knn_rann_kd <- function(X, k) {
    k <- as.integer(k+1)
    res <- RANN::nn2(X, k=k, treetype = "kd")
    res[[1]] <- res[[1]][,-1,drop=FALSE]
    res[[2]] <- res[[2]][,-1,drop=FALSE]
    `names<-`(res, c("index", "dist"))
}

knn_rann_bd <- function(X, k) {
    k <- as.integer(k+1)
    res <- RANN::nn2(X, k=k, treetype = "bd")
    res[[1]] <- res[[1]][,-1,drop=FALSE]
    res[[2]] <- res[[2]][,-1,drop=FALSE]
    `names<-`(res, c("index", "dist"))
}

knn_dbscan <- function(X, k) {
    k <- as.integer(k)
    res <- dbscan::kNN(X, k, query = NULL, approx=0)
    `names<-`(res[c("id", "dist")], c("index", "dist"))
}

knn_nabor <- function(X, k) {
    k <- as.integer(k+1)
    res <- nabor::knn(X, k=k, eps=0)
    res[[1]] <- res[[1]][,-1,drop=FALSE]
    res[[2]] <- res[[2]][,-1,drop=FALSE]
    `names<-`(res, c("index", "dist"))
}

knn_Rnanoflann <- function(X, k) {
    k <- as.integer(k+1)
    res <- Rnanoflann::nn(X, X, k=k, eps=0)#, parallel=1, cores=nthreads)
    res[[1]] <- res[[1]][,-1,drop=FALSE]
    res[[2]] <- res[[2]][,-1,drop=FALSE]
    `names<-`(res, c("index", "dist"))
}

funs_knn <- list(
    r_genieclust_brute=knn_genieclust_brute,
    r_genieclust_kdtree=knn_genieclust_kdtree,
    # r_Rnanoflann=knn_Rnanoflann, # something's wrong: extremely slow...
    pico_tree=knn_pico_tree  # K-d tree, max_leaf_size=12
)

funs_knn_single <- list(
    r_dbscan=knn_dbscan,     # ANN library, but supports self-queries; K-d tree with bucketSize=10 and the sliding midpoint rule as the splitting criterion
    r_mlpack_kdtree_single=knn_mlpack_kd_singletree,  # K-d trees, leaf_size=20
    r_mlpack_kdtree_dual=knn_mlpack_kd_dualtree,  # K-d trees, leaf_size=20, dual tree algorithm
    r_nabor=knn_nabor,       # libnabo, K-d tree, sliding midpoint bucket size=8(?)
    #r_rann_bd=knn_rann_bd,  # slower than knn_rann_kd; box-decomposition tree
    r_rann_kdtree=knn_rann_kd  # ANN library, bucket size=1(?), sliding midpoint rule
)

funs_knn_py <- list(
    py_genieclust_kdtree=knn_py_genieclust_fastmst,
    py_pykdtree=knn_py_pykdtree,  # leafsize=16
    py_sklearn_kdtree=knn_py_sklearn_neighbours  # sklearn.neighbors.NearestNeighbors algorithm="kd_tree", leaf_size=30
)

for (d in ds) for (n in ns) {
    if (n == 1208592 && d == 0) {
        X <- as.matrix(read.table("~/Python/genieclust/.devel/benchmark_data/thermogauss_scan001.3d.gz"))
    } else {
        set.seed(123)
        X <- matrix(rnorm(n*d), ncol=d)
    }

    Xpy <- py_numpy$asarray(X, order="C", copy=TRUE)

    benchmark <- function(i, X, k, funs) {
        f <- funs[[i]]
        t <- system.time(y <- f(X, k))
        if (is.null(y)) return(NULL)
        stopifnot(dim(y[[1]])==dim(y[[2]]), dim(y[[1]])==c(n, k))
        list(time=t, index=y[[1]], dist=y[[2]])
    }

    for (k in ks) {
        cat(sprintf("n=%d, d=%d, k=%d, OMP_NUM_THREADS=%d\n", n, d, k, nthreads))

        for (i in 1:ntries) {
            res <- list()
            res <- c(res, lapply(`names<-`(seq_along(funs_knn), names(funs_knn)), benchmark, X, k, funs_knn))
            res <- c(res, lapply(`names<-`(seq_along(funs_knn_py), names(funs_knn_py)), benchmark, Xpy, k, funs_knn_py))
            if (nthreads == 1)
                res <- c(res, lapply(`names<-`(seq_along(funs_knn_single), names(funs_knn_single)), benchmark, X, k, funs_knn_single))

            res <- res[!sapply(res, is.null)]

            data <- as.data.frame(`row.names<-`(cbind(
                method=names(res),
                as.data.frame(t(sapply(res, `[[`, 1)))[, 1:3],
                Δdist=sapply(res, function(e) sum(e$dist)-sum(res[[1]]$dist)),
                Σdist=sapply(res, function(e) sum(e$dist)),
                Δidx=sapply(res, function(e) sum(e$index != res[[1]]$index)),
                n=n,
                d=d,
                k=k,
                nthreads=nthreads,
                time=as.integer(Sys.time()),
                host=host
            ), NULL))

            write.table(data, sprintf("perf_knn_202506-%s.csv", host),
                row.names=FALSE, col.names=FALSE, append=TRUE, sep=",", dec=".")
            print(data)
        }
    }
}





*/

/*
                   method       n d  k nthreads elapsed
1               pico_tree 1208592 0 10        1   1.187
2    py_genieclust_kdtree 1208592 0 10        1   1.189
3             py_pykdtree 1208592 0 10        1   1.481
4       py_sklearn_kdtree 1208592 0 10        1   3.413
5                r_dbscan 1208592 0 10        1   2.571
6     r_genieclust_kdtree 1208592 0 10        1   1.171
7    r_mlpack_kdtree_dual 1208592 0 10        1   2.437
8  r_mlpack_kdtree_single 1208592 0 10        1   2.877
9                 r_nabor 1208592 0 10        1   1.443
10          r_rann_kdtree 1208592 0 10        1   2.403

                   method       n d  k nthreads elapsed  nthreads elapsed
1               pico_tree 1208592 3 10        1   3.877         6   1.238
2    py_genieclust_kdtree 1208592 3 10        1   1.666         6   0.691
3             py_pykdtree 1208592 3 10        1   5.179         6   1.621
4       py_sklearn_kdtree 1208592 3 10        1   9.741         6   7.311
5                r_dbscan 1208592 3 10        1   8.496         6   8.505
6     r_genieclust_kdtree 1208592 3 10        1   1.668         6   0.655
7    r_mlpack_kdtree_dual 1208592 3 10        1   3.652         6   3.645
8  r_mlpack_kdtree_single 1208592 3 10        1   3.920         6   3.924
9                 r_nabor 1208592 3 10        1   4.793         6   4.830
10          r_rann_kdtree 1208592 3 10        1  10.190         6  10.187

*/
