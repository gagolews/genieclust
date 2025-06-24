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
#include "perf_knn_202506-pico_tree.h"


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



// CXX_DEFS="-O3 -march=native" R CMD INSTALL ~/Python/genieclust --preclean
// OMP_NUM_THREADS=6 CXX_DEFS="-std=c++17 -O3 -march=native" Rscript -e 'Rcpp::sourceCpp("~/Python/genieclust/.devel/perf_knn_202506.cpp", echo=FALSE)'



/*** R

options(width=200, echo=FALSE)
nthreads <- as.integer(Sys.getenv("OMP_NUM_THREADS", 1))

# CXX_DEFS="-O3 -march=native" Rscript -e 'install.packages(c("RANN", "Rnanoflann", "dbscan", "nabor", "reticulate"))'
# CPPFLAGS="-O3 -march=native" pip3 install pykdtree --force --no-binary="pykdtree" --verbose

# (data, query, k)
# FNN::get.knn algorithm="kd_tree" uses ANN
# list nn.index, nn.dist
#
#
# RANN::nn2 treetype = "kd" ANN
# RANN::nn2 treetype = "bd" box-decomposition tree
#
# dbscan::kNN(x, k, query = NULL, approx=0) ANN library
#
# nabor::(data, query, k)  # libnabo
# # nn.idx, nn.dists
#
# Rnanoflann::nn(data, points, k, parallel=(nthreads>1), cores=nthreads)
#
# RcppHNSW::hnsw_knn - approximate

suppressPackageStartupMessages(library("RANN"))
suppressPackageStartupMessages(library("nabor"))
suppressPackageStartupMessages(library("dbscan"))
suppressPackageStartupMessages(library("Rnanoflann"))

suppressPackageStartupMessages(library("reticulate"))
use_virtualenv("/home/gagolews/.virtualenvs/python3-default/")
py_numpy <- import("numpy", convert=FALSE)
py_genieclust <- import("genieclust", convert=FALSE)
py_pykdtree <- import("pykdtree", convert=FALSE)
py_sklearn <- import("sklearn", convert=FALSE)



# TODO: which support nthreads???
# TODO: cleanup R interface, use the one from the R package...

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
    res <- py_sklearn$neighbors$NearestNeighbors(n_neighbors=k, n_jobs=nthreads)$fit(X)$kneighbors()
    res <- py_to_r(res)[c(2, 1)]
    res[[1]] <- res[[1]]+1
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

knn_genieclust_kdtree <- function(X, k) {
    test_knn(X, k, use_kdtree=TRUE, max_leaf_size=32)
}

knn_genieclust_brute <- function(X, k) {
    test_knn(X, k, use_kdtree=FALSE)
}

funs_knn <- list(
    r_genieclust_brute=knn_genieclust_brute,
    r_genieclust_fastmst=knn_genieclust_kdtree,
    pico_tree=knn_pico_tree,
    r_dbscan=knn_dbscan,    # ANN library, but supports self-queries
    r_rann_kd=knn_rann_kd,  # ANN library
    r_rann_bd=knn_rann_bd,  # slower than knn_rann_kd
    # r_Rnanoflann=knn_Rnanoflann, # something's wrong: extremely slow...
    r_nabor=knn_nabor
)

funs_knn_py <- list(
    py_genieclust_fastmst=knn_py_genieclust_fastmst,
    py_pykdtree=knn_py_pykdtree,
    py_sklearn_neighbours=knn_py_sklearn_neighbours
)

data <- list()
ntries <- 3L
n <- 100000L
for (d in c(5L)) {
    if (n == 1208592 && is.na(d)) {
        X <- as.matrix(read.table("~/Python/genieclust/.devel/benchmark_data/thermogauss_scan001.3d.gz"))
    } else {
        set.seed(123)
        X <- matrix(rnorm(n*d), ncol=d)
    }

    Xpy <- py_numpy$asarray(X, order="C", copy=TRUE)

    benchmark <- function(i, X, k, funs) {
        f <- funs[[i]]
        t <- system.time(y <- f(X, k))
        stopifnot(dim(y[[1]])==dim(y[[2]]), dim(y[[1]])==c(n, k))
        list(time=t, index=y[[1]], dist=y[[2]])
    }

    for (k in c(1L, 10L)) {
        cat(sprintf("n=%d, d=%d, k=%d, OMP_NUM_THREADS=%d\n", n, d, k, nthreads))

        for (i in 1:ntries) {
            res <- list()
            res <- c(res, lapply(`names<-`(seq_along(funs_knn), names(funs_knn)), benchmark, X, k, funs_knn))
            res <- c(res, lapply(`names<-`(seq_along(funs_knn_py), names(funs_knn_py)), benchmark, Xpy, k, funs_knn_py))

            this_data <- `row.names<-`(cbind(
                method=names(res),
                as.data.frame(t(sapply(res, `[[`, 1)))[, 1:3],
                Δdist=sapply(res, function(e) sum(e$dist)-sum(res[[1]]$dist)),
                Σdist=sapply(res, function(e) sum(e$dist)),
                Δidx=sapply(res, function(e) sum(e$index != res[[1]]$index)),
                n=n,
                d=d,
                k=k,
                nthreads=nthreads,
                time=as.integer(Sys.time())
            ), NULL)

            data[[length(data)+1]] <- this_data

            print(this_data)
        }
    }
}
data <- do.call(rbind.data.frame, data)
write.table(data, "perf_knn_202506.csv", row.names=FALSE, col.names=FALSE, append=TRUE, sep=",", dec=".")
print(aggregate(data[c("elapsed")], data[c("method", "n", "d", "k", "nthreads")], min))


*/

/*
                     method       n d  k nthreads elapsed  nthreads elapsed
1                 pico_tree 1208592 3  1        6   0.232         1   0.510
2  py_genieclust_fastmst_32 1208592 3  1        6   0.224         1   0.478
3               py_pykdtree 1208592 3  1        6   0.264         1   0.627
4     py_sklearn_neighbours 1208592 3  1        6   2.238         1   1.982
5    r_genieclust_fastmst_1 1208592 3  1        6   0.423         1   0.819
6  r_genieclust_fastmst_128 1208592 3  1        6   0.214         1   0.516
7   r_genieclust_fastmst_16 1208592 3  1        6   0.247         1   0.498
8    r_genieclust_fastmst_2 1208592 3  1        6   0.370         1   0.729
9   r_genieclust_fastmst_32 1208592 3  1        6   0.228         1   0.466
10   r_genieclust_fastmst_4 1208592 3  1        6   0.322         1   0.640
11  r_genieclust_fastmst_64 1208592 3  1        6   0.213         1   0.467
12   r_genieclust_fastmst_8 1208592 3  1        6   0.281         1   0.559
13                  r_nabor 1208592 3  1        6   0.654         1   0.651
14                pico_tree 1208592 3 10        6   0.438         1   1.151
15 py_genieclust_fastmst_32 1208592 3 10        6   0.535         1   1.204
16              py_pykdtree 1208592 3 10        6   0.641         1   1.535
17    py_sklearn_neighbours 1208592 3 10        6   3.693         1   3.370
18   r_genieclust_fastmst_1 1208592 3 10        6   0.768         1   1.947
19 r_genieclust_fastmst_128 1208592 3 10        6   0.504         1   1.387
20  r_genieclust_fastmst_16 1208592 3 10        6   0.488         1   1.153
21   r_genieclust_fastmst_2 1208592 3 10        6   0.723         1   1.757
22  r_genieclust_fastmst_32 1208592 3 10        6   0.473         1   1.107
23   r_genieclust_fastmst_4 1208592 3 10        6   0.641         1   1.527
24  r_genieclust_fastmst_64 1208592 3 10        6   0.475         1   1.207
25   r_genieclust_fastmst_8 1208592 3 10        6   0.581         1   1.311
26                  r_nabor 1208592 3 10        6   1.476         1   1.477
*/
