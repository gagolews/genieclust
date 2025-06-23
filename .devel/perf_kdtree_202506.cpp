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
#include "c_pico_tree.h"


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
// OMP_NUM_THREADS=6 CXX_DEFS="-std=c++17 -O3 -march=native" Rscript -e 'Rcpp::sourceCpp("/home/gagolews/Python/genieclust/.devel/perf_kdtree_202506.cpp", echo=FALSE)'



/*** R

options(width=200, echo=FALSE)
nthreads <- as.integer(Sys.getenv("OMP_NUM_THREADS", 1))

# CXX_DEFS="-O3 -march=native" Rscript -e 'install.packages(c("RANN", "Rnanoflann", "dbscan", "nabor"))'
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
py_numpy <- import("numpy")
py_genieclust <- import("genieclust")
py_pykdtree <- import("pykdtree")
py_sklearn <- import("sklearn")



TODO: which support nthreads???
TODO: cleanup R interface, use the one from the R package...

knn_py_pykdtree <- function(X, k) {
    res <- py_pykdtree$kdtree$KDTree(X)$query(X, as.integer(k+1))[c(2, 1)]
    res[[1]] <- as.matrix(res[[1]])[,-1]
    res[[2]] <- as.matrix(res[[2]])[,-1]
    res[[1]] <- res[[1]]+1
    res
}

knn_rann_kd <- function(X, k) {
    res <- RANN::nn2(X, k=k+1, treetype = "kd")
    res[[1]] <- res[[1]][,-1]
    res[[2]] <- res[[2]][,-1]
    res
}

knn_rann_bd <- function(X, k) {
    res <- RANN::nn2(X, k=k+1, treetype = "bd")
    res[[1]] <- res[[1]][,-1]
    res[[2]] <- res[[2]][,-1]
    res
}

knn_dbscan <- function(X, k) {
    res <- dbscan::kNN(X, k, query = NULL, approx=0)
    res[c("id", "dist")]
}

knn_nabor <- function(X, k) {
    res <- nabor::knn(X, k=k+1, eps=0)
    res[[1]] <- res[[1]][,-1]
    res[[2]] <- res[[2]][,-1]
    res
}

knn_Rnanoflann <- function(X, k) {
    res <- Rnanoflann::nn(X, X, k=k+1, eps=0)#, parallel=1, cores=nthreads)
    res[[1]] <- res[[1]][,-1]
    res[[2]] <- res[[2]][,-1]
    res
}

knn_py_genieclust_fastmst <- function(X, k) {
    res <- py_genieclust$fastmst$knn_euclid(X, as.integer(k))[c(2, 1)]
    res[[1]] <- res[[1]]+1
    res
}

knn_py_sklearn_neighbours <- function(X, k) {
    res <- py_sklearn$neighbors$NearestNeighbors(n_neighbors=as.integer(k), n_jobs=nthreads)$fit(X)$kneighbors()[c(2,1)]
    res[[1]] <- res[[1]]+1
    res
}


funs_knn <- list(
    #genieclust_brute=function(X, k) test_knn(X, k, use_kdtree=FALSE),
    r_genieclust_fastmst_1=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=1),
    r_genieclust_fastmst_2=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=2),
    r_genieclust_fastmst_4=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=4),
    r_genieclust_fastmst_8=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=8),
    r_genieclust_fastmst_16=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=16),
    r_genieclust_fastmst_32=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=32),
    r_genieclust_fastmst_64=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=64),
    r_genieclust_fastmst_128=function(X, k) test_knn(X, k, use_kdtree=TRUE, max_leaf_size=128),
    pico_tree=function(X, k) knn_pico_tree(X, k),
    py_genieclust_fastmst_32=knn_py_genieclust_fastmst,
    py_pykdtree=knn_py_pykdtree,
    py_sklearn_neighbours=knn_py_sklearn_neighbours,
    # r_dbscan=knn_dbscan,  # ANN library, but supports self-queries
    # r_rann_kd=knn_rann_kd,  # ANN library
    # r_rann_bd=knn_rann_bd,  # slower than knn_rann_kd
    # r_Rnanoflann=knn_Rnanoflann, # very slow...
    r_nabor=knn_nabor
)



data <- list()
ntries <- 3
n <- 1000000
for (d in c(2, 5)) for (k in c(1, 10)) {
    set.seed(123)
    X <- matrix(rnorm(n*d), ncol=d)
    cat(sprintf("n=%d, d=%d, k=%d, OMP_NUM_THREADS=%d\n", n, d, k, nthreads))

    res <- lapply(rep(`names<-`(seq_along(funs_knn), names(funs_knn)), ntries), function(i) {
        f <- funs_knn[[i]]
        t <- system.time(y <- f(X, k))
        list(time=t, index=y[[1]], dist=y[[2]])
    })

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
data <- do.call(rbind.data.frame, data)
write.csv(data, "perf_mst_202506.csv", row.names=FALSE, append=TRUE)
print(aggregate(data[c("elapsed")], data[c("method", "n", "d", "k", "nthreads")], min))
stop()













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



for (n in c(100000)) for (d in c(2, 5)) {
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

apollo @ 2025-06-23 19:44

n=1000000, d=2, k=1, OMP_NUM_THREADS=1
           user.self sys.self elapsed Δdist Δidx
new_kdtree     0.320    0.002   0.321     0    0
nabor          1.045    0.010   1.055     0    0
n=1000000, d=2, k=10, OMP_NUM_THREADS=1
           user.self sys.self elapsed Δdist Δidx
new_kdtree     0.849    0.023   0.871     0    0
nabor          2.282    0.043   2.326     0    0
n=1000000, d=5, k=1, OMP_NUM_THREADS=1
           user.self sys.self elapsed Δdist Δidx
new_kdtree     1.941    0.007   1.950     0    0
nabor          6.061    0.040   6.107     0    0
n=1000000, d=5, k=10, OMP_NUM_THREADS=1
           user.self sys.self elapsed Δdist Δidx
new_kdtree     5.580    0.038   5.621     0    0
nabor         17.292    0.053  17.347     0    0
n=100000, d=2, M=1, OMP_NUM_THREADS=1
                 user.self sys.self elapsed Δdist    Σdist Δidx
new_16_0             0.147    0.000   0.148     0 1013.976    0
dtb_4_16             0.113    0.000   0.114     0 1013.976    0
genieclust_brute     8.886    0.009   8.897     0 1013.976    0
n=100000, d=2, M=10, OMP_NUM_THREADS=1
                 user.self sys.self elapsed Δdist    Σdist Δidx
new_16_0             0.126    0.000   0.126     0 2625.667    0
dtb_4_16             0.163    0.001   0.164     0 2625.667    0
genieclust_brute    20.241    0.017  20.263     0 2625.667    0
n=100000, d=5, M=1, OMP_NUM_THREADS=1
                 user.self sys.self elapsed Δdist    Σdist Δidx
new_16_0             1.111    0.000   1.111     0 30703.02    0
dtb_4_16             1.730    0.000   1.730     0 30703.02    0
genieclust_brute    11.744    0.021  11.767     0 30703.02    0
n=100000, d=5, M=10, OMP_NUM_THREADS=1
                 user.self sys.self elapsed Δdist    Σdist Δidx
new_16_0             0.672    0.000   0.673     0 47511.27    0
dtb_4_16             1.258    0.000   1.258     0 47511.27    0
genieclust_brute    29.424    0.035  29.473     0 47511.27   19

apollo @ 2025-06-19 10:00
n=1000000, d=2, M=1, OMP_NUM_THREADS=1                                   n=1000000, d=2, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0             1.591    0.020   1.613     0 3227.846    0          new_16_0             0.792     0 3227.846    0
dtb_4_16             1.199    0.028   1.229     0 3227.846    0          dtb_4_16             1.101     0 3227.846    0

n=1000000, d=2, M=10, OMP_NUM_THREADS=1                                  n=1000000, d=2, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0             1.472    0.020   1.493     0 8381.135    0          new_16_0             0.703     0 8381.135    0
dtb_4_16             1.987    0.035   2.022     0 8381.135    0          dtb_4_16             1.665     0 8381.135    0

n=1000000, d=5, M=1, OMP_NUM_THREADS=1                                   n=1000000, d=5, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0            13.556    0.001  13.559     0 195160.6    0          new_16_0             3.889     0 195160.6    0
dtb_4_16            20.947    0.052  21.007     0 195160.6    0          dtb_4_16            19.424     0 195160.6    0

n=1000000, d=5, M=10, OMP_NUM_THREADS=1                                  n=1000000, d=5, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                             elapsed Δdist    Σdist Δidx
new_16_0             7.833    0.054   7.891     0 302141.8    0          new_16_0             2.315     0 302141.8    0
dtb_4_16            16.725    0.025  16.755     0 302141.8    0          dtb_4_16            12.205     0 302141.8    0


n=10000000, d=2, M=1, OMP_NUM_THREADS=1                                  n=10000000, d=2, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                            elapsed Δdist    Σdist Δidx
new_16_0            18.309    0.190  18.506     0 10240.63    0          new_16_0            8.511     0 10240.63    0
dtb_4_16            12.956    0.264  13.226     0 10240.63    0          dtb_4_16           11.882     0 10240.63    0

n=10000000, d=2, M=10, OMP_NUM_THREADS=1                                 n=10000000, d=2, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist    Σdist Δidx                            elapsed Δdist    Σdist Δidx
new_16_0            16.977    0.263  17.245     0 26618.46    0          new_16_0            7.788     0 26618.46    0
dtb_4_16            26.528    0.368  26.915     0 26618.46    0          dtb_4_16           23.109     0 26618.46    0

n=10000000, d=5, M=1, OMP_NUM_THREADS=1                                  n=10000000, d=5, M=1, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist   Σdist Δidx                             elapsed Δdist   Σdist Δidx
new_16_0           156.041    0.241 156.369     0 1238081    0           new_16_0           50.679     0 1238081    0
dtb_4_16           244.618    0.511 245.301     0 1238081    0           dtb_4_16          229.648     0 1238081    0

n=10000000, d=5, M=10, OMP_NUM_THREADS=1                                 n=10000000, d=5, M=10, OMP_NUM_THREADS=6
                 user.self sys.self elapsed Δdist   Σdist Δidx                             elapsed Δdist   Σdist Δidx
new_16_0            86.213    0.304  86.571     0 1916435    0           new_16_0           27.088     0 1916435    0
dtb_4_16           304.133    0.441 304.848     0 1916435    0           dtb_4_16          260.267     0 1916435    0




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






# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOTE TMP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if (FALSE) {
    n <- 100
    d <- 5

    set.seed(123)
    X <- matrix(rnorm(n*d), ncol=d)
    for (M in c(1, 10)) {
        cat(sprintf("n=%d, d=%d, M=%d, OMP_NUM_THREADS=%d\n", n, d, M, nthreads))

        res <- lapply(`names<-`(seq_along(funs_mst_mutreach), names(funs_mst_mutreach))[1], function(i) {
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

    stop()
}

# n=10000000, d=5, M=1, OMP_NUM_THREADS=6
# build_tree                                                      : time=       2.276 s
# tree init                                                       : time=       2.342 s
# find_mst_first                                                  : time=       5.806 s
# update_min_dcore                                                : time=       0.000 s
# find_mst iter #1                                                : time=       7.085 s
# find_mst iter #2                                                : time=       8.408 s
# find_mst iter #3                                                : time=       8.200 s
# find_mst iter #4                                                : time=       6.499 s
# find_mst iter #5                                                : time=       5.229 s
# find_mst iter #6                                                : time=       4.550 s
# mst call                                                        : time=      45.888 s
# mst sort                                                        : time=       1.141 s
# Cmst_euclid_kdtree finalise                                     : time=       1.268 s
#          user.self sys.self elapsed Δdist   Σdist Δidx
# new_16_0   239.573    0.284  49.705     0 1238081    0
# n=10000000, d=5, M=10, OMP_NUM_THREADS=6
# build_tree                                                      : time=       2.283 s
# tree init                                                       : time=       2.358 s
# find_mst_first                                                  : time=      19.778 s
# update_min_dcore                                                : time=       0.045 s
# find_mst iter #1                                                : time=       3.777 s
# find_mst iter #2                                                : time=       1.579 s
# find_mst iter #3                                                : time=       0.355 s
# find_mst iter #4                                                : time=       0.128 s
# find_mst iter #5                                                : time=       0.113 s
# mst call                                                        : time=      26.088 s
# mst sort                                                        : time=       1.129 s
# Cmst_euclid_kdtree finalise                                     : time=       1.274 s

# n=100000000, d=5, M=1, OMP_NUM_THREADS=6
# build_tree                                                      : time=      26.973 s
# tree init                                                       : time=      27.588 s
# find_mst_first                                                  : time=      70.429 s
# update_min_dcore                                                : time=       0.000 s
# find_mst iter #1                                                : time=      92.692 s
# find_mst iter #2                                                : time=      94.121 s
# find_mst iter #3                                                : time=      88.599 s
# find_mst iter #4                                                : time=      73.766 s
# find_mst iter #5                                                : time=      58.392 s
# find_mst iter #6                                                : time=      49.972 s
# find_mst iter #7                                                : time=      47.349 s
# mst call                                                        : time=     576.402 s
# mst sort                                                        : time=      12.989 s
# Cmst_euclid_kdtree finalise                                     : time=      14.391 s
#          user.self sys.self elapsed Δdist   Σdist Δidx
# new_16_0  3138.294    2.833 620.394     0 7833186    0
# n=100000000, d=5, M=10, OMP_NUM_THREADS=6
# build_tree                                                      : time=      27.000 s
# tree init                                                       : time=      28.066 s
# find_mst_first                                                  : time=     215.217 s
# update_min_dcore                                                : time=       0.448 s
# find_mst iter #1                                                : time=      42.560 s
# find_mst iter #2                                                : time=      21.648 s
# find_mst iter #3                                                : time=       7.198 s
# find_mst iter #4                                                : time=       1.623 s
# find_mst iter #5                                                : time=       0.936 s
# mst call                                                        : time=     293.107 s
# mst sort                                                        : time=      12.733 s
# Cmst_euclid_kdtree finalise                                     : time=      14.132 s
#          user.self sys.self elapsed Δdist    Σdist Δidx
# new_16_0  1543.238   11.914 337.967     0 12125760

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOTE TMP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


*/



