/*  The "new" (2025) functions to compute k-nearest neighbours
 *  and minimum spanning trees with respect to the Euclidean metric
 *  and thereon-based mutual reachability distance.
 *  The module provides access to a quite fast implementation of K-d trees.
 *
 *  For best speed, consider building the package from sources
 *  using, e.g., `-O3 -march=native` compiler flags.
 *
 *  Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>
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


#include "c_common.h"
#include "c_matrix.h"
// #include "c_oldmst.h"
#include "c_fastmst.h"
#include <cmath>

using namespace Rcpp;


//' @title
//' Get or Set the Number of Threads
//'
//' @description
//' These functions get or set the maximal number of OpenMP threads that
//' can be used by \code{\link{knn_euclid}} and \code{\link{mst_euclid}},
//' amongst others.
//'
//' @param n_threads maximum number of threads to use;
//'
//' @return
//' \code{omp_get_max_threads} returns the maximal number
//' of threads that will be used during the next call to a parallelised
//' function, not the maximal number of threads possibly available.
//' It there is no built-in support for OpenMP, 1 is always returned.
//'
//' For \code{omp_set_num_threads}, the previous value of \code{max_threads}
//' is output.
//'
//'
//' @rdname omp
//' @export
// [[Rcpp::export("omp_set_num_threads")]]
int Romp_set_num_threads(int n_threads)
{
    return Comp_set_num_threads(n_threads);
}



//' @rdname omp
//' @export
// [[Rcpp::export("omp_get_max_threads")]]
int Romp_get_max_threads()
{
    return Comp_get_max_threads();
}



// template<typename T>
// NumericMatrix internal_compute_mst(CDistance<T>* D, Py_ssize_t n, Py_ssize_t M, bool verbose)
// {
//     NumericMatrix ret(n-1, 3);
//
//     CDistance<T>* D2 = NULL;
//     if (M >= 2) { // yep, we need it for M==2 as well
//         if (verbose) GENIECLUST_PRINT("[genieclust] Determining the core distance.\n");
//
//         Py_ssize_t k = M-1;
//         CMatrix<Py_ssize_t> nn_i(n, k);
//         CMatrix<T> nn_d(n, k);
//         Cknn_from_complete(D, n, k, nn_d.data(), nn_i.data());
//
//         NumericMatrix nn_r(n, k);
//
//         std::vector<T> d_core(n);
//         for (Py_ssize_t i=0; i<n; ++i) {
//             d_core[i] = nn_d(i, k-1); // distance to the k-th nearest neighbour
//             GENIECLUST_ASSERT(std::isfinite(d_core[i]));
//
//             for (Py_ssize_t j=0; j<k; ++j) {
//                 GENIECLUST_ASSERT(nn_i(i,j) != i);
//                 nn_r(i,j) = nn_i(i,j)+1; // 1-based indexing
//             }
//         }
//
//         ret.attr("nn") = nn_r;
//
//         D2 = new CDistanceMutualReachability<T>(d_core.data(), n, D);
//     }
//
//     CMatrix<Py_ssize_t> mst_i(n-1, 2);
//     std::vector<T>  mst_d(n-1);
//
//     if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST.\n");
//     Cmst_from_complete<T>(D2?D2:D, n, mst_d.data(), mst_i.data(), verbose);
//     if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");
//
//     if (D2) delete D2;
//
//     for (Py_ssize_t i=0; i<n-1; ++i) {
//         GENIECLUST_ASSERT(mst_i(i,0) < mst_i(i,1));
//         GENIECLUST_ASSERT(std::isfinite(mst_d[i]));
//         ret(i,0) = mst_i(i,0)+1;  // R-based indexing
//         ret(i,1) = mst_i(i,1)+1;  // R-based indexing
//         ret(i,2) = mst_d[i];
//     }
//
//     return ret;
// }
//
//
//
//
//
// template<typename T>
// NumericMatrix internal_mst_default(
//     NumericMatrix X,
//     String distance,
//     Py_ssize_t M,
//     /*bool use_mlpack, */
//     bool verbose)
// {
//     Py_ssize_t n = X.nrow();
//     Py_ssize_t d = X.ncol();
//     NumericMatrix ret;
//
//     if (M < 1 || M >= n-1)
//         stop("`M` must be an integer in [1, n-1)");
//
//     CMatrix<T> X2(REAL(SEXP(X)), n, d, false); // Fortran- to C-contiguous
//
//     T* _X2 = X2.data();
//     for (Py_ssize_t i=0; i<n*d; i++) {
//         if (!std::isfinite(_X2[i]))
//             Rf_error("All elements in the input matrix must be finite and non-missing.");
//     }
//
//
// #if 1
//     // Special case (most frequently used)
//     if (distance == "euclidean" || distance == "l2")
//     {
//         NumericMatrix ret(n-1, 3);
//         CMatrix<Py_ssize_t> mst_i(n-1, 2);
//         std::vector<T>  mst_d(n-1);
//
//         if (M == 1) { // yes, M==2 needs 1-nns
//             if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST.\n");
//             Cmst_euclid<T>(_X2, n, d, mst_d.data(), mst_i.data(), nullptr, verbose);
//             if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");
//         }
//         else {
//             if (verbose) GENIECLUST_PRINT("[genieclust] Computing the nearest neighbours.\n");
//             Py_ssize_t k = M-1;
//             CMatrix<Py_ssize_t> nn_i(n, k);
//             CMatrix<T> nn_d(n, k);
//             Cknn_sqeuclid_brute(_X2, n, d, k, nn_d.data(), nn_i.data(), verbose);
//
//             NumericMatrix nn_r(n, k);
//
//             std::vector<T> d_core(n);
//             for (Py_ssize_t i=0; i<n; ++i) {
//                 d_core[i] = nn_d(i, k-1); // distance to the k-th nearest neighbour
//                 GENIECLUST_ASSERT(std::isfinite(d_core[i]));
//
//                 for (Py_ssize_t j=0; j<k; ++j) {
//                     GENIECLUST_ASSERT(nn_i(i,j) != i);
//                     nn_r(i,j) = nn_i(i,j)+1; // 1-based indexing
//                 }
//             }
//
//             ret.attr("nn") = nn_r;
//
//             if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST.\n");
//             Cmst_euclid<T>(_X2, n, d, mst_d.data(), mst_i.data(), d_core.data(), verbose);
//             if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");
//         }
//
//         for (Py_ssize_t i=0; i<n-1; ++i) {
//             GENIECLUST_ASSERT(mst_i(i,0) < mst_i(i,1));
//             GENIECLUST_ASSERT(std::isfinite(mst_d[i]));
//             ret(i,0) = mst_i(i,0)+1;  // R-based indexing
//             ret(i,1) = mst_i(i,1)+1;  // R-based indexing
//             ret(i,2) = mst_d[i];
//         }
//
//         return ret;
//     }
// #endif
//
//
//     CDistance<T>* D = NULL;
//     if (distance == "euclidean" || distance == "l2")
//        D = (CDistance<T>*)(new CDistanceEuclideanSquared<T>(X2.data(), n, d));
//     else if (distance == "manhattan" || distance == "cityblock" || distance == "l1")
//         D = (CDistance<T>*)(new CDistanceManhattan<T>(X2.data(), n, d));
//     else if (distance == "cosine")
//         D = (CDistance<T>*)(new CDistanceCosine<T>(X2.data(), n, d));
//     else
//         stop("given `distance` is not supported (yet)");
//
//     ret = internal_compute_mst<T>(D, n, M, verbose);
//     delete D;
//
//     if (distance == "euclidean" || distance == "l2") {
//         for (Py_ssize_t i=0; i<n-1; ++i) {
//             ret(i,2) = sqrt(ret(i,2));
//         }
//     }
//
//     return ret;
// }
//
//
//
//
//
//
// // [[Rcpp::export(".mst.default")]]
// NumericMatrix dot_mst_default(
//     NumericMatrix X,
//     String distance="euclidean",
//     int M=1,
//     bool cast_float32=true,
//     bool verbose=false)
// {
//     if (cast_float32)
//         return internal_mst_default<float >(X, distance, M, verbose);
//     else
//         return internal_mst_default<double>(X, distance, M, verbose);
// }
//
//
//
// // [[Rcpp::export(".mst.dist")]]
// NumericMatrix dot_mst_dist(
//     NumericVector d,
//     int M=1,
//     bool verbose=false)
// {
//     Py_ssize_t n = (Py_ssize_t)round((sqrt(1.0+8.0*d.size())+1.0)/2.0);
//     GENIECLUST_ASSERT(n*(n-1)/2 == d.size());
//
//     CDistancePrecomputedVector<double> D(REAL(SEXP(d)), n);
//
//     return internal_compute_mst<double>(&D, n, M, verbose);
// }




//' @title Quite Fast Euclidean Nearest Neighbours
//'
//' @description
//' If \code{Y} is \code{NULL}, then the function determines the first \code{k}
//' amongst the nearest neighbours of each point in \code{X} with respect
//' to the Euclidean distance. It is assumed that each query point is
//' not its own neighbour.
//'
//' Otherwise, for each point in \code{Y}, this function determines the \code{k}
//' nearest points thereto from \code{X}.
//'
//' @details
//' The implemented algorithms, see the \code{algorithm} parameter, assume
//' that \code{k} is rather small; say, \code{k <= 20}.
//'
//' Our implementation of K-d trees (Bentley, 1975) has been quite optimised;
//' amongst others, it has good locality of reference, features the sliding
//' midpoint (midrange) rule suggested by Maneewongvatana and Mound (1999),
//' and a node pruning strategy inspired by the discussion
//' by Sample et al. (2001).  Still, it is well-known that K-d trees
//' perform well only in spaces of low intrinsic dimensionality.  Thus,
//' due to the so-called curse of dimensionality, for high \code{d},
//' the brute-force algorithm is recommended.
//'
//' The number of threads used is controlled via the \code{OMP_NUM_THREADS}
//' environment variable or via the \code{\link{omp_set_num_threads}} function.
//' For best speed, consider building the package
//' from sources using, e.g., \code{-O3 -march=native} compiler flags.
//'
//' @references
//' J.L. Bentley, Multidimensional binary search trees used for associative
//' searching, \emph{Communications of the ACM} 18(9), 509–517, 1975,
//' \doi{10.1145/361002.361007}.
//'
//' S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
//' are fat, \emph{4th CGC Workshop on Computational Geometry}, 1999.
//'
//' N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
//' strategies in K-d Trees, \emph{5th WSES/IEEE Conf. on Circuits, Systems,
//' Communications & Computers} (CSCC'01), 2001.
//'
//'
//' @param X the "database"; a matrix of shape (n,d)
//' @param k number of nearest neighbours (should be rather small, say, <= 20)
//' @param Y the "query points"; \code{NULL} or a matrix of shape (m,d);
//'     note that setting \code{Y=X}, contrary to \code{NULL},
//'     will include the query points themselves amongst their own neighbours
//' @param algorithm
//'     K-d trees can only be used for d between 2 and 20 only;
//'     \code{"auto"} selects \code{"kd_tree"} in low-dimensional spaces
//' @param max_leaf_size maximal number of points in the K-d tree leaves;
//'        smaller leaves use more memory, yet are not necessarily faster
//' @param squared whether to return the squared Euclidean distance
//' @param verbose whether to print diagnostic messages
//'
//'
//' @return
//' A list with two elements, \code{nn.index} and \code{nn.dist}.
//'
//' \code{nn.dist} has shape (n,k) or (m,k);
//' \code{nn.dist[i,]} is sorted nondecreasingly for all \code{i}.
//' \code{nn.dist[i,j]} gives the weight of the edge \code{{i, ind[i,j]}},
//' i.e., the distance between the \code{i}-th point and its \code{j}-th NN.
//'
//' \code{nn.index} is of the same shape.
//' \code{nn.index[i,j]} is the index (between \code{1} and \code{n})
//' of the \code{j}-th nearest neighbour of \code{i}.
//'
//'
//' @examples
//' library("datasets")
//' data("iris")
//' X <- jitter(as.matrix(iris[1:2]))  # some data
//' neighbours <- knn_euclid(X, 1)  # 1-NNs of each point
//' plot(X, asp=1, las=1)
//' segments(X[,1], X[,2], X[neighbours$nn.index,1], X[neighbours$nn.index,2])
//'
//' knn_euclid(X, 5, matrix(c(6, 4), nrow=1))  # five closest points to (6, 4)
//'
//'
//' @seealso mst_euclid
//'
//' @rdname fastknn
//' @export
// [[Rcpp::export("knn_euclid")]]
List knn_euclid(
    SEXP X,
    int k=1,
    SEXP Y=R_NilValue,
    Rcpp::String algorithm="auto",
    int max_leaf_size=32,
    bool squared=false,
    bool verbose=false
) {
    using FLOAT = double;  // float is not faster..

    Rcpp::NumericMatrix _X;
    if (!Rf_isMatrix(X))  _X = Rcpp::internal::convert_using_rfunction(X, "as.matrix");
    else _X = X;

    Py_ssize_t n = (Py_ssize_t)_X.nrow();
    Py_ssize_t d = (Py_ssize_t)_X.ncol();
    Py_ssize_t m;
    bool use_kdtree;

    if (n < 1 || d <= 1) stop("X is ill-shaped");
    if (k < 1) stop("`k` must be >= 1");

    if (algorithm == "auto") {
        if (2 <= d && d <= 20)
            algorithm = "kd_tree";
        else
            algorithm = "brute";
    }

    if (algorithm == "kd_tree") {
        if (d < 2 || d > 20) stop("kd_tree can only be used for 2 <= d <= 20");
        if (max_leaf_size <= 0) stop("max_leaf_size must be positive");
        use_kdtree = true;
    }
    else if (algorithm == "brute")
        use_kdtree = false;
    else
        stop("invalid 'algorithm'");


    std::vector<FLOAT> XC(n*d);
    Py_ssize_t j = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        for (Py_ssize_t u=0; u<d; ++u)
            XC[j++] = (FLOAT)_X(i, u);  // row-major


    std::vector<FLOAT>      nn_dist;
    std::vector<Py_ssize_t> nn_ind;
    if (Rf_isNull(Y)) {
        if (k >= n) stop("too many neighbours requested");
        m = n;

        nn_dist.resize(n*k);
        nn_ind.resize(n*k);

        if (use_kdtree)
            Cknn1_euclid_kdtree(
                XC.data(), n, d, k, nn_dist.data(),
                nn_ind.data(), max_leaf_size, squared, verbose
            );
        else
            Cknn1_euclid_brute(
                XC.data(), n, d, k, nn_dist.data(),
                nn_ind.data(), squared, verbose
            );
    }
    else {
        if (k >  n) stop("too many neighbours requested");

        Rcpp::NumericMatrix _Y;
        if (!Rf_isMatrix(Y)) _Y = Rcpp::internal::convert_using_rfunction(Y, "as.matrix");
        else _Y = Y;

        m = (Py_ssize_t)_Y.nrow();
        if (_Y.ncol() != d) stop("Y's dimensionality does not match that of X");

        nn_dist.resize(m*k);
        nn_ind.resize(m*k);

        std::vector<FLOAT> YC(m*d);
        Py_ssize_t j = 0;
        for (Py_ssize_t i=0; i<m; ++i)
            for (Py_ssize_t u=0; u<d; ++u)
                YC[j++] = (FLOAT)_Y(i, u);  // row-major

        if (use_kdtree)
            Cknn2_euclid_kdtree(
                XC.data(), n, YC.data(), m, d, k, nn_dist.data(), nn_ind.data(),
                max_leaf_size, squared, verbose
            );
        else
            Cknn2_euclid_brute(
                XC.data(), n, YC.data(), m, d, k, nn_dist.data(), nn_ind.data(),
                squared, verbose
            );
    }

    Rcpp::IntegerMatrix out_ind(m, k);
    Rcpp::NumericMatrix out_dist(m, k);
    Py_ssize_t u = 0;
    for (Py_ssize_t i=0; i<m; ++i) {
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
