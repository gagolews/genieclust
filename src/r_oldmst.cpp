/*  The "old" (<= 2025) functions to compute minimum spanning trees
 *  wrt different distances (slower but generic).
 *
 *  Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>
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
#include "c_oldmst.h"
#include <cmath>

using namespace Rcpp;



template<typename T>
NumericMatrix internal_oldmst_compute(
    CDistance<T>* D, Py_ssize_t n, Py_ssize_t M, bool verbose
) {
    NumericMatrix ret(n-1, 3);

    CDistance<T>* D2 = NULL;
    if (M >= 2) {
        // TODO we need it for M==2 as well, but this data can be read from the
        // MST data below!
        if (verbose) GENIECLUST_PRINT("[genieclust] Determining the core distance.\n");

        Py_ssize_t k = M-1;
        CMatrix<Py_ssize_t> nn_i(n, k);
        CMatrix<T> nn_d(n, k);
        Cknn_from_complete(D, n, k, nn_d.data(), nn_i.data());

        IntegerMatrix out_nn_ind(n, k);
        NumericMatrix out_nn_dist(n, k);

        std::vector<T> d_core(n);
        for (Py_ssize_t i=0; i<n; ++i) {
            d_core[i] = nn_d(i, k-1); // distance to the k-th nearest neighbour
            GENIECLUST_ASSERT(std::isfinite(d_core[i]));

            for (Py_ssize_t j=0; j<k; ++j) {
                GENIECLUST_ASSERT(nn_i(i,j) != i);
                out_nn_ind(i,j)  = nn_i(i,j)+1; // 1-based indexing
                out_nn_dist(i,j) = nn_d(i,j);
            }
        }

        ret.attr("nn.index") = out_nn_ind;
        ret.attr("nn.dist")  = out_nn_dist;

        D2 = new CDistanceMutualReachability<T>(d_core.data(), n, D);
    }

    CMatrix<Py_ssize_t> mst_i(n-1, 2);
    std::vector<T>  mst_d(n-1);

    if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST.\n");
    Cmst_from_complete<T>(D2?D2:D, n, mst_d.data(), mst_i.data(), verbose);
    if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");

    if (D2) delete D2;

    for (Py_ssize_t i=0; i<n-1; ++i) {
        GENIECLUST_ASSERT(mst_i(i,0) < mst_i(i,1));
        GENIECLUST_ASSERT(std::isfinite(mst_d[i]));
        ret(i,0) = mst_i(i,0)+1;  // R-based indexing
        ret(i,1) = mst_i(i,1)+1;  // R-based indexing
        ret(i,2) = mst_d[i];
    }

    return ret;
}





template<typename T>
NumericMatrix internal_oldmst_matrix(
    NumericMatrix X,
    String distance,
    Py_ssize_t M,
    /*bool use_mlpack, */
    bool verbose
) {
    Py_ssize_t n = X.nrow();
    Py_ssize_t d = X.ncol();
    NumericMatrix ret;

    if (M < 1 || M >= n-1)
        stop("`M` must be an integer in [1, n-1)");

    CMatrix<T> X2(REAL(SEXP(X)), n, d, false); // Fortran- to C-contiguous

    T* _X2 = X2.data();
    for (Py_ssize_t i=0; i<n*d; i++) {
        if (!std::isfinite(_X2[i]))
            Rf_error("All elements in the input matrix must be finite and non-missing.");
    }


    CDistance<T>* D = NULL;
    if (distance == "euclidean" || distance == "l2")
       D = (CDistance<T>*)(new CDistanceEuclideanSquared<T>(X2.data(), n, d));
    else if (distance == "manhattan" || distance == "cityblock" || distance == "l1")
        D = (CDistance<T>*)(new CDistanceManhattan<T>(X2.data(), n, d));
    else if (distance == "cosine")
        D = (CDistance<T>*)(new CDistanceCosine<T>(X2.data(), n, d));
    else
        stop("given `distance` is not supported (yet)");

    ret = internal_oldmst_compute<T>(D, n, M, verbose);
    delete D;

    if (distance == "euclidean" || distance == "l2") {
        for (Py_ssize_t i=0; i<n-1; ++i) {
            ret(i,2) = sqrt(ret(i,2));
        }

        if (M > 1) {
            Rcpp::NumericMatrix out_nn_dist = ret.attr("nn.dist");
            for (Py_ssize_t i=0; i<n; ++i) {
                for (Py_ssize_t j=0; j<M-1; ++j) {
                    out_nn_dist(i,j) = sqrt(out_nn_dist(i,j));
                }
            }
        }
    }

    return ret;
}


// [[Rcpp::export(".oldmst.matrix")]]
NumericMatrix dot_oldmst_matrix(
    NumericMatrix X,
    String distance="euclidean",
    int M=1,
    bool cast_float32=false,
    bool verbose=false
) {
    if (cast_float32)
        return internal_oldmst_matrix<float >(X, distance, M, verbose);
    else
        return internal_oldmst_matrix<double>(X, distance, M, verbose);
}


// [[Rcpp::export(".oldmst.dist")]]
NumericMatrix dot_oldmst_dist(
    NumericVector d,
    int M=1,
    bool verbose=false
) {
    Py_ssize_t n = (Py_ssize_t)round((sqrt(1.0+8.0*d.size())+1.0)/2.0);
    GENIECLUST_ASSERT(n*(n-1)/2 == d.size());

    CDistancePrecomputedVector<double> D(REAL(SEXP(d)), n);

    return internal_oldmst_compute<double>(&D, n, M, verbose);
}
