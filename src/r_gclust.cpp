/*  The Genie++ Clustering Algorithm - R Wrapper
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
#include "c_distance.h"
#include "c_mst.h"
#include "c_genie.h"
#include "c_postprocess.h"
#include <cmath>

using namespace Rcpp;




/* This function was originally part of our `genie` package for R */
void internal_generate_merge(Py_ssize_t n, NumericMatrix links, NumericMatrix merge)
{
    std::vector<Py_ssize_t> elements(n+1, 0);
    std::vector<Py_ssize_t> parents(n+1, 0);

    Py_ssize_t clusterNumber = 1;
    for (Py_ssize_t k=0; k<n-1; ++k, ++clusterNumber) {
        Py_ssize_t i = (Py_ssize_t)links(k, 0);
        Py_ssize_t j = (Py_ssize_t)links(k, 1);
        Py_ssize_t si = elements[i];
        Py_ssize_t sj = elements[j];
        elements[i] = clusterNumber;
        elements[j] = clusterNumber;

        if (si == 0)
            merge(k, 0) = -(double)i;
        else {
            while (parents[si] != 0) {
                size_t sinew = parents[si];
                parents[si] = clusterNumber;
                si = sinew;
            }
            if (si != 0) parents[si] = clusterNumber;
            merge(k,0) = (double)si;
        }

        if (sj == 0)
            merge(k, 1) = -(double)j;
        else {
            while (parents[sj] != 0) {
                Py_ssize_t sjnew = parents[sj];
                parents[sj] = clusterNumber;
                sj = sjnew;
            }
            if (sj != 0) parents[sj] = clusterNumber;
            merge(k,1) = (double)sj;
        }

        if (merge(k, 0) < 0) {
            if (merge(k, 1) < 0 && merge(k, 0) < merge(k, 1))
                std::swap(merge(k, 0), merge(k, 1));
        }
        else {
            if (merge(k, 0) > merge(k, 1))
                std::swap(merge(k, 0), merge(k, 1));
        }
    }
}


/* Originally, this function was part of our `genie` package for R */
void internal_generate_order(Py_ssize_t n, NumericMatrix merge, NumericVector order)
{
   std::vector< std::list<double> > relord(n+1);
   Py_ssize_t clusterNumber = 1;
   for (Py_ssize_t k=0; k<n-1; ++k, ++clusterNumber) {
      double i = merge(k, 0);
      if (i < 0)
         relord[clusterNumber].push_back(-i);
      else
         relord[clusterNumber].splice(relord[clusterNumber].end(), relord[(size_t)i]);

      double j = merge(k, 1);
      if (j < 0)
         relord[clusterNumber].push_back(-j);
      else
         relord[clusterNumber].splice(relord[clusterNumber].end(), relord[(size_t)j]);
   }

   GENIECLUST_ASSERT(relord[n-1].size() == (size_t)n);
   Py_ssize_t k = 0;
   for (std::list<double>::iterator it = relord[n-1].begin();
         it != relord[n-1].end(); ++it) {
      order[k++] = (*it);
   }
}


template<typename T>
NumericMatrix internal_compute_mst(CDistance<T>* D, Py_ssize_t n, Py_ssize_t M, bool verbose)
{
    if (M < 1 || M >= n-1)
        stop("`M` must be an integer in [1, n-1)");

    NumericMatrix ret(n-1, 3);

    CDistance<T>* D2 = NULL;
    if (M >= 2) { // yep, we need it for M==2 as well
        if (verbose) GENIECLUST_PRINT("[genieclust] Determining the core distance.\n");

        Py_ssize_t k = M-1;
        CMatrix<Py_ssize_t> nn_i(n, k);
        CMatrix<T> nn_d(n, k);
        Cknn_from_complete(D, n, k, nn_d.data(), nn_i.data());

        NumericMatrix nn_r(n, k);

        std::vector<T> d_core(n);
        for (Py_ssize_t i=0; i<n; ++i) {
            d_core[i] = nn_d(i, k-1); // distance to the k-th nearest neighbour
            GENIECLUST_ASSERT(std::isfinite(d_core[i]));

            for (Py_ssize_t j=0; j<k; ++j) {
                GENIECLUST_ASSERT(nn_i(i,j) != i);
                nn_r(i,j) = nn_i(i,j)+1; // 1-based indexing
            }
        }

        ret.attr("nn") = nn_r;

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
NumericMatrix internal_mst_default(
    NumericMatrix X,
    String distance,
    Py_ssize_t M,
    /*bool use_mlpack, */
    bool verbose)
{
    Py_ssize_t n = X.nrow();
    Py_ssize_t d = X.ncol();
    NumericMatrix ret;

    CMatrix<T> X2(REAL(SEXP(X)), n, d, false); // Fortran- to C-contiguous

    T* _X2 = X2.data();
    for (Py_ssize_t i=0; i<n*d; i++) {
        if (!std::isfinite(_X2[i]))
            Rf_error("All elements in the input matrix must be finite and non-missing.");
    }


#if 1
    // Special case (most frequently used)
    if (M == 1 && (distance == "euclidean" || distance == "l2"))
    {
        NumericMatrix ret(n-1, 3);
        CMatrix<Py_ssize_t> mst_i(n-1, 2);
        std::vector<T>  mst_d(n-1);
        if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST.\n");
        Cmst_euclidean<T>(_X2, n, d, mst_d.data(), mst_i.data(), verbose);
        if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");

        for (Py_ssize_t i=0; i<n-1; ++i) {
            GENIECLUST_ASSERT(mst_i(i,0) < mst_i(i,1));
            GENIECLUST_ASSERT(std::isfinite(mst_d[i]));
            ret(i,0) = mst_i(i,0)+1;  // R-based indexing
            ret(i,1) = mst_i(i,1)+1;  // R-based indexing
            ret(i,2) = mst_d[i];
        }

        return ret;
    }
#endif


    CDistance<T>* D = NULL;
    if (distance == "euclidean" || distance == "l2")
       D = (CDistance<T>*)(new CDistanceEuclideanSquared<T>(X2.data(), n, d));
    else if (distance == "manhattan" || distance == "cityblock" || distance == "l1")
        D = (CDistance<T>*)(new CDistanceManhattan<T>(X2.data(), n, d));
    else if (distance == "cosine")
        D = (CDistance<T>*)(new CDistanceCosine<T>(X2.data(), n, d));
    else
        stop("given `distance` is not supported (yet)");

    ret = internal_compute_mst<T>(D, n, M, verbose);
    delete D;

    if (distance == "euclidean" || distance == "l2") {
        for (Py_ssize_t i=0; i<n-1; ++i) {
            ret(i,2) = sqrt(ret(i,2));
        }
    }

    return ret;
}






// [[Rcpp::export(".mst.default")]]
NumericMatrix dot_mst_default(
    NumericMatrix X,
    String distance="euclidean",
    int M=1,
    bool cast_float32=true,
    bool verbose=false)
{
    if (cast_float32)
        return internal_mst_default<float >(X, distance, M, verbose);
    else
        return internal_mst_default<double>(X, distance, M, verbose);
}



// [[Rcpp::export(".mst.dist")]]
NumericMatrix dot_mst_dist(
    NumericVector d,
    int M=1,
    bool verbose=false)
{
    Py_ssize_t n = (Py_ssize_t)round((sqrt(1.0+8.0*d.size())+1.0)/2.0);
    GENIECLUST_ASSERT(n*(n-1)/2 == d.size());

    CDistancePrecomputedVector<double> D(REAL(SEXP(d)), n);

    return internal_compute_mst<double>(&D, n, M, verbose);
}



// [[Rcpp::export(".genie")]]
IntegerVector dot_genie(
        NumericMatrix mst,
        int k,
        double gini_threshold,
        String postprocess,
        bool detect_noise,
        bool verbose)
{
    if (verbose) GENIECLUST_PRINT("[genieclust] Determining clusters.\n");

    if (gini_threshold < 0.0 || gini_threshold > 1.0)
        stop("`gini_threshold` must be in [0, 1]");

    if (postprocess == "boundary" && detect_noise && Rf_isNull(mst.attr("nn")))
        stop("`nn` attribute of the MST not set; unable to proceed with this postprocessing action");

    Py_ssize_t n = mst.nrow()+1;

    if (k < 1 || k > n) stop("invalid requested number of clusters, `k`");

    CMatrix<Py_ssize_t> mst_i(n-1, 2);
    std::vector<double>  mst_d(n-1);

    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_i(i, 0) = (Py_ssize_t)mst(i, 0)-1;  // 1-based to 0-based indices
        mst_i(i, 1) = (Py_ssize_t)mst(i, 1)-1;  // 1-based to 0-based indices
        mst_d[i] = mst(i, 2);
    }

    CGenie<double> g(mst_d.data(), mst_i.data(), n, detect_noise);
    g.compute(k, gini_threshold);


    if (verbose) GENIECLUST_PRINT("[genieclust] Postprocessing the outputs.\n");

    std::vector<Py_ssize_t> xres(n);
    Py_ssize_t k_detected = g.get_labels(k, xres.data());

    if (k_detected != k)
        Rf_warning("The number of clusters detected is different from the requested one due to the presence of noise points.");

    if (detect_noise && postprocess == "boundary") {
        NumericMatrix nn_r = mst.attr("nn");
        GENIECLUST_ASSERT(nn_r.nrow() == n);
        Py_ssize_t M = nn_r.ncol()+1;
        GENIECLUST_ASSERT(M < n);
        CMatrix<Py_ssize_t> nn_i(n, M-1);
        for (Py_ssize_t i=0; i<n; ++i) {
            for (Py_ssize_t j=0; j<M-1; ++j) {
                GENIECLUST_ASSERT(nn_r(i,j) >= 1);
                GENIECLUST_ASSERT(nn_r(i,j) <= n);
                nn_i(i,j) = (Py_ssize_t)nn_r(i,j)-1; // 0-based indexing
            }
        }

        Cmerge_boundary_points(mst_i.data(), n-1, nn_i.data(),
                               M-1, M, xres.data(), n);
    }
    else if (detect_noise && postprocess == "all") {
        Cmerge_noise_points(mst_i.data(), n-1, xres.data(), n);
    }

    IntegerVector res(n);
    for (Py_ssize_t i=0; i<n; ++i) {
        if (xres[i] < 0) res[i] = NA_INTEGER; // noise point
        else res[i] = xres[i] + 1;
    }

    if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");

    return res;
}



// [[Rcpp::export(".gclust")]]
List dot_gclust(
        NumericMatrix mst,
        double gini_threshold,
        bool verbose)
{
    if (verbose) GENIECLUST_PRINT("[genieclust] Determining clusters.\n");

    if (gini_threshold < 0.0 || gini_threshold > 1.0)
        stop("`gini_threshold` must be in [0, 1]");

    Py_ssize_t n = mst.nrow()+1;
    CMatrix<Py_ssize_t> mst_i(n-1, 2);
    std::vector<double>  mst_d(n-1);

    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_i(i, 0) = (Py_ssize_t)mst(i, 0)-1; // 1-based to 0-based indices
        mst_i(i, 1) = (Py_ssize_t)mst(i, 1)-1; // 1-based to 0-based indices
        mst_d[i] = mst(i, 2);
    }

    CGenie<double> g(mst_d.data(), mst_i.data(), n/*, noise_leaves=M>1*/);
    g.compute(1, gini_threshold);


    if (verbose) GENIECLUST_PRINT("[genieclust] Postprocessing the outputs.\n");

    std::vector<Py_ssize_t> links(n-1);
    g.get_links(links.data());



    NumericMatrix links2(n-1, 2);
    NumericVector height(n-1, NA_REAL);
    Py_ssize_t k = 0;
    for (Py_ssize_t i=0; i<n-1; ++i) {
        if (links[i] >= 0) {
            links2(k, 0) = mst_i(links[i], 0) + 1;
            links2(k, 1) = mst_i(links[i], 1) + 1;
            height(k) = mst_d[ links[i] ];
            ++k;
        }
    }
    for (; k<n-1; ++k) {
        links2(k, 0) = links2(k, 1) = NA_REAL;
    }


    NumericMatrix merge(n-1, 2);
    internal_generate_merge(n, links2, merge);

    NumericVector order(n, NA_REAL);
    internal_generate_order(n, merge, order);

    if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");

    return List::create(
        _["merge"]  = merge,
        _["height"] = height,
        _["order"]  = order
    );
}




template<typename FLOAT>
List internal_knn_sqeuclid(NumericMatrix X, Py_ssize_t k, bool verbose)
{
    Py_ssize_t n = (Py_ssize_t)X.nrow();
    Py_ssize_t d = (Py_ssize_t)X.ncol();
    if (k < 1 || k >= n)
        stop("`k` must be an integer in [1, n-1]");

    if (n < 1) stop("The matrix has no rows.");

    std::vector<FLOAT> XC(n*d);
    Py_ssize_t j = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        for (Py_ssize_t u=0; u<d; ++u)
            XC[j++] = X(i, u);  // row-major

    std::vector<Py_ssize_t> nn_ind(n*k);
    std::vector<FLOAT> nn_dist(n*k);

    Cknn_sqeuclid_brute<FLOAT>(XC.data(), n, d, k, nn_dist.data(), nn_ind.data(), verbose);

    Rcpp::IntegerMatrix out_ind(n, k);
    Rcpp::NumericMatrix out_dist(n, k);
    Py_ssize_t u = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        for (Py_ssize_t j=0; j<k; ++j) {
            out_ind(i, j)  = nn_ind[u]+1;  // R-based indexing
            out_dist(i, j) = nn_dist[u];
            u++;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("nn.index")=out_ind,
        Rcpp::Named("nn.dist")=out_dist
    );
}


// [[Rcpp::export("knn_sqeuclid")]]
List knn_sqeuclid(
    NumericMatrix X,
    int k,
    bool cast_float32=true,
    bool verbose=false)
{
    if (cast_float32)
        return internal_knn_sqeuclid<float >(X, k, verbose);
    else
        return internal_knn_sqeuclid<double>(X, k, verbose);
}
