/*  The Genie++ Clustering Algorithm - R Wrapper
 *
 *  Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 *  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "c_common.h"
#include "c_matrix.h"
#include "c_distance.h"
#include "c_mst.h"
#include "c_genie.h"
#include <cmath>
#include <Rcpp.h>
using namespace Rcpp;



/* This function was originally part of our `genie` package for R */
void generate_merge(ssize_t n, NumericMatrix links, NumericMatrix merge)
{
    std::vector<ssize_t> elements(n+1, 0);
    std::vector<ssize_t> parents(n+1, 0);

    ssize_t clusterNumber = 1;
    for (ssize_t k=0; k<n-1; ++k, ++clusterNumber) {
        ssize_t i = (ssize_t)links(k, 0);
        ssize_t j = (ssize_t)links(k, 1);
        ssize_t si = elements[i];
        ssize_t sj = elements[j];
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
                ssize_t sjnew = parents[sj];
                parents[sj] = clusterNumber;
                sj = sjnew;
            }
            if (sj != 0) parents[sj] = clusterNumber;
            merge(k,1) = (double)sj;
        }

        if (merge(k, 0) < 0) {
            if (merge(k, 1) < 0 && merge(k, 0) < merge(k, 1)) std::swap(merge(k, 0), merge(k, 1));
        }
        else {
            if (merge(k, 0) > merge(k, 1)) std::swap(merge(k, 0), merge(k, 1));
        }
    }
}


/* This function was originally part of our `genie` package for R */
void generate_order(ssize_t n, NumericMatrix merge, NumericVector order)
{
   std::vector< std::list<double> > relord(n+1);
   ssize_t clusterNumber = 1;
   for (ssize_t k=0; k<n-1; ++k, ++clusterNumber) {
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
   ssize_t k = 0;
   for (std::list<double>::iterator it = relord[n-1].begin();
         it != relord[n-1].end(); ++it) {
      order[k++] = (*it);
   }
}


template<typename T>
List __gclust(CDistance<T>* D,
            ssize_t n,
            double gini_threshold,
            ssize_t M,
            String postprocess,
            bool verbose)
{
    if (gini_threshold < 0.0 || gini_threshold > 1.0)
        stop("`gini_threshold` must be in [0, 1]");
    if (M < 1 || M >= n-1)
        stop("`M` must be an integer in [1, n-1)");


    CDistance<T>* D2 = NULL;
    if (M > 2) {
        // clustering w.r.t. mutual reachability distance
        // M == 2 is like the original distance, but with noise points detection
        stop("M > 2 is not supported yet.");

        // k = M-1
        // cdef np.ndarray[ssize_t,ndim=2] ind  = np.empty((n, k), dtype=np.intp)
        // cdef np.ndarray[floatT,ndim=2]  dist = np.empty((n, k),
        // Cknn_from_complete(D, n, k, &dist[0,0], &ind[0,0]) ...

        std::vector<T> d_core(n);
        // TODO
        D2 = new CDistanceMutualReachability<T>(d_core.data(), n, D);
    }

    std::vector<ssize_t> mst_i((n-1)*2);
    std::vector<T>  mst_d(n-1);
    Cmst_from_complete<T>(D2?D2:D, n, mst_d.data(), mst_i.data(), verbose);

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining clusters.\n");

    CGenie<T> g(mst_d.data(), mst_i.data(), n, /*noise_leaves*/M>1);
    g.apply_genie(1, gini_threshold);


    if (verbose) GENIECLUST_PRINT("[genieclust] Postprocessing the outputs.\n");

    std::vector<ssize_t> links(n-1);
    g.get_links(links.data());

    if (M > 1) {
        // noise points post-processing might be requested
        if (postprocess == "boundary") {

        }
        else if (postprocess == "none") {

        }
        else if (postprocess == "all") {

        }
        else
            stop("incorrect `postprocess`");

        stop("M > 1 is not supported yet.");

        if (D2) delete D2;
    }


    NumericMatrix links2(n-1, 2);
    NumericVector height(n-1, NA_REAL);
    ssize_t k = 0;
    for (ssize_t i=0; i<n-1; ++i) {
        if (links[i] >= 0) {
            links2(k, 0) = mst_i[ links[i]*2 + 0 ] + 1;
            links2(k, 1) = mst_i[ links[i]*2 + 1 ] + 1;
            height(k) = mst_d[ links[i] ];
            ++k;
        }
    }
    for (; k<n-1; ++k) {
        links2(k, 0) = links2(k, 1) = NA_REAL;
    }


    NumericMatrix merge(n-1, 2);
    generate_merge(n, links2, merge);

    NumericVector order(n, NA_REAL);
    generate_order(n, merge, order);

    return List::create(
        _["merge"]  = merge,
        _["height"] = height,
        _["order"]  = order
    );
}


template<typename T>
List __gclust_default(NumericMatrix X,
    double gini_threshold,
    int M,
    String postprocess,
    String distance,
    bool verbose)
{
    CDistance<T>* D = NULL;
    ssize_t n = X.nrow();
    ssize_t d = X.ncol();

    if (verbose) GENIECLUST_PRINT("[genieclust] Initialising data.\n");

    matrix<T> X2(REAL(SEXP(X)), n, d, false); // Fortran- to C-contiguous

    if (distance == "euclidean" || distance == "l2")
        D = (CDistance<T>*)(new CDistanceEuclideanSquared<T>(X2.data(), n, d));
    else if (distance == "manhattan" || distance == "cityblock" || distance == "l1")
        D = (CDistance<T>*)(new CDistanceManhattan<T>(X2.data(), n, d));
    else if (distance == "cosine")
        D = (CDistance<T>*)(new CDistanceCosine<T>(X2.data(), n, d));
    else
        stop("given `distance` is not supported (yet)");

    List ret = __gclust<T>(D, n, gini_threshold, M, postprocess, verbose);
    delete D;

    if (distance == "euclidean" || distance == "l2") {
        NumericVector height = ret["height"];
        for (ssize_t i=0; i<n-1; ++i)
            height[i] = sqrt(height[i]);
    }

    if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");
    return ret;
}




template List __gclust_default<float>(NumericMatrix X, double gini_threshold,
    int M, String postprocess, String distance, bool verbose);

template List __gclust_default<double>(NumericMatrix X, double gini_threshold,
    int M, String postprocess, String distance, bool verbose);






// [[Rcpp::export(".gclust.default")]]
List gclust_default(NumericMatrix X,
    double gini_threshold=0.3,
    int M=1,
    String postprocess="boundary",
    String distance="euclidean",
    bool cast_float32=true,
    bool verbose=false)
{
    if (cast_float32)
        return __gclust_default<float>(X, gini_threshold, M, postprocess, distance, verbose);
    else
        return __gclust_default<double>(X, gini_threshold, M, postprocess, distance, verbose);
}



// [[Rcpp::export(".gclust.dist")]]
List gclust_dist(NumericVector d,
    double gini_threshold=0.3,
    int M=1,
    String postprocess="boundary",
    bool verbose=false)
{
    ssize_t n = (ssize_t)round((sqrt(1.0+8.0*d.size())+1.0)/2.0);
    GENIECLUST_ASSERT(n*(n-1)/2 == d.size());

    if (verbose) GENIECLUST_PRINT("[genieclust] Initialising data.\n");

    List ret;
    CDistancePrecomputedVector<double> D(REAL(SEXP(d)), n);
    ret = __gclust<double>((CDistance<double>*)&D, n, gini_threshold, M, postprocess, verbose);

    if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");
    return ret;
}

