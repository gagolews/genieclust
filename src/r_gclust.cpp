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

#include "c_matrix.h"
#include "c_genie.h"
#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;


List gclust(CDistance* D,
            size_t n,
            double gini_threshold,
            int M,
            String postprocess)
{
    if (gini_threshold < 0.0 || gini_threshold > 1.0)
        stop("`gini_threshold` must be in [0, 1]");
    if (M < 1 || M >= n-1)
        stop("`M` must be an integer in [1, n-1)");


    CDistance* D2 = NULL;
    if (M > 1) {
        // clustering w.r.t. mutual reachability distance
        stop("M > 1 is not supported yet.");
    }





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

        delete D2;
    }

    return List::create(
        _["merge"] = merge,
        _["height"] = height,
        _["order"] = order
    );
}


// [[Rcpp::export(".gclust.default")]]
List gclust_default(NumericMatrix d,
    double gini_threshold=0.3,
    int M=1,
    String postprocess="boundary",
    String distance="euclidean")
{
    CDistance* D = NULL;
    ssize_t n = d.nrow();
    ssize_t d = d.ncol();

    if (distance == "euclidean" || distance == "l2")
        D = (CDistance*)new CDistanceEuclidean(REAL(SEXP(d)), n, d);
    else if (distance == "manhattan" || distance == "cityblock" || distance == "l1")
        D = (CDistance*)new CDistanceManhattan(REAL(SEXP(d)), n, d);
    else if (distance == "cosine")
        D = (CDistance*)new CDistanceCosine(REAL(SEXP(d)), n, d);
    else
        stop("given `distance` is not supported (yet)");

    List ret = gclust(D, n, gini_threshold, M, postprocess);
    delete D;
    return ret;
}


// [[Rcpp::export(".gclust.dist")]]
List gclust_dist(NumericVector d,
    double gini_threshold=0.3,
    int M=1,
    String postprocess="boundary")
{
    size_t n = (size_t)round((sqrt(1.0+8.0*d.size())+1.0)/2.0);
    GENIECLUST_ASSERT(n*(n-1)/2 == d.size());
    CDistancePrecomputedVector D(REAL(SEXP(d)), n);

    return gclust(&D, n, gini_threshold, M, postprocess);
}
