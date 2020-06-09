/*  Economic Inequity (Inequality) Measures.
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

#include "c_compare_partitions.h"
#include <Rcpp.h>
#include <algorithm>
#include <vector>
using namespace Rcpp;



/**
 *  @param x vector or contingency table (matrix)
 *  @param y R_NilValue or vector of size x.size() if x is not a matrix
 *  @param xc [out]
 *  @param yc [out]
 *
 *  @return flat, contiguous c_style vector representing the contingency table
 *   with xc rows and yc columns
 */
std::vector<int> __get_contingency_matrix(RObject x, RObject y,
                                          ssize_t* xc, ssize_t* yc)
{
    if (Rf_isMatrix(x)) {
        if (!Rf_isNull(y))
            stop("if x is a contingency matrix, y must be NULL");

        IntegerMatrix _C(x);
        *xc = _C.nrow();
        *yc = _C.ncol();
        std::vector<int> C((*xc)*(*yc));
        ssize_t k=0;
        for (ssize_t i=0; i<*xc; ++i)
            for (ssize_t j=0; j<*yc; ++j)
                C[k++] = _C(i, j); // Fortran -> C-style
        return C;
    }
    else {
        if (Rf_isNull(y))
            stop("if x is not a contingency matrix, y must not be NULL");

        IntegerVector _x(x);
        IntegerVector _y(y);

        ssize_t n = _x.size();
        if (_y.size() != n)
            stop("x and y must be of equal lengths");

        int xmin, xmax;
        Cminmax(INTEGER(SEXP(_x)), n, &xmin, &xmax);
        *xc = (xmax-xmin+1);

        int ymin, ymax;
        Cminmax(INTEGER(SEXP(_y)), n, &ymin, &ymax);
        *yc = (ymax-ymin+1);

        std::vector<int> C((*xc)*(*yc));
        Ccontingency_table(C.data(), *xc, *yc,
            xmin, ymin, INTEGER(SEXP(_x)), INTEGER(SEXP(_y)), n);
        return C;
    }
}





//' adjusted_rand_index(x, y), adjusted_rand_index(table(x, y))
//' @export
//[[Rcpp::export]]
double adjusted_rand_index(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        __get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_pairs(C.data(), xc, yc).ar;
}
