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

#include "c_inequity.h"
#include <Rcpp.h>
#include <algorithm>



//' @title The Normalised Gini Index
//'
//' @description
//' The normalised  Gini index is given by:
//' \deqn{
//'     G(x_1,\dots,x_n) = \frac{
//'     \sum_{i=1}^{n-1} \sum_{j=i+1}^n |x_i-x_j|
//'     }{
//'     (n-1) \sum_{i=1}^n x_i
//'     }.
//' }
//'
//' Time complexity: \eqn{O(n)} for sorted (increasingly) data.
//' Otherwise, the vector will be sorted.
//' Note that for sorted data, it holds:
//' \deqn{
//'     G(x_1,\dots,x_n) = \frac{
//'     \sum_{i=1}^{n} (n-2i+1) x_{\sigma(n-i+1)}
//'     }{
//'     (n-1) \sum_{i=1}^n x_i
//'     },
//' }
//' where \eqn{\sigma} is an ordering permutation of \eqn{(x_1,\dots,x_n)}.
//'
//'
//' @param x numeric vector
//'
//' @return The value of the inequity index, a number in [0,1].
//'
//' @examples
//' gini(c(2, 2, 2, 2, 2))  # no inequality
//' gini(c(0, 0, 10, 0, 0)) # one has it all
//' gini(c(7, 0, 3, 0, 0))  # give to the poor, take away from the rich
//' gini(c(6, 0, 3, 1, 0))  # (a.k.a. Pigou-Dalton principle)
//'
//' @seealso bonferroni
//'
//' @export
// [[Rcpp::export]]
double gini(Rcpp::NumericVector x)
{
    size_t n = x.size();

    // check if sorted; if not, sort.
    for (size_t i=1; i<n; ++i) {
        if (x[i] > x[i+1]) {
            x = Rcpp::clone(x);
            std::sort(x.begin(), x.end());
            break;
        }
    }

    return Cgini_sorted(REAL(SEXP(x)), n);
}





//' @title The Normalised Bonferroni Index
//'
//' @description
//' The normalised Bonferroni index is given by:
//' \deqn{
//'     B(x_1,\dots,x_n) = \frac{
//'     \sum_{i=1}^{n}  \left( n-\sum_{j=1}^i \frac{n}{n-j+1} \right)
//'          x_{\sigma(n-i+1)}
//'     }{
//'     (n-1) \sum_{i=1}^n x_i
//' }   },
//'
//' Time complexity: \eqn{O(n)} for sorted (increasingly) data.
//' Otherwise, the vector will be sorted.
//'
//'
//' @param x numeric vector
//'
//' @return The value of the inequity index, a number in [0,1].
//'
//' @examples
//' bonferroni(c(2, 2, 2, 2, 2))  # no inequality
//' bonferroni(c(0, 0, 10, 0, 0)) # one has it all
//' bonferroni(c(7, 0, 3, 0, 0))  # give to the poor, take away from the rich
//' bonferroni(c(6, 0, 3, 1, 0))  # (a.k.a. Pigou-Dalton principle)
//'
//' @seealso gini
//'
//' @export
// [[Rcpp::export]]
double bonferroni(Rcpp::NumericVector x)
{
    size_t n = x.size();

    // check if sorted; if not, sort.
    for (size_t i=1; i<n; ++i) {
        if (x[i] > x[i+1]) {
            x = Rcpp::clone(x);
            std::sort(x.begin(), x.end());
            break;
        }
    }

    return Cbonferroni_sorted(REAL(SEXP(x)), n);
}

