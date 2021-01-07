/*  Calls RcppMLPACK::DualTreeBoruvka::ComputeMST
 *
 *  Copyleft (C) 2018-2021, Marek Gagolewski <https://www.gagolewski.com>
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

#include <RcppMLPACK.h>
// #include <RcppArmadillo.h>
// #include <Rcpp.h>
#include "c_common.h"



// Euclidean MST via MLPACK
// Calls RcppMLPACK::DualTreeBoruvka::ComputeMST
// [[Rcpp::export(".emst_mlpack")]]
Rcpp::NumericMatrix dot_emst_mlpack(Rcpp::NumericMatrix X)
{
    ssize_t n = X.nrow();
    ssize_t d = X.ncol();

    // Let aX = transpose(X)
    arma::Mat<double> aX(d, n);
    for (ssize_t i=0; i<n; ++i)
        for (ssize_t j=0; j<d; ++j)
            aX(j, i) = X(i, j);

    arma::Mat<double> aret;
    mlpack::emst::DualTreeBoruvka<>(aX).ComputeMST(aret);

    Rcpp::NumericMatrix ret(n-1, 3);
    for (ssize_t i=0; i<n-1; ++i) {
        ret(i, 0) = aret(0, i)+1; // lesser edge index (0-based -> 1-based)
        ret(i, 1) = aret(1, i)+1; // greater edge index
        ret(i, 2) = aret(2, i);   // distance between the pair of points
        GENIECLUST_ASSERT(ret(i, 0) < ret(i, 1));
        GENIECLUST_ASSERT(i == 0 || ret(i-1, 2) <= ret(i, 2));
    }

    return ret;
}
