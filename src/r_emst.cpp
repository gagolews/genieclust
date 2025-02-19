/*  Calls RcppMLPACK::DualTreeBoruvka::ComputeMST
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


#include <mlpack.h>
#include <mlpack/methods/emst/dtb.hpp>
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "c_common.h"


// [[Rcpp::export(".emst_mlpack")]]
Rcpp::NumericMatrix dot_emst_mlpack(
    Rcpp::NumericMatrix X,
    int leaf_size=1,
    bool verbose=false)
{
    // NOTE bool cast_float32=true, - can't pass arma::Mat<float> to DTB
    // TODO other Distances are available

    if (leaf_size < 1)
        Rcpp::stop("`leaf_size` must be >= 1.");

    mlpack::Log::Info.ignoreInput = !verbose;

    Py_ssize_t n = X.nrow();
    Py_ssize_t d = X.ncol();


    // Let aX = transpose(X)
    arma::Mat<double> aX(d, n);
    const double* _X = REAL(X);
    for (Py_ssize_t j=0; j<d; ++j)
        for (Py_ssize_t i=0; i<n; ++i)
            aX(j, i) = *(_X++);

//     arma::Mat<double> aret;
//
//     mlpack::DualTreeBoruvka<>(aX).ComputeMST(aret);
//
//     Rcpp::NumericMatrix ret(n-1, 3);
//     for (Py_ssize_t i=0; i<n-1; ++i) {
//         ret(i, 0) = aret(0, i)+1;  // lesser edge index (0-based -> 1-based)
//         ret(i, 1) = aret(1, i)+1;  // greater edge index
//         ret(i, 2) = aret(2, i);    // distance between the pair of points
//         GENIECLUST_ASSERT(ret(i, 0) < ret(i, 1));
//         GENIECLUST_ASSERT(i == 0 || ret(i-1, 2) <= ret(i, 2));
//     }
    //return ret;

    // based on     mlpack/src/mlpack/methods/emst/emst_main.cpp


    mlpack::Log::Info << "Building the K-d tree." << std::endl;
    std::vector<size_t> idx_map;
    mlpack::KDTree<mlpack::EuclideanDistance, mlpack::DTBStat, arma::mat> tree(aX, idx_map, leaf_size);
    mlpack::LMetric<2, true> metric;

    mlpack::Log::Info << "Calculating the minimum spanning tree." << std::endl;
    arma::mat aret;
    mlpack::DualTreeBoruvka<>(&tree, metric).ComputeMST(aret);

    // Unmap the results.
    Rcpp::NumericMatrix ret(n-1, 3);
    for (Py_ssize_t i = 0; i < n-1; ++i) {
        size_t i0 = idx_map[size_t(aret(0, i))];
        size_t i1 = idx_map[size_t(aret(1, i))];

        if (i0 < i1) {
            ret(i, 0) = (double)i0+1.0;  // lesser edge index (0-based -> 1-based)
            ret(i, 1) = (double)i1+1.0;  // greater edge index
        }
        else {
            ret(i, 0) = (double)i1+1.0;  // lesser edge index (0-based -> 1-based)
            ret(i, 1) = (double)i0+1.0;  // greater edge index
        }

        ret(i, 2) = aret(2, i);

        GENIECLUST_ASSERT(ret(i, 0) < ret(i, 1));
        GENIECLUST_ASSERT(i == 0 || ret(i-1, 2) <= ret(i, 2));
    }
    return ret;
}
