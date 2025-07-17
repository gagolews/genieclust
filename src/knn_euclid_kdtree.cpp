/*  This file is part of the 'quitefastmst' package.
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


#include "c_fastmst.h"
#include "c_common.h"
#include <cmath>
#include "c_kdtree.h"



/**
 * helper function called by Cknn2_euclid_kdtree below
 */
template <class FLOAT, Py_ssize_t D>
void _knn_sqeuclid_kdtree(
    FLOAT* X, const size_t n,
    const FLOAT* Y, const Py_ssize_t m,
    const size_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, size_t max_leaf_size, bool /*verbose=false*/)
{
    using DISTANCE=quitefastkdtree::kdtree_distance_sqeuclid<FLOAT,D>;

    quitefastkdtree::kdtree<FLOAT, D, DISTANCE> tree(X, n, max_leaf_size);
    if (!Y)
        quitefastkdtree::kneighbours<FLOAT, D>(tree, nn_dist, nn_ind, k);
    else
        quitefastkdtree::kneighbours<FLOAT, D>(tree, Y, m, nn_dist, nn_ind, k);
}




/*! Get the k nearest neighbours of each point w.r.t. the Euclidean distance,
 *  using a K-d tree to speed up the computations.
 *
 *  The implemented algorithm assumes that `k` is rather small; say, `k <= 20`.
 *
 *  Our implementation of K-d trees [1]_ has been quite optimised; amongst
 *  others, it has good locality of reference, features the sliding midpoint
 *  (midrange) rule suggested in [2]_, and a node pruning strategy inspired
 *  by the discussion in [3]_.  Still, it is well-known that K-d trees
 *  perform well only in spaces of low intrinsic dimensionality.  Thus,
 *  due to the so-called curse of dimensionality, for high `d`, a brute-force
 *  algorithm is recommended.
 *
 *  [1] J.L. Bentley, Multidimensional binary search trees used for associative
 *  searching, Communications of the ACM 18(9), 509–517, 1975,
 *  DOI:10.1145/361002.361007.
 *
 *  [2] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
 *  are fat, 4th CGC Workshop on Computational Geometry, 1999.
 *
 *  [3] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
 *  strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems,
 *  Communications & Computers (CSCC'01), 2001.
 *
 *
 *
 * @param X [destroyable] data: a C-contiguous data matrix [destroyable]
 * @param n number of rows in X
 * @param Y query points: a C-contiguous data matrix [destroyable]
 * @param m number of rows in Y
 * @param d number of columns in X and in Y
 * @param k number of nearest neighbours to look for
 * @param nn_dist [out] vector(matrix) of length n*k in Y is NULL or m*k otherwise; distances to NNs
 * @param nn_ind [out] vector(matrix) of the same length as nn_ind; indexes of NNs
 * @param max_leaf_size maximal number of points in the K-d tree's leaves
 * @param squared return the squared Euclidean distance?
 * @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn2_euclid_kdtree(
    FLOAT* X, const Py_ssize_t n,
    const FLOAT* Y, const Py_ssize_t m,
    const Py_ssize_t d, const Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size, bool squared, bool verbose
) {
    Py_ssize_t nknn;
    if (n <= 0)     throw std::domain_error("n <= 0");
    if (k <= 0)     throw std::domain_error("k <= 0");
    if (!Y) {
        if (k >= n) throw std::domain_error("k >= n");
        nknn = n;
    }
    else {
        if (m <= 0) throw std::domain_error("m <= 0");
        if (k > n)  throw std::domain_error("k > n");
        nknn = m;
    }

    if (max_leaf_size <= 0) throw std::domain_error("max_leaf_size <= 0");

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining the nearest neighbours... ");

    #define ARGS_knn_sqeuclid_kdtree X, n, Y, m, k, nn_dist, nn_ind, max_leaf_size, verbose
    /* LMAO; templates... */
    /**/ if (d ==  2) _knn_sqeuclid_kdtree<FLOAT,  2>(ARGS_knn_sqeuclid_kdtree);
    else if (d ==  3) _knn_sqeuclid_kdtree<FLOAT,  3>(ARGS_knn_sqeuclid_kdtree);
    else if (d ==  4) _knn_sqeuclid_kdtree<FLOAT,  4>(ARGS_knn_sqeuclid_kdtree);
    else if (d ==  5) _knn_sqeuclid_kdtree<FLOAT,  5>(ARGS_knn_sqeuclid_kdtree);
    else if (d ==  6) _knn_sqeuclid_kdtree<FLOAT,  6>(ARGS_knn_sqeuclid_kdtree);
    else if (d ==  7) _knn_sqeuclid_kdtree<FLOAT,  7>(ARGS_knn_sqeuclid_kdtree);
    else if (d ==  8) _knn_sqeuclid_kdtree<FLOAT,  8>(ARGS_knn_sqeuclid_kdtree);
    else if (d ==  9) _knn_sqeuclid_kdtree<FLOAT,  9>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 10) _knn_sqeuclid_kdtree<FLOAT, 10>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 11) _knn_sqeuclid_kdtree<FLOAT, 11>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 12) _knn_sqeuclid_kdtree<FLOAT, 12>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 13) _knn_sqeuclid_kdtree<FLOAT, 13>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 14) _knn_sqeuclid_kdtree<FLOAT, 14>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 15) _knn_sqeuclid_kdtree<FLOAT, 15>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 16) _knn_sqeuclid_kdtree<FLOAT, 16>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 17) _knn_sqeuclid_kdtree<FLOAT, 17>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 18) _knn_sqeuclid_kdtree<FLOAT, 18>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 19) _knn_sqeuclid_kdtree<FLOAT, 19>(ARGS_knn_sqeuclid_kdtree);
    else if (d == 20) _knn_sqeuclid_kdtree<FLOAT, 20>(ARGS_knn_sqeuclid_kdtree);
    else {
        throw std::runtime_error("d should be between 2 and 20");
    }

    if (!squared) {
        for (Py_ssize_t i=0; i<nknn*k; ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}




/*! Get the k nearest neighbours of each point w.r.t. the Euclidean distance,
 * using a K-d tree to speed up the computations.
 *
 * Fast for small d, small k, but large n
 *
 * It is assumed that each point is not its own nearest neighbour.
 *
 * For more details, see the man page of Cknn2_euclid_kdtree.
 *
 * @param X [destroyable] a C-contiguous data matrix [destroyable]
 * @param n number of rows in X
 * @param d number of columns in X
 * @param k number of nearest neighbours to look for
 * @param nn_dist [out] vector(matrix) of length n*k; distances to NNs
 * @param nn_ind [out] vector(matrix) of the same length as nn_ind; indexes of NNs
 * @param max_leaf_size maximal number of points in the K-d tree's leaves
 * @param squared return the squared Euclidean distance?
 * @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn1_euclid_kdtree(
    FLOAT* X, const Py_ssize_t n,
    const Py_ssize_t d, const Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size, bool squared, bool verbose
) {
    Cknn2_euclid_kdtree(
        X, n, (const FLOAT*)nullptr, -1, d, k, nn_dist, nn_ind,
        max_leaf_size, squared, verbose
    );
}


// instantiate:
template void Cknn2_euclid_kdtree<float>(
    float* X, const Py_ssize_t n,
    const float* Y, const Py_ssize_t m,
    const Py_ssize_t d, const Py_ssize_t k,
    float* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size=32, bool squared, bool verbose
);

template void Cknn2_euclid_kdtree<double>(
    double* X, const Py_ssize_t n,
    const double* Y, const Py_ssize_t m,
    const Py_ssize_t d, const Py_ssize_t k,
    double* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size=32, bool squared, bool verbose
);


template void Cknn1_euclid_kdtree<float>(
    float* X, const Py_ssize_t n,
    const Py_ssize_t d, const Py_ssize_t k,
    float* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size=32, bool squared, bool verbose
);


template void Cknn1_euclid_kdtree<double>(
    double* X, const Py_ssize_t n,
    const Py_ssize_t d, const Py_ssize_t k,
    double* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size=32, bool squared, bool verbose
);
