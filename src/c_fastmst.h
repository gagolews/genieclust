/*  Minimum Spanning Tree and k-nearest neighbour algorithms
 *  (the "new" (2025) interface)
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


#ifndef __c_fastmst_h
#define __c_fastmst_h

#include "c_common.h"
#include "c_mst_triple.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include "c_disjoint_sets.h"
#include "c_kdtree.h"
#include "c_dtb.h"
// #include "c_picotree.h"


#ifdef _OPENMP
void Comp_set_num_threads(Py_ssize_t n_threads) {
    if (n_threads <= 0)
        n_threads = omp_get_max_threads();
    omp_set_num_threads(n_threads);
}
#else
void Comp_set_num_threads(Py_ssize_t /*n_threads*/) {
    ;
}
#endif






/*! Determine the k nearest neighbours of each point
 *  wrt the squared Euclidean distance
 *
 *  Exactly n*(n-1)/2 distance computations are performed.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 *  Worst-case time complexity: O(n*(n-1)/2*d*k).
 *  So, use for small ks like k<=20.
 *
 *
 *  @param X the n input points in R^d; a c_contiguous array, shape (n,d)
 *  @param n number of points
 *  @param d number of features
 *  @param k number of nearest neighbours requested
 *  @param nn_dist [out]  a c_contiguous array, shape (n,k),
 *         dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 *  @param nn_ind [out]   a c_contiguous array, shape (n,k),
 *         (undirected) edge definition, interpreted as {i, ind[i,j]}
 *  @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn_sqeuclid_brute(
    const FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, bool verbose=false
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining the nearest neighbours... ");

    for (Py_ssize_t i=0; i<n*k; ++i) {
        nn_dist[i] = INFINITY;
        nn_ind[i] = -1;
    }

    std::vector<FLOAT> dij(n);
    for (Py_ssize_t i=0; i<n-1; ++i) {
        const FLOAT* x_cur = X+i*d;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=i+1; j<n; ++j) {
            FLOAT dd = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                dd += square(x_cur[u]-X[j*d+u]);
            dij[j] = dd;

            if (dd < nn_dist[j*k+k-1]) {
                // i might be amongst k-NNs of j;
                // insert into an ordered sequence
                Py_ssize_t l = k-1;
                while (l > 0 && dd < nn_dist[j*k+l-1]) {
                    nn_dist[j*k+l] = nn_dist[j*k+l-1];
                    nn_ind[j*k+l]  = nn_ind[j*k+l-1];
                    l -= 1;
                }
                nn_dist[j*k+l] = dd;
                nn_ind[j*k+l]  = i;
            }
        }


        // This part can't be parallelised
        for (Py_ssize_t j=i+1; j<n; ++j) {
            if (dij[j] < nn_dist[i*k+k-1]) {
                // j might be amongst k-NNs of i
                Py_ssize_t l = k-1;
                while (l > 0 && dij[j] < nn_dist[i*k+l-1]) {
                    nn_dist[i*k+l] = nn_dist[i*k+l-1];
                    nn_ind[i*k+l]  = nn_ind[i*k+l-1];
                    l -= 1;
                }
                nn_dist[i*k+l] = dij[j];
                nn_ind[i*k+l]  = j;
            }
        }

        // if (verbose) GENIECLUST_PRINT_int("\b\b\b\b%3d%%", (n-1+n-i-1)*(i+1)*100/n/(n-1));

        #if GENIECLUST_R
        Rcpp::checkUserInterrupt();
        #elif GENIECLUST_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}





/*! A Jarnik (Prim/Dijkstra)-like algorithm for determining
 *  a(*) Euclidean minimum spanning tree (MST) or
 *  one that corresponds to an M-mutual reachability distance.
 *
 *  Time complexity: O(n^2).
 *  It is assumed that M is rather small (say, M<=20).
 *  If M>1, all pairwise the distances are computed twice (first for
 *  the neighbours, then to determine the tree).
 *
 *  (*) Note that there might be multiple minimum trees spanning a given graph.
 *
 *
 *  References:
 *  ----------
 *
 *  V. Jarnik, O jistem problemu minimalnim,
 *  Prace Moravske Prirodovedecke Spolecnosti 6 (1930) 57-63.
 *
 *  C.F. Olson, Parallel algorithms for hierarchical clustering,
 *  Parallel Comput. 21 (1995) 1313-1325.
 *
 *  R. Prim, Shortest connection networks and some generalisations,
 *  Bell Syst. Tech. J. 36 (1957) 1389-1401.
 *
 *  R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
 *  on hierarchical density estimates, Lecture Notes in Computer Science 7819
 *  (2013) 160–172. DOI: 10.1007/978-3-642-37456-2_14.
 *
 *
 * @param X [destroyable] a C-contiguous data matrix, shape n*d
 * @param n number of rows
 * @param d number of columns
 * @param M the level of the "core" distance if M > 1
 * @param mst_dist [out] vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_ind [out] vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 * @param d_core [out] NULL for M==1; distances to the points'
 *        (M-1)-th neighbours
 * @param verbose should we output diagnostic/progress messages?
 */
template <class FLOAT>
void Cmst_euclid_brute(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind, FLOAT* d_core=nullptr,
    bool verbose=false
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (M <= 0)   throw std::domain_error("M <= 0");
    if (M-1 >= n) throw std::domain_error("M >= n-1");

    std::vector<FLOAT> nn_dist;
    std::vector<Py_ssize_t> nn_ind;
    if (M > 1) {
        // M == 2 needs d_core too
        GENIECLUST_ASSERT(d_core);
        nn_dist.resize(n*(M-1));
        nn_ind.resize(n*(M-1));
        Cknn_sqeuclid_brute(X, n, d, M-1, nn_dist.data(), nn_ind.data(), verbose);
        for (Py_ssize_t i=0; i<n; ++i) d_core[i] = nn_dist[i*(M-1)+(M-2)];
    }
    else
        GENIECLUST_ASSERT(!d_core);
    // TODO: actually, for M==2, we could compute d_core (1-nn) distance on the fly...


    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the MST... %3d%%", 0);


    // ind_nn[j] is the vertex from the current tree closest to vertex j
    std::vector<Py_ssize_t> ind_nn(n);
    std::vector<FLOAT> dist_nn(n, INFINITY);  // dist_nn[j] = d(j, ind_nn[j])

    //std::vector<FLOAT> distances(n);
    //FLOAT* _distances = distances.data();

    std::vector<Py_ssize_t> ind_left(n);  // aka perm
    for (Py_ssize_t i=0; i<n; ++i) ind_left[i] = i;

    std::vector< CMstTriple<FLOAT> > mst(n-1);

    for (Py_ssize_t i=1; i<n; ++i) {
        // i-1 is the vertex most recently added to the tree
        // i, i+1, ..., n-1 - vertices not yet in the tree

        FLOAT* x_cur = X+(i-1)*d;

        // compute the distances
        // between ind_cur=ind_left[i-1] and all j=i, i+1, ..., n-1:
#if 0
        // two-stage - slower
#else
        if (M <= 2) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (Py_ssize_t j=i; j<n; ++j) {
                FLOAT dd = 0.0;
                for (Py_ssize_t u=0; u<d; ++u)
                    dd += square(x_cur[u]-X[j*d+u]);

                if (dd < dist_nn[j]) {
                    ind_nn[j] = i-1;
                    dist_nn[j] = dd;
                }
            }
        }
        else
        {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (Py_ssize_t j=i; j<n; ++j) {
                FLOAT dd = 0.0;
                for (Py_ssize_t u=0; u<d; ++u)
                    dd += square(x_cur[u]-X[j*d+u]);
                if (dd < dist_nn[j]) {
                    // pulled-away from each other, but ordered w.r.t. the original pairwise distances (increasingly)
                    FLOAT d_core_max = std::max(d_core[i-1], d_core[j]);
                    if (dd <= d_core_max)
                        dd = d_core_max+dd/DCORE_DIST_ADJ;

                    if (dd < dist_nn[j]) {
                        ind_nn[j] = i-1;
                        dist_nn[j] = dd;
                    }
                }
            }
        }
#endif

// for we want to include the vertex that is closest to the vertices
        // of the tree constructed so far
        Py_ssize_t best_j = i;
        for (Py_ssize_t j=i+1; j<n; ++j) {
            if (dist_nn[j] < dist_nn[best_j]) {
                best_j = j;
            }
        }

        // don't visit i again
        // with swapping we get better locality of reference
        std::swap(ind_left[best_j], ind_left[i]);
        std::swap(dist_nn[best_j], dist_nn[i]);
        std::swap(ind_nn[best_j], ind_nn[i]);
        for (Py_ssize_t u=0; u<d; ++u) {
            std::swap(X[best_j*d+u], X[i*d+u]);
        }

        if (M > 2) {
            std::swap(d_core[best_j], d_core[i]);
            // recompute the distance without the ambiguity correction
            dist_nn[i] = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                dist_nn[i] += square(X[i*d+u]-X[ind_nn[i]*d+u]);
            dist_nn[i] = max3(dist_nn[i], d_core[ind_nn[i]], d_core[i]);
        }

        // connect best_ind_left with the tree: add a new edge {best_ind_left, ind_nn[best_ind_left]}
        GENIECLUST_ASSERT(ind_nn[i] < i);
        mst[i-1] = CMstTriple<FLOAT>(ind_left[ind_nn[i]], ind_left[i], dist_nn[i], /*order=*/true);


        if (verbose) GENIECLUST_PRINT_int("\b\b\b\b%3d%%", (n-1+n-i-1)*(i+1)*100/n/(n-1));

        #if GENIECLUST_R
        Rcpp::checkUserInterrupt();
        #elif GENIECLUST_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    // sort the resulting MST edges in increasing order w.r.t. d
    std::sort(mst.begin(), mst.end());

    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_dist[i]    = sqrt(mst[i].d);
        mst_ind[2*i+0] = mst[i].i1; // i1 < i2
        mst_ind[2*i+1] = mst[i].i2;
    }

    if (M > 2) {
        for (Py_ssize_t i=0; i<n; ++i)
            d_core[i] = sqrt(nn_dist[i*(M-1)+(M-2)]);
    }

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");
}









/**
 * helper function called by Cknn_sqeuclid_kdtree below
 */
template <class FLOAT, Py_ssize_t D>
void Cknn_sqeuclid_kdtree(FLOAT* X, const size_t n, const size_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, size_t max_leaf_size, bool /*verbose=false*/)
{
    using DISTANCE=mgtree::kdtree_distance_sqeuclid<FLOAT,D>;

    mgtree::kdtree<FLOAT, D, DISTANCE> tree(X, n, max_leaf_size);
    mgtree::kneighbours<FLOAT, D>(tree, nn_dist, nn_ind, k);

    // std::vector<FLOAT_INTERNAL> XC(n*D);
    // for (size_t i=0; i<n*D; ++i)
    //     XC[i] = (FLOAT_INTERNAL)X[i];

    // std::vector<FLOAT_INTERNAL>  _nn_dist(n*k);
    // std::vector<size_t> _nn_ind(n*k);
    //
    // mgtree::kdtree<FLOAT_INTERNAL, D, DISTANCE> tree(XC.data(), n, max_leaf_size);
    // mgtree::kneighbours<FLOAT_INTERNAL, D>(tree, _nn_dist.data(), _nn_ind.data(), k);
    //
    // #ifdef _OPENMP
    // #pragma omp parallel for schedule(static)
    // #endif
    // for (size_t i=0; i<n; ++i) {
    //     const FLOAT* x_cur = X+i*D;
    //     for (size_t j=0; j<k; ++j) {
    //         nn_ind[i*k+j]  = (Py_ssize_t)_nn_ind[i*k+j];
    //
    //         // recompute the distance using FLOAT's precision
    //         const FLOAT* x_other = X+nn_ind[i*k+j]*D;
    //         FLOAT _d = 0.0;
    //         for (size_t u=0; u<D; ++u) {
    //             FLOAT _df = x_cur[u]-x_other[u];
    //             _d += _df*_df;
    //         }
    //         nn_dist[i*k+j] = _d;
    //     }
    // }
}


/*! Get the k nearest neighbours of each point w.r.t. the Euclidean distance,
 * using a K-d tree to speed up the computations.
 *
 * Fast for small d, small k, but large n
 *
 * @param X [destroyable] a C-contiguous data matrix [destroyable]
 * @param n number of rows
 * @param d number of columns
 * @param k number of nearest neighbours to look for
 * @param nn_dist [out] vector(matrix) of length n*k, distances to NNs
 * @param nn_ind [out] vector(matrix) of length n*k, indexes of NNs
 * @param max_leaf_size ......TODO
 * @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn_sqeuclid_kdtree(FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, Py_ssize_t max_leaf_size=32, bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (max_leaf_size < 0) throw std::domain_error("max_leaf_size < 0");
    else if (max_leaf_size == 0) max_leaf_size = 32;  // default

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining the nearest neighbours... ");

    /* LMAO; templates... */
    /**/ if (d ==  2)  Cknn_sqeuclid_kdtree<FLOAT,  2>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  3)  Cknn_sqeuclid_kdtree<FLOAT,  3>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  4)  Cknn_sqeuclid_kdtree<FLOAT,  4>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  5)  Cknn_sqeuclid_kdtree<FLOAT,  5>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  6)  Cknn_sqeuclid_kdtree<FLOAT,  6>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  7)  Cknn_sqeuclid_kdtree<FLOAT,  7>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  8)  Cknn_sqeuclid_kdtree<FLOAT,  8>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  9)  Cknn_sqeuclid_kdtree<FLOAT,  9>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 10)  Cknn_sqeuclid_kdtree<FLOAT, 10>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 11)  Cknn_sqeuclid_kdtree<FLOAT, 11>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 12)  Cknn_sqeuclid_kdtree<FLOAT, 12>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 13)  Cknn_sqeuclid_kdtree<FLOAT, 13>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 14)  Cknn_sqeuclid_kdtree<FLOAT, 14>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 15)  Cknn_sqeuclid_kdtree<FLOAT, 15>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 16)  Cknn_sqeuclid_kdtree<FLOAT, 16>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 17)  Cknn_sqeuclid_kdtree<FLOAT, 17>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 18)  Cknn_sqeuclid_kdtree<FLOAT, 18>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 19)  Cknn_sqeuclid_kdtree<FLOAT, 19>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 20)  Cknn_sqeuclid_kdtree<FLOAT, 20>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else {
        throw std::runtime_error("d should be between 2 and 20");
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}


/**
 * helper function called by Cmst_euclid_kdtree below
 */
template <class FLOAT, Py_ssize_t D>
void Cmst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind, FLOAT* d_core,
    Py_ssize_t max_leaf_size,
    Py_ssize_t first_pass_max_brute_size,
    bool /*verbose*/
) {
    using DISTANCE=mgtree::kdtree_distance_sqeuclid<FLOAT, D>;
    mgtree::dtb<FLOAT, D, DISTANCE> tree(X, n, M, max_leaf_size, first_pass_max_brute_size);
    mgtree::mst<FLOAT, D>(tree, mst_dist, mst_ind, d_core);

    for (Py_ssize_t i=0; i<n-1; ++i)
        mst_dist[i] = sqrt(mst_dist[i]);

    if (d_core) {
        for (Py_ssize_t i=0; i<n; ++i)
            d_core[i] = sqrt(d_core[i]);
    }
}


/*! A Dual-tree Boruvka-like algorithm based on K-d trees for determining
 *  a(*) Euclidean minimum spanning tree (MST) or
 *  one that corresponds to an M-mutual reachability distance.
 *  Fast in low-dimensional spaces.
 *
 *  (*) Note that there might be multiple minimum trees spanning a given graph.
 *
 *
 *  References:
 *  ----------
 *
 *  TODO ......................
 *
 *  R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
 *  on hierarchical density estimates, Lecture Notes in Computer Science 7819
 *  (2013) 160–172. DOI: 10.1007/978-3-642-37456-2_14.
 *
 *
 * @param X [destroyable] a C-contiguous data matrix, shape n*d
 * @param n number of rows
 * @param d number of columns
 * @param M the level of the "core" distance if M > 1
 * @param mst_dist [out] vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_ind [out] vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 * @param d_core [out] NULL for M==1; distances to the points'
 *        (M-1)-th neighbours
 * @param max_leaf_size TODO ...
 * @param first_pass_max_brute_size TODO ...
 * @param verbose should we output diagnostic/progress messages?
 */
template <class FLOAT>
void Cmst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind, FLOAT* d_core=nullptr,
    Py_ssize_t max_leaf_size=0,
    Py_ssize_t first_pass_max_brute_size=0,
    bool verbose=false
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (M <= 0)   throw std::domain_error("M <= 0");
    if (M-1 >= n) throw std::domain_error("M >= n-1");

    if (max_leaf_size < 0)
        throw std::domain_error("max_leaf_size < 0");
    else if (max_leaf_size == 0)
        max_leaf_size = 4;  // default

    if (first_pass_max_brute_size < 0)
        throw std::domain_error("first_pass_max_brute_size < 0");
    else if (first_pass_max_brute_size == 0)
        first_pass_max_brute_size = 16;  // default


    if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST... ");

    /* LMAO; templates... */
    /**/ if (d ==  2)  Cmst_euclid_kdtree<FLOAT,  2>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d ==  3)  Cmst_euclid_kdtree<FLOAT,  3>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d ==  4)  Cmst_euclid_kdtree<FLOAT,  4>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d ==  5)  Cmst_euclid_kdtree<FLOAT,  5>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d ==  6)  Cmst_euclid_kdtree<FLOAT,  6>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d ==  7)  Cmst_euclid_kdtree<FLOAT,  7>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d ==  8)  Cmst_euclid_kdtree<FLOAT,  8>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d ==  9)  Cmst_euclid_kdtree<FLOAT,  9>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 10)  Cmst_euclid_kdtree<FLOAT, 10>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 11)  Cmst_euclid_kdtree<FLOAT, 11>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 12)  Cmst_euclid_kdtree<FLOAT, 12>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 13)  Cmst_euclid_kdtree<FLOAT, 13>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 14)  Cmst_euclid_kdtree<FLOAT, 14>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 15)  Cmst_euclid_kdtree<FLOAT, 15>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 16)  Cmst_euclid_kdtree<FLOAT, 16>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 17)  Cmst_euclid_kdtree<FLOAT, 17>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 18)  Cmst_euclid_kdtree<FLOAT, 18>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 19)  Cmst_euclid_kdtree<FLOAT, 19>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else if (d == 20)  Cmst_euclid_kdtree<FLOAT, 20>(X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, verbose);
    else {
        throw std::runtime_error("d should be between 2 and 20");
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}

#endif
