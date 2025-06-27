/*  Minimum spanning tree and k-nearest neighbour algorithms
 *  (the "new">=2025 interface, quite fast, currently Euclidean distance
 *  only)
 *
 *
 *  [1] V. Jarník, O jistém problému minimálním,
 *  Práce Moravské Přírodovědecké Společnosti 6, 1930, 57–63.
 *
 *  [2] C.F. Olson, Parallel algorithms for hierarchical clustering,
 *  Parallel Comput. 21, 1995, 1313–1325.
 *
 *  [3] R. Prim, Shortest connection networks and some generalizations,
 *  Bell Syst. Tech. J. 36, 1957, 1389–1401.
 *
 *  [4] O. Borůvka, O jistém problému minimálním. Práce Mor. Přírodověd. Spol.
 *  V Brně III 3, 1926, 37–58.
 *
 *  [5] W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
 *  tree: algorithm, analysis, and applications, Proc. 16th ACM SIGKDD Intl.
 *  Conf. Knowledge Discovery and Data Mining (KDD '10), 2010, 603–612.
 *
 *  [6] J.L. Bentley, Multidimensional binary search trees used for associative
 *  searching, Communications of the ACM 18(9), 509–517, 1975,
 *  DOI:10.1145/361002.361007.
 *
 *  [7] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
 *  are fat, The 4th CGC Workshop on Computational Geometry, 1999.
 *
 *  [8] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
 *  strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems,
 *  Communications & Computers (CSCC'01), 2001.
 *
 *  [9] R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
 *  on hierarchical density estimates, Lecture Notes in Computer Science 7819,
 *  2013, 160–172. DOI: 10.1007/978-3-642-37456-2_14.
 *
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
#include "c_kdtree_boruvka.h"
// #include "c_picotree.h"


#define MST_OMP_CHUNK_SIZE 1024



/*! Order the n-1 edges of a spanning tree of n points in place,
 * wrt the weights increasingly, resolving ties if needed based on
 * the points' IDs.
 *
 * @param n
 * @param mst_dist [in/out] size m
 * @param mst_ind [in/out] size m*2
 */
template <class FLOAT>
void Ctree_order(Py_ssize_t m, FLOAT* tree_dist, Py_ssize_t* tree_ind)
{
    GENIECLUST_PROFILER_USE

    std::vector< CMstTriple<FLOAT> > mst(m);

    for (Py_ssize_t i=0; i<m; ++i) {
        mst[i] = CMstTriple<FLOAT>(tree_ind[2*i+0], tree_ind[2*i+1], tree_dist[i]);
    }


    GENIECLUST_PROFILER_START
    std::sort(mst.begin(), mst.end());
    GENIECLUST_PROFILER_STOP("mst sort");

    for (Py_ssize_t i=0; i<m; ++i) {
        tree_dist[i]    = mst[i].d;
        tree_ind[2*i+0] = mst[i].i1;  // i1 < i2
        tree_ind[2*i+1] = mst[i].i2;
    }
}


/*! Determine the k nearest neighbours of each point
 *  wrt the Euclidean distance
 *
 *  Exactly n*(n-1)/2 distance computations are performed.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 *  Worst-case time complexity: O(n*(n-1)/2*d*k).
 *  So, use for small k, say, k<=20.
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
 *  @param squared return the squared Euclidean distance?
 *  @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn1_euclid_brute(
    const FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, bool squared=false, bool verbose=false
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining the nearest neighbours... ");

    for (Py_ssize_t i=0; i<n*k; ++i) nn_dist[i] = INFINITY;
    for (Py_ssize_t i=0; i<n*k; ++i) nn_ind[i] = -1;

    std::vector<FLOAT> dij(n);
    for (Py_ssize_t i=0; i<n-1; ++i) {
        const FLOAT* x_cur = X+i*d;

        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static,MST_OMP_CHUNK_SIZE)  /* chunks get smaller and smaller... */
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

        // This part can't be (naively) parallelised
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

        // if (verbose) GENIECLUST_PRINT("\b\b\b\b%3d%%", (n-1+n-i-1)*(i+1)*100/n/(n-1));

        if (i % MST_OMP_CHUNK_SIZE == MST_OMP_CHUNK_SIZE-1) {
            #if GENIECLUST_R
            Rcpp::checkUserInterrupt();  // throws an exception, not a longjmp
            #elif GENIECLUST_PYTHON
            if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
            #endif
        }
    }

    if (!squared) {
        for (Py_ssize_t i=0; i<k*n; ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}


/*! Determine the k nearest neighbours of each point
 *  wrt the Euclidean distance
 *
 *  Use for small k, say, k<=20.
 *
 *
 *  @param X the n input points in R^d; a c_contiguous array, shape (n,d)
 *  @param n number of points
 *  @param Y the m query points in R^d; a c_contiguous array, shape (m,d)
 *  @param m number of points
 *  @param d number of features
 *  @param k number of nearest neighbours requested
 *  @param nn_dist [out]  a c_contiguous array, shape (m,k),
 *         dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 *  @param nn_ind [out]   a c_contiguous array, shape (m,k),
 *         (undirected) edge definition, interpreted as {i, ind[i,j]}
 *  @param squared return the squared Euclidean distance?
 *  @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn2_euclid_brute(
    const FLOAT* X, Py_ssize_t n, const FLOAT* Y, Py_ssize_t m,
    Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, bool squared=false, bool verbose=false
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (m <= 0)   throw std::domain_error("m <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >  n)   throw std::domain_error("k > n");

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining the nearest neighbours... ");

    for (Py_ssize_t i=0; i<m*k; ++i) nn_dist[i] = INFINITY;
    for (Py_ssize_t i=0; i<m*k; ++i) nn_ind[i] = -1;

    #if OPENMP_IS_ENABLED
    #pragma omp parallel for schedule(static)
    #endif
    for (Py_ssize_t i=0; i<m; ++i) {
        const FLOAT* y_cur = Y+i*d;

        const FLOAT* x_cur = X;
        for (Py_ssize_t j=0; j<n; ++j) {
            FLOAT dd = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                dd += square(y_cur[u]-x_cur[u]);
            x_cur += d;

            if (dd < nn_dist[i*k+k-1]) {
                Py_ssize_t l = k-1;
                while (l > 0 && dd < nn_dist[i*k+l-1]) {
                    nn_dist[i*k+l] = nn_dist[i*k+l-1];
                    nn_ind[i*k+l]  = nn_ind[i*k+l-1];
                    l -= 1;
                }
                nn_dist[i*k+l] = dd;
                nn_ind[i*k+l]  = j;
            }
        }
    }

    if (!squared) {
        for (Py_ssize_t i=0; i<m*k; ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}


/*! A Jarnik (Prim/Dijkstra)-like algorithm for determining
 *  a(*) Euclidean minimum spanning tree (MST) or
 *  one that corresponds to an M-mutual reachability distance.
 *
 *  Time complexity: O(n^2). It is assumed that M is rather small
 *  (say, M<=20). If M>1, all pairwise the distances are computed twice
 *  (first for the neighbours/core distance, then to determine the tree).
 *
 *  (*) Note that there might be multiple minimum trees spanning a given graph.
 *
 *
 *  References:
 *  ----------
 *
 *  [1] V. Jarník, O jistém problému minimálním,
 *  Práce Moravské Přírodovědecké Společnosti 6, 1930, 57–63.
 *
 *  [2] C.F. Olson, Parallel algorithms for hierarchical clustering,
 *  Parallel Comput. 21, 1995, 1313–1325.
 *
 *  [3] R. Prim, Shortest connection networks and some generalizations,
 *  Bell Syst. Tech. J. 36, 1957, 1389–1401.
 *
 *  [9] R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
 *  on hierarchical density estimates, Lecture Notes in Computer Science 7819,
 *  2013, 160–172. DOI: 10.1007/978-3-642-37456-2_14.
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
    if (M > 2) {
        GENIECLUST_ASSERT(d_core);
        nn_dist.resize(n*(M-1));
        nn_ind.resize(n*(M-1));
        Cknn1_euclid_brute(X, n, d, M-1, nn_dist.data(), nn_ind.data(),
                           /*squared=*/true, verbose);
        for (Py_ssize_t i=0; i<n; ++i) d_core[i] = nn_dist[i*(M-1)+(M-2)];

        // for M==2, we can fetch d_core from MST, as nearest neighbours
        // are connected by an edge (see below)
    }

    if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST... %3d%%", 0);


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
        // between the (i-1)-th vertex and all j=i, i+1, ..., n-1:
#if 0
        // NOTE two-stage Euclidean distance computation: slower -> removed
#else
        if (M <= 2) {
            #if OPENMP_IS_ENABLED
            #pragma omp parallel for schedule(static,MST_OMP_CHUNK_SIZE)  /* chunks get smaller and smaller... */
            #endif
            for (Py_ssize_t j=i; j<n; ++j) {
                FLOAT dd = 0.0;
                for (Py_ssize_t u=0; u<d; ++u)
                    dd += square(x_cur[u]-X[j*d+u]);

                if (dd < dist_nn[j]) {
                    dist_nn[j] = dd;
                    ind_nn[j] = i-1;
                }
            }
        }
        else
        {
            #if OPENMP_IS_ENABLED
            #pragma omp parallel for schedule(static,MST_OMP_CHUNK_SIZE)
            #endif
            for (Py_ssize_t j=i; j<n; ++j) {
                FLOAT dd = 0.0;
                for (Py_ssize_t u=0; u<d; ++u)
                    dd += square(x_cur[u]-X[j*d+u]);
                if (dd < dist_nn[j]) {  // otherwise why bother
                    // pulled-away from each other, but ordered w.r.t. the original pairwise distances (increasingly)
                    FLOAT d_core_max = std::max(d_core[i-1], d_core[j]);
                    if (dd <= d_core_max)
                        dd = d_core_max + dd/DCORE_DIST_ADJ;
                    else
                        dd = dd + dd/DCORE_DIST_ADJ;

                    if (dd < dist_nn[j]) {
                        dist_nn[j] = dd;
                        ind_nn[j] = i-1;
                    }
                }
            }
        }
#endif

        // we want to include the vertex that is closest to the vertices
        // of the tree constructed so far
        Py_ssize_t best_j = i;
        for (Py_ssize_t j=i+1; j<n; ++j)
            if (dist_nn[j] < dist_nn[best_j])
                best_j = j;


        // with swapping we get better locality of reference
        std::swap(ind_left[best_j], ind_left[i]);
        std::swap(dist_nn[best_j], dist_nn[i]);
        std::swap(ind_nn[best_j], ind_nn[i]);
        for (Py_ssize_t u=0; u<d; ++u) std::swap(X[best_j*d+u], X[i*d+u]);


        if (M > 2) {
            std::swap(d_core[best_j], d_core[i]);
            // recompute the distance without the ambiguity correction
            dist_nn[i] = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                dist_nn[i] += square(X[i*d+u]-X[ind_nn[i]*d+u]);
            dist_nn[i] = max3(dist_nn[i], d_core[ind_nn[i]], d_core[i]);
        }

        // don't visit i again - it's being added to the tree

        // connect best_ind_left with the tree: add a new edge {best_ind_left, ind_nn[best_ind_left]}
        GENIECLUST_ASSERT(ind_nn[i] < i);
        mst[i-1] = CMstTriple<FLOAT>(ind_left[ind_nn[i]], ind_left[i], dist_nn[i], /*order=*/true);


        if (verbose) GENIECLUST_PRINT("\b\b\b\b%3d%%", (int)((n-1+n-i-1)*(i+1)*100/n/(n-1)));

        if (i % MST_OMP_CHUNK_SIZE == MST_OMP_CHUNK_SIZE-1) {
            #if GENIECLUST_R
            Rcpp::checkUserInterrupt();  // throws an exception, not a longjmp
            #elif GENIECLUST_PYTHON
            if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
            #endif
        }
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
    else if (M == 2) {
        // for M==2 we just need the nearest neighbours, and the MST connects
        // them with each other
        for (Py_ssize_t i=0; i<n; ++i)
            d_core[i] = INFINITY;

        for (Py_ssize_t i=0; i<n-1; ++i) {
            if (d_core[mst_ind[2*i+0]] > mst_dist[i])
                d_core[mst_ind[2*i+0]] = mst_dist[i];
            if (d_core[mst_ind[2*i+1]] > mst_dist[i])
                d_core[mst_ind[2*i+1]] = mst_dist[i];
        }

    }

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");
}





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
    using DISTANCE=mgtree::kdtree_distance_sqeuclid<FLOAT,D>;

    mgtree::kdtree<FLOAT, D, DISTANCE> tree(X, n, max_leaf_size);
    if (!Y)
        mgtree::kneighbours<FLOAT, D>(tree, nn_dist, nn_ind, k);
    else
        mgtree::kneighbours<FLOAT, D>(tree, Y, m, nn_dist, nn_ind, k);
}


/*! Get the k nearest neighbours of each point w.r.t. the Euclidean distance,
 * using a K-d tree to speed up the computations.
 *
 * Fast for small d, small k, but large n
 *
 *
 *  [6] J.L. Bentley, Multidimensional binary search trees used for associative
 *  searching, Communications of the ACM 18(9), 509–517, 1975,
 *  DOI:10.1145/361002.361007.
 *
 *  [7] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
 *  are fat, The 4th CGC Workshop on Computational Geometry, 1999.
 *
 *  [8] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
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
    Py_ssize_t max_leaf_size=32, bool squared=false, bool verbose=false
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
    Py_ssize_t max_leaf_size=32, bool squared=false, bool verbose=false
) {
    Cknn2_euclid_kdtree(
        X, n, (const FLOAT*)nullptr, -1, d, k, nn_dist, nn_ind,
        max_leaf_size, squared, verbose
    );
}


/**
 * helper function called by Cmst_euclid_kdtree below
 */
template <class FLOAT, Py_ssize_t D>
void _mst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind, FLOAT* d_core,
    Py_ssize_t max_leaf_size,
    Py_ssize_t first_pass_max_brute_size,
    bool use_dtb,
    bool /*verbose*/
) {
    using DISTANCE=mgtree::kdtree_distance_sqeuclid<FLOAT, D>;

    GENIECLUST_PROFILER_USE

    GENIECLUST_PROFILER_START
    mgtree::kdtree_boruvka<FLOAT, D, DISTANCE> tree(X, n, M,
        max_leaf_size, first_pass_max_brute_size, use_dtb);
    GENIECLUST_PROFILER_STOP("tree init")

    GENIECLUST_PROFILER_START
    mgtree::mst<FLOAT, D>(tree, mst_dist, mst_ind, d_core);
    GENIECLUST_PROFILER_STOP("mst call")

    GENIECLUST_PROFILER_START
    for (Py_ssize_t i=0; i<n-1; ++i)
        mst_dist[i] = sqrt(mst_dist[i]);

    Ctree_order(n-1, mst_dist, mst_ind);
    if (d_core) {
        for (Py_ssize_t i=0; i<n; ++i)
            d_core[i] = sqrt(d_core[i]);
    }
    GENIECLUST_PROFILER_STOP("Cmst_euclid_kdtree finalise")

}


/*! A Boruvka-like algorithm based on K-d trees for determining
 *  a(*) Euclidean minimum spanning tree (MST) or
 *  one that corresponds to an M-mutual reachability distance.
 *  Quite fast in low-dimensional spaces.
 *
 *  (*) Note that there might be multiple minimum trees spanning a given graph.
 *
 *  TODO ....
 *
 *  References:
 *  ----------
 *
 *  [4] O. Borůvka, O jistém problému minimálním. Práce Mor. Přírodověd. Spol.
 *  V Brně III 3, 1926, 37–58.
 *
 *  [5] W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
 *  tree: algorithm, analysis, and applications, Proc. 16th ACM SIGKDD Intl.
 *  Conf. Knowledge Discovery and Data Mining (KDD '10), 2010, 603–612.
 *
 *  [6] J.L. Bentley, Multidimensional binary search trees used for associative
 *  searching, Communications of the ACM 18(9), 509–517, 1975,
 *  DOI:10.1145/361002.361007.
 *
 *  [7] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
 *  are fat, The 4th CGC Workshop on Computational Geometry, 1999.
 *
 *  [8] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
 *  strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems,
 *  Communications & Computers (CSCC'01), 2001.
 *
 *  [9] R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
 *  on hierarchical density estimates, Lecture Notes in Computer Science 7819,
 *  2021, 160–172. DOI: 10.1007/978-3-642-37456-2_14.
 *
 *
 * @param X [destroyable] a C-contiguous data matrix, shape n*d
 * @param n number of rows
 * @param d number of columns, 2<=d<=20
 * @param M the level of the "core" distance if M > 1
 * @param mst_dist [out] a vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_ind [out] a vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 * @param d_core [out] NULL for M==1; distances to the points'
 *        (M-1)-th neighbours
 * @param max_leaf_size maximal number of points in the K-d tree's leaves
 * @param first_pass_max_brute_size minimal number of points in a node to treat it as a leaf (unless it's actually a leaf) in the first iteration of the algorithm
 * @param use_dtb whether a dual or a single-tree Boruvka algorithm should be used
 * @param verbose should we output diagnostic/progress messages?
 */
template <class FLOAT>
void Cmst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind, FLOAT* d_core=nullptr,
    Py_ssize_t max_leaf_size=32,
    Py_ssize_t first_pass_max_brute_size=32,
    bool use_dtb=false,
    bool verbose=false
) {
    GENIECLUST_PROFILER_USE

    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (M <= 0)   throw std::domain_error("M <= 0");
    if (M-1 >= n) throw std::domain_error("M >= n-1");

    if (max_leaf_size <= 0)
        throw std::domain_error("max_leaf_size <= 0");

    //if (first_pass_max_brute_size <= 0)
    // does no harm - will have no effect

    if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST... ");

    #define ARGS_mst_euclid_kdtree X, n, M, mst_dist, mst_ind, d_core, max_leaf_size, first_pass_max_brute_size, use_dtb, verbose

    /* LMAO; templates... */
    GENIECLUST_PROFILER_START
    /**/ if (d ==  2) _mst_euclid_kdtree<FLOAT,  2>(ARGS_mst_euclid_kdtree);
    else if (d ==  3) _mst_euclid_kdtree<FLOAT,  3>(ARGS_mst_euclid_kdtree);
    else if (d ==  4) _mst_euclid_kdtree<FLOAT,  4>(ARGS_mst_euclid_kdtree);
    else if (d ==  5) _mst_euclid_kdtree<FLOAT,  5>(ARGS_mst_euclid_kdtree);
    else if (d ==  6) _mst_euclid_kdtree<FLOAT,  6>(ARGS_mst_euclid_kdtree);
    else if (d ==  7) _mst_euclid_kdtree<FLOAT,  7>(ARGS_mst_euclid_kdtree);
    else if (d ==  8) _mst_euclid_kdtree<FLOAT,  8>(ARGS_mst_euclid_kdtree);
    else if (d ==  9) _mst_euclid_kdtree<FLOAT,  9>(ARGS_mst_euclid_kdtree);
    else if (d == 10) _mst_euclid_kdtree<FLOAT, 10>(ARGS_mst_euclid_kdtree);
    else if (d == 11) _mst_euclid_kdtree<FLOAT, 11>(ARGS_mst_euclid_kdtree);
    else if (d == 12) _mst_euclid_kdtree<FLOAT, 12>(ARGS_mst_euclid_kdtree);
    else if (d == 13) _mst_euclid_kdtree<FLOAT, 13>(ARGS_mst_euclid_kdtree);
    else if (d == 14) _mst_euclid_kdtree<FLOAT, 14>(ARGS_mst_euclid_kdtree);
    else if (d == 15) _mst_euclid_kdtree<FLOAT, 15>(ARGS_mst_euclid_kdtree);
    else if (d == 16) _mst_euclid_kdtree<FLOAT, 16>(ARGS_mst_euclid_kdtree);
    else if (d == 17) _mst_euclid_kdtree<FLOAT, 17>(ARGS_mst_euclid_kdtree);
    else if (d == 18) _mst_euclid_kdtree<FLOAT, 18>(ARGS_mst_euclid_kdtree);
    else if (d == 19) _mst_euclid_kdtree<FLOAT, 19>(ARGS_mst_euclid_kdtree);
    else if (d == 20) _mst_euclid_kdtree<FLOAT, 20>(ARGS_mst_euclid_kdtree);
    else {
        // TODO: does it work for d==1?
        // although then a trivial, faster algorithm exists...
        throw std::runtime_error("d should be between 2 and 20");
    }
    GENIECLUST_PROFILER_STOP("Cmst_euclid_kdtree");

    if (verbose) GENIECLUST_PRINT("done.\n");
}

#endif
