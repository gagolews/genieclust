/*  Minimum spanning tree and k-nearest neighbour algorithms
 *  (quite fast in low-dimensional spaces, currently Euclidean distance only)
 *
 *
 *  [1] V. Jarník, O jistém problému minimálním,
 *  Práce Moravské Přírodovědecké Společnosti 6, 1930, 57–63.
 *
 *  [2] C.F. Olson, Parallel algorithms for hierarchical clustering,
 *  Parallel Computing 21(8), 1995, 1313–1325.
 *
 *  [3] R. Prim, Shortest connection networks and some generalizations,
 *  The Bell System Technical Journal 36(6), 1957, 1389–1401.
 *
 *  [4] O. Borůvka, O jistém problému minimálním,
 *  Práce Moravské Přírodovědecké Společnosti 3, 1926, 37–58.
 *
 *  [5] W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
 *  tree: Algorithm, analysis, and applications, Proc. 16th ACM SIGKDD Intl.
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
 *  2013, 160–172. DOI:10.1007/978-3-642-37456-2_14.
 *
 *  [10] R.J.G.B. Campello, D. Moulavi, A. Zimek. J. Sander, Hierarchical
 *  density estimates for data clustering, visualization, and outlier detection,
 *  ACM Transactions on Knowledge Discovery from Data (TKDD) 10(1),
 *  2015, 1–51, DOI:10.1145/2733381.
 *
 *  [11] L. McInnes, J. Healy, Accelerated hierarchical density-based
 *  clustering, IEEE Intl. Conf. Data Mining Workshops (ICMDW), 2017, 33–42,
 *  DOI:10.1109/ICDMW.2017.12.
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
#include <limits>
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


/*! A Jarník (Prim/Dijkstra)-like algorithm for determining
 *  a(*) Euclidean minimum spanning tree (MST) or
 *  one wrt an M-mutual reachability distance.
 *
 *  If `M>2`, the spanning tree is the smallest wrt the degree-`M`
 *  mutual reachability distance [9]_ given by
 *  :math:`d_M(i, j)=\\max\\{ c_M(i), c_M(j), d(i, j)\\}`, where :math:`d(i,j)`
 *  is the Euclidean distance between the `i`-th and the `j`-th point,
 *  and :math:`c_M(i)` is the `i`-th `M`-core distance defined as the distance
 *  between the `i`-th point and its `(M-1)`-th nearest neighbour
 *  (not including the query points themselves).
 *
 *  (\*) We note that if there are many pairs of equidistant points,
 *  there can be many minimum spanning trees. In particular, it is likely
 *  that there are point pairs with the same mutual reachability distances.
 *  To make the definition less ambiguous (albeit with no guarantees),
 *  internally, we rely on the adjusted distance:
 *  :math:`d_M(i, j)=\\max\\{c_M(i), c_M(j), d(i, j)\\}+\\varepsilon d(i, j)` or
 *  :math:`d_M(i, j)=\\max\\{c_M(i), c_M(j), d(i, j)\\}-\\varepsilon \\min\\{c_M(i), c_M(j)\\}`,
 *  where :math:`\\varepsilon` is close to 0. ``|mutreach_adj| < 1`` selects
 *  the former (ε=``mutreach_adj``) whilst ``1 < |mutreach_adj| < 2``
 *  chooses the latter (ε=``mutreach_adj``±1).
 *
 *  Time complexity: O(n^2). It is assumed that M is rather small
 *  (say, M<=20). If M>2, all pairwise the distances are computed twice
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
 *  Parallel Computing 21(8), 1995, 1313–1325.
 *
 *  [3] R. Prim, Shortest connection networks and some generalizations,
 *  The Bell System Technical Journal 36(6), 1957, 1389–1401.
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
 * @param nn_dist [out] NULL for M==1 or the n*(M-1) distances to the n points'
 *        (M-1) nearest neighbours
 * @param nn_ind [out] NULL for M==1 or the n*(M-1) indexes of the n points'
 *        (M-1) nearest neighbours
 * @param mutreach_adj adjustment for mutual reachability distance ambiguity (for M>2) whose fractional part should be close to 0:
 *        values in `(-1,0)` prefer connecting to farther NNs,
 *        values in `(0, 1)` fall for closer NNs,
 *        values in `(-2,-1)` prefer connecting to points with smaller core distances,
 *        values in `(1, 2)` favour larger core distances;
 *        see above for more details
 *
 * @param verbose should we output diagnostic/progress messages?
 */
template <class FLOAT>
void Cmst_euclid_brute(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    FLOAT mutreach_adj=-1.00000011920928955078125,
    bool verbose=false
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (M <= 0)   throw std::domain_error("M <= 0");
    if (M-1 >= n) throw std::domain_error("M >= n-1");
    GENIECLUST_ASSERT(mst_dist);
    GENIECLUST_ASSERT(mst_ind);

    if (std::abs(mutreach_adj)>=2) throw std::domain_error("|mutreach_adj|>=2");
    bool mutreach_adj_via_dcore = (std::abs(mutreach_adj) >= 1);
    mutreach_adj = mutreach_adj - std::trunc(mutreach_adj);  // fractional part
    if (std::abs(mutreach_adj) < 2.0*std::numeric_limits<FLOAT>::epsilon())
        mutreach_adj = 0.0;




    std::vector<FLOAT> d_core;
    if (M > 2) {
        d_core.resize(n);
        GENIECLUST_ASSERT(nn_dist);
        GENIECLUST_ASSERT(nn_ind);
        Cknn1_euclid_brute(X, n, d, M-1, nn_dist, nn_ind,
                           /*squared=*/true, verbose);
        for (Py_ssize_t i=0; i<n; ++i) d_core[i] = nn_dist[i*(M-1)+(M-2)];

        // for M==2, we can fetch d_core from MST, as nearest neighbours
        // are connected by an edge (see below)
    }

    if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST... %3d%%", 0);


    // ind_nn[j] is the vertex from the current tree closest to vertex j
    std::vector<Py_ssize_t> ind_nn(n);
    std::vector<FLOAT> dist_nn(n, INFINITY);  // dist_nn[j] = d(j, ind_nn[j])

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
                // if (d_core[i-1] > dist_nn[j]) continue;
                FLOAT dd = 0.0;
                for (Py_ssize_t u=0; u<d; ++u)
                    dd += square(x_cur[u]-X[j*d+u]);

                if (mutreach_adj_via_dcore) {
                    if (d_core[i-1] <= d_core[j]) {
                        if (dd <= d_core[j])
                            dd = d_core[j]   - mutreach_adj*d_core[i-1];  // minus
                        else
                            dd = dd          - mutreach_adj*d_core[i-1];
                    }
                    else {  // d_core[j] < d_core[i-1]
                        if (dd <= d_core[i-1])
                            dd = d_core[i-1] - mutreach_adj*d_core[j];
                        else
                            dd = dd          - mutreach_adj*d_core[j];
                    }
                }
                else {
                    dd = max3(dd, d_core[i-1], d_core[j]) + mutreach_adj*dd; // plus
                }

                if (dd < dist_nn[j]) {
                    dist_nn[j] = dd;
                    ind_nn[j] = i-1;
                }
            }
        }
#endif

        // we want to include the vertex that is closest to
        // the vertices of the tree constructed so far
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
            if (mutreach_adj != 0.0) {
                // recompute the distance without the ambiguity correction
                dist_nn[i] = 0.0;
                for (Py_ssize_t u=0; u<d; ++u)
                    dist_nn[i] += square(X[i*d+u]-X[ind_nn[i]*d+u]);
                dist_nn[i] = max3(dist_nn[i], d_core[ind_nn[i]], d_core[i]);
            }
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
        for (Py_ssize_t i=0; i<n*(M-1); ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }
    else if (M == 2) {
        // for M==2 we just need the nearest neighbours,
        // and the MST connects them with each other
        for (Py_ssize_t i=0; i<n; ++i)
            nn_dist[i] = INFINITY;

        for (Py_ssize_t i=0; i<n-1; ++i) {
            if (nn_dist[mst_ind[2*i+0]] > mst_dist[i]) {
                nn_dist[mst_ind[2*i+0]] = mst_dist[i];
                nn_ind[mst_ind[2*i+0]] = mst_ind[2*i+1];
            }
            if (nn_dist[mst_ind[2*i+1]] > mst_dist[i]) {
                nn_dist[mst_ind[2*i+1]] = mst_dist[i];
                nn_ind[mst_ind[2*i+1]] = mst_ind[2*i+0];
            }
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
    FLOAT* mst_dist, Py_ssize_t* mst_ind,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size,
    Py_ssize_t first_pass_max_brute_size,
    bool use_dtb,
    FLOAT mutreach_adj,
    bool /*verbose*/
) {
    using DISTANCE=quitefastkdtree::kdtree_distance_sqeuclid<FLOAT, D>;

    GENIECLUST_PROFILER_USE

    GENIECLUST_PROFILER_START
    quitefastkdtree::kdtree_boruvka<FLOAT, D, DISTANCE> tree(X, n, M,
        max_leaf_size, first_pass_max_brute_size, use_dtb, mutreach_adj);
    GENIECLUST_PROFILER_STOP("tree init")

    GENIECLUST_PROFILER_START
    quitefastkdtree::mst<FLOAT, D, DISTANCE>(
        tree, mst_dist, mst_ind, nn_dist, nn_ind
    );
    GENIECLUST_PROFILER_STOP("mst call")

    GENIECLUST_PROFILER_START
    for (Py_ssize_t i=0; i<n-1; ++i)
        mst_dist[i] = sqrt(mst_dist[i]);

    if (M>1) {
        for (Py_ssize_t i=0; i<n*(M-1); ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }

    Ctree_order(n-1, mst_dist, mst_ind);
    GENIECLUST_PROFILER_STOP("Cmst_euclid_kdtree finalise")

}


/*! The function determines the/a(\*) minimum spanning tree (MST) of a set
 *  of `n` points, i.e., an acyclic undirected graph whose vertices represent
 *  the points, and `n-1` edges with the minimal sum of weights, given by
 *  the pairwise distances.  MSTs have many uses in, amongst others,
 *  topological data analysis (clustering, dimensionality reduction, etc.).
 *
 *  For `M<=2`, we get a spanning tree that minimises the sum of Euclidean
 *  distances between the points. If `M==2`, the function additionally returns
 *  the distance to each point's nearest neighbour.
 *
 *  If `M>2`, the spanning tree is the smallest wrt the degree-`M`
 *  mutual reachability distance [9]_ given by
 *  :math:`d_M(i, j)=\\max\\{ c_M(i), c_M(j), d(i, j)\\}`, where :math:`d(i,j)`
 *  is the Euclidean distance between the `i`-th and the `j`-th point,
 *  and :math:`c_M(i)` is the `i`-th `M`-core distance defined as the distance
 *  between the `i`-th point and its `(M-1)`-th nearest neighbour
 *  (not including the query points themselves).
 *  In clustering and density estimation, `M` plays the role of a smoothing
 *  factor; see [10]_ and the references therein for discussion. This parameter
 *  corresponds to the ``hdbscan`` Python package's ``min_samples=M-1``.
 *
 *  (\*) We note that if there are many pairs of equidistant points,
 *  there can be many minimum spanning trees. In particular, it is likely
 *  that there are point pairs with the same mutual reachability distances.
 *  See ``mutreach_adj`` for an adjustment to address this (partially).
 *
 *  The implemented algorithm assumes that `M` is rather small; say, `M <= 20`.
 *
 *  Our implementation of K-d trees [6]_ has been quite optimised; amongst
 *  others, it has good locality of reference (at the cost of making a
 *  copy of the input dataset), features the sliding midpoint (midrange) rule
 *  suggested in [7]_, and a node pruning strategy inspired by the discussion
 *  in [8]_.
 *
 *  The "single-tree" version of the Borůvka algorithm is naively
 *  parallelisable: in every iteration, it seeks each point's nearest "alien",
 *  i.e., the nearest point thereto from another cluster.
 *  The "dual-tree" Borůvka version of the algorithm is, in principle, based
 *  on [5]_. As far as our implementation is concerned, the dual-tree approach
 *  is only faster in 2- and 3-dimensional spaces, for `M<=2`, and in
 *  a single-threaded setting.  For another (approximate) adaptation
 *  of the dual-tree algorithm to the mutual reachability distance, see [11]_.
 *
 *  Nevertheless, it is well-known that K-d trees perform well only in spaces
 *  of low intrinsic dimensionality (a.k.a. the "curse").
 *
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
 *  [10] R.J.G.B. Campello, D. Moulavi, A. Zimek. J. Sander, Hierarchical
 *  density estimates for data clustering, visualization, and outlier detection,
 *  ACM Transactions on Knowledge Discovery from Data (TKDD) 10(1),
 *  2015, 1–51, DOI:10.1145/2733381.
 *
 *  [11] L. McInnes, J. Healy, Accelerated hierarchical density-based
 *  clustering, IEEE Intl. Conf. Data Mining Workshops (ICMDW), 2017, 33–42,
 *  DOI:10.1109/ICDMW.2017.12.
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
 * @param nn_dist [out] NULL for M==1 or the n*(M-1) distances to the n points'
 *        (M-1) nearest neighbours
 * @param nn_ind [out] NULL for M==1 or the n*(M-1) indexes of the n points'
 *        (M-1) nearest neighbours
 * @param max_leaf_size maximal number of points in the K-d tree's leaves
 * @param first_pass_max_brute_size minimal number of points in a node to treat
 *        it as a leaf (unless it's actually a leaf) in the first iteration
 *        of the algorithm
 * @param use_dtb whether a dual or a single-tree Borůvka algorithm
 *        should be used
 * @param mutreach_adj (M>2 only) adjustment for mutual reachability distance
 *        ambiguity (for M>2):
 *        values in `(-1,0)` prefer connecting to farther NNs,
 *        values in `(0, 1)` fall for closer NNs,
 *        values in `(-2,-1)` prefer connecting to points with smaller core distances,
 *        values in `(1, 2)` favour larger core distances
 * @param verbose should we output diagnostic/progress messages?
 */
template <class FLOAT>
void Cmst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind,
    FLOAT* nn_dist=nullptr, Py_ssize_t* nn_ind=nullptr,
    Py_ssize_t max_leaf_size=32,
    Py_ssize_t first_pass_max_brute_size=32,
    bool use_dtb=false,
    FLOAT mutreach_adj=-1.00000011920928955078125,
    bool verbose=false
) {
    GENIECLUST_PROFILER_USE

    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (M <= 0)   throw std::domain_error("M <= 0");
    if (M-1 >= n) throw std::domain_error("M >= n-1");
    if (std::abs(mutreach_adj)>=2) throw std::domain_error("|mutreach_adj|>=2");
    GENIECLUST_ASSERT(mst_dist);
    GENIECLUST_ASSERT(mst_ind);

    if (max_leaf_size <= 0) throw std::domain_error("max_leaf_size <= 0");


    //if (first_pass_max_brute_size <= 0)
    // does no harm - will have no effect

    if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST... ");

    #define IF_d_CALL_MST_EUCLID_KDTREE(D_) \
        if (d == D_) \
            _mst_euclid_kdtree<FLOAT,  D_>(\
                X, n, M, mst_dist, mst_ind, \
                nn_dist, nn_ind, max_leaf_size, first_pass_max_brute_size, \
                use_dtb, mutreach_adj, verbose \
            )

    /* LMAO; templates... */
    GENIECLUST_PROFILER_START
    /**/ IF_d_CALL_MST_EUCLID_KDTREE(2);
    else IF_d_CALL_MST_EUCLID_KDTREE(3);
    else IF_d_CALL_MST_EUCLID_KDTREE(4);
    else IF_d_CALL_MST_EUCLID_KDTREE(5);
    else IF_d_CALL_MST_EUCLID_KDTREE(6);
    else IF_d_CALL_MST_EUCLID_KDTREE(7);
    else IF_d_CALL_MST_EUCLID_KDTREE(8);
    else IF_d_CALL_MST_EUCLID_KDTREE(9);
    else IF_d_CALL_MST_EUCLID_KDTREE(10);
    else IF_d_CALL_MST_EUCLID_KDTREE(11);
    else IF_d_CALL_MST_EUCLID_KDTREE(12);
    else IF_d_CALL_MST_EUCLID_KDTREE(13);
    else IF_d_CALL_MST_EUCLID_KDTREE(14);
    else IF_d_CALL_MST_EUCLID_KDTREE(15);
    else IF_d_CALL_MST_EUCLID_KDTREE(16);
    else IF_d_CALL_MST_EUCLID_KDTREE(17);
    else IF_d_CALL_MST_EUCLID_KDTREE(18);
    else IF_d_CALL_MST_EUCLID_KDTREE(19);
    else IF_d_CALL_MST_EUCLID_KDTREE(20);
    else {
        // TODO: does it work for d==1?
        // although then a trivial, faster algorithm exists...
        throw std::runtime_error("d should be between 2 and 20");
    }
    GENIECLUST_PROFILER_STOP("Cmst_euclid_kdtree");

    if (verbose) GENIECLUST_PRINT("done.\n");
}

#endif
