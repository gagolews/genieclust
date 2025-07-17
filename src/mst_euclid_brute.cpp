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
#include "c_mst_triple.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>


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
    FLOAT mutreach_adj,
    bool verbose
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


// instantiate:
template void Ctree_order<float>(Py_ssize_t m, float* tree_dist, Py_ssize_t* tree_ind);

template void Ctree_order<double>(Py_ssize_t m, double* tree_dist, Py_ssize_t* tree_ind);

template void Cmst_euclid_brute<float>(
    float* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    float* mst_dist, Py_ssize_t* mst_ind,
    float* nn_dist, Py_ssize_t* nn_ind,
    float mutreach_adj,
    bool verbose
);

template void Cmst_euclid_brute<double>(
    double* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    double* mst_dist, Py_ssize_t* mst_ind,
    double* nn_dist, Py_ssize_t* nn_ind,
    double mutreach_adj,
    bool verbose
);
