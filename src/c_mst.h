/*  Minimum Spanning Tree Algorithms:
 *  a. Prim-Jarník's for Complete Undirected Graphs,
 *  b. Kruskal's for k-NN graphs.
 *
 *  Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If not, see <https://www.gnu.org/licenses/>.
 */


#ifndef __c_mst_h
#define __c_mst_h

#include "c_common.h"
#include <vector>
#include <algorithm>
#include <queue>
#include <deque>
#include <cmath>
#include "c_argfuns.h"
#include "c_disjoint_sets.h"
#include "c_distance.h"



/*! Represents an undirected edge in a weighted graph.
 *  Main purpose: a comparer used to sort MST edges w.r.t. decreasing weights
 */
template <class T>
struct CMstTriple {
    ssize_t i1; //!< first  vertex defining an edge
    ssize_t i2; //!< second vertex defining an edge
    T d;        //!< edge weight

    CMstTriple() {}

    CMstTriple(ssize_t i1, ssize_t i2, T d, bool order=true) {
        this->d = d;
        if (!order || (i1 < i2)) {
            this->i1 = i1;
            this->i2 = i2;
        }
        else {
            this->i1 = i2;
            this->i2 = i1;
        }
    }

    bool operator<(const CMstTriple<T>& other) const {
        if (d == other.d) {
            if (i1 == other.i1)
                return i2 > other.i2;
            else
                return i1 > other.i1;
        }
        else
            return d > other.d;
    }
};






/*! Computes a minimum spanning forest of a (<=k)-nearest neighbour
 *  (i.e., one that consists of 1-, 2-, ..., k-neighbours = the first k
 *  nearest neighbours) graph using Kruskal's algorithm, and orders
 *  its edges w.r.t. increasing weights.
 *
 *  Note that, in general, an MST of the (<=k)-nearest neighbour graph
 *  might not be equal to the MST of the complete Pairwise Distances Graph.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 * @param dist   a c_contiguous array, shape (n,k),
 *        dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 * @param ind    a c_contiguous array, shape (n,k),
 *        (undirected) edge definition, interpreted as {i, ind[i,j]}
 * @param n number of nodes
 * @param k minimal degree of all the nodes
 * @param mst_dist [out] c_contiguous vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order;
 *        refer to the function's return value for the actual number
 *        of edges generated (if this is < n-1, the object is padded with INFTY)
 * @param mst_ind [out] c_contiguous matrix of size (n-1)*2, defining the edges
 *        corresponding to mst_d, with mst_i[j,0] <= mst_i[j,1] for all j;
 *        refer to the function's return value for the actual number
 *        of edges generated (if this is < n-1, the object is padded with -1)
 * @param maybe_inexact [out] true indicates that k should be increased to
 *        guarantee that the resulting tree would be the same if a complete
 *        pairwise distance graph was given.
 * @param verbose output diagnostic/progress messages?
 *
 * @return number of edges in the minimal spanning forest
 */
template <class T>
ssize_t Cmst_from_nn(const T* dist, const ssize_t* ind,
    ssize_t n, ssize_t k,
    T* mst_dist, ssize_t* mst_ind, bool* maybe_inexact,
    bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");
    ssize_t nk = n*k;

    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the MST... %3d%%", 0);

    // determine the ordering permutation of dist
    // we're using O(nk) memory anyway
    std::vector<ssize_t> arg_dist(nk);
    Cargsort(arg_dist.data(), dist, nk, true); // stable sort
    std::vector<ssize_t> nn_used(n, 0);

    // slower than arg_dist:
    // std::priority_queue< CMstTriple<T>, std::deque< CMstTriple<T> > > pq;
    // for (ssize_t i=0; i<n; ++i) {
    //     pq.push(CMstTriple<T>(i, ind[k*i+0], dist[k*i+0], false));
    // }
    // std::vector<ssize_t> nn_used(n, 1);




    ssize_t arg_dist_cur = 0;
    ssize_t mst_edge_cur = 0;
    *maybe_inexact = false;
    CDisjointSets ds(n);
    while (mst_edge_cur < n-1) {
        if (arg_dist_cur == nk /*pq.empty()*/) {
            // The input graph is not connected (we have a forest)
            ssize_t ret = mst_edge_cur;
            while (mst_edge_cur < n-1) {
                mst_ind[2*mst_edge_cur+0] = -1;
                mst_ind[2*mst_edge_cur+1] = -1;
                mst_dist[mst_edge_cur]    = INFTY;
                mst_edge_cur++;
            }
            return ret;
        }

        //ssize_t u = pq.top().i1;
        //ssize_t v = pq.top().i2;
        //T d = pq.top().d;

        ssize_t u = arg_dist[arg_dist_cur]/k; // u is the arg_dist_cur-th edge
        GENIECLUST_ASSERT(nn_used[u] < k && u >= 0 && u < n);
        ssize_t v = ind[k*u+nn_used[u]];      // v is its nn_used[u]-th NN
        T d = dist[k*u+nn_used[u]];
        arg_dist_cur++;

        //pq.pop();

        //if (nn_used[u] < k) {
        //    pq.push(CMstTriple<T>(u, ind[k*u+nn_used[u]], dist[k*u+nn_used[u]], false));
        //    nn_used[u]++;                         // mark u's NN as used
        //}
        //else
        //    *maybe_inexact = true; // we've run out of elems
        nn_used[u]++;
        if (nn_used[u] == k) *maybe_inexact = true;

        if (ds.find(u) == ds.find(v))
            continue;

        if (u > v) std::swap(u, v);
        mst_ind[2*mst_edge_cur+0] = u;
        mst_ind[2*mst_edge_cur+1] = v;
        mst_dist[mst_edge_cur]    = d;

        GENIECLUST_ASSERT(mst_edge_cur == 0 || mst_dist[mst_edge_cur] >= mst_dist[mst_edge_cur-1]);

        ds.merge(u, v);
        mst_edge_cur++;

        if (verbose) GENIECLUST_PRINT_int("\b\b\b\b%3d%%", mst_edge_cur*100/(n-1));

        #if GENIECLUST_R
        Rcpp::checkUserInterrupt();
        #elif GENIECLUST_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");

    return mst_edge_cur;
}



/*! Determine the first k nearest neighbours of each point.
 *
 *  Exactly n*(n-1) distance computations are performed.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 * Worst-case time complexity: O(n*(n-1)/2*d*k)
 *
 *
 * @param D a callable CDistance object such that a call to
 *        <T*>D(j, <ssize_t*>M, ssize_t l) returns an n-ary array
 *        with the distances from the j-th point to l points whose indices
 *        are given in array M
 * @param n number of points
 * @param k number of nearest neighbours,
 * @param dist [out]  a c_contiguous array, shape (n,k),
 *        dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 * @param ind [out]   a c_contiguous array, shape (n,k),
 *        (undirected) edge definition, interpreted as {i, ind[i,j]}
 * @param verbose output diagnostic/progress messages?
 */
template <class T>
void Cknn_from_complete(CDistance<T>* D, ssize_t n, ssize_t k,
    T* dist, ssize_t* ind, bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the K-nn graph... %3d%%", 0);


    for (ssize_t i=0; i<n*k; ++i) {
        dist[i] = INFTY;
        ind[i] = -1;
    }

    std::vector<ssize_t> M(n);
    for (ssize_t i=0; i<n; ++i) M[i] = i;

    for (ssize_t i=0; i<n-1; ++i) {
        // pragma omp parallel for inside:
        const T* dij = (*D)(i, M.data()+i+1, n-i-1);
        // let dij[j] == d(x_i, x_j)


        // TODO: the 2nd if below can be OpenMP'd
        for (ssize_t j=i+1; j<n; ++j) {

            if (dij[j] < dist[i*k+k-1]) {
                // j might be amongst k-NNs of i
                ssize_t l = k-1;
                while (l > 0 && dij[j] < dist[i*k+l-1]) {
                    dist[i*k+l] = dist[i*k+l-1];
                    ind[i*k+l]  = ind[i*k+l-1];
                    l -= 1;
                }
                dist[i*k+l] = dij[j];
                ind[i*k+l]  = j;
            }

            if (dij[j] < dist[j*k+k-1]) {
                // i might be amongst k-NNs of j
                ssize_t l = k-1;
                while (l > 0 && dij[j] < dist[j*k+l-1]) {
                    dist[j*k+l] = dist[j*k+l-1];
                    ind[j*k+l]  = ind[j*k+l-1];
                    l -= 1;
                }
                dist[j*k+l] = dij[j];
                ind[j*k+l]  = i;
            }
        }

        if (verbose) GENIECLUST_PRINT_int("\b\b\b\b%3d%%", (n-1+n-i-1)*(i+1)*100/n/(n-1));

        #if GENIECLUST_R
        Rcpp::checkUserInterrupt();
        #elif GENIECLUST_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");
}




/*! A Jarník (Prim/Dijkstra)-like algorithm for determining
 *  a(*) minimum spanning tree (MST) of a complete undirected graph
 *  with weights given by, e.g., a symmetric n*n matrix.
 *
 *  However, the distances can be computed on the fly, so that O(n) memory is used.
 *
 *  (*) Note that there might be multiple minimum trees spanning a given graph.
 *
 *
 *  References:
 *  ----------
 *
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
 *  Information Sciences 363 (2016) 8–23.
 *
 *  V. Jarník, O jistém problému minimálním,
 *  Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.
 *
 *  C.F. Olson, Parallel algorithms for hierarchical clustering,
 *  Parallel Comput. 21 (1995) 1313–1325.
 *
 *  R. Prim, Shortest connection networks and some generalisations,
 *  Bell Syst. Tech. J. 36 (1957) 1389–1401.
 *
 *
 * @param D a callable CDistance object such that a call to
 *        <T*>D(j, <ssize_t*>M, ssize_t k) returns an n-ary array
 *        with the distances from the j-th point to k points whose indices
 *        are given in array M
 * @param n number of points
 * @param mst_d [out] vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_i [out] vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 * @param verbose output diagnostic/progress messages?
 */
template <class T>
void Cmst_from_complete(CDistance<T>* D, ssize_t n,
    T* mst_dist, ssize_t* mst_ind, bool verbose=false)
{
    std::vector<T> Dnn(n, INFTY);
    std::vector<ssize_t> Fnn(n);
    std::vector<ssize_t> M(n);
    std::vector< CMstTriple<T> > res(n-1);

    for (ssize_t i=0; i<n; ++i) M[i] = i;

    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the MST... %3d%%", 0);

    ssize_t lastj = 0, bestj, bestjpos;
    for (ssize_t i=0; i<n-1; ++i) {
        // M[1], ... M[n-i-1] - points not yet in the MST

        // compute the distances from lastj (on the fly)
        // dist_from_lastj[j] == d(lastj, j)
        // pragma omp parallel for inside:
        const T* dist_from_lastj = (*D)(lastj, M.data()+1, n-i-1);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ssize_t j=1; j<n-i; ++j) {
            ssize_t M_j = M[j];
            T curdist = dist_from_lastj[M_j]; // d(lastj, M_j)
            if (curdist < Dnn[M_j]) {
                Dnn[M_j] = curdist;
                Fnn[M_j] = lastj;
            }
        }

        // find min and argmin in Dnn:
        bestjpos = bestj = 0;
        for (ssize_t j=1; j<n-i; ++j) {
            ssize_t M_j = M[j];
            if (Dnn[M_j] < Dnn[bestj]) {        // Dnn[0] == INFTY
                bestj = M_j;
                bestjpos = j;
            }
        }

        M[bestjpos] = M[n-i-1]; // don't visit bestj again
        lastj = bestj;          // start from bestj next time

        // and an edge to MST: (smaller index first)
        res[i] = CMstTriple<T>(Fnn[bestj], bestj, Dnn[bestj], true);

        if (verbose) GENIECLUST_PRINT_int("\b\b\b\b%3d%%", (n-1+n-i-1)*(i+1)*100/n/(n-1));

        #if GENIECLUST_R
        Rcpp::checkUserInterrupt();
        #elif GENIECLUST_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    // sort the resulting MST edges in nondecreasing order w.r.t. d
    std::sort(res.begin(), res.end());

    for (ssize_t i=0; i<n-1; ++i) {
        mst_dist[i]    = res[n-i-2].d;
        mst_ind[2*i+0] = res[n-i-2].i1; // i1 < i2
        mst_ind[2*i+1] = res[n-i-2].i2;
    }

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");
}

#endif
