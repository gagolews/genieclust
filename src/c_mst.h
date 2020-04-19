/*  Minimum Spanning Tree Algorithms:
 *  a. Prim-Jarník's for Complete Undirected Graphs,
 *  b. Kruskal's for k-NN graphs.
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






/*! Computes a minimum spanning forest of a (<=k)-nearest neighbour graph
 *  using Kruskal's algorithm, and orders its edges w.r.t. increasing weights.
 *
 *  Note that, in general, an MST of the (<=k)-nearest neighbour graph
 *  might not be equal to the MST of the complete Pairwise Distances Graph.
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
 *
 * @return number of edges in the minimal spanning forest
 */
template <class T>
ssize_t Cmst_from_nn(const T* dist, const ssize_t* ind,
    ssize_t n, ssize_t k,
    T* mst_dist, ssize_t* mst_ind, bool* maybe_inexact)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    ssize_t nk = n*k;

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
    }

    return mst_edge_cur;
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
 * @param dist a callable CDistance object such that a call to
 *        <T*>dist(j, <ssize_t*>M, ssize_t k) returns an n-ary array
 *        with the distances from the j-th point to k points whose indices
 *        are given in array M
 * @param n number of points
 * @param mst_d [out] vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_i [out] vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 */
template <class T>
void Cmst_from_complete(CDistance<T>* dist, ssize_t n,
    T* mst_dist, ssize_t* mst_ind)
{
    std::vector<T>  Dnn(n, INFTY);
    std::vector<ssize_t> Fnn(n);
    std::vector<ssize_t> M(n);
    std::vector< CMstTriple<T> > res(n-1);

    for (ssize_t i=0; i<n; ++i) M[i] = i;

    ssize_t lastj = 0, bestj, bestjpos;
    for (ssize_t i=0; i<n-1; ++i) {
        // M[1], ... M[n-i-1] - points not yet in the MST

        // compute the distances from lastj (on the fly)
        // dist_from_lastj[j] == d(lastj, j)
        // pragma omp parallel for inside::
        const T* dist_from_lastj = (*dist)(lastj, M.data()+1, n-i-1);

        bestjpos = bestj = 0;
        for (ssize_t j=1; j<n-i; ++j) {
            // T curdist = dist[n*lastj+M_j]; // d(lastj, M_j)
            ssize_t M_j = M[j];
            T curdist = dist_from_lastj[M_j];
            if (curdist < Dnn[M_j]) {
                Dnn[M_j] = curdist;
                Fnn[M_j] = lastj;
            }
            if (Dnn[M_j] < Dnn[bestj]) {        // D[0] == INFTY
                bestj = M_j;
                bestjpos = j;
            }
        }

        M[bestjpos] = M[n-i-1]; // never ever visit bestj again
        lastj = bestj;          // next time, start from bestj

        // and an edge to MST: (smaller index first)
        res[i] = CMstTriple<T>(Fnn[bestj], bestj, Dnn[bestj], true);
    }

    // sort the resulting MST edges in nondecreasing order w.r.t. d
    std::sort(res.begin(), res.end());

    for (ssize_t i=0; i<n-1; ++i) {
        mst_dist[i]    = res[n-i-2].d;
        mst_ind[2*i+0] = res[n-i-2].i1; // i1 < i2
        mst_ind[2*i+1] = res[n-i-2].i2;
    }
}

#endif
