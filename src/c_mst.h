/*  Minimum Spanning Tree Algorithms:
 *  1. Prim-Jarnik's for complete undirected graphs,
 *  2. Kruskal's for k-NN graphs.
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


#ifndef __c_mst_h
#define __c_mst_h

#include "c_common.h"
#include <vector>
#include <algorithm>
#include <cmath>
// #include <queue>
// #include <deque>
//#include "c_argfuns.h"
#include "c_disjoint_sets.h"
#include "c_distance.h"



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



/*! Represents an undirected edge in a weighted graph.
 *  Features a comparer used to sort MST edges w.r.t. increasing weights.
 */
template <class T>
class CMstTriple
{
public:
    Py_ssize_t i1;  //!< first  vertex defining an edge
    Py_ssize_t i2;  //!< second vertex defining an edge
    T d;            //!< edge weight

    CMstTriple() {}

    CMstTriple(Py_ssize_t i1, Py_ssize_t i2, T d, bool order=true)
    {
        GENIECLUST_ASSERT(i1 != i2);
        GENIECLUST_ASSERT(i1 >= 0);
        GENIECLUST_ASSERT(i2 >= 0);
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

    bool operator<(const CMstTriple<T>& other) const
    {
        if (d == other.d) {
            if (i1 == other.i1)
                return i2 < other.i2;
            else
                return i1 < other.i1;
        }
        else
            return d < other.d;
    }
};




/*! Specialised version of 'Cmst_from_complete' for Euclidean distance
/* (has better locality of reference)
 *
 *
 * @param X [destroyable] a C-contiguous data matrix
 * @param n number of rows
 * @param d number of columns
 * @param mst_d [out] vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_i [out] vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 * @param verbose output diagnostic/progress messages?
 */
template <class T>
void Cmst_euclidean(T* X, Py_ssize_t n, Py_ssize_t d,
    T* mst_dist, Py_ssize_t* mst_ind, bool verbose=false)
{
    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the MST... %3d%%", 0);

    // see Cmst_from_complete for comments

    std::vector<Py_ssize_t> ind_nn(n);
    std::vector<T> dist_nn(n, INFTY);

    //std::vector<T> distances(n);
    //T* _distances = distances.data();

    std::vector<Py_ssize_t> ind_left(n);
    for (Py_ssize_t i=0; i<n; ++i) ind_left[i] = i;


    // TODO: optimise distance computation for the Euclidean and EuclideanSquared distances
    // cache sum(x_i^2) in a vector d
    // note that sum((x_i-x_j)^2) = sum(x_i^2) - 2*sum(x_i*x_j) + sum(x_j^2)
    //                            = -2 * t(x_j)*x_i + 1 * d[j] + d[i]
    // d[i] = const in each iter
    // BLAS GEMV can be used in the remaining part


    std::vector< CMstTriple<T> > mst(n-1);

    Py_ssize_t ind_cur = 0;
    for (Py_ssize_t i=1; i<n; ++i) {
        // i, i+1, ..., n-1 - vertices not yet in the tree

        GENIECLUST_ASSERT(ind_left[i-1] == ind_cur);

        T* x_cur = X+(i-1)*d;

        // compute the distances
        // between ind_cur=ind_left[i-1] and all j=i, i+1, ..., n-1:
#if 0
        // two-stage - slower
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=i; j<n; ++j) {
            _distances[j] = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                _distances[j] += (x_cur[u]-X[j*d+u])*(x_cur[u]-X[j*d+u]);
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=i; j<n; ++j) {
            if (_distances[j] < dist_nn[ind_left[j]]) {
                ind_nn[ind_left[j]] = ind_cur;
                dist_nn[ind_left[j]] = _distances[j];
            }
        }
#else
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=i; j<n; ++j) {
            T dd = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                dd += (x_cur[u]-X[j*d+u])*(x_cur[u]-X[j*d+u]);

            if (dd < dist_nn[ind_left[j]]) {
                ind_nn[ind_left[j]] = ind_cur;
                dist_nn[ind_left[j]] = dd;
            }
        }
#endif

        // let best_ind_left and best_ind_left_pos = min and argmin of dist_nn,
        // for we want to include the vertex that is closest to the vertices
        // of the tree constructed so far
        Py_ssize_t best_ind_left_pos = i;
        Py_ssize_t best_ind_left = ind_left[i];
        for (Py_ssize_t j=i+1; j<n; ++j) {
            if (dist_nn[ind_left[j]] < dist_nn[best_ind_left]) {
                best_ind_left_pos = j;
                best_ind_left = ind_left[j];
            }
        }

        // connect best_ind_left with the tree: add a new edge {best_ind_left, ind_nn[best_ind_left]}
        mst[i-1] = CMstTriple<T>(best_ind_left, ind_nn[best_ind_left], dist_nn[best_ind_left], /*order=*/true);


        // don't visit best_ind_left again
        std::swap(ind_left[best_ind_left_pos], ind_left[i]);
        for (Py_ssize_t u=0; u<d; ++u) {
            std::swap(X[i*d+u], X[best_ind_left_pos*d+u]);
        }
        // this has better locality of reference

        ind_cur = best_ind_left;  // start from best_ind_left next time

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

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");
}



/*! A Jarnik (Prim/Dijkstra)-like algorithm for determining
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
 *  Information Sciences 363 (2016) 8-23.
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
 *
 * @param D a CDistance object such that a call to
 *        <T*>D(j, <Py_ssize_t*>M, Py_ssize_t k) returns a length-n array
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
void Cmst_from_complete(CDistance<T>* D, Py_ssize_t n,
    T* mst_dist, Py_ssize_t* mst_ind, bool verbose=false)
{
    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the MST... %3d%%", 0);

    // NOTE: all changes should also be mirrored in Cmst_euclidean()

    // ind_nn[j] is the vertex from the current tree closest to vertex j
    std::vector<Py_ssize_t> ind_nn(n);
    std::vector<T> dist_nn(n, INFTY);  // dist_nn[j] = d(j, ind_nn[j])

    std::vector<Py_ssize_t> ind_left(n);
    for (Py_ssize_t i=0; i<n; ++i) ind_left[i] = i;

    std::vector< CMstTriple<T> > mst(n-1);

    Py_ssize_t ind_cur = 0;  // start with the first vertex (because we can start with any)
    for (Py_ssize_t i=1; i<n; ++i) {
        // ind_cur is the vertex most recently added to the tree
        // ind_left[i], ind_left[i+1], ..., ind_left[n-1] - vertices not yet in the tree

        // compute the distances (on the fly)
        // between ind_cur and all j=ind_left[i], ind_left[i+1], ..., ind_left[n-1]:
        // dist_from_ind_cur[j] == d(ind_cur, j)
        // pragma omp parallel for inside:
        const T* dist_from_ind_cur = (*D)(ind_cur, ind_left.data()+i, n-i);


        // update ind_nn and dist_nn as maybe now ind_cur (recently added to the tree)
        // is closer to some of the remaining vertices?
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=i; j<n; ++j) {
            Py_ssize_t cur_ind_left = ind_left[j];
            T cur_dist = dist_from_ind_cur[cur_ind_left]; // d(ind_cur, cur_ind_left)
            if (cur_dist < dist_nn[cur_ind_left]) {
                ind_nn[cur_ind_left] = ind_cur;
                dist_nn[cur_ind_left] = cur_dist;
            }
        }

        // let best_ind_left and best_ind_left_pos = min and argmin of dist_nn,
        // for we want to include the vertex that is closest to the vertices
        // of the tree constructed so far
        Py_ssize_t best_ind_left_pos = i;
        Py_ssize_t best_ind_left = ind_left[i];
        for (Py_ssize_t j=i+1; j<n; ++j) {
            Py_ssize_t cur_ind_left = ind_left[j];
            if (dist_nn[cur_ind_left] < dist_nn[best_ind_left]) {
                best_ind_left = cur_ind_left;
                best_ind_left_pos = j;
            }
        }

        // connect best_ind_left with the tree: add a new edge {best_ind_left, ind_nn[best_ind_left]}
        mst[i-1] = CMstTriple<T>(best_ind_left, ind_nn[best_ind_left], dist_nn[best_ind_left], /*order=*/true);


        // don't visit best_ind_left again
#if 0
        std::swap(ind_left[best_ind_left_pos], ind_left[i]);
#else
        // keep ind_left sorted (a bit better locality of reference) (#62)
        for (Py_ssize_t j=best_ind_left_pos; j>i; --j)
            ind_left[j] = ind_left[j-1];
        ind_left[i] = best_ind_left;  // for readability only
#endif


        ind_cur = best_ind_left;  // start from best_ind_left next time

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
        mst_dist[i]    = mst[i].d;
        mst_ind[2*i+0] = mst[i].i1; // i1 < i2
        mst_ind[2*i+1] = mst[i].i2;
    }

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");
}



/*! Determine the first k nearest neighbours of each point.
 *
 *  Exactly n*(n-1)/2 distance computations are performed.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 *  Worst-case time complexity: O(n*(n-1)/2*d*k)
 *
 *
 *  @param D a callable CDistance object such that a call to
 *         <T*>D(j, <Py_ssize_t*>M, Py_ssize_t l) returns an n-ary array
 *         with the distances from the j-th point to l points whose indices
 *         are given in array M
 *  @param n number of points
 *  @param k number of nearest neighbours,
 *  @param dist [out]  a c_contiguous array, shape (n,k),
 *         dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 *  @param ind [out]   a c_contiguous array, shape (n,k),
 *         (undirected) edge definition, interpreted as {i, ind[i,j]}
 *  @param verbose output diagnostic/progress messages?
 */
template <class T>
void Cknn_from_complete(CDistance<T>* D, Py_ssize_t n, Py_ssize_t k,
    T* dist, Py_ssize_t* ind, bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the K-nn graph... %3d%%", 0);


    for (Py_ssize_t i=0; i<n*k; ++i) {
        dist[i] = INFTY;
        ind[i] = -1;
    }

    std::vector<Py_ssize_t> M(n);
    for (Py_ssize_t i=0; i<n; ++i) M[i] = i;

    for (Py_ssize_t i=0; i<n-1; ++i) {
        // pragma omp parallel for inside:
        const T* dij = (*D)(i, M.data()+i+1, n-i-1);
        // let dij[j] == d(x_i, x_j)


        // TODO: the 2nd `if` below can be OpenMP'd
        for (Py_ssize_t j=i+1; j<n; ++j) {

            if (dij[j] < dist[i*k+k-1]) {
                // j might be amongst k-NNs of i
                Py_ssize_t l = k-1;
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
                Py_ssize_t l = k-1;
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



/*! Computes a minimum spanning forest of a (<=k)-nearest neighbour graph
 *  (i.e., one that connects no more than the first k nearest neighbours
 *  (of each point)  using Kruskal's algorithm, and orders
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
 *        (undirected) edge definition, interpreted as {i, ind[i,j]};
 *        negative indices as well as those such that ind[i,j]==i are ignored
 * @param d_core "core" distance (or NULL);
 *        if not NULL then the distance between 2 points will be
 *        d(i, ind[i,j]) = max(d(i, ind[i,j]), d_core[i], d_core[ind[i,j]])
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
Py_ssize_t Cmst_from_nn(
    const T* dist,
    const Py_ssize_t* ind,
    const T* d_core,
    Py_ssize_t n,
    Py_ssize_t k,
    T* mst_dist,
    Py_ssize_t* mst_ind,
    bool* maybe_inexact,
    bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");
    Py_ssize_t nk = n*k;

    if (verbose) GENIECLUST_PRINT_int("[genieclust] Computing the MST... %3d%%", 0);

    std::vector< CMstTriple<T> > nns(nk);
    Py_ssize_t c = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        for (Py_ssize_t j = 0; j < k; ++j) {
            Py_ssize_t i2 = ind[k*i+j];
            if (i2 >= 0 && i2 != i) {
                double d = dist[k*i+j];
                if (d_core) {
                    // d(i, i2) = max(d(i,i2), d_core[i], d_core[i2])
                    if (d < d_core[i])  d = d_core[i];
                    if (d < d_core[i2]) d = d_core[i2];
                }
                nns[c++] = CMstTriple<T>(i, i2, d, true);
            }
        }
    }

    std::stable_sort(nns.data(), nns.data()+c);


    Py_ssize_t triple_cur = 0;
    Py_ssize_t mst_edge_cur = 0;

    CDisjointSets ds(n);
    while (mst_edge_cur < n-1) {
        if (triple_cur == c) {
            // The input graph is not connected (we have a forest)
            Py_ssize_t ret = mst_edge_cur;
            while (mst_edge_cur < n-1) {
                mst_ind[2*mst_edge_cur+0] = -1;
                mst_ind[2*mst_edge_cur+1] = -1;
                mst_dist[mst_edge_cur]    = INFTY;
                mst_edge_cur++;
            }
            if (verbose)
                GENIECLUST_PRINT_int("\b\b\b\b%3d%%", mst_edge_cur*100/(n-1));
            return ret;
        }

        Py_ssize_t u = nns[triple_cur].i1;
        Py_ssize_t v = nns[triple_cur].i2;
        T d = nns[triple_cur].d;
        triple_cur++;

        if (ds.find(u) == ds.find(v))
            continue;

        mst_ind[2*mst_edge_cur+0] = u;
        mst_ind[2*mst_edge_cur+1] = v;
        mst_dist[mst_edge_cur]    = d;

        GENIECLUST_ASSERT(mst_edge_cur == 0 || mst_dist[mst_edge_cur] >= mst_dist[mst_edge_cur-1]);

        ds.merge(u, v);
        mst_edge_cur++;


        if (verbose)
            GENIECLUST_PRINT_int("\b\b\b\b%3d%%", mst_edge_cur*100/(n-1));

        #if GENIECLUST_R
        Rcpp::checkUserInterrupt();
        #elif GENIECLUST_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");

    return mst_edge_cur;
}





// *! Computes a minimum spanning forest of a near-neighbour
// *  graph using Kruskal's algorithm, and orders
// *  its edges w.r.t. increasing weights.
// *
// *  Points can have different numbers of nearest neighbours determined.
// *  Hence, the input graph is given in a "list"-like form.
// *
// *  In general, the MST of the (<=k)-nearest neighbour graph
// *  might not be equal to the MST of the complete pairwise distance graph.
// *
// *
// * @param nns [in/out]  a c_contiguous array, shape (c,), of CMstTriple elements
// *        defining the near-neighbour graphs. Loops are ignored.
// *        The array is sorted in-place.
// * @param c number of elements in `nns`.
// * @param n number of nodes in the graph
// * @param mst_dist [out] c_contiguous vector of length n-1, gives weights of the
// *        resulting MST edges in nondecreasing order;
// *        refer to the function's return value for the actual number
// *        of edges generated (if this is < n-1, the object is padded with INFTY)
// * @param mst_ind [out] c_contiguous matrix of size (n-1)*2, defining the edges
// *        corresponding to mst_d, with mst_i[j,0] <= mst_i[j,1] for all j;
// *        refer to the function's return value for the actual number
// *        of edges generated (if this is < n-1, the object is padded with -1)
// * @param verbose output diagnostic/progress messages?
// *
// * @return number of edges in the minimal spanning forest
// *
// template <class T>
// Py_ssize_t Cmst_from_nn_list(CMstTriple<T>* nns, Py_ssize_t c,
//     Py_ssize_t n, T* mst_dist, Py_ssize_t* mst_ind, bool verbose=false)
// {
//     if (n <= 0)   throw std::domain_error("n <= 0");
//     if (c <= 0)   throw std::domain_error("c <= 0");
//
//     if (verbose)
//         GENIECLUST_PRINT_int("[genieclust] Computing the MST... %3d%%", 0);
//
//     std::sort(nns, nns+c); // unstable sort (do we need stable here?)
//
//     Py_ssize_t triple_cur = 0;
//     Py_ssize_t mst_edge_cur = 0;
//
//     CDisjointSets ds(n);
//     while (mst_edge_cur < n-1) {
//         if (triple_cur == c) {
//             // The input graph is not connected (we have a forest)
//             Py_ssize_t ret = mst_edge_cur;
//             while (mst_edge_cur < n-1) {
//                 mst_ind[2*mst_edge_cur+0] = -1;
//                 mst_ind[2*mst_edge_cur+1] = -1;
//                 mst_dist[mst_edge_cur]    = INFTY;
//                 mst_edge_cur++;
//             }
//             if (verbose)
//                 GENIECLUST_PRINT_int("\b\b\b\b%3d%%", mst_edge_cur*100/(n-1));
//             return ret;
//         }
//
//         Py_ssize_t u = nns[triple_cur].i1;
//         Py_ssize_t v = nns[triple_cur].i2;
//         T d = nns[triple_cur].d;
//         triple_cur++;
//
//         if (u > v) std::swap(u, v); // assure u < v
//         if (u < 0 || ds.find(u) == ds.find(v))
//             continue;
//
//         mst_ind[2*mst_edge_cur+0] = u;
//         mst_ind[2*mst_edge_cur+1] = v;
//         mst_dist[mst_edge_cur]    = d;
//
//         GENIECLUST_ASSERT(mst_edge_cur == 0 || mst_dist[mst_edge_cur] >= mst_dist[mst_edge_cur-1]);
//
//         ds.merge(u, v);
//         mst_edge_cur++;
//
//
//         if (verbose)
//             GENIECLUST_PRINT_int("\b\b\b\b%3d%%", mst_edge_cur*100/(n-1));
//
//         #if GENIECLUST_R
//         Rcpp::checkUserInterrupt();
//         #elif GENIECLUST_PYTHON
//         if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
//         #endif
//     }
//
//     if (verbose) GENIECLUST_PRINT("\b\b\b\bdone.\n");
//
//     return mst_edge_cur;
// }



#endif
