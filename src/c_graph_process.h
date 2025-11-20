/*  Graph pre/post-processing and other routines
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


#ifndef __c_graph_process_h
#define __c_graph_process_h

#include <stdexcept>
#include <algorithm>
#include "c_common.h"
#include "c_argfuns.h"


/*! Translate indexes based on a skip array.
 *
 * If skip=[False, True, False, False, True, False, False],
 * then indexes are mapped in such a way that:
 * 0 -> 0,
 * 1 -> 2,
 * 2 -> 3,
 * 3 -> 5,
 * 4 -> 6
 *
 * @param ind [in/out] Array of indexes to translate
 * @param m size of ind
 * @param skip Boolean array
 * @param n size of skip
 */
template<class T> void Ctranslate_skipped_indexes(
    Py_ssize_t* ind, Py_ssize_t m,
    T* skip, Py_ssize_t n
) {
    if (m <= 0) return;

    std::vector<Py_ssize_t> o(m);
    Cargsort(o.data(), ind, m, false);

    Py_ssize_t j = 0;
    Py_ssize_t k = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        if (skip[i]) continue;

        if (ind[o[k]] == j) {
            ind[o[k]] = i;
            k++;

            if (k == m) return;
            GENIECLUST_ASSERT(o[k] != j);
        }

        j++;
    }

    GENIECLUST_ASSERT(false);
}


/*! Compute the degree of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 * @param ind c_contiguous matrix of size m*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n.
 *     Edges with ind[i,0] < 0 or ind[i,1] < 0 are purposely ignored.
 * @param m number of edges (rows in ind)
 * @param n number of vertices
 * @param deg [out] array of size n, where
 *     deg[i] will give the degree of the i-th vertex.
 */
void Cget_graph_node_degrees(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    Py_ssize_t* deg /*out*/
) {
    for (Py_ssize_t i=0; i<n; ++i)
        deg[i] = 0;

    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];

        if (u < 0) {
            GENIECLUST_ASSERT(v < 0);
            continue; // represents a no-edge -> ignore
        }
        GENIECLUST_ASSERT(v >= 0);

        if (u >= n || v >= n)
            throw std::domain_error("All elements must be < n");
        if (u == v)
            throw std::domain_error("Self-loops are not allowed");

        deg[u]++;
        deg[v]++;
    }
}





/*! Merge all midliers with their nearest clusters
 *
 *  The i-th node is a midlier if it is a leaf in the spanning tree
 *  (and hence it meets c[i] < 0) which is amongst the
 *  M nearest neighbours of its adjacent vertex, j.
 *
 *  This procedure allocates c[i] to its its closest cluster, c[j].
 *
 *
 *  @param tree_ind c_contiguous matrix of size num_edges*2,
 *     where {tree_ind[k,0], tree_ind[k,1]} specifies the k-th (undirected) edge
 *     in the spanning tree (or forest); 0 <= tree_ind[i,j] < n;
 *     edges with tree_ind[i,0] < 0 or tree_ind[i,1] < 0 are ignored.
 *  @param num_edges number of rows in tree_ind (edges)
 *  @param nn_ind c_contiguous matrix of size n*num_neighbours;
 *     nn[i,:] gives the indexes of the i-th point's
 *     nearest neighbours; -1 indicates a "missing value"
 *  @param num_neighbours number of columns in nn
 *  @param M smoothing factor, 1 <= M <= num_neighbours
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster ID (in {-1, 0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1.  Class -1 represents the leaves of the
 *      input spanning tree
 *  @param n length of c and the number of vertices in the spanning tree
 */
void Cmerge_midliers(
    const Py_ssize_t* tree_ind,
    Py_ssize_t num_edges,
    const Py_ssize_t* nn_ind,
    Py_ssize_t num_neighbours,
    Py_ssize_t M,
    Py_ssize_t* c,
    Py_ssize_t n
) {
    if (M < 1 || M > num_neighbours)
        throw std::domain_error("incorrect smoothing factor M");

    for (Py_ssize_t i=0; i<num_edges; ++i) {
        Py_ssize_t u = tree_ind[2*i+0];
        Py_ssize_t v = tree_ind[2*i+1];
        if (u<0 || v<0)
            continue; // represents a no-edge -> ignore
        if (u>=n || v>=n)
            throw std::domain_error("all elements must be <= n");
        if (c[u] < 0 && c[v] < 0)
            continue; // throw std::domain_error("!(c[u] < 0 && c[v] < 0)");

        if (c[u] >= 0 && c[v] >= 0)
            continue;

        if (c[v] < 0)
            std::swap(u, v);

        GENIECLUST_ASSERT(c[u] <  0);  // u is a leaf
        GENIECLUST_ASSERT(c[v] >= 0);  // v is a non-leaf

        // check if u is amongst v's M nearest neighbours

        //c[u] = -1; // it's negative anyway
        for (Py_ssize_t j=0; j<M; ++j) {
            // -1s are ignored (they should be at the end of the array btw)
            if (nn_ind[v*num_neighbours+j] == u) {
                // yes, it's a midlier point
                c[u] = c[v];
                break;
            }
        }
    }
}


/*! Merge all outliers and midliers with their nearest clusters
 *
 *  For each leaf in the MST, i (and hence a vertex which meets c[i] < 0),
 *  this procedure allocates c[i] to its its closest cluster, c[j],
 *  where j is the vertex adjacent to i.
 *
 *
 *  @param tree_ind c_contiguous matrix of size num_edges*2,
 *     where {tree_ind[k,0], tree_ind[k,1]} specifies the k-th (undirected) edge
 *     in the spanning tree (or forest); 0 <= tree_ind[i,j] < n;
 *     edges with tree_ind[i,0] < 0 or tree_ind[i,1] < 0 are ignored.
 *  @param num_edges number of rows in ind (edges)
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster ID (in {-1, 0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1.  Class -1 represents the leaves of the
 *      input spanning tree
 *  @param n length of c and the number of vertices in the spanning tree
 */
void Cmerge_all(
    const Py_ssize_t* tree_ind,
    Py_ssize_t num_edges,
    Py_ssize_t* c,
    Py_ssize_t n
) {
    for (Py_ssize_t i=0; i<num_edges; ++i) {
        Py_ssize_t u = tree_ind[2*i+0];
        Py_ssize_t v = tree_ind[2*i+1];
        if (u<0 || v<0)
            continue; // represents a no-edge -> ignore
        if (u>=n || v>=n)
            throw std::domain_error("all elements must be <= n");
        if (c[u] < 0 && c[v] < 0)
            throw std::domain_error("!(c[u] < 0 && c[v] < 0)");

        if (c[u] < 0)
            c[u] = c[v];
        else if (c[v] < 0)
            c[v] = c[u];
        else
           continue;
    }
}


#endif
