/*  Noisy k-partition post-processing
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


#ifndef __c_postprocess_h
#define __c_postprocess_h

#include "c_common.h"
#include <algorithm>



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
            throw std::domain_error("!(c[u] < 0 && c[v] < 0)");

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
