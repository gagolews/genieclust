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



/*! Merge all "boundary" points with their nearest "core" points  TODO
 *
 *
 *  For all the boundary points i, set c[i] = c[j], where {i,j} is an edge
 *  in a spanning forest given by an adjacency matrix ind.
 *
 *  The i-th point is a boundary point if it is a noise point, i.e., c[i] < 0,
 *  and it is amongst j's M nearest neighbours.
 *
 *
 *  @param ind c_contiguous matrix of size num_edges*2,
 *     where {ind[i,0], ind[i,1]} specifies the i-th (undirected) edge
 *     in a spanning tree or forest; ind[i,j] < n.
 *     Edges with ind[i,0] < 0 or ind[i,1] < 0 are purposely ignored.
 *  @param num_edges number of rows in ind (edges)
 *  @param nn c_contiguous matrix of size n*num_neighbours;
 *     nn[i,:] gives the indices of the i-th point's
 *     nearest neighbours; -1 indicates a "missing value"
 *  @param num_neighbours number of columns in nn
 *  @param M smoothing factor, 1 <= M <= num_neighbours
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster id
 *      (in {-1, 0, 1, ..., k-1} for some k) of the i-th object, i=0,...,n-1.
 *      Class -1 denotes the `noise' cluster.
 *  @param n length of c and the number of vertices in the spanning forest
 */
void Cmerge_boundary_points(
    const Py_ssize_t* ind,
    Py_ssize_t num_edges,
    const Py_ssize_t* nn,
    Py_ssize_t num_neighbours,
    Py_ssize_t M,
    Py_ssize_t* c,
    Py_ssize_t n
) {
    if (M < 1 || M > num_neighbours)
        throw std::domain_error("Incorrect smoothing factor M");

    for (Py_ssize_t i=0; i<num_edges; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];
        if (u<0 || v<0)
            continue; // represents a no-edge -> ignore
        if (u>=n || v>=n)
            throw std::domain_error("all elements must be <= n");
        if (c[u] < 0 && c[v] < 0)
            throw std::domain_error("there is an edge between two noise points");

        if (c[u] >= 0 && c[v] >= 0)
            continue;

        if (c[v] < 0)
            std::swap(u, v);

        GENIECLUST_ASSERT(c[u] <  0);  // u is marked as a noise point
        GENIECLUST_ASSERT(c[v] >= 0);  // v is a core point

        // a noise point is not necessarily a boundary point:
        // u is a boundary point if u is amongst v's M-1 nearest neighbours

        //c[u] = -1; // it's negative anyway
        for (Py_ssize_t j=0; j<M; ++j) {
            // -1s are ignored (they should be at the end of the array btw)
            if (nn[v*num_neighbours+j] == u) {
                // yes, it's a boundary point
                c[u] = c[v];
                break;
            }
        }
    }
}


/*! Merge all noise points with their nearest clusters  TODO
 *
 *  For all the points i with some cluster identifier c[i] < 0, i.e., for
 *  all the noise points, set c[i] = c[j],
 *  where {i,j} is an edge in a spanning forest given by adjacency matrix ind.
 *
 *
 *  @param ind c_contiguous matrix of size num_edges*2,
 *     where {ind[i,0], ind[i,1]} specifies the i-th (undirected) edge
 *     in a spanning tree or forest; ind[i,j] < n.
 *     Edges with ind[i,0] < 0 or ind[i,1] < 0 are purposely ignored.
 *  @param num_edges number of rows in ind (edges)
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster id
 *      (in {-1, 0, 1, ..., k-1} for some k) of the i-th object, i=0,...,n-1.
 *      Class -1 denotes the `noise' cluster.
 *  @param n length of c and the number of vertices in the spanning forest
 */
void Cmerge_noise_points(
        const Py_ssize_t* ind,
        Py_ssize_t num_edges,
        Py_ssize_t* c,
        Py_ssize_t n
) {
    for (Py_ssize_t i=0; i<num_edges; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];
        if (u<0 || v<0)
            continue; // represents a no-edge -> ignore
        if (u>=n || v>=n)
            throw std::domain_error("all elements must be <= n");
        if (c[u] < 0 && c[v] < 0)
            throw std::domain_error("there is an edge between two noise points");

        if (c[u] < 0)
            c[u] = c[v];
        else if (c[v] < 0)
            c[v] = c[u];
        //else
        //    continue;
    }
}


#endif
