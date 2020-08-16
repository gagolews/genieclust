/*  Graph pre-processing and other routines
 *
 *  Copyleft (C) 2018-2020, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_preprocess_h
#define __c_preprocess_h

#include "c_common.h"
#include <algorithm>
#include <vector>
#include <deque>
#include <cmath>

#include "c_gini_disjoint_sets.h"
#include "c_int_dict.h"


/*! Compute the degree of each vertex in an undirected graph
 * over vertex set {0,...,n-1}
 *
 * @param ind c_contiguous matrix of size num_edges*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge
 *     with ind[i,j] < n.
 *     Edges with ind[i,0] < 0 or ind[i,1] < 0 are purposely ignored.
 * @param num_edges number of edges (rows in ind)
 * @param n number of vertices
 * @param deg [out] array of size n, where
 *     deg[i] will give the degree of the i-th vertex.
 */
void Cget_graph_node_degrees(
    const ssize_t* ind,
    ssize_t num_edges,
    ssize_t n,
    ssize_t* deg)
{
    for (ssize_t i=0; i<n; ++i)
        deg[i] = 0;

    for (ssize_t i=0; i<num_edges; ++i) {
        ssize_t u = ind[2*i+0];
        ssize_t v = ind[2*i+1];
        if (u<0 || v<0)
            continue; // represents a no-edge → ignore
        if (u>=n || v>=n)
            throw std::domain_error("All elements must be <= n");
        if (u == v)
            throw std::domain_error("Self-loops are not allowed");

        deg[u]++;
        deg[v]++;
    }
}

#endif
