/*  Graph pre-processing and other routines
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


#ifndef __c_preprocess_h
#define __c_preprocess_h

#include <stdexcept>
#include "c_common.h"


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



/*! Compute the incidence list of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 * @param ind c_contiguous matrix of size m*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n.
 *     Edges with ind[i,0] < 0 or ind[i,1] < 0 are purposely ignored.
 * @param m number of edges (rows in ind)
 * @param n number of vertices
 * @param deg array of size n, where deg[i] gives the degree of the i-th vertex.
 * @param data [out] a data buffer of length 2*m, provides data for adj
 * @param adj [out] an array of length n+1, where adj[i] will be an array
 *     of length deg[i] giving the edges incident on the i-th vertex;
 *     adj[n] is a sentinel element
 */
void Cget_graph_node_inclists(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    const Py_ssize_t* deg,
    Py_ssize_t* data,
    Py_ssize_t** inc
) {
    Py_ssize_t cumdeg = 0;
    inc[0] = data;
    for (Py_ssize_t i=0; i<n; ++i) {
        inc[i+1] = data+cumdeg;
        cumdeg += deg[i];
    }

    GENIECLUST_ASSERT(cumdeg <= 2*m);

    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];
        if (u < 0 || v < 0)
            continue; // represents a no-edge -> ignore

#ifdef DEBUG
        if (u >= n || v >= n)
            throw std::domain_error("All elements must be < n");
        if (u == v)
            throw std::domain_error("Self-loops are not allowed");
#endif

        *(inc[u+1]) = i;
        ++(inc[u+1]);

        *(inc[v+1]) = i;
        ++(inc[v+1]);
    }

#ifdef DEBUG
    cumdeg = 0;
    inc[0] = data;
    for (Py_ssize_t i=0; i<n; ++i) {
        GENIECLUST_ASSERT(inc[i] == data+cumdeg);
        cumdeg += deg[i];
    }
#endif
}

#endif
