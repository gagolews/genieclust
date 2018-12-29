# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Kruskal's Minimum Spanning Tree Algorithm for M-Nearest Neighbor Graphs

Copyright (C) 2018 Marek.Gagolewski.com
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""



cimport cython
cimport numpy as np
import numpy as np
from numpy.math cimport INFINITY
import warnings

from libcpp.vector cimport vector
from . cimport disjoint_sets
from . cimport argfuns





cpdef tuple MST_pair2(np.double_t[:,::1] dist, np.int_t[:,::1] ind): # [:,::1]==c_contiguous
    """
    Computes a minimum spanning tree of an M-Nearest Neighbor Graph
    using Kruskal's algorithm, and orders its edges w.r.t. increasing weights.

    Note that in general, an MST of the M-Nearest Neighbor Graph
    might not be the MST of the complete Pairwise Distances Graph.

    In case of an unconnected graph, an exception is raised.


    Parameters:
    ----------

    dist : a c_contiguous ndarray, shape (n,M)
        dist[i,:] is sorted increasingly for all i,
        dist[i,j] gives the weight of the edge {i, ind[i,j]}

    ind : a c_contiguous ndarray, shape (n,M)
        edge definition, interpreted as {i, ind[i,j]}

    Returns:
    -------

    pair : tuple
         A pair (indices_matrix, corresponding weights);
         the results are ordered w.r.t. the weights
         (and then the 1st, and the the 2nd index)
    """
    if not (dist.shape[0] == ind.shape[0]
            and dist.shape[1] == ind.shape[1]):
        raise ValueError("shapes of dist and ind must match")
    #if not dist.data.c_contiguous:
        #raise ValueError("dist must be a c_contiguous array")
    #if not ind.data.c_contiguous:
        #raise ValueError("ind must be a c_contiguous array")

    cdef np.uint_t n = dist.shape[0]
    cdef np.uint_t n_neighbors = dist.shape[1]
    cdef np.uint_t nm = n*n_neighbors

    cdef vector[np.uint_t] nn_used = vector[np.uint_t](n, 0)
    cdef vector[np.uint_t] arg_dist = vector[np.uint_t](nm)
    argfuns.argsort(arg_dist.data(), &dist[0,0], nm, False)

    cdef np.uint_t arg_dist_cur = 0
    cdef np.uint_t mst_edge_cur = 0
    cdef np.ndarray[np.int_t,ndim=2] mst_i = np.empty((n-1, 2), dtype=np.int_)
    cdef np.ndarray[np.double_t]     mst_d = np.empty(n-1, dtype=np.float_)

    cdef np.uint_t u, v
    cdef np.double_t d

    cdef disjoint_sets.DisjointSets ds = disjoint_sets.DisjointSets(n)

    while mst_edge_cur < n-1:
        if arg_dist_cur >= nm:
            raise RuntimeError("the graph is not connected. increase n_neighbors")

        u = arg_dist[arg_dist_cur]//n_neighbors
        v = ind[u, nn_used[u]]
        d = dist[u, nn_used[u]]
        nn_used[u] += 1
        arg_dist_cur += 1

        if ds.find(u) == ds.find(v):
            continue

        if u > v: u, v = v, u
        mst_i[mst_edge_cur,0] = u
        mst_i[mst_edge_cur,1] = v
        mst_d[mst_edge_cur]   = d

        ds.merge(u, v)
        mst_edge_cur += 1

    return mst_i, mst_d
