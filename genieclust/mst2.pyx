# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Kruskal's Minimum Spanning Tree Algorithm for Complete Undirected Graphs

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
from . import internal




cpdef tuple MST_pair2(np.double_t[:,:] dist, np.int_t[:,:] ind):
    assert dist.shape[0] == ind.shape[0]
    assert dist.shape[1] == ind.shape[1]
    cdef np.int_t n = dist.shape[0]
    cdef np.int_t n_neighbors = dist.shape[1]

    cdef np.int_t[:] arg_dist = np.argsort(dist, axis=None)
    cdef np.int_t[:] nn_used = np.zeros(n, dtype=np.int_)
    cdef np.int_t arg_dist_cur = 0
    cdef np.int_t mst_edge_cur = 0
    cdef np.ndarray[np.int_t,ndim=2] mst_i = np.empty((n-1, 2), dtype=np.int_)
    cdef np.ndarray[np.double_t]     mst_d = np.empty(n-1, dtype=np.float_)

    cdef np.int_t u, v
    cdef np.double_t d

    ds = internal.DisjointSets(n)

    while mst_edge_cur < n-1:
        if arg_dist_cur >= arg_dist.shape[0]:
            raise Exception("graph is not connected. increase n_neighbors")

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
        ds.union(u, v)
        mst_edge_cur += 1

    return mst_i, mst_d
