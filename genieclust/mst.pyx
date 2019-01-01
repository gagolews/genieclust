# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
The Prim-Jarník Minimum Spanning Tree Algorithm for Complete Undirected Graphs

Copyright (C) 2018-2019 Marek.Gagolewski.com
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
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np
import numpy as np
from libc.math cimport fabs, sqrt
from numpy.math cimport INFINITY
from libcpp.vector cimport vector
from . cimport c_disjoint_sets
from . cimport c_argfuns
import warnings

ctypedef unsigned long long ulonglong


cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int num, int size,
                int(*compar)(const_void*, const_void*)) nogil



cdef struct MST_triple:
    ulonglong i1
    ulonglong i2
    double w


cdef int MST_triple_comparer(const_void* _a, const_void* _b):
    cdef MST_triple a = (<MST_triple*>_a)[0]
    cdef MST_triple b = (<MST_triple*>_b)[0]
    if a.w < b.w:
        return -1
    elif a.w > b.w:
        return 1
    elif a.i1 != b.i1:
        return a.i1-b.i1
    else:
        return a.i2-b.i2


cpdef np.ndarray[ulonglong,ndim=2] MST(double[:,:] D):
    """
    A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of a complete undirected graph
    with weights given by a symmetric n*n matrix.

    (*) Note that there might be multiple minimum trees spanning a given graph.


    References:
    ----------

    M. Gagolewski, M. Bartoszuk, A. Cena,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363 (2016) 8–23.

    V. Jarník, O jistém problému minimálním,
    Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Comput. 21 (1995) 1313–1325.

    R. Prim, Shortest connection networks and some generalizations,
    Bell Syst. Tech. J. 36 (1957) 1389–1401.


    Parameters:
    ----------

    D : ndarray, shape (n,n)
        Edges' weights.
        It is assumed that D[i,j] == D[j,i] for all i != j.


    Returns:
    -------

    I : ndarray, shape (n-1,2)
        An (n-1)*2 matrix I such that {I[i,0], I[i,1]}
        defines the i-th edge of the resulting MST, I[i,0] < I[i,1].

    """
    if not D.shape[0] == D.shape[1]:
        raise ValueError("D must be a square matrix")
    cdef ulonglong n = D.shape[0] # D is a square matrix
    cdef ulonglong i, j
    cdef np.ndarray[ulonglong,ndim=2] I = np.empty((n-1, 2), dtype=np.ulonglong)

    cpdef double*     Dnn = <double*> PyMem_Malloc(n * sizeof(double))
    cpdef ulonglong*  Fnn = <ulonglong*> PyMem_Malloc(n * sizeof(ulonglong))
    cpdef ulonglong*  M   = <ulonglong*> PyMem_Malloc(n * sizeof(ulonglong))
    for i in range(n):
        Dnn[i] = INFINITY
        #Fnn[i] = 0xffffffff
        M[i] = i

    cdef ulonglong lastj = 0, bestj, bestjpos
    for i in range(n-1):
        # M[1], ... M[n-i-1] - points not yet in the MST
        bestjpos = bestj = 0
        for j in range(1, n-i):
            if D[lastj, M[j]] < Dnn[M[j]]:
                Dnn[M[j]] = D[lastj, M[j]]
                Fnn[M[j]] = lastj
            if Dnn[M[j]] < Dnn[bestj]:        # D[0] == INFTY
                bestj = M[j]
                bestjpos = j
        M[bestjpos] = M[n-i-1] # never visit bestj again
        lastj = bestj          # next time, start from bestj
        # and an edge to MST:
        I[i,0], I[i,1] = (Fnn[bestj], bestj) if Fnn[bestj]<bestj else (bestj, Fnn[bestj])

    PyMem_Free(Fnn)
    PyMem_Free(Dnn)
    PyMem_Free(M)

    return I


cpdef tuple MST_pair(double[:,:] D):
    """
    Computes a minimum spanning tree of a complete undirected graph,
    see MST(), and orders its edges w.r.t. increasing weights.


    Parameters:
    ----------

    D : ndarray, shape (n,n)


    Returns:
    -------

    pair : tuple
         A pair (indices_matrix, corresponding weights);
         the results are ordered w.r.t. the weights
         (and then the 1st, and the the 2nd index);
         indices_matrix -- see MST()
    """
    if not D.shape[0] == D.shape[1]:
        raise ValueError("D must be a square matrix")
    cdef np.ndarray[ulonglong,ndim=2] mst_i = MST(D)
    cdef ulonglong n = mst_i.shape[0]+1, i
    cpdef MST_triple* d = <MST_triple*>PyMem_Malloc((n-1) * sizeof(MST_triple))
    for i in range(n-1):
        d[i].i1 = mst_i[i,0]
        d[i].i2 = mst_i[i,1]

    for i in range(n-1):
        d[i].w  = D[d[i].i1, d[i].i2]

    qsort(<void*>(d), n-1, sizeof(MST_triple), MST_triple_comparer)

    for i in range(n-1):
        mst_i[i,0] = d[i].i1
        mst_i[i,1] = d[i].i2

    cdef np.ndarray[double] mst_d = np.empty(n-1, dtype=np.double)
    for i in range(n-1):
        mst_d[i]   = d[i].w

    PyMem_Free(d)

    return mst_i, mst_d




cpdef tuple MST_nn_pair(double[:,::1] dist, ulonglong[:,::1] ind): # [:,::1]==c_contiguous
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

    cdef ulonglong n = dist.shape[0]
    cdef ulonglong n_neighbors = dist.shape[1]
    cdef ulonglong nm = n*n_neighbors

    cdef vector[ulonglong] nn_used  = vector[ulonglong](n, 0)
    cdef vector[ulonglong] arg_dist = vector[ulonglong](nm)
    c_argfuns.Cargsort(arg_dist.data(), &dist[0,0], nm, False)

    cdef ulonglong arg_dist_cur = 0
    cdef ulonglong mst_edge_cur = 0
    cdef np.ndarray[ulonglong,ndim=2] mst_i = np.empty((n-1, 2), dtype=np.ulonglong)
    cdef np.ndarray[double]           mst_d = np.empty(n-1, dtype=np.double)

    cdef ulonglong u, v
    cdef double d

    cdef c_disjoint_sets.CDisjointSets ds = c_disjoint_sets.CDisjointSets(n)

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

