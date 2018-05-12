#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
Minimum Spanning Tree Algorithm
Copyright (C) 2018 Marek.Gagolewski.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


cimport numpy as np
import numpy as np
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from numpy.math cimport INFINITY


cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int num, int size,
                int(*compar)(const_void*, const_void*)) nogil


cdef struct MST_triple:
    np.int_t i1
    np.int_t i2
    np.double_t w


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


cpdef np.ndarray[np.int_t,ndim=2] MST(np.double_t[:,:] D):
    """
    A Prim-like algorithm for determining the MST
    based on a pre-computed pairwise n*n distance matrix
    (defining edge weights of the complete undirected loop-free graph
    with vertices set {0,1,...n-1}), where
    D[i,j] = D[j,i] denotes the distance between point i and j.

    Returns an (n-1)*2 matrix I such that {I[i,0], I[i,1]} gives the
    ith MST edge, I[i,0] < I[i,1].

    References:

    M. Gagolewski, M. Bartoszuk, A. Cena,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363 (2016) 8–23.

    C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Comput. 21 (1995) 1313–1325.

    R. Prim, Shortest connection networks and some generalizations,
    Bell Syst. Tech. J. 36 (1957) 1389–1401.
    """
    cdef np.int_t n = D.shape[0] # D is a square matrix
    cdef np.int_t i, j
    cdef np.ndarray[np.int_t,ndim=2] I = np.empty((n-1, 2), dtype=np.int_)

    cpdef np.double_t* Dnn = <np.double_t*> PyMem_Malloc(n * sizeof(np.double_t))
    cpdef np.int_t*    Fnn = <np.int_t*> PyMem_Malloc(n * sizeof(np.int_t))
    cpdef np.int_t*    M   = <np.int_t*> PyMem_Malloc(n * sizeof(np.int_t))
    for i in range(n):
        Dnn[i] = INFINITY
        Fnn[i] = 0xffffffff
        M[i] = i

    cdef np.int_t lastj = 0, bestj, bestjpos
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
        lastj = bestj       # start from bestj next time
        # and an edge to MST:
        I[i,0], I[i,1] = (Fnn[bestj], bestj) if Fnn[bestj]<bestj else (bestj, Fnn[bestj])

    PyMem_Free(Fnn)
    PyMem_Free(Dnn)
    PyMem_Free(M)

    return I


cpdef tuple MST_pair(np.double_t[:,:] D):
    """
    MST: Return a pair (indices_matrix, corresponding distances);
    the results are ordered w.r.t. the distances (and then the 1st, 2nd index)
    """
    cdef np.ndarray[np.int_t,ndim=2] mst_i = MST(D)
    cdef np.int_t n = mst_i.shape[0], i
    cpdef MST_triple* d = <MST_triple*>PyMem_Malloc(n * sizeof(MST_triple))
    for i in range(n):
        d[i].i1 = mst_i[i,0]
        d[i].i2 = mst_i[i,1]

    for i in range(n):
        d[i].w  = D[d[i].i1, d[i].i2]

    qsort(<void*>(d), n, sizeof(MST_triple), MST_triple_comparer)

    for i in range(n):
        mst_i[i,0] = d[i].i1
        mst_i[i,1] = d[i].i2

    cdef np.ndarray[np.double_t] mst_d = np.empty(n, dtype=np.double)
    for i in range(n):
        mst_d[i]   = d[i].w

    PyMem_Free(d)

    return mst_i, mst_d


#def MST_pair_old(D):
    #"""
    #MST: Return a pair (indices_matrix, corresponding distances);
    #the results are ordered w.r.t. the distances
    #"""
    #cdef object mst_i = MST(D)
    #cdef object mst_d = D[mst_i[:,0], mst_i[:,1]]
    #mst_o = np.argsort(mst_i[:,0]) # sort w.r.t. first index
    #mst_i, mst_d = mst_i[mst_o,:], mst_d[mst_o]
    #mst_o = np.argsort(mst_d, kind="mergesort") # sort w.r.t. dist (stable)
    #return mst_i[mst_o,:], mst_d[mst_o]


#cpdef list MST_list(np.double_t[:,:] D):
    #"""
    #MST: Return a list (index1, index2, D[index1,index2];
    #the results are ordered w.r.t. the distances
    #"""
    #mst_i, mst_d = MST_pair(D)
    #return [ (mst_i[i,0], mst_i[i,1], mst_d[i]) for i in range(len(mst_i)) ]
