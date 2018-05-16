# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Minimum Spanning Tree Algorithm

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
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np
import numpy as np
from libc.math cimport fabs, sqrt
from numpy.math cimport INFINITY
import scipy.spatial.distance
import warnings




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
    A Prim-like algorithm for determining a minimum spanning tree (MST)
    based on a precomputed pairwise n*n distance matrix
    (defining edge weights of the complete undirected loop-free graph
    with vertices set {0,1,...n-1}), where
    D[i,j] = D[j,i] denotes the distance between point i and j.


    @TODO@: write a version of the algorithm that computes
    the pairwise distances (for a range of metrics) on the flight,
    so that the memory use is better than O(n**2). Also,
    use OpenMP to parallelize the inner loop.
    However, we will still need function to compute an MST based
    on the HDBSCAN*'s mutual reachability distance.


    References:
    ----------

    M. Gagolewski, M. Bartoszuk, A. Cena,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363 (2016) 8–23.

    C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Comput. 21 (1995) 1313–1325.

    R. Prim, Shortest connection networks and some generalizations,
    Bell Syst. Tech. J. 36 (1957) 1389–1401.


    Parameters:
    ----------

    D : ndarray, shape (n,n)


    Returns:
    -------

    I : ndarray, shape (n,2)
        An (n-1)*2 matrix I such that {I[i,0], I[i,1]}
        gives the i-th edge of the resulting MST, I[i,0] < I[i,1].

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
    Computes a minimum spanning tree of a given pairwise distance matrix,
    see MST().


    Parameters:
    ----------

    D : ndarray, shape (n,n)


    Returns:
    -------

    pair : tuple
         A pair (indices_matrix, corresponding distances);
         the results are ordered w.r.t. the distances
         (and then the 1st, and the the 2nd index)
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
