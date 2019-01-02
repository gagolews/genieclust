# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
The Genie+ clustering algorithm (with extras)

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
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport fabs, sqrt
from numpy.math cimport INFINITY

from . cimport c_argfuns
from . cimport c_gini_disjoint_sets


from libcpp.vector cimport vector



import numpy as np
import scipy.spatial.distance
import warnings



ctypedef fused intT:
    int
    long
    long long

ctypedef fused T:
    int
    long
    long long
    float
    double


cdef T square(T x):
    return x*x



#############################################################################
# HDBSCAN* Clustering Algorithm - auxiliary functions
#############################################################################


cpdef np.ndarray[int] merge_boundary_points(
            tuple mst,
            np.ndarray[int] cl,
            np.ndarray[double,ndim=2] D,
            np.ndarray[double] Dcore):
    """
    A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all "boundary" noise points with their nearest
    "core" points.


    Parameters:
    ----------

    mst : tuple
        See genieclust.internal.MST_wrt_mutual_reachability_distance()

    cl : ndarray, shape (n_samples,)
        An integer vector c with c[i] denoting the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.

    D : ndarray, shape (n_samples,n_samples)
        A pairwise n*n distance matrix.

    Dcore : ndarray, shape (n_samples,)
        The core distance, see genieclust.internal.core_distance()


    Returns:
    -------

    cl : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {-1, 0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[int] cl2 = np.array(cl, dtype=np.intc)
    cdef ssize_t n = cl2.shape[0], i
    cdef ssize_t j0, j1
    cdef np.ndarray[ssize_t,ndim=2] mst_i = mst[0]
    assert (mst_i.shape[0] + 1) == n

    for i in range(n-1):
        assert cl2[mst_i[i,0]] >= 0 or cl2[mst_i[i,1]] >= 0
        if cl2[mst_i[i,0]] < 0:
            j0, j1 = mst_i[i,0],  mst_i[i,1]
        elif cl2[mst_i[i,1]] < 0:
            j0, j1 = mst_i[i,1],  mst_i[i,0]
        else:
            continue

        if D[j1, j0] <= Dcore[j1]:
            cl2[j0] = cl2[j1]

    return cl2


cpdef np.ndarray[int] merge_leaves_with_nearest_clusters(
            tuple mst,
            np.ndarray[int] cl):
    """
    A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all noise points with their nearest
    clusters.


    Parameters:
    ----------

    mst : tuple
        See genieclust.internal.MST_wrt_mutual_reachability_distance().

    cl : ndarray, shape (n_samples,)
        An integer vector c with c[i] denoting the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.


    Returns:
    -------

    cl : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[int] cl2 = np.array(cl, dtype=np.intc)
    cdef ssize_t n = cl2.shape[0], i
    cdef np.ndarray[ssize_t,ndim=2] mst_i = mst[0]
    assert (mst_i.shape[0] + 1) == n

    for i in range(n-1):
        assert cl2[mst_i[i,0]] >= 0 or cl2[mst_i[i,1]] >= 0
        if cl2[mst_i[i,0]] < 0:
            cl2[mst_i[i,0]] = cl2[mst_i[i,1]]
        elif cl2[mst_i[i,1]] < 0:
            cl2[mst_i[i,1]] = cl2[mst_i[i,0]]

    return cl2


cpdef np.ndarray[double] core_distance(np.ndarray[double,ndim=2] D, ssize_t M):
    """
    Given a pairwise distance matrix, computes the "core distance", i.e.,
    the distance of each point to its M-th nearest neighbor.
    Note that M==1 always yields all the distances equal to 0.0.
    The core distances are needed when computing the mutual reachability
    distance in the HDBSCAN* algorithm.

    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi: 10.1145/2733381.

    The input distance matrix for a given point cloud X may be computed,
    e.g., via a call to
    `scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))`.


    Parameters:
    ----------

    D : ndarray, shape (n_samples,n_samples)
        A pairwise n*n distance matrix.

    M : int
        A smoothing factor >= 1.


    Returns:
    -------

    Dcore : ndarray, shape (n_samples,)
        Dcore[i] gives the distance between the i-th point and its M-th nearest
        neighbor. The i-th point's 1st nearest neighbor is the i-th point itself.
    """
    cdef ssize_t n = D.shape[0], i, j
    cdef double v
    cdef np.ndarray[double] Dcore = np.zeros(n, np.double)
    cdef double[::1] row

    if M < 1: raise ValueError("M < 1")
    if D.shape[1] != n: raise ValueError("not a square matrix")
    if M >= n: raise ValueError("M >= matrix size")

    if M == 1: return Dcore

    cdef vector[ssize_t] buf = vector[ssize_t](M)
    for i in range(n):
        row = D[i,:]
        j = c_argfuns.Cargkmin(&row[0], row.shape[0], M-1, buf.data())
        Dcore[i] = D[i, j]

    return Dcore


cpdef np.ndarray[double,ndim=2] mutual_reachability_distance(
        np.ndarray[double,ndim=2] D,
        ssize_t M
):
    """
    Given a pairwise distance matrix,
    computes the mutual reachability distance w.r.t. a smoothing
    factor M >= 1. Note that for M <= 2 the mutual reachability distance
    is equivalent to the original distance measure.

    Note that M == 1 should not be used, as in such a case the HDBSCAN* algorithm
    reduces to the single linkage clustering.

    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi: 10.1145/2733381.

    The input distance matrix for a given point cloud X
    may be computed, e.g., via a call to
    `scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))`.


    Parameters:
    ----------

    D : ndarray, shape (n_samples,n_samples)
        A pairwise n*n distance matrix.

    M : int
        A smoothing factor >= 1.


    Returns:
    -------

    R : ndarray, shape (n_samples,n_samples)
        A new distance matrix, giving the mutual reachability distance w.r.t. M.
    """
    cdef ssize_t n = D.shape[0], i, j
    cdef double v
    if M < 1: raise ValueError("M < 1")
    if D.shape[1] != n: raise ValueError("not a square matrix")

    cdef np.ndarray[double,ndim=2] R = np.array(D, dtype=np.double)
    cdef np.ndarray[double] Dcore = core_distance(D, M)
    if M > 2:
        for i in range(0, n-1):
            for j in range(i+1, n):
                v = D[i, j]
                if v < Dcore[i]: v = Dcore[i]
                if v < Dcore[j]: v = Dcore[j]
                R[i, j] = R[j, i] = v

    return R


cpdef np.ndarray[ssize_t] get_tree_node_degrees(np.ndarray[ssize_t,ndim=2] I):
    """
    Given an adjacency list I representing an undirected tree with vertex
    set {0,...,n-1}, return an array d with d[i] denoting
    the degree of the i-th vertex. For instance, d[i]==1 marks a leaf node.


    Parameters:
    ----------

    I : ndarray
        A 2-column matrix with elements in {0, ..., n-1},
        where n = I.shape[0]+1.


    Returns:
    -------

    d : ndarray, shape(n,)
        An integer array of length I.shape[0]+1.
    """
    cdef ssize_t n = I.shape[0]+1, i
    cdef np.ndarray[ssize_t] d = np.zeros(n, dtype=np.intp)
    for i in range(n-1):
        if I[i,0] < 0 or I[i,0] >= n:
            raise ValueError("Detected an element not in {0, ..., n-1}")
        d[I[i,0]] += 1
        if I[i,1] < 0 or I[i,1] >= n:
            raise ValueError("Detected an element not in {0, ..., n-1}")
        d[I[i,1]] += 1

    return d



cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int num, int size,
                int(*compar)(const_void*, const_void*)) nogil



cdef struct MST_triple:
    ssize_t i1
    ssize_t i2
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


cpdef tuple MST_wrt_mutual_reachability_distance(double[:,:] D, double[:] Dcore):
    """
    A Jarník (Prim/Dijkstra)-like algorithm for determining
    a minimum spanning tree (MST) based on a precomputed pairwise
    n*n mutual reachability distance matrix DM, where
    DM[i,j] = max{D[i,j], Dcore[i], Dcore[j]} denotes the (augmented)
    distance between point i and j.

    Note that there may be multiple minimum trees spanning a given vertex set.

    @TODO@: write a version of the algorithm that computes
    the pairwise distances (for a range of metrics) on the fly,
    so that the memory use is better than O(n**2). Also,
    use OpenMP to parallelize the inner loop.
    However, we will still need function to compute an MST based
    on the HDBSCAN*'s mutual reachability distance.


    References:
    ----------

    R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi: 10.1145/2733381.

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

    D : ndarray, shape (n_samples,n_samples)
        A pairwise n*n distance matrix.

    Dcore : ndarray, shape (n_samples,)
        The core distance, see genieclust.internal.core_distance()


    Returns:
    -------

    pair : tuple
         A pair (indices_matrix, corresponding distances);
         the results are ordered w.r.t. the distances
         (and then the 1st, and then the 2nd index)
         The indices_matrix is an (n-1)*2 matrix I such that {I[i,0], I[i,1]}
         gives the i-th edge of the resulting MST, I[i,0] < I[i,1].
    """

    cdef ssize_t n = D.shape[0] # D is a square matrix
    cdef ssize_t i, j
    cdef double curd
    cpdef MST_triple* d = <MST_triple*>PyMem_Malloc(n * sizeof(MST_triple))


    cpdef double*  Dnn = <double*> PyMem_Malloc(n * sizeof(double))
    cpdef ssize_t* Fnn = <ssize_t*> PyMem_Malloc(n * sizeof(ssize_t))
    cpdef ssize_t* M   = <ssize_t*> PyMem_Malloc(n * sizeof(ssize_t))
    for i in range(n):
        Dnn[i] = INFINITY
        #Fnn[i] = 0xffffffff
        M[i] = i

    cdef ssize_t lastj = 0, bestj, bestjpos
    for i in range(n-1):
        # M[1], ... M[n-i-1] - points not yet in the MST
        bestjpos = bestj = 0
        for j in range(1, n-i):
            curd = D[lastj, M[j]]
            if curd < Dcore[lastj]: curd = Dcore[lastj]
            if curd < Dcore[M[j]]: curd = Dcore[M[j]]

            if curd < Dnn[M[j]]:
                Dnn[M[j]] = curd
                Fnn[M[j]] = lastj
            if Dnn[M[j]] < Dnn[bestj]:        # D[0] == INFTY
                bestj = M[j]
                bestjpos = j
        M[bestjpos] = M[n-i-1] # never visit bestj again
        lastj = bestj          # start from bestj next time
        # and an edge to MST:
        d[i].i1, d[i].i2 = (Fnn[bestj], bestj) if Fnn[bestj]<bestj else (bestj, Fnn[bestj])
        d[i].w = Dnn[bestj]

    PyMem_Free(Fnn)
    PyMem_Free(Dnn)
    PyMem_Free(M)




    qsort(<void*>(d), n-1, sizeof(MST_triple), MST_triple_comparer)

    cdef np.ndarray[ssize_t,ndim=2] mst_i = np.empty((n-1, 2), dtype=np.intp)
    for i in range(n-1):
        mst_i[i,0] = d[i].i1
        mst_i[i,1] = d[i].i2

    cdef np.ndarray[double] mst_d = np.empty(n-1, dtype=np.double)
    for i in range(n-1):
        mst_d[i]   = d[i].w

    PyMem_Free(d)

    return mst_i, mst_d


#############################################################################
# The Genie+ Clustering Algorithm (internal)
#############################################################################

cpdef np.ndarray[int] genie_from_mst(tuple mst,
                     ssize_t n_clusters=2,
                     double gini_threshold=0.3,
                     bint noise_leaves=False):
    """
    Compute a k-partition based on a precomputed MST.

    The Genie+ Clustering Algorithm (with extensions)

    Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

    A new hierarchical clustering linkage criterion: the Genie algorithm
    links two clusters in such a way that a chosen economic inequity measure
    (here, the Gini index) of the cluster sizes does not increase drastically
    above a given threshold. Benchmarks indicate a high practical
    usefulness of the introduced method: it most often outperforms
    the Ward or average linkage, k-means, spectral clustering,
    DBSCAN, Birch, and others in terms of the clustering
    quality while retaining the single linkage speed.

    This is a new implementation of the O(n sqrt(n))-time version
    of the original algorithm. Additionally, MST leaves can be
    marked as noise points (if `noise_leaves==True`). This is useful,
    if the Genie algorithm is applied on the MST with respect to
    the HDBSCAN-like mutual reachability distance.

    The MST may, for example, be determined as follows:


    If gini_threshold==1.0 and noise_leaves==False, then basically this
    is the single linkage algorithm. Set gini_threshold==1.0 and
    noise_leaves==True to get a HDBSCAN-like behavior (and make sure
    the MST is computed w.r.t. the mutual reachability distance).


    Parameters:
    ----------

    mst : tuple
        See genieclust.internal.MST_wrt_mutual_reachability_distance()

    n_clusters : int, default=2
        Number of clusters the data is split into.

    gini_threshold : float, default=0.3
        The threshold for the Genie correction

    noise_leaves : bool
        Mark leaves as noise

    Returns:
    -------

    labels_ : ndarray, shape (n,)
        Predicted labels, representing a partition of X.
        labels_[i] gives the cluster id of the i-th input point.
        If noise_leaves==True, then label -1 denotes a noise point.
    """
    cdef ssize_t n, i, j, curidx, m, i1, i2, lastm, lastidx, previdx
    cdef ssize_t noise_count

    cdef np.ndarray[ssize_t,ndim=2] mst_i = mst[0]
    cdef np.ndarray[ssize_t] deg = get_tree_node_degrees(mst_i)
    n = mst_i.shape[0]+1

    cdef vector[ssize_t] denoise_index     = vector[ssize_t](n)
    cdef vector[ssize_t] denoise_index_rev = vector[ssize_t](n)


    # Create the non-noise points' translation table (for GiniDisjointSets)
    noise_count = 0
    if noise_leaves:
        j = 0
        for i in range(n):
            if deg[i] == 1: # a leaf
                noise_count += 1
                denoise_index_rev[i] = -1
            else:           # a non-leaf
                denoise_index[j] = i
                denoise_index_rev[i] = j
                j += 1
        assert noise_count >= 2
        assert j + noise_count == n
    else:
        for i in range(n):
            denoise_index[i]     = i
            denoise_index_rev[i] = i

    if n-noise_count-n_clusters <= 0:
        raise RuntimeError("The requested number of clusters is too large \
            with this many detected noise points")

    # When the Genie correction is on, some MST edges will be chosen
    # in a non-consecutive order. An array-based skiplist will speed up
    # searching within the not-yet-consumed edges.
    cdef vector[ssize_t] next_edge = vector[ssize_t](n)
    cdef vector[ssize_t] prev_edge = vector[ssize_t](n)
    if noise_leaves:
        curidx = -1
        lastidx = -1
        for i in range(n-1):
            i1, i2 = mst_i[i,0], mst_i[i,1]
            if deg[i1] > 1 and deg[i2] > 1:
                # a non-leaf:
                if curidx < 0:
                    curidx = i # the first non-leaf edge
                    prev_edge[i] = -1
                else:
                    next_edge[lastidx] = i
                    prev_edge[i] = lastidx
                lastidx = i

        next_edge[lastidx] = n-1
        lastidx = curidx # first non-leaf
    else:
        curidx  = 0
        lastidx = 0
        for i in range(n-1):
            next_edge[i] = i+1
            prev_edge[i] = i-1

    cdef c_gini_disjoint_sets.CGiniDisjointSets ds = \
        c_gini_disjoint_sets.CGiniDisjointSets(n-noise_count)

    lastm = 0 # last minimal cluster size
    for i in range(n-noise_count-n_clusters):
        if ds.get_gini() > gini_threshold:
            m = ds.get_smallest_count()
            if m != lastm or lastidx < curidx:
                lastidx = curidx
            assert 0 <= lastidx < n-1

            while ds.get_count(denoise_index_rev[mst_i[lastidx,0]]) != m and \
                  ds.get_count(denoise_index_rev[mst_i[lastidx,1]]) != m:
                lastidx = next_edge[lastidx]
                assert 0 <= lastidx < n-1

            i1, i2 = mst_i[lastidx,0], mst_i[lastidx,1]

            assert lastidx >= curidx
            if lastidx == curidx:
                curidx = next_edge[curidx]
                lastidx = curidx
            else:
                previdx = prev_edge[lastidx]
                lastidx = next_edge[lastidx]
                assert 0 <= previdx
                assert previdx < lastidx
                assert lastidx < n
                next_edge[previdx] = lastidx
                prev_edge[lastidx] = previdx
            lastm = m

        else: # single linkage-like
            assert 0 <= curidx < n-1
            i1, i2 = mst_i[curidx,0], mst_i[curidx,1]
            curidx = next_edge[curidx]

        ds.merge(denoise_index_rev[i1], denoise_index_rev[i2])




    cdef np.ndarray[int] res = np.empty(n, dtype=np.intc)
    cdef vector[int] res_cluster_id = vector[int](n)
    for i in range(n): res_cluster_id[i] = -1
    cdef int c = 0
    for i in range(n):
        if denoise_index_rev[i] >= 0:
            # a non-noise point
            j = denoise_index[ds.find(denoise_index_rev[i])]
            assert 0 <= j < n
            if res_cluster_id[j] < 0:
                res_cluster_id[j] = c
                c += 1
            res[i] = res_cluster_id[j]
        else:
            res[i] = -1

    return res
