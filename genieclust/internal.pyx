# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""The Genie+ clustering algorithm (with extras)

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
from numpy.math cimport INFINITY

from . cimport c_argfuns
from . cimport c_gini_disjoint_sets
# from . cimport c_mst

from libcpp.vector cimport vector

import numpy as np


ctypedef fused floatT:
    float
    double

# type convention:
# 1. cluster labels == int (int32, np.intc)
# 2. points == double/float
# 3. indexes == ssize_t (Py_ssize_t, np.intp)
# 4. integer params to cpdef functions -- int


#############################################################################
# HDBSCAN* Clustering Algorithm - auxiliary functions (for testing)
#############################################################################


cpdef np.ndarray[floatT] core_distance(np.ndarray[floatT,ndim=2] dist, int M):
    """Given a pairwise distance matrix, computes the "core distance", i.e.,
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


    Parameters
    ----------

    dist : ndarray, shape (n_samples,n_samples)
        A pairwise n*n distance matrix.
    M : int
        A smoothing factor >= 1.


    Returns
    -------

    d_core : ndarray, shape (n_samples,)
        d_core[i] gives the distance between the i-th point and its M-th nearest
        neighbor. The i-th point's 1st nearest neighbor is the i-th point itself.
    """
    cdef ssize_t n = dist.shape[0], i, j
    cdef floatT v
    cdef np.ndarray[floatT] d_core = np.zeros(n,
        dtype=np.float32 if floatT is float else np.float64)
    cdef floatT[::1] row

    if M < 1: raise ValueError("M < 1")
    if dist.shape[1] != n: raise ValueError("not a square matrix")
    if M >= n: raise ValueError("M >= matrix size")

    if M == 1: return d_core # zeros

    cdef vector[ssize_t] buf = vector[ssize_t](M)
    for i in range(n):
        row = dist[i,:]
        j = c_argfuns.Cargkmin(&row[0], row.shape[0], M-1, buf.data())
        d_core[i] = dist[i, j]

    return d_core


cpdef np.ndarray[floatT,ndim=2] mutual_reachability_distance(
        np.ndarray[floatT,ndim=2] dist,
        np.ndarray[floatT] d_core):
    """Given a pairwise distance matrix,
    computes the mutual reachability distance w.r.t. the given
    core distance vector, see genieclust.internal.core_distance().

    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi: 10.1145/2733381.

    The input distance matrix for a given point cloud X
    may be computed, e.g., via a call to
    `scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))`.


    Parameters
    ----------

    dist : ndarray, shape (n_samples,n_samples)
        A pairwise n*n distance matrix.
    d_core : ndarray, shape (n_samples,)
        See genieclust.internal.core_distance().


    Returns
    -------

    R : ndarray, shape (n_samples,n_samples)
        A new distance matrix, giving the mutual reachability distance.
    """
    cdef ssize_t n = dist.shape[0], i, j
    cdef floatT v
    if dist.shape[1] != n: raise ValueError("not a square matrix")

    cdef np.ndarray[floatT,ndim=2] R = np.array(dist,
        dtype=np.float32 if floatT is float else np.float64)
    for i in range(0, n-1):
        for j in range(i+1, n):
            v = dist[i, j]
            if v < d_core[i]: v = d_core[i]
            if v < d_core[j]: v = d_core[j]
            R[i, j] = R[j, i] = v

    return R



#############################################################################
##### Noisy k-partition post-processing #####################################
#############################################################################

cpdef np.ndarray[int] merge_boundary_points(
        np.ndarray[floatT] mst_d,
        np.ndarray[ssize_t,ndim=2] mst_i,
        np.ndarray[int] cl,
        np.ndarray[ssize_t,ndim=2] nn_i,
        int M):
    """A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all "boundary" noise points with their nearest
    "core" points.


    Parameters
    ----------

    mst_d, mst_i : ndarray
        See genieclust.mst.mst_from_distance()
    cl : ndarray, shape (n_samples,)
        An integer vector c with c[i] denoting the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.
    nn_i : ndarray, shape (n_samples,n_neighbors)
        nn_ind[i,:] gives the indexes of the i'th point's
        nearest neighbors.
    M : int
        smoothing factor, M>=2


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {-1, 0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[int] cl2 = np.array(cl, dtype=np.intc)
    cdef ssize_t n = cl2.shape[0], i, j
    cdef ssize_t j0, j1

    if not (mst_i.shape[0] + 1) == n or not nn_i.shape[0] == n:
        raise ValueError("arrays' shapes do not match")
    if M < 2 or M-2 >= nn_i.shape[1]:
        raise ValueError("incorrect M")

    for i in range(n-1):
        assert cl2[mst_i[i,0]] >= 0 or cl2[mst_i[i,1]] >= 0
        if cl2[mst_i[i,0]] < 0:
            j0, j1 = mst_i[i,0],  mst_i[i,1]
        elif cl2[mst_i[i,1]] < 0:
            j0, j1 = mst_i[i,1],  mst_i[i,0]
        else:
            continue

        assert cl2[j0] <  0  # j0 is marked as a noise point
        assert cl2[j1] >= 0  # j1 is a core point
        # j0 is a boundary point if j0 is among j1's M-1 nearest neighbors
        #if dist[j1, j0] <= d_core[j1]:
        #    cl2[j0] = cl2[j1]

        cl2[j0] = -1
        for j in range(M-1):
            if nn_i[j1,j] == j0:
                cl2[j0] = cl2[j1]

    return cl2


cpdef np.ndarray[int] merge_leaves_with_nearest_clusters(
        np.ndarray[floatT] mst_d,
        np.ndarray[ssize_t,ndim=2] mst_i,
        np.ndarray[int] cl):
    """A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all noise points with their nearest
    clusters.


    Parameters
    ----------

    mst_d, mst_i : ndarray
        See genieclust.internal.MST_wrt_mutual_reachability_distance().
    cl : ndarray, shape (n_samples,)
        An integer vector c with c[i] denoting the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[int] cl2 = np.array(cl, dtype=np.intc)
    cdef ssize_t n = cl2.shape[0], i
    assert (mst_i.shape[0] + 1) == n

    for i in range(n-1):
        assert cl2[mst_i[i,0]] >= 0 or cl2[mst_i[i,1]] >= 0
        if cl2[mst_i[i,0]] < 0:
            cl2[mst_i[i,0]] = cl2[mst_i[i,1]]
        elif cl2[mst_i[i,1]] < 0:
            cl2[mst_i[i,1]] = cl2[mst_i[i,0]]

    return cl2


#############################################################################
#############################################################################
#############################################################################

cpdef np.ndarray[ssize_t] get_graph_node_degrees(
        np.ndarray[ssize_t,ndim=2] ind,
        int n):
    """Given an adjacency list representing an undirected simple graph over
    vertex set {0,...,n-1}, return an array deg with deg[i] denoting
    the degree of the i-th vertex. For instance, deg[i]==1 marks a leaf node.


    Parameters
    ----------

    ind : ndarray, shape (m,2)
        A 2-column matrix such that {ind[i,0], ind[i,1]} represents
        undirected edges. Negative indexes are ignored.
    n : int
        Number of vertices.


    Returns
    -------

    deg : ndarray, shape(n,)
        An integer array of length n.
    """
    cdef ssize_t num_edges = ind.shape[0], i
    assert ind.shape[1] == 2
    cdef np.ndarray[ssize_t] deg = np.zeros(n, dtype=np.intp)
    for i in range(num_edges):
        if ind[i,0] < 0  or ind[i,1] < 0:
            continue # represents a no-edge → ignore
        if ind[i,0] >= n or ind[i,1] >= n:
            raise ValueError("Detected an element not in {0, ..., n-1}")
        if ind[i,0] == ind[i,1]:
            raise ValueError("Self-loops are not allowed")

        deg[ind[i,0]] += 1
        deg[ind[i,1]] += 1

    return deg




#############################################################################
# The Genie+ Clustering Algorithm (internal)
#############################################################################

cpdef np.ndarray[int] genie_from_mst(
        np.ndarray[floatT] mst_d,
        np.ndarray[ssize_t,ndim=2] mst_i,
        ssize_t n_clusters=2,
        double gini_threshold=0.3,
        bint noise_leaves=False):
    """Compute a k-partition based on a precomputed MST.

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


    Parameters
    ----------

    mst_d, mst_i : ndarray
        Minimal spanning tree defined by a pair (mst_i, mst_d),
        see genieclust.mst.
    n_clusters : int, default=2
        Number of clusters the data is split into.
    gini_threshold : float, default=0.3
        The threshold for the Genie correction
    noise_leaves : bool
        Mark leaves as noise;
        Prevents forming singleton-clusters.


    Returns
    -------

    labels_ : ndarray, shape (n,)
        Predicted labels, representing a partition of X.
        labels_[i] gives the cluster id of the i-th input point.
        If noise_leaves==True, then label -1 denotes a noise point.
    """
    cdef ssize_t n, i, j, curidx, m, i1, i2, lastm, lastidx, previdx
    cdef ssize_t noise_count
    n = mst_i.shape[0]+1
    cdef np.ndarray[ssize_t] deg = get_graph_node_degrees(mst_i, n)

    if not 1 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")
    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")
    if not 0.0 <= gini_threshold <= 1.0:
        raise ValueError("incorrect gini_threshold")

    cdef vector[ssize_t] denoise_index     = vector[ssize_t](n)
    cdef vector[ssize_t] denoise_index_rev = vector[ssize_t](n)

    for i in range(1, n-1):
        if not mst_d[i] >= mst_d[i-1]:
            raise ValueError("mst_d unsorted")


    # Create the non-noise points' translation table (for GiniDisjointSets)
    # Also count the number of noise points
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
        # start with a list that skips all edges that lead to noise points
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
        # no noise leaves
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
