#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
HDBSCAN* Clustering Algorithm
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

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np
import numpy as np
from libc.math cimport fabs

include "disjoint_sets.pyx"
include "argkmin.pyx"



cpdef np.ndarray[np.int_t] merge_leaves_with_closets_clusters(tuple mst, np.ndarray[np.int_t] cl):
    """
    Given a k-partition (with noise points included),
    merges all noise points with their nearest
    clusters.

    Parameters:
    ----------

    mst : tuple
        A tuple as returned by MST_pair().

    cl : mdarray, shape (n_samples,)
        An integer vector c with c[i] denoting the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.

    Returns:
    -------

    cl : mdarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {0, ..., k-1}) of the i-th object.
    """
    cl = cl.copy()
    cpdef np.int_t n = cl.shape[0], i, j
    cdef np.ndarray[np.int_t,ndim=2] mst_i = mst[0]
    assert mst_i.shape[0] + 1 == n

    for i in range(n-1):
        assert cl[mst_i[i,0]] >= 0 or cl[mst_i[i,1]] >= 0
        if cl[mst_i[i,0]] < 0:
            cl[mst_i[i,0]] = cl[mst_i[i,1]]
        elif cl[mst_i[i,1]] < 0:
            cl[mst_i[i,1]] = cl[mst_i[i,0]]

    return cl


cpdef np.ndarray[np.double_t,ndim=2] mutual_reachability_distance(np.ndarray[np.double_t,ndim=2] D, np.int_t M):
    """
    Given a pairwise distance matrix,
    computes the so-called mutual reachability distance w.r.t. a smoothing
    factor M >= 2. Note that for M <= 2 the mutual reachability distance
    is equivalent to the original distance measure.

    M == 1 is disallowed here, as in such a case the HDBSCAN* algorithm
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
        A pairwise n*n distance matrix

    M : int
        A smoothing factor >= 2


    Returns:
    -------

    R : ndarray, shape (n_samples,n_samples)
        A new distance matrix D'
    """
    cdef np.int_t n = D.shape[0], i, j
    cdef np.double_t v
    cdef np.double_t* Dcore
    cdef np.ndarray[np.double_t] row

    if M < 2: raise Exception("M < 2")
    if D.shape[1] != n: raise Exception("not a square matrix")
    if M >= n: raise Exception("M >= matrix size")

    cdef np.ndarray[np.double_t,ndim=2] R = D.copy()
    if M > 2:
        Dcore = <np.double_t*>PyMem_Malloc(n*sizeof(np.double_t))
        for i in range(n):
            row = D[i,:]
            j = argkmin(row, M-1)
            Dcore[i] = D[i, j]
        for i in range(0, n-1):
            for j in range(i+1, n):
                v = D[i, j]
                if v < Dcore[i]: v = Dcore[i]
                if v < Dcore[j]: v = Dcore[j]
                R[i, j] = R[j, i] = v

        PyMem_Free(Dcore)

    return R


cpdef np.ndarray[np.int_t] get_tree_node_degrees(np.ndarray[np.int_t,ndim=2] I):
    """
    Given an adjacency list I representing an undirected tree with vertex
    set {0,...,n-1}, return an array d with d[i] denoting
    the degree of the i-th vertex. For instance, d[i]==1 marks a leaf node.

    Parameters:
    ----------

    I : ndarray
        a 2-column matrix with elements in {0, ..., n-1},
        where n = I.shape[0]+1

    Returns:
    -------

    d : ndarray, shape(n,)
        An integer array of length I.shape[0]+1
    """
    cdef np.int_t n = I.shape[0]+1, i
    cdef np.ndarray[np.int_t] d = np.zeros(n, dtype=np.int_)
    for i in range(n-1):
        if I[i,0] < 0 or I[i,0] >= n:
            raise Exception("Detected an element not in {0, ..., n-1}")
        d[I[i,0]] += 1
        if I[i,1] < 0 or I[i,1] >= n:
            raise Exception("Detected an element not in {0, ..., n-1}")
        d[I[i,1]] += 1

    return d


cpdef np.ndarray[np.int_t] HDBSCAN(tuple mst, np.int_t k):
    """
    The HDBSCAN* Clustering Algorithm

    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi: 10.1145/2733381.

    Basically this is the single linkage algorithm that marks all
    MST leaves as noise points. The algorithm finds exactly k clusters.
    The mutual_reachability_distance() function returns a pairwise distance
    matrix that enables to take the `smoothing factor' into account.

    Note that for smoothing factor M == 1, you should use the ordinary
    single linkage algorithm, i.e., mark no points as noise.

    The authors of the original manuscript suggest some post-processing
    of the results, as in practice the number of noise points tends
    to be very large. For instance, "cluster boundary points"
    can be merged back with the nearest clusters.

    Another option is just to merge all `noise points' with their
    nearest clusters, see `merge_leaves_with_nearest_clusters()`.
    This yields a classical k-partition of a data set (with no notion
    of noise).


    Example call:

    >>> HDBSCAN(MST_pair(
        mutual_reachability_distance(
            scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X)),
            M
        )), k)

    Arguments:
    * mst - a tuple as returned by MST_pair()
    * k - the number of clusters to find

    Returns:
    * An integer vector c with c[i] denoting the cluster id (in {0, 1, ..., k})
      of the i-th object. Class 0 denotes the `noise' cluster.
    """
    cdef np.ndarray[np.int_t,ndim=2] mst_i = mst[0]
    cdef np.int_t n = mst_i.shape[0]+1, i, i1, i2
    cdef np.int_t num_leaves = 0, first_leaf = -1, first_leaf_cluster_id
    cdef np.ndarray[np.int_t] ret
    cdef DisjointSets ds = DisjointSets(n)
    cdef np.ndarray[np.int_t] deg = get_tree_node_degrees(mst_i)

    for i in range(n):
        # get the number of leaves, id of the first leaf,
        # and construct the "noise cluster"
        if deg[i] == 1:
            num_leaves += 1
            if num_leaves == 1:
                first_leaf = i
            else:
                ds.union(first_leaf, i)
    assert num_leaves >= 2

    for i in range(n-1):
        i1, i2 = mst_i[i,0], mst_i[i,1]
        if deg[i1] == 1 or deg[i2] == 1:
            # a leaf -> ignore
            continue

        ds.union(i1, i2)
        if len(ds) == k:
            break
    else:
        raise Exception("The requested number of clusters is too large \
            with this many noise points")

    ret = ds.to_list_normalized()
    first_leaf_cluster_id = ret[first_leaf]
    for i in range(n):
        if   ret[i] == first_leaf_cluster_id: ret[i]  = 0
        elif ret[i] >  first_leaf_cluster_id: ret[i] -= 1

    return ret
