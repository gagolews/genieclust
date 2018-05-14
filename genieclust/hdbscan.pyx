#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
HDBSCAN* Clustering Algorithm

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
from libc.math cimport fabs
#from sklearn.base import BaseEstimator, ClusterMixin
import scipy.spatial.distance

include "disjoint_sets.pyx"
include "argkmin.pyx"
include "mst.pyx"


cpdef np.ndarray[np.int_t] merge_leaves_with_closets_clusters(tuple mst, np.ndarray[np.int_t] cl):
    """
    A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all noise points with their nearest
    clusters.


    Parameters:
    ----------

    mst : tuple
        See genieclust.mst.MST_pair().

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
    computes the mutual reachability distance w.r.t. a smoothing
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
        A pairwise n*n distance matrix.

    M : int
        A smoothing factor >= 2.


    Returns:
    -------

    R : ndarray, shape (n_samples,n_samples)
        A new distance matrix, giving the mutual reachability distance w.r.t. M.
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
        A 2-column matrix with elements in {0, ..., n-1},
        where n = I.shape[0]+1.


    Returns:
    -------

    d : ndarray, shape(n,)
        An integer array of length I.shape[0]+1.
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




cdef class HDBSCAN(): # (BaseEstimator, ClusterMixin):
    """
    An implementation of the HDBSCAN* Clustering Algorithm,
    that yields a specific number of clusters, and hence
    is not dependent on the original DBSCAN's somehow magical
    parameter eps.

    @TODO@: The current implementation runs in O(n**2) and
    uses O(n**2) memory.


    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi:10.1145/2733381.

    Basically this is the single linkage algorithm that marks all
    leaves in the corresponding minimum spanning tree as noise points.
    The mutual_reachability_distance() function returns a pairwise distance
    matrix that enables to take the smoothing factor M into account.

    Note that for smoothing factor M == 1, you should use the ordinary
    single linkage algorithm, i.e., mark no points as noise.

    The authors of the original manuscript suggest some post-processing
    of the results, as in practice the number of noise points tends
    to be very large. For instance, "cluster boundary points"
    can be merged back with the nearest clusters.

    Another option is just to merge all noise points with their
    nearest clusters, see merge_leaves_with_nearest_clusters().
    This yields a classical n_clusters-partition of a data set (with no notion
    of noise).


    Parameters:
    ----------

    n_clusters : int, default=2
        Number of clusters the data is split into.

    M : int, default=4
        Smoothing factor.

    metric : str or function, default="euclidean"
        See scipy.spatial.distance.pdist()


    Attributes:
    --------

    labels_ : ndarray, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit():
        an integer vector c with c[i] denoting the cluster id
        (in {0, ..., n_clusters-1}) of the i-th object.
        -1 denotes that a noise point.
    """

    cdef np.int_t n_clusters
    cdef np.int_t M
    cdef str metric
    cdef object labels_

    def __cinit__(self,
                  np.int_t n_clusters=2,
                  np.int_t M=4,
                  metric="euclidean"):
        self.n_clusters = n_clusters
        self.M = M
        self.metric = metric
        self.labels_ = None


    cpdef np.ndarray[np.int_t] fit_predict(self, np.double_t[:,:] X, y=None):
        """
        Compute a k-partition and return the predicted labels.

        @TODO@: do not compute the whole distance matrix.
        The current version requires O(n**2) memory.


        Parameters:
        ----------

        X : ndarray, shape (n,d)
            A matrix defining n points in a d-dimensional vector space.

        y : None
            Ignored.


        Returns:
        -------

        labels_ : ndarray, shape (n,)
            Predicted labels, representing a partition of X.
            labels_[i] gives the cluster id of the i-th input point.
        """
        self.fit(X)
        return self.labels_


    cpdef np.ndarray[np.int_t] fit_predict_from_mst(self, tuple mst):
        """
        Compute a k-partition based on a precomputed MST
        (w.r.t. the mutual reachability distance)
        and return the predicted labels.

        This method ignores self.M and self.metric.


        The MST may, for example, be determined as follows:

        mst = genieclust.mst.MST_pair(
            mutual_reachability_distance(
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(X, metric)),
                M)
        )


        Parameters:
        ----------

        mst : tuple
            See genieclust.mst.MST_pair()


        Returns:
        -------

        labels_ : ndarray, shape (n,)
            Predicted labels, representing a partition of X.
            labels_[i] gives the cluster id of the i-th input point.
        """
        self.fit_from_mst(mst)
        return self.labels_


    cpdef fit(self, np.double_t[:,:] X, y=None):
        """
        Compute a k-partition.

        @TODO@: do not compute the whole distance matrix.
        The current version requires O(n**2) memory.


        Parameters:
        ----------

        X : ndarray, shape (n,d)
            A matrix defining n points in a d-dimensional vector space.

        y : None
            Ignored.
        """
        mst = MST_pair(
            mutual_reachability_distance(
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(X, self.metric)),
                self.M)
        )
        self.fit_from_mst(mst)


    cpdef fit_from_mst(self, tuple mst):
        """
        Compute a k-partition based on a precomputed MST.

        This method ignores self.M and self.metric.


        The MST may, for example, be determined as follows:

        mst = genieclust.mst.MST_pair(
            mutual_reachability_distance(
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(X, metric)),
                M)
        )


        Parameters:
        ----------

        mst : tuple
            See genieclust.mst.MST_pair()
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
            if len(ds) == self.n_clusters:
                break
        else:
            raise Exception("The requested number of clusters is too large \
                with this many noise points")

        ret = ds.to_list_normalized()
        first_leaf_cluster_id = ret[first_leaf]
        for i in range(n):
            if   ret[i] == first_leaf_cluster_id: ret[i]  = 0
            elif ret[i] >  first_leaf_cluster_id: ret[i] -= 1

        self.labels_ = ret
