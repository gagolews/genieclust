# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


# Deprecated functions
#
# Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License
# Version 3, 19 November 2007, published by the Free Software Foundation.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License Version 3 for more details.
# You should have received a copy of the License along with this program.
# If not, see <https://www.gnu.org/licenses/>.


cimport cython
cimport numpy as np
import numpy as np

from . cimport c_argfuns

from libcpp.vector cimport vector

ctypedef fused floatT:
    float
    double



#############################################################################
# HDBSCAN* Clustering Algorithm - auxiliary functions (for testing)
#############################################################################


cpdef np.ndarray[floatT] core_distance(np.ndarray[floatT,ndim=2] dist, int M):
    """Given a pairwise distance matrix, computes the "core distance", i.e.,
    the distance of each point to its M-th nearest neighbour.
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
        neighbour. The i-th point's 1st nearest neighbour is the i-th point itself.
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


