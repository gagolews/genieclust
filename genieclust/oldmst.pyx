# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
The "old" (<=2025), slow yet more universal functions to compute
k-nearest neighbours and minimum spanning trees.

See the `quitefastmst` <https://quitefastmst.gagolewski.com/>
package for faster algorithms working in the Euclidean space.

Minimum spanning tree algorithms:
(a) Prim-Jarník's for complete undirected Graphs,
(b) Kruskal's for k-NN graphs (approximate MSTs) [DEPRECATED].
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import numpy as np
cimport numpy as np
np.import_array()
import os
import warnings

cimport libc.math
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport INFINITY


ctypedef fused T:
    int
    long
    long long
    Py_ssize_t
    float
    double

ctypedef fused floatT:
    float
    double




cdef extern from "../src/c_oldmst.h":

    cdef cppclass CDistance[T]:
        pass

    cdef cppclass CDistanceMutualReachability[T]: # inherits from CDistance
        CDistanceMutualReachability()
        CDistanceMutualReachability(const T* d_core, Py_ssize_t n, CDistance[T]* d_pairwise)

    cdef cppclass CDistanceEuclidean[T]: # inherits from CDistance
        CDistanceEuclidean()
        CDistanceEuclidean(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceEuclideanSquared[T]: # inherits from CDistance
        CDistanceEuclideanSquared()
        CDistanceEuclideanSquared(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceManhattan[T]: # inherits from CDistance
        CDistanceManhattan()
        CDistanceManhattan(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceCosine[T]: # inherits from CDistance
        CDistanceCosine()
        CDistanceCosine(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistancePrecomputedMatrix[T]: # inherits from CDistance
        CDistancePrecomputedMatrix()
        CDistancePrecomputedMatrix(T* d, Py_ssize_t n)

    cdef cppclass CDistancePrecomputedVector[T]: # inherits from CDistance
        CDistancePrecomputedVector()
        CDistancePrecomputedVector(T* d, Py_ssize_t n)



    void Cknn_from_complete[T](
        CDistance[T]* D, Py_ssize_t n, Py_ssize_t k,
        T* dist, Py_ssize_t* ind, bint verbose) except +

    void Cmst_from_complete[T](
        CDistance[T]* D, Py_ssize_t n,
        T* mst_dist, Py_ssize_t* mst_ind, bint verbose) except +

    Py_ssize_t Cmst_from_nn[T](  # TODO: remove
        T* dist, Py_ssize_t* ind, const T* d_core, Py_ssize_t n, Py_ssize_t k,
        T* mst_dist, Py_ssize_t* mst_ind, int* maybe_inexact, bint verbose) except +



################################################################################

cpdef tuple mst_from_nn(  # TODO: remove
    floatT[:,::1] dist,
    Py_ssize_t[:,::1] ind,
    floatT[::1] d_core=None,
    bint stop_disconnected=True,
    bint stop_inexact=False,
    bint verbose=False):
    """
    genieclust.oldmst.mst_from_nn(dist, ind, d_core=None, stop_disconnected=True, stop_inexact=False, verbose=False)

    Computes a minimum spanning forest for a (<=k)-nearest neighbour graph [DEPRECATED]


    Parameters
    ----------

    dist : ndarray
        A ``c_contiguous`` `ndarray` of shape (n, k).
        ``dist[i,:]`` is sorted nondecreasingly for all ``i``,
        ``dist[i,j]`` gives the weight of the edge ``{i, ind[i,j]}``
    ind : a ``c_contiguous`` ndarray, shape (n,k)
        Defines edges of the input graph, interpreted as ``{i, ind[i,j]}``.
    d_core : a ``c_contiguous`` ndarray, shape (n,), or ``None``
        Core distances for computing the mutual reachability distance,
        can be ``None``.
    stop_disconnected : bool
        Whether to raise an exception if the input graph is not connected.
    stop_inexact : bool
        Whether to raise an exception if the return MST is definitely
        subobtimal.
    verbose : bool
        Whether to print diagnostic messages.


    Returns
    -------

    tuple like ``(mst_dist, mst_ind)``
        Defines the `n-1` edges of the resulting MST.
        The `(n-1)`-ary array ``mst_dist`` is such that
        ``mst_dist[i]`` gives the weight of the ``i``-th edge.
        Moreover, ``mst_ind`` is a matrix with `n-1` rows and 2 columns,
        where ``{mst_ind[i,0], mst_ind[i,1]}`` defines the ``i``-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
        For each ``i``, it holds ``mst_ind[i,0] < mst_ind[i,1]``.

        If `stop_disconnected` is ``False``, then the weights of the
        last `c-1` edges are set to infinity and the corresponding indexes
        are set to -1, where `c` is the number of connected components
        in the resulting minimum spanning forest.



    See also
    --------

    Kruskal's algorithm is used.

    Note that in general, the sum of weights in an MST of the (<= k)-nearest
    neighbour graph might be greater than the sum of weights in a minimum
    spanning tree of the complete pairwise distances graph.

    If the input graph is not connected, the result is a forest.

    """
    cdef Py_ssize_t n = dist.shape[0]
    cdef Py_ssize_t k = dist.shape[1]

    if not (ind.shape[0] == n and ind.shape[1] == k):
        raise ValueError("shapes of dist and ind must match")

    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]         mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef int maybe_inexact

    cdef floatT* d_core_ptr = NULL
    if d_core is not None:
        if not (d_core.shape[0] == n):
            raise ValueError("shapes of dist and d_core must match")
        d_core_ptr = &d_core[0]

    # _openmp_set_num_threads()
    cdef Py_ssize_t n_edges = Cmst_from_nn(
        &dist[0,0], &ind[0,0],
        d_core_ptr,
        n, k,
        &mst_dist[0], &mst_ind[0,0], &maybe_inexact, verbose)

    if stop_disconnected and n_edges < n-1:
        raise ValueError("graph is disconnected")

    if stop_inexact and maybe_inexact:
        raise ValueError("MST maybe inexact")

    return mst_dist, mst_ind




cpdef tuple mst_from_complete(
    floatT[:,::1] X,
    bint verbose=False): # [:,::1]==c_contiguous
    """
    genieclust.oldmst.mst_from_complete(X, verbose=False)

    A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of a complete undirected graph
    with weights given by a symmetric n*n matrix
    or a distance vector of length n*(n-1)/2.

    The number of threads used is controlled via the
    OMP_NUM_THREADS environment variable or via
    `quitefastmst.omp_set_num_threads` at runtime.

    (*) Note that there might be multiple minimum trees spanning a given graph.


    References
    ----------

    .. [1]
        V. Jarník, O jistém problému minimálním,
        Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    .. [2]
        C.F. Olson, Parallel algorithms for hierarchical clustering,
        Parallel Computing 21(8) (1995) 1313–1325.

    .. [3]
        R. Prim, Shortest connection networks and some generalizations,
        The Bell System Technical Journal 36(6) (1957) 1389–1401.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n*(n-1)/2, 1) or (n,n)
        distance vector or matrix
    verbose: bool
        whether to print diagnostic messages


    Returns
    -------

    pair : tuple
        A pair (mst_dist, mst_ind) defining the n-1 edges of the MST:
          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;
          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
        For each i, it holds mst[i,0]<mst[i,1].
    """
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t n = X.shape[0]
    if d == 1:
        n = <Py_ssize_t>libc.math.round((libc.math.sqrt(1.0+8.0*n)+1.0)/2.0)
        assert n*(n-1)//2 == X.shape[0]

    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]            mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef CDistance[floatT]* D = NULL
    if d == 1:
        D = <CDistance[floatT]*>new CDistancePrecomputedVector[floatT](&X[0,0], n)
    else:
        assert d == n
        D = <CDistance[floatT]*>new CDistancePrecomputedMatrix[floatT](&X[0,0], n)

    # _openmp_set_num_threads()
    Cmst_from_complete(D, n, &mst_dist[0], &mst_ind[0,0], verbose)

    if D:  del D

    return mst_dist, mst_ind




cpdef tuple mst_from_distance(
    floatT[:,::1] X,
    str metric="euclidean",
    floatT[::1] d_core=None,
    bint verbose=False):
    """
    genieclust.oldmst.mst_from_distance(X, metric="euclidean", d_core=None, verbose=False)

    A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of X with respect to a given metric
    (distance). Distances are computed on the fly.
    Memory use: O(n*d).

    The number of threads used is controlled via the
    OMP_NUM_THREADS environment variable or via
    `quitefastmst.omp_set_num_threads` at runtime.

    Possible ties in mutual reachability distances [4]_ are resolved in such
    a way that neighbours with smaller core distances are preferred; see [5]_.


    References
    ----------

    .. [1]
        Jarník V., O jistém problému minimálním,
        Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    .. [2]
        Olson C.F., Parallel algorithms for hierarchical clustering,
        Parallel Computing 21(8) (1995) 1313–1325.

    .. [3]
        Prim R., Shortest connection networks and some generalizations,
        The Bell System Technical Journal 36(6) (1957) 1389–1401.

    .. [4] Campello R.J.G.B., Moulavi D., Sander J.,
        Density-based clustering based on hierarchical density estimates,
        Lecture Notes in Computer Science 7819 (2013) 160-172.

    .. [5]
        Gagolewski M., TODO, 2025


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n,d) or,
            if metric == "precomputed", (n*(n-1)/2,1) or (n,n)
        n data points in a feature space of dimensionality d
        or pairwise distances between n points
    metric : string
        one of ``"euclidean"`` (a.k.a. ``"l2"``),
        ``"manhattan"`` (synonyms: ``"cityblock"``, ``"l1"``),
        ``"cosine"`` (a.k.a. ``"cosinesimil"``), or ``"precomputed"``.
        More metrics/distances might be supported in future versions.
    d_core : c_contiguous ndarray of length n; optional (default=None)
        core distances for computing the mutual reachability distance
    verbose: bool
        whether to print diagnostic messages

    Returns
    -------

    pair : tuple
        A pair (mst_dist, mst_ind) defining the n-1 edges of the MST:
          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;
          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
        For each i, it holds mst[i,0]<mst[i,1].
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    if d == 1 and metric == "precomputed":
        n = <Py_ssize_t>libc.math.round((libc.math.sqrt(1.0+8.0*n)+1.0)/2.0)
        assert n*(n-1)//2 == X.shape[0]
    #cdef Py_ssize_t i
    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT] mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef CDistance[floatT]* D = NULL
    cdef CDistance[floatT]* D2 = NULL

    if metric == "euclidean" or metric == "l2":
        D = <CDistance[floatT]*>new CDistanceEuclidean[floatT](&X[0,0], n, d)
    elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
        D = <CDistance[floatT]*>new CDistanceManhattan[floatT](&X[0,0], n, d)
    elif metric == "cosine" or metric == "cosinesimil":
        D = <CDistance[floatT]*>new CDistanceCosine[floatT](&X[0,0], n, d)
    elif metric == "precomputed":
        if d == 1:
            D = <CDistance[floatT]*>new CDistancePrecomputedVector[floatT](&X[0,0], n)
        else:
            assert d == n
            D = <CDistance[floatT]*>new CDistancePrecomputedMatrix[floatT](&X[0,0], n)
    else:
        raise NotImplementedError("given `metric` is not supported (yet)")

    if d_core is not None:
        D2 = D  # must be deleted separately
        D  = <CDistance[floatT]*>new CDistanceMutualReachability[floatT](&d_core[0], n, D2)

    # _openmp_set_num_threads()
    Cmst_from_complete(D, n, &mst_dist[0], &mst_ind[0,0], verbose)

    if D:  del D
    if D2: del D2

    return mst_dist, mst_ind



cpdef tuple knn_from_distance(floatT[:,::1] X, Py_ssize_t k,
       str metric="euclidean", floatT[::1] d_core=None, bint verbose=False):
    """
    genieclust.oldmst.knn_from_distance(X, k, metric="euclidean", d_core=None, verbose=False)

    Determines the first k nearest neighbours of each point in X,
    with respect to a given metric (distance).
    Distances are computed on the fly.
    Memory use: O(n*k).

    It is assumed that each query point is not its own neighbour.

    The number of threads used is controlled via the
    OMP_NUM_THREADS environment variable or via
    `quitefastmst.omp_set_num_threads` at runtime.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n,d) or,
            if metric == "precomputed", (n*(n-1)/2,1) or (n,n)
        n data points in a feature space of dimensionality d
        or pairwise distances between n points
    k : int < n
        number of nearest neighbours
    metric : string
        one of ``"euclidean"`` (a.k.a. ``"l2"``),
        ``"manhattan"`` (synonyms: ``"cityblock"``, ``"l1"``),
        ``"cosine"`` (a.k.a. ``"cosinesimil"``), or ``"precomputed"``.
        More metrics/distances might be supported in future versions.
    d_core : c_contiguous ndarray of length n; optional (default=None)
        core distances for computing the mutual reachability distance
    verbose: bool
        whether to print diagnostic messages

    Returns
    -------

    pair : tuple
        A pair (dist, ind) representing the k-NN graph, where:
            dist : a c_contiguous ndarray, shape (n,k)
                dist[i,:] is sorted nondecreasingly for all i,
                dist[i,j] gives the weight of the edge {i, ind[i,j]},
                i.e., the distance between the i-th point and its j-th NN.
            ind : a c_contiguous ndarray, shape (n,k)
                edge definition, interpreted as {i, ind[i,j]};
                ind[i,j] is the index of the j-th nearest neighbour of i.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    if d == 1 and metric == "precomputed":
        n = <Py_ssize_t>libc.math.round((libc.math.sqrt(1.0+8.0*n)+1.0)/2.0)
        assert n*(n-1)//2 == X.shape[0]

    if k >= n:
        raise ValueError("too many nearest neighbours requested")

    cdef Py_ssize_t i
    cdef np.ndarray[Py_ssize_t,ndim=2] ind  = np.empty((n, k), dtype=np.intp)
    cdef np.ndarray[floatT,ndim=2]  dist = np.empty((n, k),
        dtype=np.float32 if floatT is float else np.float64)
    cdef CDistance[floatT]* D = NULL
    cdef CDistance[floatT]* D2 = NULL

    if metric == "euclidean" or metric == "l2":
        D = <CDistance[floatT]*>new CDistanceEuclidean[floatT](&X[0,0], n, d)
    elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
        D = <CDistance[floatT]*>new CDistanceManhattan[floatT](&X[0,0], n, d)
    elif metric == "cosine" or metric == "cosinesimil":
        D = <CDistance[floatT]*>new CDistanceCosine[floatT](&X[0,0], n, d)
    elif metric == "precomputed":
        if d == 1:
            D = <CDistance[floatT]*>new CDistancePrecomputedVector[floatT](&X[0,0], n)
        else:
            assert d == n
            D = <CDistance[floatT]*>new CDistancePrecomputedMatrix[floatT](&X[0,0], n)
    else:
        raise NotImplementedError("given `metric` is not supported (yet)")

    if d_core is not None:
        D2 = D # must be deleted separately
        D  = <CDistance[floatT]*>new CDistanceMutualReachability[floatT](&d_core[0], n, D2)

    # _openmp_set_num_threads()
    Cknn_from_complete(D, n, k, &dist[0,0], &ind[0,0], verbose)

    if D:  del D
    if D2: del D2

    return dist, ind

