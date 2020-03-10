# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""Minimum Spanning Tree Algorithms:
(a) Prim-Jarník's for Complete Undirected Graphs,
(b) Kruskal's for k-NN graphs.

Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
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
from . cimport c_mst
import numpy as np
cimport libc.math


ctypedef fused floatT:
    float
    double


cpdef tuple mst_from_complete(floatT[:,::1] dist): # [:,::1]==c_contiguous
    """A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of a complete undirected graph
    with weights given by a symmetric n*n matrix.

    (*) Note that there might be multiple minimum trees spanning a given graph.


    References:
    ----------

    [1] M. Gagolewski, M. Bartoszuk, A. Cena,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363 (2016) 8–23.

    [2] V. Jarník, O jistém problému minimálním,
    Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    [3] C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Comput. 21 (1995) 1313–1325.

    [4] R. Prim, Shortest connection networks and some generalizations,
    Bell Syst. Tech. J. 36 (1957) 1389–1401.


    Parameters
    ----------

    dist : c_contiguous ndarray, shape (n,n)


    Returns
    -------

    pair : tuple
        A pair (mst_dist, mst_ind) defining the n-1 edges of the MST:
          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;
          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
        (and then the 1st, and the the 2nd index).
        For each i, it holds mst[i,0]<mst[i,1].
    """
    if not dist.shape[0] == dist.shape[1]:
        raise ValueError("D must be a square matrix")

    cdef ssize_t n = dist.shape[0]
    cdef np.ndarray[ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]         mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef c_mst.CDistanceCompletePrecomputed[floatT] dist_complete = \
        c_mst.CDistanceCompletePrecomputed[floatT](&dist[0,0], n)

    c_mst.Cmst_from_complete(<c_mst.CDistance[floatT]*>(&dist_complete),
        n, &mst_dist[0], &mst_ind[0,0])

    return mst_dist, mst_ind




cpdef tuple mst_from_distance(floatT[:,::1] X,
       str metric="euclidean", metric_params=None):
    """A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of X with respect to a given metric
    (distance). Distances are computed on the fly.
    Memory use: O(n).


    References
    ----------

    [1] M. Gagolewski, M. Bartoszuk, A. Cena,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363 (2016) 8–23.

    [2] V. Jarník, O jistém problému minimálním,
    Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    [3] C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Comput. 21 (1995) 1313–1325.

    [4] R. Prim, Shortest connection networks and some generalizations,
    Bell Syst. Tech. J. 36 (1957) 1389–1401.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n,d)
        n data points in a feature space of dimensionality d.
    metric : string
        one of `"euclidean"` (a.k.a. `"l2"`),
        `"manhattan"` (synonyms: `"cityblock"`, `"l1"`), or
        `"cosine"`.
        More metrics/distances might be supported in future versions.
    metric_params : dict, optional (default=None)
        Additional keyword arguments for the metric function, including:
        * `d_core` - core distances for computing the mutual reachability distance


    Returns
    -------

    pair : tuple
        A pair (mst_dist, mst_ind) defining the n-1 edges of the MST:
          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;
          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
        (and then the 1st, and the the 2nd index).
        For each i, it holds mst[i,0]<mst[i,1].
    """
    cdef ssize_t n = X.shape[0]
    cdef ssize_t d = X.shape[1]
    cdef ssize_t i
    cdef np.ndarray[ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]         mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)
    cdef floatT[::1] d_core
    cdef c_mst.CDistance[floatT]* dist = NULL
    cdef c_mst.CDistance[floatT]* dist2 = NULL
    cdef dict metric_params_dict

    if metric == "euclidean" or metric == "l2":
        dist = <c_mst.CDistance[floatT]*>new c_mst.CDistanceEuclidean[floatT](&X[0,0], n, d)
    elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
        dist = <c_mst.CDistance[floatT]*>new c_mst.CDistanceManhattan[floatT](&X[0,0], n, d)
    elif metric == "cosine":
        dist = <c_mst.CDistance[floatT]*>new c_mst.CDistanceCosine[floatT](&X[0,0], n, d)
    else:
        raise NotImplementedError("given `metric` is not supported (yet)")

    if metric_params is not None:
        metric_params_dict = metric_params
        if "d_core" in metric_params_dict:
            d_core = metric_params_dict["d_core"]
            dist2 = dist # must be deleted separately
            dist  = <c_mst.CDistance[floatT]*>new c_mst.CDistanceMutualReachability[floatT](&d_core[0], n, dist2)

    c_mst.Cmst_from_complete(dist, n, &mst_dist[0], &mst_ind[0,0])

    if dist:  del dist
    if dist2: del dist2

    return mst_dist, mst_ind





cpdef tuple mst_from_nn(floatT[:,::1] dist, ssize_t[:,::1] ind,
        bint stop_disconnected=True):
    """Computes a minimum spanning tree(*) of a (<=k)-nearest neighbour graph
    using Kruskal's algorithm, and orders its edges w.r.t. increasing weights.

    Note that in general, the sum of weights in an MST of the (<=k)-nearest
    neighbour graph might be greater than the sum of weights in a minimum
    spanning tree of the complete pairwise distances graph.

    (*) or forest, if the input graph is unconnected. However,
    if stop_disconnected is True, an exception is raised when there is
    no tree spanning a given (<=k)-nn graph.


    Parameters
    ----------

    dist : a c_contiguous ndarray, shape (n,k)
        dist[i,:] is sorted nondecreasingly for all i,
        dist[i,j] gives the weight of the edge {i, ind[i,j]}
    ind : a c_contiguous ndarray, shape (n,k)
        edge definition, interpreted as {i, ind[i,j]}
    stop_disconnected : bool
        raise an exception if the input graph is not connected


    Returns:
    -------

    pair : tuple
        A pair (mst_dist, mst_ind) defining the n-1 edges of the MST:
          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;
          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
          (and then the 1st, and the the 2nd index).
          For each i, it holds mst[i,0]<mst[i,1].

        If stop_disconnected is False, then the weights of the
        last c-1 edges are set to infinity and the corresponding indices
        are set to -1, where c is the number of connected components
        in the resulting minimum spanning forest.
    """
    if not (dist.shape[0] == ind.shape[0] and
            dist.shape[1] == ind.shape[1]):
        raise ValueError("shapes of dist and ind must match")

    cdef ssize_t n = dist.shape[0]
    cdef ssize_t k = dist.shape[1]

    cdef np.ndarray[ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]         mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef int maybe_inexact

    cdef ssize_t n_edges = c_mst.Cmst_from_nn(&dist[0,0], &ind[0,0], n, k,
             &mst_dist[0], &mst_ind[0,0], &maybe_inexact)

    if stop_disconnected and n_edges < n-1:
        raise ValueError("graph is disconnected")

    # TODO use maybe_inexact ...

    return mst_dist, mst_ind
