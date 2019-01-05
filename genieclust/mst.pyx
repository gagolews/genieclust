# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Minimum Spanning Tree Algorithms:
a. Prim-Jarník's for Complete Undirected Graphs,
b. Kruskal's for k-NN graphs.

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
from . cimport c_mst
import numpy as np
cimport libc.math




cpdef tuple mst_complete(double[:,::1] dist): # [:,::1]==c_contiguous
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

    dist : c_contiguous ndarray, shape (n,n)


    Returns:
    -------

    pair : tuple
        A pair (indices_matrix, corresponding weights) giving
        the n-1 MST edges.
        The results are ordered w.r.t. increasing weights.
        (and then the 1st, and the the 2nd index).
    """
    if not dist.shape[0] == dist.shape[1]:
        raise ValueError("D must be a square matrix")

    cdef ssize_t n = dist.shape[0]
    cdef np.ndarray[ssize_t,ndim=2] mst_i = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[double]         mst_d = np.empty(n-1, dtype=np.double)

    cdef c_mst.CDistanceCompletePrecomputed dist_complete = \
        c_mst.CDistanceCompletePrecomputed(&dist[0,0], n)

    c_mst.Cmst_complete(<c_mst.CDistance*>(&dist_complete), n, &mst_d[0], &mst_i[0,0])

    return mst_i, mst_d



cpdef tuple mst_from_distance(double[:,::1] X,
        metric="euclidean", metric_params=None):
    """
    A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of X with respect to a given metric
    (distance). Distances are computed on the fly.
    Memory use: O(n).


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

    X : c_contiguous ndarray, shape (n,d)
        n data points

    metric : string
        one of `"euclidean"` (a.k.a. `"l2"`),
        `"sqeuclidean"`,
        `"manhattan"` (synonyms: `"cityblock"`, `"l1"`), or
        `"cosine"`.
        More metrics/distances might be supported in future versions.

    metric_params : dict, optional (default=None)
        Additional keyword arguments for the metric function,
        currently ignored.


    Returns:
    -------

    pair : tuple
        A pair (indices_matrix, corresponding weights) giving
        the n-1 MST edges.
        The results are ordered w.r.t. increasing weights.
        (and then the 1st, and the the 2nd index).
    """
    cdef ssize_t n = X.shape[0]
    cdef ssize_t d = X.shape[1]
    cdef ssize_t i
    cdef np.ndarray[ssize_t,ndim=2] mst_i = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[double]         mst_d = np.empty(n-1, dtype=np.double)

    cdef c_mst.CDistance* dist

    if metric in ("euclidean", "l2", "sqeuclidean"):
        # MST w.r.t. L2^2 == MST w.r.t. L2
        dist = <c_mst.CDistance*>new c_mst.CDistanceSquaredEuclidean(&X[0,0], n, d)
        c_mst.Cmst_complete(dist, n, &mst_d[0], &mst_i[0,0])
        if metric != "sqeuclidean":
            for i in range(n-1):
                mst_d[i] = libc.math.sqrt(mst_d[i])
        del dist
    elif metric in ("manhattan", "cityblock", "l1"):
        dist = <c_mst.CDistance*>new c_mst.CDistanceManhattan(&X[0,0], n, d)
        c_mst.Cmst_complete(dist, n, &mst_d[0], &mst_i[0,0])
        del dist
    elif metric in ("cosine",):
        dist = <c_mst.CDistance*>new c_mst.CDistanceCosine(&X[0,0], n, d)
        c_mst.Cmst_complete(dist, n, &mst_d[0], &mst_i[0,0])
        del dist
    else:
        raise NotImplementedError("given `metric` is not supported")

    return mst_i, mst_d





cpdef tuple mst_nn(double[:,::1] dist, ssize_t[:,::1] ind, stop_disconnected=True):
    """
    Computes a minimum spanning tree of a k-Nearest Neighbor Graph
    using Kruskal's algorithm, and orders its edges w.r.t. increasing weights.

    Note that in general, an MST of the k-Nearest Neighbor Graph
    might not be the MST of the complete Pairwise Distances Graph.

    In case of an unconnected graph, an exception is raised.


    Parameters:
    ----------

    dist : a c_contiguous ndarray, shape (n,k)
        dist[i,:] is sorted increasingly for all i,
        dist[i,j] gives the weight of the edge {i, ind[i,j]}

    ind : a c_contiguous ndarray, shape (n,k)
        edge definition, interpreted as {i, ind[i,j]}

    stop_disconnected : bool
        raise exception if given graph is not connected


    Returns:
    -------

    pair : tuple
        A pair (indices_matrix, corresponding weights);
        the results are ordered w.r.t. the weights
        (and then the 1st, and the the 2nd index).
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

    cdef np.ndarray[ssize_t,ndim=2] mst_i = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[double]         mst_d = np.empty(n-1, dtype=np.double)

    cdef ssize_t n_edges = c_mst.Cmst_nn(&dist[0,0], &ind[0,0], n, k,
             &mst_d[0], &mst_i[0,0])

    if stop_disconnected and n_edges < n-1:
        raise ValueError("graph is disconnected")

    return mst_i, mst_d
