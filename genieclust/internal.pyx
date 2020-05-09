# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3




"""
Internal functions

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
import numpy as np


cimport libc.math
from libcpp cimport bool
from libcpp.vector cimport vector



ctypedef fused T:
    int
    long
    long long
    ssize_t
    float
    double

ctypedef fused floatT:
    float
    double




from . cimport c_mst
from . cimport c_preprocess
from . cimport c_postprocess
from . cimport c_disjoint_sets
from . cimport c_gini_disjoint_sets
from . cimport c_genie





################################################################################
# Minimum Spanning Tree Algorithms:
# (a) Prim-Jarník's for Complete Undirected Graphs,
# (b) Kruskal's for k-NN graphs.
################################################################################





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
       str metric="euclidean", floatT[::1] d_core=None):
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
    d_core : c_contiguous ndarray of length n; optional (default=None)
        core distances for computing the mutual reachability distance


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
    cdef c_mst.CDistance[floatT]* dist = NULL
    cdef c_mst.CDistance[floatT]* dist2 = NULL
    cdef dict metric_params_dict

    if metric == "euclidean" or metric == "l2":
        # get squared(!) Euclidean if d_core is None
        dist = <c_mst.CDistance[floatT]*>new c_mst.CDistanceEuclidean[floatT](&X[0,0], n, d, d_core is None)
    elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
        dist = <c_mst.CDistance[floatT]*>new c_mst.CDistanceManhattan[floatT](&X[0,0], n, d)
    elif metric == "cosine":
        dist = <c_mst.CDistance[floatT]*>new c_mst.CDistanceCosine[floatT](&X[0,0], n, d)
    elif metric == "precomputed":
        dist = <c_mst.CDistance[floatT]*>new c_mst.CDistanceCompletePrecomputed[floatT](&X[0,0], n)
    else:
        raise NotImplementedError("given `metric` is not supported (yet)")

    if d_core is not None:
        dist2 = dist # must be deleted separately
        dist  = <c_mst.CDistance[floatT]*>new c_mst.CDistanceMutualReachability[floatT](&d_core[0], n, dist2)

    c_mst.Cmst_from_complete(dist, n, &mst_dist[0], &mst_ind[0,0])

    if d_core is None and (metric == "euclidean" or metric == "l2"):
        for i in range(n-1):
            mst_dist[i] = libc.math.sqrt(mst_dist[i])

    if dist:  del dist
    if dist2: del dist2

    return mst_dist, mst_ind





cpdef tuple mst_from_nn(floatT[:,::1] dist, ssize_t[:,::1] ind,
        bint stop_disconnected=True,
        bint stop_inexact=False):
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

    cdef bool maybe_inexact

    cdef ssize_t n_edges = c_mst.Cmst_from_nn(&dist[0,0], &ind[0,0], n, k,
             &mst_dist[0], &mst_ind[0,0], &maybe_inexact)

    if stop_disconnected and n_edges < n-1:
        raise ValueError("graph is disconnected")

    if stop_inexact and maybe_inexact:
        raise ValueError("MST maybe inexact")

    return mst_dist, mst_ind






################################################################################
# Graph pre-processing routines
################################################################################





cpdef np.ndarray[ssize_t] get_graph_node_degrees(ssize_t[:,::1] ind, ssize_t n):
    """Given an adjacency list representing an undirected simple graph over
    vertex set {0,...,n-1}, return an array deg with deg[i] denoting
    the degree of the i-th vertex. For instance, deg[i]==1 marks a leaf node.


    Parameters
    ----------

    ind : ndarray, shape (m,2)
        A 2-column matrix such that {ind[i,0], ind[i,1]} represents
        one of m undirected edges. Negative indices are ignored.
    n : int
        Number of vertices.


    Returns
    -------

    deg : ndarray, shape(n,)
        An integer array of length n.
    """
    cdef ssize_t num_edges = ind.shape[0]
    assert ind.shape[1] == 2
    cdef np.ndarray[ssize_t] deg = np.empty(n, dtype=np.intp)

    c_preprocess.Cget_graph_node_degrees(&ind[0,0], num_edges, n, &deg[0])

    return deg







################################################################################
# Noisy k-partition and other post-processing routines
################################################################################


cpdef np.ndarray[ssize_t] merge_boundary_points(
        ssize_t[:,::1] mst_i,
        ssize_t[::1] c,
        ssize_t[:,::1] nn_i,
        ssize_t M):
    """A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all "boundary" noise points with their nearest
    "core" points.


    Parameters
    ----------

    mst_i : c_contiguous array
        See genieclust.mst.mst_from_distance()
    c : c_contiguous array of shape (n_samples,)
        c[i] gives the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.
    nn_i : c_contiguous matrix of shape (n_samples,n_neighbors)
        nn_ind[i,:] gives the indices of the i-th point's
        nearest neighbours.
    M : int
        smoothing factor, M>=2


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {-1, 0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[ssize_t] cl2 = np.array(c, dtype=np.intp)

    c_postprocess.Cmerge_boundary_points(
        &mst_i[0,0], mst_i.shape[0],
        &nn_i[0,0], nn_i.shape[1], M,
        &cl2[0], cl2.shape[0])

    return cl2


cpdef np.ndarray[ssize_t] merge_noise_points(
        ssize_t[:,::1] mst_i,
        ssize_t[::1] c):
    """A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all noise points with their nearest
    clusters.


    Parameters
    ----------

    mst_i : c_contiguous array
        See genieclust.mst.mst_from_distance()
    c : c_contiguous array of shape (n_samples,)
        c[i] gives the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[ssize_t] cl2 = np.array(c, dtype=np.intp)

    c_postprocess.Cmerge_noise_points(
        &mst_i[0,0], mst_i.shape[0],
        &cl2[0], cl2.shape[0])

    return cl2


cpdef dict get_linkage_matrix(ssize_t[::1] links,
                              floatT[::1] mst_d,
                              ssize_t[:,::1] mst_i):
    """

    Parameters
    ----------

    links : ndarray
        see return value of genieclust.internal.genie_from_mst.
    mst_d, mst_i : ndarray
        Minimal spanning tree defined by a pair (mst_i, mst_d),
        see genieclust.mst.


    Returns
    -------

    Z : dict
        A dictionary with 3 keys: children, distances, counts,
        see the description of Z[:,:2], Z[:,2] and Z[:,3], respectively,
        in scipy.cluster.hierarchy.linkage.
    """
    cdef ssize_t n = mst_i.shape[0]+1
    cdef ssize_t i, i1, i2, par, w, num_unused, j

    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")
    if not n-1 == links.shape[0]:
        raise ValueError("ill-defined MST")

    cdef c_gini_disjoint_sets.CGiniDisjointSets ds = \
        c_gini_disjoint_sets.CGiniDisjointSets(n)

    cdef np.ndarray[ssize_t,ndim=2] children_  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]         distances_ = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)
    cdef np.ndarray[ssize_t]        counts_    = np.empty(n-1, dtype=np.intp)
    cdef np.ndarray[ssize_t]        used       = np.zeros(n-1, dtype=np.intp)
    cdef np.ndarray[ssize_t]        ids        = np.empty(n, dtype=np.intp)

    num_unused = n-1
    for i in range(n-1):
        if links[i] < 0: break # no more mst edges
        if links[i] >= n-1: raise ValueError("ill-defined links")
        used[links[i]] += 1
        num_unused -= 1

    for i in range(n):
        ids[i] = i

    w = -1
    for i in range(n-1):
        if i < num_unused:
            # get the next unused edge (links a leaf node)
            while True:
                w += 1
                assert w < n-1
                if not used[w]: break
        else:
            assert 0 <= i-num_unused < n-1
            w = links[i-num_unused]

        assert 0 <= w < n-1
        i1 = mst_i[w, 0]
        i2 = mst_i[w, 1]
        if not 0 <= i1 < n: raise ValueError("ill-defined MST")
        if not 0 <= i2 < n: raise ValueError("ill-defined MST")

        i1 = ds.find(i1)
        i2 = ds.find(i2)
        children_[i, 0] = ids[i1]
        children_[i, 1] = ids[i2]
        par = ds.merge(i1, i2)
        ids[par] = n+i # see scipy.cluster.hierarchy.linkage
        distances_[i] = mst_d[w] if i >= num_unused else 0.0
        counts_[i] = ds.get_count(par)


    return dict(
        children=children_,
        distances=distances_,
        counts=counts_
    )


################################################################################
# Augmented DisjointSets
################################################################################



cdef class DisjointSets:
    """
    Disjoint Sets (Union-Find)

    Represents a partition of the set {0,1,...,n-1}

    Path compression for find() is implemented,
    but the union() operation is naive (neither
    it is union by rank nor by size),
    see https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
    This is by design, as some other operations in the current
    package rely on the assumption that the parent id of each
    element is always <= than itself.


    Parameters:
    ----------

    n : ssize_t
        The cardinality of the set whose partitions are generated.
    """
    cdef c_disjoint_sets.CDisjointSets ds

    def __cinit__(self, ssize_t n):
        self.ds = c_disjoint_sets.CDisjointSets(n)

    def __len__(self):
        """
        Returns the number of subsets-1,
        a.k.a. how many calls to union() can we still perform.

        Returns:
        -------

        len : ssize_t
            A value in {0,...,n-1}.
        """
        return self.ds.get_k()-1


    cpdef ssize_t get_n(self):
        """
        Returns the number of elements in the set being partitioned.
        """
        return self.ds.get_n()


    cpdef ssize_t get_k(self):
        """
        Returns the current number of subsets.
        """
        return self.ds.get_k()


    cpdef ssize_t find(self, ssize_t x):
        """
        Finds the subset id for a given x.


        Parameters:
        ----------

        x : ssize_t
            An integer in {0,...,n-1}, representing an element to find.


        Returns:
        -------

        parent_x : ssize_t
            The id of the parent of x.
        """
        return self.ds.find(x)


    cpdef ssize_t union(self, ssize_t x, ssize_t y):
        """
        Merges the sets containing given x and y.

        Let px be the parent id of x, and py be the parent id of y.
        If px < py, then the new parent id of py will be set to py.
        Otherwise, px will have py as its parent.

        If x and y are already members of the same subset,
        an exception is thrown.

        Parameters:
        ----------

        x, y : ssize_t
            Integers in {0,...,n-1}, representing elements
            of two sets to merge.


        Returns:
        -------

        parent : ssize_t
            The id of the parent of x or y, whichever is smaller.
        """

        return self.ds.merge(x, y)


    cpdef np.ndarray[ssize_t] to_list(self):
        """
        Get parent ids of all the elements


        Returns:
        -------

        parents : ndarray, shape (n,)
            A list m such that m[x] denotes the (recursive) parent id of x,
            for x=0,1,...,n.
        """
        cdef ssize_t i
        cdef np.ndarray[ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        for i in range(self.ds.get_n()):
            m[i] = self.ds.find(i)
        return m


    cpdef np.ndarray[ssize_t] to_list_normalized(self):
        """
        Get the normalised elements' membership information.


        Returns:
        -------

        set_ids : ndarray, shape (n,)
            A list m such that m[x] denotes the normalised parent id of x.
            The resulting values are in {0,1,...,k-1}, where k is the current
            number of subsets in the partition.
        """
        cdef ssize_t i, j
        cdef np.ndarray[ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        cdef np.ndarray[ssize_t] v = np.zeros(self.ds.get_n(), dtype=np.intp)
        cdef ssize_t c = 1
        for i in range(self.ds.get_n()):
            j = self.ds.find(i)
            if v[j] == 0:
                v[j] = c
                c += 1
            m[i] = v[j]-1
        return m


    def to_lists(self):
        """
        Returns a list of lists representing the current partition.
        This is a slow operation. Do you really need this?


        Returns:
        -------

        partition : list of lists
            A list of length k, where k is the current number
            of sets in a partition. Each list element is a list
            with values in {0,...,n-1}
        """
        cdef ssize_t i
        cdef list tou, out

        tou = [ [] for i in range(self.ds.get_n()) ]
        for i in range(self.ds.get_n()):
            tou[self.ds.find(i)].append(i)

        out = []
        for i in range(self.ds.get_n()):
            if tou[i]: out.append(tou[i])

        return out


    def __repr__(self):
        """
        Calls self.to_lists()
        """
        return "DisjointSets("+repr(self.to_lists())+")"





################################################################################
# Disjoint Sets (Union-Find)
# A Python class to represent partitions of the set {0,1,...,n-1} for any n
################################################################################




cdef class GiniDisjointSets():
    """
    Augmented disjoint sets (Union-Find) over {0,1,...,n-1}.
    The class allows to compute the normalised Gini index of the
    distribution of subset sizes, i.e.,
    $$
        G(x_1,\dots,x_k) = \frac{
        \sum_{i=1}^{n-1} \sum_{j=i+1}^n |x_i-x_j|
        }{
        (n-1) \sum_{i=1}^n x_i
        }.
    $$

    For a use case, see: Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003


    Parameters:
    ----------

    n : ssize_t
        The cardinality of the set whose partitions are generated.
    """
    cdef c_gini_disjoint_sets.CGiniDisjointSets ds

    def __cinit__(self, ssize_t n):
        self.ds = c_gini_disjoint_sets.CGiniDisjointSets(n)

    def __len__(self):
        """
        Returns the number of subsets-1,
        a.k.a. how many calls to union() can we still perform.

        Returns:
        -------

        len : ssize_t
            A value in {0,...,n-1}.
        """
        return self.ds.get_k()-1


    cpdef ssize_t get_n(self):
        """
        Returns the number of elements in the set being partitioned.
        """
        return self.ds.get_n()


    cpdef ssize_t get_k(self):
        """
        Returns the current number of subsets.
        """
        return self.ds.get_k()


    cpdef double get_gini(self):
        """
        Returns the Gini index of the distribution of subsets' sizes.

        Run time: O(1), as the Gini index is updated during a call
        to union().
        """
        return self.ds.get_gini()


    cpdef ssize_t get_count(self, ssize_t x):
        """
        Returns the size of the subset containing x.

        Run time: the cost of find(x)
        """
        return self.ds.get_count(x)


    cpdef ssize_t get_smallest_count(self):
        """
        Returns the size of the smallest subset.

        Run time: O(1)
        """
        return self.ds.get_smallest_count()


    cpdef ssize_t find(self, ssize_t x):
        """
        Finds the subset id for a given x.


        Parameters:
        ----------

        x : int
            An integer in {0,...,n-1}, representing an element to find.


        Returns:
        -------

        parent_x : int
            The id of the parent of x.
        """
        return self.ds.find(x)


    cpdef ssize_t union(self, ssize_t x, ssize_t y):
        """
        Merges the sets containing given x and y.

        Let px be the parent id of x, and py be the parent id of y.
        If px < py, then the new parent id of py will be set to py.
        Otherwise, px will have py as its parent.

        If x and y are already members of the same subset,
        an exception is thrown.

        Update time: pessimistically O(sqrt(n)).


        Parameters:
        ----------

        x, y : int
            Integers in {0,...,n-1}, representing elements
            of two sets to merge.


        Returns:
        -------

        parent : int
            The id of the parent of x or y, whichever is smaller.
        """

        return self.ds.merge(x, y)



    cpdef np.ndarray[ssize_t] to_list(self):
        """
        Get parent ids of all the elements


        Returns:
        -------

        parents : ndarray, shape (n,)
            A list m such that m[x] denotes the (recursive) parent id of x,
            for x=0,1,...,n.
        """
        cdef ssize_t i
        cdef np.ndarray[ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        for i in range(self.ds.get_n()):
            m[i] = self.ds.find(i)
        return m


    cpdef np.ndarray[ssize_t] to_list_normalized(self):
        """
        Get the normalized elements' membership information.


        Returns:
        -------

        set_ids : ndarray, shape (n,)
            A list m such that m[x] denotes the normalised parent id of x.
            The resulting values are in {0,1,...,k-1}, where k is the current
            number of subsets in the partition.
        """
        cdef ssize_t i, j
        cdef np.ndarray[ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        cdef np.ndarray[ssize_t] v = np.zeros(self.ds.get_n(), dtype=np.intp)
        cdef ssize_t c = 1
        for i in range(self.ds.get_n()):
            j = self.ds.find(i)
            if v[j] == 0:
                v[j] = c
                c += 1
            m[i] = v[j]-1
        return m


    def to_lists(self):
        """
        Returns a list of lists representing the current partition.
        This is a slow operation. Do you really need this?

        Returns:
        -------

        partition : list of lists
            A list of length k, where k is the current number
            of sets in a partition. Each list element is a list
            with values in {0,...,n-1}
        """
        cdef ssize_t i
        cdef list tou, out

        tou = [ [] for i in range(self.ds.get_n()) ]
        for i in range(self.ds.get_n()):
            tou[self.ds.find(i)].append(i)

        out = []
        for i in range(self.ds.get_n()):
            if tou[i]: out.append(tou[i])

        return out


    def get_counts(self):
        """
        Generates an array of subsets' sizes.
        The resulting vector is ordered nondecreasingly.

        Run time: O(k), where k is the current number of subsets.
        """
        cdef ssize_t k = self.ds.get_k()
        cdef np.ndarray[ssize_t] out = np.empty(k, dtype=np.intp)
        self.ds.get_counts(&out[0])
        return out


    def __repr__(self):
        """
        Calls self.to_lists()
        """
        return "GiniDisjointSets("+repr(self.to_lists())+")"




#############################################################################
# The Genie+ Clustering Algorithm (internal)
#############################################################################

cpdef dict genie_from_mst(
        floatT[::1] mst_d,
        ssize_t[:,::1] mst_i,
        ssize_t n_clusters=1,
        double gini_threshold=0.3,
        bint noise_leaves=False,
        bint compute_full_tree=True,
        bint compute_all_cuts=False):
    """Compute a k-partition based on a precomputed MST.

    The Genie+ Clustering Algorithm (with extensions)

    Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

    A new hierarchical clustering linkage criterion: the Genie algorithm
    links two clusters in such a way that a chosen economic inequity measure
    (here, the Gini index) of the cluster sizes does not increase drastically
    above a given threshold. The introduced method most often outperforms
    the Ward or average linkage, k-means, spectral clustering,
    DBSCAN, Birch, and others in terms of the clustering
    quality on benchmark data while retaining the single linkage speed.

    This is a new implementation of the O(n sqrt(n))-time version
    of the original algorithm. Additionally, MST leaves can be
    marked as noise points (if `noise_leaves==True`). This is useful,
    if the Genie algorithm is applied on the MST with respect to
    the HDBSCAN-like mutual reachability distance.


    gini_threshold==1.0 and noise_leaves==False, gives the single linkage
    algorithm. Set gini_threshold==1.0 and noise_leaves==True to get
    a HDBSCAN-like behaviour (and make sure
    the MST is computed w.r.t. the mutual reachability distance).


    Parameters
    ----------

    mst_d, mst_i : ndarray
        Minimal spanning tree defined by a pair (mst_i, mst_d),
        see genieclust.mst.
    n_clusters : int, default=1
        Number of clusters the dataset is split into.
        Use n_clusters==1 to generate the complete hierarchy,
        see also `compute_full_tree`.
    gini_threshold : float, default=0.3
        The threshold for the Genie correction
    noise_leaves : bool
        Mark leaves as noise;
        Prevents forming singleton-clusters.
    compute_full_tree : bool
        Compute the whole merge sequence or stop early?
    compute_all_cuts : bool
        Compute the n_clusters and all the more coarse-grained ones?


    Returns
    -------

    res : dict, with the following elements:
        labels : ndarray, shape (n,) or (n_clusters, n) or None
            Is None if n_clusters==0.

            If compute_all_cuts==False, this gives the predicted labels,
            representing an n_clusters-partition of X.
            labels[i] gives the cluster id of the i-th input point.
            If noise_leaves==True, then label -1 denotes a noise point.

            If compute_all_cuts==True, then
            labels[i,:] gives the (i+1)-partition, i=0,...,n_cluster-1.

        links : ndarray, shape (n-1,)
            links[i] gives the MST edge merged at the i-th iteration
            of the algorithm.

        iters : int
            number of merge steps performed

        n_cluster : integer
            actual number of clusters found, 0 if labels is None

    """
    cdef ssize_t n = mst_i.shape[0]+1

    if not 0 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")
    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")
    if not 0.0 <= gini_threshold <= 1.0:
        raise ValueError("incorrect gini_threshold")


    cdef np.ndarray[ssize_t] tmp_labels_1
    cdef np.ndarray[ssize_t,ndim=2] tmp_labels_2

    cdef np.ndarray[ssize_t] links_  = np.empty(n-1, dtype=np.intp)
    cdef ssize_t n_clusters_ = 0, iters_
    labels_ = None # on request, see below

    cdef c_genie.CGenie[floatT] g
    g = c_genie.CGenie[floatT](&mst_d[0], &mst_i[0,0], n, noise_leaves)

    g.apply_genie(1 if compute_full_tree else n_clusters, gini_threshold)

    iters_ = g.get_links(&links_[0])

    if n_clusters >= 1:
        if g.get_max_n_clusters() < n_clusters:
            n_clusters_ = g.get_max_n_clusters()
        else:
            n_clusters_ = n_clusters

        if compute_all_cuts:
            tmp_labels_2 = np.empty((n_clusters_, n), dtype=np.intp)
            g.get_labels_matrix(n_clusters_, &tmp_labels_2[0,0])
            labels_ = tmp_labels_2
        else:
            # just one cut:
            tmp_labels_1 = np.empty(n, dtype=np.intp)
            g.get_labels(n_clusters_, &tmp_labels_1[0])
            labels_ = tmp_labels_1

    return dict(labels=labels_,
                n_clusters=n_clusters_,
                links=links_,
                iters=iters_)





#############################################################################
# The Genie+Information Criterion (G+IC) Clustering Algorithm
# (experimental, under construction)
#############################################################################


cpdef dict gic_from_mst(
        floatT[::1] mst_d,
        ssize_t[:,::1] mst_i,
        double n_features,
        ssize_t n_clusters=1,
        ssize_t add_clusters=0,
        double[::1] gini_thresholds=None,
        bint noise_leaves=False,
        bint compute_full_tree=True,
        bint compute_all_cuts=False):
    """GIc (Genie+Information Criterion) Information-Theoretic
    Hierarchical Clustering Algorithm

    Computes a k-partition based on a pre-computed MST
    maximising (heuristically) the information criterion [2].

    GIc has been proposed by Anna Cena in [1] and was inspired
    by Mueller's (et al.) ITM [2] and Gagolewski's (et al.) Genie [3]

    GIc uses a bottom-up, agglomerative approach (as opposed to the ITM,
    which follows a divisive scheme). It greedily selects for merging
    a pair of clusters that maximises the information criterion [2].
    By default, the initial partition is determined by considering
    the intersection of clusterings found by the Genie methods with
    thresholds 0.1, 0.3, 0.5 and 0.7.


    References
    ==========

    [1] Cena A., Adaptive hierarchical clustering algorithms based on
    data aggregation methods, PhD Thesis, Systems Research Institute,
    Polish Academy of Sciences 2018.

    [2] Mueller A., Nowozin S., Lampert C.H., Information Theoretic
    Clustering using Minimum Spanning Trees, DAGM-OAGM 2012.

    [3] Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003



    Parameters
    ----------

    mst_d, mst_i : ndarray
        Minimal spanning tree defined by a pair (mst_i, mst_d),
        see genieclust.mst.
    n_features : double
        number of features in the data set
        [can be fractional if you know what you're doing]
    n_clusters : int, default=1
        Number of clusters the dataset is split into.
        Use n_clusters==1 to generate the complete hierarchy,
        see also `compute_full_tree`.
    add_clusters: int, default=0
        Number of additional clusters to work with internally.
    gini_thresholds : ndarray or None for the default
        Gini index thresholds to use when computing the initial
        partition. Multiple runs of the Genie algorithm with different
        thresholds are explored and the intersection of the resulting
        clusterings is taken as the entry point.
        If gini_thresholds is an empty array, `add_clusters`
        is ignored and the procedure starts from a weak clustering
        (singletons), which we call Agglomerative-IC (ICA).
        If gini_thresholds is of length 1 and add_clusters==0,
        then the procedure is equivalent to the classical Genie algorithm.
    noise_leaves : bool
        Mark leaves as noise;
        Prevents forming singleton-clusters.
    compute_full_tree : bool
        Compute the whole merge sequence or stop early?
    compute_all_cuts : bool
        Compute the n_clusters and all the more coarse-grained ones?


    Returns
    -------

    res : dict, with the following elements:
        labels : ndarray, shape (n,) or None
            Predicted labels, representing an n_clusters-partition of X.
            labels[i] gives the cluster id of the i-th input point.
            If noise_leaves==True, then label -1 denotes a noise point.
            Is None if n_clusters==0.

        links : ndarray, shape (n-1,)
            links[i] gives the MST edge merged at the i-th iteration
            of the algorithm.

        iters : int
            number of merge steps performed

        n_cluster : integer
            actual number of clusters found, 0 if labels is None
    """
    cdef ssize_t n = mst_i.shape[0]+1

    if not 0 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")
    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")

    if gini_thresholds is None:
        gini_thresholds = np.r_[0.1, 0.3, 0.5, 0.7]


    cdef np.ndarray[ssize_t] tmp_labels_1
    cdef np.ndarray[ssize_t,ndim=2] tmp_labels_2

    cdef np.ndarray[ssize_t] links_  = np.empty(n-1, dtype=np.intp)
    cdef ssize_t n_clusters_ = 0, iters_
    labels_ = None # on request, see below

    cdef c_genie.CGIc[floatT] g
    g = c_genie.CGIc[floatT](&mst_d[0], &mst_i[0,0], n, noise_leaves)

    g.apply_gic(1 if compute_full_tree else n_clusters,
                n_clusters-1+add_clusters if compute_full_tree else add_clusters,
                n_features,
            &gini_thresholds[0], gini_thresholds.shape[0])

    iters_ = g.get_links(&links_[0])

    if n_clusters >= 1:
        if g.get_max_n_clusters() < n_clusters:
            n_clusters_ = g.get_max_n_clusters()
        else:
            n_clusters_ = n_clusters

        if compute_all_cuts:
            tmp_labels_2 = np.empty((n_clusters_, n), dtype=np.intp)
            g.get_labels_matrix(n_clusters_, &tmp_labels_2[0,0])
            labels_ = tmp_labels_2
        else:
            # just one cut:
            tmp_labels_1 = np.empty(n, dtype=np.intp)
            g.get_labels(n_clusters_, &tmp_labels_1[0])
            labels_ = tmp_labels_1

    return dict(labels=labels_,
                n_clusters=n_clusters_,
                links=links_,
                iters=iters_)
