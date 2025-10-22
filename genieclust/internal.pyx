# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Internal functions and classes
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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
from libcpp.vector cimport vector

cdef extern from "../src/c_argfuns.h":
    Py_ssize_t Cargkmin[T](T* x, Py_ssize_t n, Py_ssize_t k, Py_ssize_t* buf)



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





################################################################################
# DisjointSets (Union-Find)
################################################################################


cdef extern from "../src/c_disjoint_sets.h":
    cdef cppclass CDisjointSets:
        CDisjointSets() except +
        CDisjointSets(Py_ssize_t) except +
        Py_ssize_t get_k()
        Py_ssize_t get_n()
        Py_ssize_t find(Py_ssize_t)
        Py_ssize_t merge(Py_ssize_t, Py_ssize_t)


cdef extern from "../src/c_gini_disjoint_sets.h":
    cdef cppclass CGiniDisjointSets:
        CGiniDisjointSets() except +
        CGiniDisjointSets(Py_ssize_t) except +
        Py_ssize_t get_k()
        Py_ssize_t get_n()
        Py_ssize_t find(Py_ssize_t)
        Py_ssize_t merge(Py_ssize_t, Py_ssize_t)
        double get_gini()
        Py_ssize_t get_smallest_count()
        Py_ssize_t get_count(Py_ssize_t)
        void get_counts(Py_ssize_t*)



cdef class DisjointSets:
    """
    Disjoint Sets (Union-Find)


    Parameters
    ----------

    n : Py_ssize_t
        The cardinality of the set whose partitions are generated.


    Notes
    -----

    Represents a partition of the set :math:`\{0,1,...,n-1\}`
    for some :math:`n`.

    Path compression for `find()` is implemented,
    but the `union()` operation is naive (neither
    it is union by rank nor by size),
    see https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
    This is by design, as some other operations in the current
    package rely on the assumption that the parent ID of each
    element is always ≤ than itself.

    """
    cdef CDisjointSets ds

    def __cinit__(self, Py_ssize_t n):
        self.ds = CDisjointSets(n)

    def __len__(self):
        """
        Returns the number of subsets-1,
        i.e., how many calls to `union()` we can still perform.


        Returns
        -------

        Py_ssize_t
            A value in `{0,...,n-1}`.
        """
        return self.ds.get_k()-1


    cpdef Py_ssize_t get_n(self):
        """
        Returns the number of elements in the set being partitioned
        """
        return self.ds.get_n()


    cpdef Py_ssize_t get_k(self):
        """
        Returns the current number of subsets
        """
        return self.ds.get_k()


    cpdef Py_ssize_t find(self, Py_ssize_t x):
        """
        Finds the subset ID for a given `x`


        Parameters
        ----------

        x : Py_ssize_t
            An integer in `{0,...,n-1}`, representing an element to find.


        Returns
        -------

        Py_ssize_t
            The ID of the parent of `x`.
        """
        return self.ds.find(x)


    cpdef Py_ssize_t union(self, Py_ssize_t x, Py_ssize_t y):
        """
        Merges the sets containing given `x` and `y`



        Parameters
        ----------

        x : Py_ssize_t
            Integer in {0,...,n-1}, representing an element of the first set
            to be merged.

        y : Py_ssize_t
            Integer in {0,...,n-1}, representing an element of the second set
            to be merged.


        Returns
        -------

        parent : Py_ssize_t
            The id of the parent of x or y, whichever is smaller.


        Notes
        -----

        Let `px` be the parent ID of `x`, and `py` be the parent ID of `y`.
        If `px < py`, then the new parent ID of `py` will be set to `py`.
        Otherwise, `px` will have `py` as its parent.

        If `x` and `y` are members of the same subset,
        an exception is thrown.

        """

        return self.ds.merge(x, y)


    cpdef np.ndarray[Py_ssize_t] to_list(self):
        """
        Gets parent IDs of all the elements


        Returns
        -------

        ndarray, shape (n,)
            A list ``m`` such that ``m[x]`` denotes the (recursive) parent ID
            of `x`, for `x = 0,1,...,n-1`.
        """
        cdef Py_ssize_t i
        cdef np.ndarray[Py_ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        for i in range(self.ds.get_n()):
            m[i] = self.ds.find(i)
        return m


    cpdef np.ndarray[Py_ssize_t] to_list_normalized(self):
        """
        Gets the normalised elements' membership information


        Returns
        -------

        ndarray, shape (n,)
            A list ``m`` such that ``m[x]`` denotes the normalised parent ID
            of `x`. The resulting values are in `{0,1,...,k-1}`,
            where `k` is the current number of subsets in the partition.
        """
        cdef Py_ssize_t i, j
        cdef np.ndarray[Py_ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        cdef np.ndarray[Py_ssize_t] v = np.zeros(self.ds.get_n(), dtype=np.intp)
        cdef Py_ssize_t c = 1
        for i in range(self.ds.get_n()):
            j = self.ds.find(i)
            if v[j] == 0:
                v[j] = c
                c += 1
            m[i] = v[j]-1
        return m


    def to_lists(self):
        """
        Returns a list of lists representing the current partition

        Returns
        -------

        list of lists
            A list of length `k`, where `k` is the current number
            of sets in the partition. Each list element is a list
            with values in `{0,...,n-1}`.


        Notes
        -----

        A slow operation. Do you really need it?

        """
        cdef Py_ssize_t i
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
        Calls `self.to_lists()`
        """
        return "DisjointSets("+repr(self.to_lists())+")"






cdef class GiniDisjointSets():
    """
    Disjoint sets (Union-Find) over `{0,1,...,n-1}` with extras.

    Parameters
    ----------

    n : Py_ssize_t
        The cardinality of the set whose partitions are generated.


    Notes
    -----

    The class allows to compute the normalised Gini index of the
    subset sizes, i.e.,
    :math:`G(x_1,\dots,x_k) = \\frac{\\sum_{i=1}^{n-1} \\sum_{j=i+1}^n |x_i-x_j|}{(n-1) \\sum_{i=1}^n x_i}`\ ,
    where :math:`x_i` is the number of elements in the `i`-th subset in
    the current partition.

    For a use case, see: Gagolewski, M., Bartoszuk, M., Cena, A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    *Information Sciences* **363**, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

    """
    cdef CGiniDisjointSets ds

    def __cinit__(self, Py_ssize_t n):
        self.ds = CGiniDisjointSets(n)

    def __len__(self):
        """
        Returns the number of subsets-1,
        i.e., how many calls to `union()` we can still perform.


        Returns
        -------

        Py_ssize_t
            A value in `{0,...,n-1}`.
        """
        return self.ds.get_k()-1


    cpdef Py_ssize_t get_n(self):
        """
        Returns the number of elements in the set being partitioned
        """
        return self.ds.get_n()


    cpdef Py_ssize_t get_k(self):
        """
        Returns the current number of subsets
        """
        return self.ds.get_k()


    cpdef double get_gini(self):
        """
        Returns the Gini index of the distribution of subsets' sizes

        Notes
        -----

        Run time is :math:`O(1)`, as the Gini index is updated during
        each call to `union()`.
        """
        return self.ds.get_gini()


    cpdef Py_ssize_t get_count(self, Py_ssize_t x):
        """
        Returns the size of the subset containing `x`


        Parameters
        ----------

        x : Py_ssize_t
            An integer in `{0,...,n-1}`, representing an element to find.

        Notes
        -----

        Run time: the cost of `find(x)`
        """
        return self.ds.get_count(x)


    cpdef Py_ssize_t get_smallest_count(self):
        """
        Returns the size of the smallest subset


        Notes
        -----

        Run time is `O(1)`.
        """
        return self.ds.get_smallest_count()


    cpdef Py_ssize_t find(self, Py_ssize_t x):
        """
        Finds the subset ID for a given `x`


        Parameters
        ----------

        x : Py_ssize_t
            An integer in `{0,...,n-1}`, representing an element to find.


        Returns
        -------

        Py_ssize_t
            The ID of the parent of `x`.
        """
        return self.ds.find(x)


    cpdef Py_ssize_t union(self, Py_ssize_t x, Py_ssize_t y):
        """
        Merges the sets containing given `x` and `y`



        Parameters
        ----------

        x : Py_ssize_t
            Integer in {0,...,n-1}, representing an element of the first set
            to be merged.

        y : Py_ssize_t
            Integer in {0,...,n-1}, representing an element of the second set
            to be merged.


        Returns
        -------

        parent : Py_ssize_t
            The id of the parent of x or y, whichever is smaller.


        Notes
        -----

        Let `px` be the parent ID of `x`, and `py` be the parent ID of `y`.
        If `px < py`, then the new parent ID of `py` will be set to `py`.
        Otherwise, `px` will have `py` as its parent.

        If `x` and `y` are members of the same subset,
        an exception is thrown.

        Update time: pessimistically :math:`O(\\sqrt{n})`,
        as the Gini index must be recomputed.

        """

        return self.ds.merge(x, y)



    cpdef np.ndarray[Py_ssize_t] to_list(self):
        """
        Gets parent IDs of all the elements


        Returns
        -------

        ndarray, shape (n,)
            A list ``m`` such that ``m[x]`` denotes the (recursive) parent ID
            of `x`, for `x = 0,1,...,n-1`.
        """
        cdef Py_ssize_t i
        cdef np.ndarray[Py_ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        for i in range(self.ds.get_n()):
            m[i] = self.ds.find(i)
        return m


    cpdef np.ndarray[Py_ssize_t] to_list_normalized(self):
        """
        Gets the normalised elements' membership information


        Returns
        -------

        ndarray, shape (n,)
            A list ``m`` such that ``m[x]`` denotes the normalised parent ID
            of `x`. The resulting values are in `{0,1,...,k-1}`,
            where `k` is the current number of subsets in the partition.
        """
        cdef Py_ssize_t i, j
        cdef np.ndarray[Py_ssize_t] m = np.empty(self.ds.get_n(), dtype=np.intp)
        cdef np.ndarray[Py_ssize_t] v = np.zeros(self.ds.get_n(), dtype=np.intp)
        cdef Py_ssize_t c = 1
        for i in range(self.ds.get_n()):
            j = self.ds.find(i)
            if v[j] == 0:
                v[j] = c
                c += 1
            m[i] = v[j]-1
        return m


    def to_lists(self):
        """
        Returns a list of lists representing the current partition



        Returns
        -------

        list of lists
            A list of length `k`, where `k` is the current number
            of sets in the partition. Each list element is a list
            with values in `{0,...,n-1}`.


        Notes
        -----

        A slow operation. Do you really need it?

        """
        cdef Py_ssize_t i
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
        Generates an array of subsets' sizes

        Notes
        -----

        The resulting vector is ordered nondecreasingly.

        Run time is :math:`O(k)`, where `k` is the current number of subsets.
        """
        cdef Py_ssize_t k = self.ds.get_k()
        cdef np.ndarray[Py_ssize_t] out = np.empty(k, dtype=np.intp)
        self.ds.get_counts(&out[0])
        return out


    def __repr__(self):
        """
        Calls `self.to_lists()`
        """
        return "GiniDisjointSets("+repr(self.to_lists())+")"





################################################################################

cpdef np.ndarray[floatT] get_d_core(
    floatT[:,::1] dist,
    Py_ssize_t[:,::1] ind,
    Py_ssize_t M):
    """
    Get "core" distance = distance to the M-th nearest neighbour
    (if available, otherwise, distance to the furthest away one at hand).

    Note that unlike in Campello et al.'s 2013 paper, the definition
    of the core distance does not include the distance to self.


    Parameters
    ----------

    dist : a c_contiguous ndarray, shape (n,k)
        dist[i,:] is sorted nondecreasingly for all i,
        dist[i,j] gives the weight of the edge {i, ind[i,j]}
    ind : a c_contiguous ndarray, shape (n,k)
        edge definition, interpreted as {i, ind[i,j]};
        -1 denotes a "missing value"
    M : int
        "smoothing factor"


    Returns
    -------

    ndarray
        of length dist.shape[0]
    """

    cdef Py_ssize_t n = dist.shape[0]
    cdef Py_ssize_t k = dist.shape[1]

    if not (ind.shape[0] == n and ind.shape[1] == k):
        raise ValueError("shapes of dist and ind must match")

    if M > k:
        raise ValueError("too few nearest neighbours provided")

    cdef np.ndarray[floatT] d_core = np.empty(n,
        dtype=np.float32 if floatT is float else np.float64)

    #Python equivalent if all NNs are available:
    #assert nn_dist.shape[1] >= cur_state["M"]
    #d_core = nn_dist[:, cur_state["M"]-1].astype(X.dtype, order="C")

    cdef Py_ssize_t i, j
    for i in range(n):
        j = M-1
        while ind[i, j] < 0:
            j -= 1
            if j < 0: raise ValueError("no nearest neighbours provided")
        d_core[i] = dist[i, j]

    return d_core



cpdef np.ndarray[floatT] _core_distance(np.ndarray[floatT,ndim=2] dist, int M):
    """
    (provided for testing only)

    Given a pairwise distance matrix, computes the "core distance", i.e.,
    the distance of each point to its `M`-th nearest neighbour.
    Note that `M==0` always yields all distances equal to zero.
    The core distances are needed when computing the mutual reachability
    distance in the HDBSCAN* algorithm.

    See Campello R.J.G.B., Moulavi D., Sander J.,
    Density-based clustering based on hierarchical density estimates,
    *Lecture Notes in Computer Science* 7819, 2013, 160-172,
    doi:10.1007/978-3-642-37456-2_14 -- but unlike to the definition therein,
    we do not consider the distance to self as part of the core distance setting.

    The input distance matrix for a given point cloud X may be computed,
    e.g., via a call to
    ``scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))``.


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
        neighbour.
    """
    cdef Py_ssize_t n = dist.shape[0], i, j
    cdef floatT v
    cdef np.ndarray[floatT] d_core = np.zeros(n,
        dtype=np.float32 if floatT is float else np.float64)
    cdef floatT[::1] row

    if M < 0: raise ValueError("M < 0")
    if dist.shape[1] != n: raise ValueError("not a square matrix")
    if M >= n: raise ValueError("M >= matrix size")

    if M == 0: return d_core  # zeros

    cdef vector[Py_ssize_t] buf = vector[Py_ssize_t](M+1)
    for i in range(n):
        row = dist[i,:]
        j = Cargkmin(&row[0], row.shape[0], M, buf.data())
        d_core[i] = dist[i, j]

    return d_core



cpdef np.ndarray[floatT,ndim=2] _mutual_reachability_distance(
        np.ndarray[floatT,ndim=2] dist,
        np.ndarray[floatT] d_core):
    """
    (provided for testing only)

    Given a pairwise distance matrix, computes the mutual reachability
    distance w.r.t. the given core distance vector;
    see ``genieclust.internal.core_distance``,
    ``new_dist[i,j] = max(dist[i,j], d_core[i], d_core[j])``.

    Note that there may be many ties in the mutual reachability distances.

    See Campello R.J.G.B., Moulavi D., Sander J.,
    Density-based clustering based on hierarchical density estimates,
    *Lecture Notes in Computer Science* 7819, 2013, 160-172,
    doi:10.1007/978-3-642-37456-2_14.

    The input distance matrix for a given point cloud X
    may be computed, e.g., via a call to
    ``scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))``.


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
    cdef Py_ssize_t n = dist.shape[0], i, j
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






cpdef tuple nn_list_to_matrix(
    list nns,
    Py_ssize_t k_max):
    """
    genieclust.internal.nn_list_to_matrix(nns, k_max)

    Converts a list of (<=`k_max`)-nearest neighbours to a matrix of `k_max` NNs


    Parameters
    ----------

    nns : list
        Each ``nns[i]`` should be a pair of ``c_contiguous`` `ndarray`\ s.
        An edge ``{i, nns[i][0][j]}`` has weight ``nns[i][1][j]``.
        Each ``nns[i][0]`` is of type `int32` and ``nns[i][1]``
        is of type `float32` (for compatibility with `nmslib`).
    k_max : int
        If `k_max` is greater than 0, `O(n*k_max)` space will be reserved
        for auxiliary data.


    Returns
    -------

    tuple like ``(nn_dist, nn_ind)``
        See `genieclust.internal.mst_from_nn`.
        Unused elements (last items in each row)
        will be filled with ``INFINITY`` and `-1`, respectively.


    See also
    --------

    genieclust.internal.mst_from_nn :
        Constructs a minimum spanning tree from a near-neighbour matrix

    """
    cdef Py_ssize_t n = len(nns)
    cdef np.ndarray[int]   nn_i
    cdef np.ndarray[float] nn_d

    cdef np.ndarray[Py_ssize_t,ndim=2] ret_nn_ind  = np.empty((n, k_max), dtype=np.intp)
    cdef np.ndarray[float,ndim=2]  ret_nn_dist = np.empty((n, k_max), dtype=np.float32)

    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t i1, i2
    cdef float d

    for i in range(n):
        nn_i = nns[i][0]
        nn_d = nns[i][1].astype(np.float32, copy=False)
        k = nn_i.shape[0]
        if nn_d.shape[0] != k:
            raise ValueError("nns has arrays of different lengths as elements")

        l = 0
        for j in range(k):
            i2 = nn_i[j]
            d = nn_d[j]
            if i2 >= 0 and i != i2:
                if l >= k_max: raise ValueError("`k_max` is too small")
                ret_nn_ind[i, l]  = i2
                ret_nn_dist[i, l] = d
                if l > 0 and ret_nn_dist[i, l] < ret_nn_dist[i, l-1]:
                    raise ValueError("nearest neighbours not sorted")
                l += 1

        while l < k_max:
            ret_nn_ind[i, l]  = -1
            ret_nn_dist[i, l] = INFINITY
            l += 1

    return ret_nn_dist, ret_nn_ind



cpdef np.ndarray[Py_ssize_t] get_graph_node_degrees(Py_ssize_t[:,::1] ind, Py_ssize_t n):
    """
    genieclust.internal.get_graph_node_degrees(ind, n)

    Given an adjacency list representing an undirected simple graph over
    a vertex set {0,...,n-1}, returns an array deg with deg[i] denoting
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
    cdef Py_ssize_t num_edges = ind.shape[0]
    assert ind.shape[1] == 2
    cdef np.ndarray[Py_ssize_t] deg = np.empty(n, dtype=np.intp)

    # _openmp_set_num_threads()
    Cget_graph_node_degrees(&ind[0,0], num_edges, n, &deg[0])

    return deg


################################################################################
# Noisy k-partition and other post-processing routines
################################################################################


cdef extern from "../src/c_preprocess.h":
    cdef void Cget_graph_node_degrees(Py_ssize_t* tree_ind, Py_ssize_t m,
            Py_ssize_t n, Py_ssize_t* deg)


cdef extern from "../src/c_postprocess.h":
    void Cmerge_midliers(const Py_ssize_t* tree_ind, Py_ssize_t num_edges,
        const Py_ssize_t* nn_ind, Py_ssize_t num_neighbours, Py_ssize_t M,
        Py_ssize_t* c, Py_ssize_t n)

    void Cmerge_all(const Py_ssize_t* tree_ind, Py_ssize_t num_edges,
        Py_ssize_t* c, Py_ssize_t n)




cpdef np.ndarray[Py_ssize_t] merge_midliers(
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t[::1] c,
        Py_ssize_t[:,::1] nn_i,
        Py_ssize_t M
    ):
    """
    genieclust.internal.merge_midliers(mst_i, c, nn_i, M)

    The `i`-th node is a midlier if it is a leaf in the spanning tree
    (and hence it meets `c[i] < 0`) which is amongst the
    M nearest neighbours of its adjacent vertex, `j`.

    This procedure allocates `c[i]` to its its closest cluster, `c[j]`.


    Parameters
    ----------

    mst_i : c_contiguous array
        See genieclust.mst.mst_from_distance()
    c : c_contiguous array of shape (n_samples,)
        c[i] gives the cluster ID (in {-1, 0, 1, ..., k-1} for some k) of
        the i-th object.  Class -1 represents the leaves of the spanning tree.
    nn_i : c_contiguous matrix of shape (n_samples,n_neighbors)
        nn_ind[i,:] gives the indices of the i-th point's
        nearest neighbours; -1 indicates a "missing value"
    M : int
        smoothing factor, M>=1


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        ID (in {-1, 0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[Py_ssize_t] cl2 = np.array(c, dtype=np.intp)

    # _openmp_set_num_threads()
    Cmerge_midliers(
        &mst_i[0,0], mst_i.shape[0],
        &nn_i[0,0], nn_i.shape[1], M,
        &cl2[0], cl2.shape[0]
    )

    return cl2


cpdef np.ndarray[Py_ssize_t] merge_all(
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t[::1] c
    ):
    """
    genieclust.internal.merge_all(mst_i, c)

    For each leaf in the MST, `i` (and hence a vertex which meets `c[i] < 0`),
    this procedure allocates `c[i]` to its its closest cluster, `c[j]`,
    where `j` is the vertex adjacent to `i`.


    Parameters
    ----------

    mst_i : c_contiguous array
        See genieclust.mst.mst_from_distance()
    c : c_contiguous array of shape (n_samples,)
        c[i] gives the cluster ID (in {-1, 0, 1, ..., k-1} for some k) of
        the i-th object.  Class -1 represents the leaves of the spanning tree.


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        ID (in {0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[Py_ssize_t] cl2 = np.array(c, dtype=np.intp)

    # _openmp_set_num_threads()
    Cmerge_all(
        &mst_i[0,0], mst_i.shape[0],
        &cl2[0], cl2.shape[0]
    )

    return cl2


cpdef dict get_linkage_matrix(Py_ssize_t[::1] links,
                              floatT[::1] mst_d,
                              Py_ssize_t[:,::1] mst_i):
    """
    genieclust.internal.get_linkage_matrix(links, mst_d, mst_i)


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
    cdef Py_ssize_t n = mst_i.shape[0]+1
    cdef Py_ssize_t i, i1, i2, par, w, num_unused, j

    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")
    if not n-1 == links.shape[0]:
        raise ValueError("ill-defined MST")

    cdef CGiniDisjointSets ds = CGiniDisjointSets(n)

    cdef np.ndarray[Py_ssize_t,ndim=2] children_  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]         distances_ = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)
    cdef np.ndarray[Py_ssize_t]        counts_    = np.empty(n-1, dtype=np.intp)
    cdef np.ndarray[Py_ssize_t]        used       = np.zeros(n-1, dtype=np.intp)
    cdef np.ndarray[Py_ssize_t]        ids        = np.empty(n, dtype=np.intp)

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
        ids[par] = n+i  # see scipy.cluster.hierarchy.linkage
        distances_[i] = mst_d[w] if i >= num_unused else 0.0
        counts_[i] = ds.get_count(par)



    # corrections for departures from ultrametricity:
    # distances_ = genieclust.tools.cummin(distances_[::-1])[::-1]
    for i in range(n-2, 0, -1):
        if distances_[i-1] > distances_[i]:
            distances_[i-1] = distances_[i]

    return dict(
        children=children_,
        distances=distances_,
        counts=counts_
    )


#############################################################################
# The Genie Clustering Algorithm (internal)
#############################################################################

cdef extern from "../src/c_genie.h":

    cdef cppclass CGenie[T]:
        CGenie() except +
        CGenie(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t n, bint skip_leaves) except +
        void compute(Py_ssize_t n_clusters, double gini_threshold) except +
        Py_ssize_t get_max_n_clusters()
        Py_ssize_t get_links(Py_ssize_t* res)
        Py_ssize_t get_labels(Py_ssize_t n_clusters, Py_ssize_t* res)
        void get_labels_matrix(Py_ssize_t n_clusters, Py_ssize_t* res)
        #void get_is_outlier(int* res)


cpdef dict genie_from_mst(
        floatT[::1] mst_d,
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t n_clusters=1,
        double gini_threshold=0.3,
        bint skip_leaves=False,
        bint compute_full_tree=True,
        bint compute_all_cuts=False
    ):
    """The Genie Clustering Algorithm (with extensions)

    Determines a dataset's partition based on a precomputed MST.

    Gagolewski, M., Bartoszuk, M., Cena, A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

    Refer to the online manual at <https://genieclust.gagolewski.com/> for
    more details.

    This is a new implementation of the original algorithm,
    which runs in O(n sqrt(n))-time. Additionally, MST leaves can be
    omitted from the clustering process and marked as outliers
    (if `skip_leaves==True`). This is useful when the Genie algorithm
    is applied on MSTs with respect to mutual reachability distances.

    The input tree can actually be any spanning tree.
    Moreover, it does not even have to be a connected graph.

    gini_threshold==1.0 and skip_leaves==False, gives the single linkage
    algorithm.


    Parameters
    ----------

    mst_d, mst_i : ndarray
        A spanning tree defined by a pair (mst_i, mst_d),
        see genieclust.mst.
    n_clusters : int
        Number of clusters the dataset is split into.
        If `compute_full_tree` is False, then only partial cluster hierarchy
        is determined.
    gini_threshold : float
        The threshold for the Genie correction
    skip_leaves : bool
        Mark leaves as outliers.
        Prevents forming singleton-clusters.
    compute_full_tree : bool
        Compute the entire merge sequence or stop early?
    compute_all_cuts : bool
        Compute the n_clusters and all the more coarse-grained ones?
        Implies `compute_full_tree`.


    Returns
    -------

    res : dict, with the following elements:
        labels : ndarray, shape (n,) or (n_clusters, n) or None
            Is None if n_clusters==0.

            If compute_all_cuts==False, this gives the predicted labels,
            representing an n_clusters-partition of X.
            labels[i] gives the cluster ID of the i-th input point.
            If skip_leaves==True, then label -1 denotes an outlier.

            If compute_all_cuts==True, then
            labels[i,:] gives the (i+1)-partition, i=0,...,n_clusters-1.

        links : ndarray, shape (n-1,)
            links[i] gives the MST edge merged at the i-th iteration
            of the algorithm.

        iters : int
            number of merge steps performed

        n_clusters : integer
            actual number of clusters found, 0 if `labels` is None

    """
    cdef Py_ssize_t n = mst_i.shape[0]+1

    if not 0 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")
    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")
    if not 0.0 <= gini_threshold <= 1.0:
        raise ValueError("incorrect gini_threshold")


    cdef np.ndarray[Py_ssize_t] tmp_labels_1
    cdef np.ndarray[Py_ssize_t,ndim=2] tmp_labels_2

    cdef np.ndarray[Py_ssize_t] links_  = np.empty(n-1, dtype=np.intp)
    cdef Py_ssize_t n_clusters_ = 0, iters_
    labels_ = None # on request, see below

    # _openmp_set_num_threads()

    cdef CGenie[floatT] g
    g = CGenie[floatT](&mst_d[0], &mst_i[0,0], n, skip_leaves)

    if compute_all_cuts:
        compute_full_tree = True

    g.compute(1 if compute_full_tree else n_clusters, gini_threshold)

    iters_ = g.get_links(&links_[0])

    if n_clusters >= 1:
        n_clusters_ = min(g.get_max_n_clusters(), n_clusters)

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
#############################################################################

cdef extern from "../src/c_genie.h":

    cdef cppclass CGIc[T]:
        CGIc() except +
        CGIc(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t n, bint skip_leaves) except +
        void compute(Py_ssize_t n_clusters, Py_ssize_t add_clusters,
            double n_features, double* gini_thresholds, Py_ssize_t n_thresholds)  except +
        Py_ssize_t get_max_n_clusters()
        Py_ssize_t get_links(Py_ssize_t* res)
        Py_ssize_t get_labels(Py_ssize_t n_clusters, Py_ssize_t* res)
        void get_labels_matrix(Py_ssize_t n_clusters, Py_ssize_t* res)



cpdef dict gic_from_mst(
        floatT[::1] mst_d,
        Py_ssize_t[:,::1] mst_i,
        double n_features,
        Py_ssize_t n_clusters=1,
        Py_ssize_t add_clusters=0,
        double[::1] gini_thresholds=None,
        bint skip_leaves=False,
        bint compute_full_tree=True,
        bint compute_all_cuts=False):
    """GIc (Genie+Information Criterion) Information-Theoretic
    Hierarchical Clustering Algorithm

    Determines a dataset's partition based on a precomputed MST,
    maximising (heuristically) an information criterion [2].

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

    .. [1]
        Cena, A., *Adaptive hierarchical clustering algorithms based on
        data aggregation methods*, PhD Thesis, Systems Research Institute,
        Polish Academy of Sciences 2018.

    .. [2]
        Mueller, A., Nowozin, S., Lampert, C.H., Information Theoretic
        Clustering using Minimum Spanning Trees, *DAGM-OAGM* 2012.

    .. [3]
        Gagolewski, M., Bartoszuk M., Cena, A.,
        Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
        *Information Sciences* 363, 2016, 8-23. DOI:10.1016/j.ins.2016.05.003



    Parameters
    ----------

    mst_d, mst_i : ndarray
        Minimal spanning tree defined by a pair (mst_i, mst_d),
        see genieclust.mst.
    n_features : double
        number of features in the data set
        [can be fractional if you know what you're doing]
    n_clusters : int
        Number of clusters the dataset is split into.
        If `compute_full_tree` is False, then only partial cluster hierarchy
        is determined.
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
    skip_leaves : bool
        Mark leaves as outliers.
        Prevents forming singleton-clusters.
    compute_full_tree : bool
        Compute the entire merge sequence or stop early?
        Implies compute_full_tree.
    compute_all_cuts : bool
        Compute the n_clusters and all the more coarse-grained ones?


    Returns
    -------

    res : dict, with the following elements:
        labels : ndarray, shape (n,) or None
            Predicted labels, representing an n_clusters-partition of X.
            labels[i] gives the cluster id of the i-th input point.
            If skip_leaves==True, then label -1 denotes an outlier.
            Is None if n_clusters==0.

        links : ndarray, shape (n-1,)
            links[i] gives the MST edge merged at the i-th iteration
            of the algorithm.

        iters : int
            number of merge steps performed

        n_clusters : integer
            actual number of clusters found, 0 if `labels` is None
    """
    cdef Py_ssize_t n = mst_i.shape[0]+1

    if not 0 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")
    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")

    if gini_thresholds is None:
        gini_thresholds = np.r_[0.1, 0.3, 0.5, 0.7]


    cdef np.ndarray[Py_ssize_t] tmp_labels_1
    cdef np.ndarray[Py_ssize_t,ndim=2] tmp_labels_2

    cdef np.ndarray[Py_ssize_t] links_  = np.empty(n-1, dtype=np.intp)
    cdef Py_ssize_t n_clusters_ = 0, iters_
    labels_ = None # on request, see below

    # _openmp_set_num_threads()

    cdef CGIc[floatT] g
    g = CGIc[floatT](&mst_d[0], &mst_i[0,0], n, skip_leaves)

    if compute_all_cuts:
        compute_full_tree = True

    g.compute(1 if compute_full_tree else n_clusters,
                n_clusters-1+add_clusters if compute_full_tree else add_clusters,
                n_features,
            &gini_thresholds[0], gini_thresholds.shape[0])

    iters_ = g.get_links(&links_[0])

    if n_clusters >= 1:
        n_clusters_ = min(g.get_max_n_clusters(), n_clusters)

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
