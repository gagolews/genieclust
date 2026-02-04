# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Auxiliary functions and classes
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
from libcpp.vector cimport vector


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

        ndarray, shape (n,) :
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

        list of lists :
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
# Auxiliary graph processing routines
################################################################################


cdef extern from "../src/c_graph_process.h":

    void Cmerge_midliers(  # [DEPRECATED]
        const Py_ssize_t* mst_i, Py_ssize_t num_edges,
        const Py_ssize_t* nn_i, Py_ssize_t num_neighbours, Py_ssize_t M,
        Py_ssize_t* c, Py_ssize_t n
    ) except+

    void Cmerge_all(  # [DEPRECATED]
        const Py_ssize_t* mst_i, Py_ssize_t num_edges,
        Py_ssize_t* c, Py_ssize_t n
    ) except+




cpdef np.ndarray[Py_ssize_t] merge_midliers(
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t[::1] c,
        Py_ssize_t[:,::1] nn_i,
        Py_ssize_t M
    ):
    """
    genieclust.core.merge_midliers(mst_i, c, nn_i, M)

    [DEPRECATED]

    The `i`-th node is a midlier if it is a leaf in the spanning tree
    (and hence it meets `c[i] < 0`) which is amongst the
    M nearest neighbours of its adjacent vertex, `j`.

    This procedure allocates `c[i]` to its its closest cluster, `c[j]`.


    Parameters
    ----------

    mst_i : c_contiguous array of shape (m, 2)
        m undirected edges of the spanning tree
    c : c_contiguous array of shape (n,)
        c[i] gives the cluster ID (in {-1, 0, 1, ..., k-1} for some k) of
        the i-th object.  Class -1 represents the missing values to be imputed.
    nn_i : c_contiguous matrix of shape (n,n_neighbors)
        nn_ind[i,:] gives the indexes of the i-th point's
        nearest neighbours; -1 indicates a "missing value"
    M : int
        smoothing factor, M>=1


    Returns
    -------

    c : ndarray, shape (n,)
        A new integer vector c with c[i] denoting the cluster
        ID (in {-1, 0, ..., k-1}) of the i-th object
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
    genieclust.core.merge_all(mst_i, c)

    [DEPRECATED]

    For each leaf in the MST, `i` (and hence a vertex which meets `c[i] < 0`),
    this procedure allocates `c[i]` to its its closest cluster, `c[j]`,
    where `j` is the vertex adjacent to `i`.


    Parameters
    ----------

    mst_i : c_contiguous array of shape (m, 2)
        m undirected edges of the spanning tree
    c : c_contiguous array of shape (n,)
        c[i] gives the cluster ID (in {-1, 0, 1, ..., k-1} for some k) of
        the i-th object.  Class -1 represents the missing values to be imputed.


    Returns
    -------

    c : ndarray, shape (n,)
        a new integer vector c with c[i] denoting the cluster
        ID (in {0, ..., k-1}) of the i-th object
    """
    cdef np.ndarray[Py_ssize_t] cl2 = np.array(c, dtype=np.intp)

    # _openmp_set_num_threads()
    Cmerge_all(
        &mst_i[0,0], mst_i.shape[0],
        &cl2[0], cl2.shape[0]
    )

    return cl2


cpdef dict get_linkage_matrix(
        Py_ssize_t[::1] links,
        floatT[::1] mst_d,
        Py_ssize_t[:,::1] mst_i
    ):
    """
    genieclust.core.get_linkage_matrix(links, mst_d, mst_i)


    Parameters
    ----------

    links : ndarray
        see return value of genieclust.core.genie_from_mst.

    mst_d, mst_i : ndarray
        minimal spanning tree defined by a pair (mst_i, mst_d),
        with mst_i of shape (n-1,2) giving the edges and mst_d providing the
        corresponding edge weights.


    Returns
    -------

    Z : dict
        a dictionary with 3 keys: children, distances, counts,
        see the description of Z[:,:2], Z[:,2] and Z[:,3], respectively,
        in scipy.cluster.hierarchy.linkage
    """
    cdef Py_ssize_t n = mst_i.shape[0]+1
    cdef Py_ssize_t i, i1, i2, par, w, num_unused, j

    if mst_i.shape[1] != 2: raise ValueError("mst_i must have two columns")

    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")
    if not n-1 == links.shape[0]:
        raise ValueError("ill-defined MST")

    cdef CGiniDisjointSets ds = CGiniDisjointSets(n)

    cdef np.ndarray[Py_ssize_t,ndim=2] children_  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]            distances_ = np.empty(n-1,
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
        CGenie(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t n, bool* skip_nodes) except +
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
        bint compute_all_cuts=False,
    ):
    """The Genie Clustering Algorithm (with extensions)

    Determines a dataset's partition based on a precomputed MST.

    Refer to the online manual at <https://genieclust.gagolewski.com/> for
    more details.

    This is a new implementation of the original algorithm,
    which runs in O(n sqrt(n))-time. Additionally, some nodes can be
    omitted from the clustering process (e.g., outliers).

    Any spanning tree can actually be fed as input.
    Moreover, it does not even have to be a connected graph.

    gini_threshold==1.0 gives the single linkage algorithm.


    Parameters
    ----------

    mst_d, mst_i : ndarray
        A spanning tree defined by a pair (mst_i, mst_d),
        with mst_i giving the edges (n-1,2) and mst_d providing the
        corresponding edge weights.

    n_clusters : int
        Number of clusters the dataset is split into.

    gini_threshold : float
        The threshold for the Genie correction

    compute_all_cuts : bool
        Determine n_clusters and all the more coarse-grained ones?


    Returns
    -------

    res : dict, with the following elements:
        labels : ndarray, shape (n,) or (n_clusters, n) or None
            Is None if n_clusters==0.

            If compute_all_cuts==False, we get the predicted labels,
            representing an n_clusters-partition of X.
            labels[i] gives the cluster ID of the i-th input point.
            Label -1 denotes a skipped point.

            If compute_all_cuts==True, then
            labels[i,:] gives the (i+1)-partition, i=0,...,n_clusters-1.

        links : ndarray, shape (n-1,)
            links[i] gives the MST edge merged at the i-th iteration
            of the algorithm.

        iters : int
            number of merge steps performed

        n_clusters : integer
            actual number of clusters found, 0 if `labels` is None


    References
    ----------

    .. [1]
        Gagolewski, M., Bartoszuk, M., Cena, A., Genie: A new, fast, and
        outlier-resistant hierarchical clustering algorithm,
        Information Sciences 363, 2016, pp. 8-23. DOI:10.1016/j.ins.2016.05.003
    """
    cdef Py_ssize_t n = mst_i.shape[0]+1

    if not 0 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")
    if not n-1 == mst_d.shape[0]:
        raise ValueError("ill-defined MST")
    if not 0.0 <= gini_threshold <= 1.0:
        raise ValueError("incorrect gini_threshold")
    if mst_i.shape[1] != 2: raise ValueError("mst_i must have two columns")


    cdef np.ndarray[Py_ssize_t] tmp_labels_1
    cdef np.ndarray[Py_ssize_t,ndim=2] tmp_labels_2

    cdef np.ndarray[Py_ssize_t] links_  = np.empty(n-1, dtype=np.intp)
    cdef Py_ssize_t n_clusters_ = 0, iters_
    labels_ = None  # on demand, see below

    # _openmp_set_num_threads()

    # TODO: remove skip_nodes:: this can be achieved more easily
    # using skip/unskip_index...
    # cdef bool* skip_nodes_ptr = NULL
    # if skip_nodes.shape[0] == 0:
    #     pass
    # elif skip_nodes.shape[0] == n:
    #     skip_nodes_ptr = &skip_nodes[0]
    # else:
    #     raise ValueError("skip_nodes should be either of size 0 or n")

    cdef CGenie[floatT] g
    g = CGenie[floatT](&mst_d[0], &mst_i[0,0], n, NULL)

    # g.compute(1 if compute_full_tree else n_clusters, gini_threshold)
    g.compute(1, gini_threshold)

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

    return dict(
        labels=labels_,
        n_clusters=n_clusters_,
        links=links_,
        iters=iters_
    )


#############################################################################
# The Genie+Information Criterion (G+IC) Clustering Algorithm
#############################################################################

cdef extern from "../src/c_genie.h":

    cdef cppclass CGIc[T]:
        CGIc() except +
        CGIc(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t n, bool* skip_nodes) except +
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
        #bool[::1] skip_nodes  TODO
        Py_ssize_t n_clusters=1,
        Py_ssize_t add_clusters=0,
        double[::1] gini_thresholds=None,
        bint compute_all_cuts=False
    ):
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
    ----------

    .. [1]
        Cena, A., *Adaptive hierarchical clustering algorithms based on
        data aggregation methods*, PhD Thesis, Systems Research Institute,
        Polish Academy of Sciences, 2018.

    .. [2]
        Mueller, A., Nowozin, S., Lampert, C.H., Information Theoretic
        Clustering using Minimum Spanning Trees, *DAGM-OAGM* 2012.

    .. [3]
        Gagolewski, M., Bartoszuk M., Cena, A., Genie: A new, fast,
        and outlier-resistant hierarchical clustering algorithm,
        *Information Sciences* 363, 2016, 8-23. DOI:10.1016/j.ins.2016.05.003



    Parameters
    ----------

    mst_d, mst_i : ndarray
        Minimal spanning tree defined by a pair (mst_i, mst_d),
        with mst_i giving the edges (n-1,2) and mst_d providing the
        corresponding edge weights.

    n_features : double
        number of features in the data set
        [can be fractional if you know what you're doing]

    n_clusters : int
        Number of clusters the dataset is split into.

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

    compute_all_cuts : bool
        Determine n_clusters and all the more coarse-grained ones?


    Returns
    -------

    res : dict, with the following elements:
        labels : ndarray, shape (n,) or None
            Predicted labels, representing an n_clusters-partition of X.
            labels[i] gives the cluster ID of the i-th input point.

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
    if mst_i.shape[1] != 2: raise ValueError("mst_i must have two columns")

    if gini_thresholds is None:
        gini_thresholds = np.r_[0.1, 0.3, 0.5, 0.7]


    cdef np.ndarray[Py_ssize_t] tmp_labels_1
    cdef np.ndarray[Py_ssize_t,ndim=2] tmp_labels_2

    cdef np.ndarray[Py_ssize_t] links_  = np.empty(n-1, dtype=np.intp)
    cdef Py_ssize_t n_clusters_ = 0, iters_
    labels_ = None # on request, see below

    # _openmp_set_num_threads()

    cdef CGIc[floatT] g
    g = CGIc[floatT](&mst_d[0], &mst_i[0,0], n, NULL)
    # TODO: skip_nodes has not been tested yet!

    g.compute(
        1,  # if compute_full_tree else n_clusters,
        n_clusters-1+add_clusters,  # if compute_full_tree else add_clusters,
        n_features,
        &gini_thresholds[0],
        gini_thresholds.shape[0]
    )

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

    return dict(
        labels=labels_,
        n_clusters=n_clusters_,
        links=links_,
        iters=iters_
    )
