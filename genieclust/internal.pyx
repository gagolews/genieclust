# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
The Genie+ clustering algorithm (with extras)
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
from libc.math cimport fabs, sqrt
from numpy.math cimport INFINITY
import scipy.spatial.distance
import warnings



ctypedef fused intT:
    np.int64_t
    np.int32_t
    np.int_t

ctypedef fused T:
    np.float64_t
    np.float32_t
    np.int64_t
    np.int32_t
    np.int_t
    np.double_t

ctypedef fused arrayT:
    np.ndarray[np.double_t]
    np.ndarray[np.int_t]

cdef T square(T x):
    return x*x



#############################################################################
# Determine the index of the k-th smallest element in an array
#############################################################################


cpdef np.int_t argkmin(arrayT x, np.int_t k):
    """
    Returns the index of the (k-1)-th smallest value in an array x,
    where argkmin(x, 0) == argmin(x), or, more generally,
    argkmin(x, k) == np.argsort(x)[k].

    Run time: O(nk), where n == len(x). Working mem: O(k).
    Does not modify x.

    In practice, very fast for small k and randomly ordered
    or almost sorted (increasingly) data.

    Example timings:                 argkmin(x, k) np.argsort(x)[k]
    (ascending)  n=100000000, k=  5:        0.058s           1.448s
    (descending)                            0.572s           2.651s
    (random)                                0.064s          20.049s
    (ascending)  n=100000000, k=100:        0.057s           1.472s
    (descending)                           18.051s           2.662s
    (random)                                0.064s          20.269s


    Parameters:
    ----------

    x : ndarray
        an integer or float vector

    k : int
        an integer in {0,...,len(x)-1}, preferably small


    Returns:
    -------

    val
        the (k-1)-th smallest value in x
    """
    cdef np.int_t n = len(x), i, j, ret
    cdef np.int_t* idx
    if k < 0:  raise Exception("k < 0")
    if k >= n: raise Exception("k >= n")

    k += 1
    idx = <np.int_t*>PyMem_Malloc(k*sizeof(np.int_t))
    for i in range(0, k):
        j = i
        idx[i] = i
        while j > 0 and x[idx[j]] < x[idx[j-1]]:
            idx[j], idx[j-1] = idx[j-1], idx[j] # KISS
            j -= 1

    for i in range(k, n):
        if x[idx[k-1]] <= x[i]:
            continue
        j = k-1
        idx[k-1] = i
        while j > 0 and x[idx[j]] < x[idx[j-1]]:
            idx[j], idx[j-1] = idx[j-1], idx[j] # KISS
            j -= 1

    ret = idx[k-1]
    PyMem_Free(idx)
    return ret



#############################################################################
# Disjoint Sets (Union-Find)
# A Python class to represent partitions of the set {0,1,...,n-1} for any n
#############################################################################

cdef class DisjointSets:
    """
    Disjoint Sets (Union-Find)

    Represents a partition of the set {0,1,...,n-1}

    Path compression for find() is implemented,
    but the union() operation is naive (neither
    it is union by rank nor by size),
    see https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
    This is by design, as some other operations in the current
    package rely on the assumption, that the parent id of each
    element is always <= than itself.


    Parameters:
    ----------

    n : int
        The cardinality of the set being partitioned.
    """
    cdef np.int_t n # number of distinct elements
    cdef np.int_t k # number of subsets
    cdef np.int_t* par

    def __cinit__(self, np.int_t n):
        if n <= 0: raise Exception("n <= 0")
        cdef np.int_t i

        self.n = n
        self.k = n

        self.par = <np.int_t*>PyMem_Malloc(self.n*sizeof(np.int_t))
        for i in range(self.n):
            self.par[i] = i


    def __dealloc__(self):
        if self.par != NULL:
            PyMem_Free(self.par)
            self.par = <np.int_t*>NULL


    def __len__(self):
        """
        Returns the number of subsets-1,
        a.k.a. how many calls to union() can we still perform.

        Returns:
        -------

        len : int
            A value in {0,...,n-1}.
        """
        return self.k-1


    cpdef np.int_t find(self, np.int_t x):
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
        if x < 0 or x >= self.n: raise Exception("elem not in {0,1,...,n-1}")
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]


    cpdef np.int_t union(self, np.int_t x, np.int_t y):
        """
        Merges the sets containing given x and y.
        Let px be the parent id of x, and py be the parent id of y.
        If px < py, then the new parent id of py will be set to py.
        Otherwise, px will have py as its parent.

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
        cdef np.int_t i, size1, size2

        x = self.find(x)
        y = self.find(y)
        if x == y: raise Exception("find(x) == find(y)")
        if y < x: x,y = y,x

        self.par[y] = x
        self.k -= 1
        return x


    cpdef np.ndarray[np.int_t] to_list(self):
        """
        Get parent ids of all the elements


        Returns:
        -------

        parents : ndarray, shape (n,)
            A list m such that m[x] denotes the (recursive) parent id of x,
            for x=0,1,...,n.
        """
        cdef np.int_t i
        cdef np.ndarray[np.int_t] m = np.empty(self.n, dtype=np.int_)
        for i in range(self.n):
            m[i] = self.find(i)
        return m


    cpdef np.ndarray[np.int_t] to_list_normalized(self):
        """
        Get the normalized elements' membership information.


        Returns:
        -------

        set_ids : ndarray, shape (n,)
            A list m such that m[x] denotes the normalized parent id of x.
            The resulting values are in {0,1,...,k-1}, where k is the current
            number of subsets in the partition.
        """
        cdef np.int_t i, j
        cdef np.ndarray[np.int_t] m = np.empty(self.n, dtype=np.int_)
        cdef np.ndarray[np.int_t] v = np.empty(self.n, dtype=np.int_)
        for i in range(self.n): v[i] = -1
        cdef np.int_t c = 0
        for i in range(self.n):
            j = self.find(i)
            if v[j] < 0:
                v[j] = c
                c += 1
            m[i] = v[j]
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
        cdef np.int_t i
        cdef list tou, out

        tou = [ [] for i in range(self.n) ]
        for i in range(self.n):
            tou[self.find(i)].append(i)

        out = []
        for i in range(self.n):
            if tou[i]: out.append(tou[i])

        return out


    def __repr__(self):
        """
        Calls self.to_lists()
        """
        return "DisjointSets("+repr(self.to_lists())+")"



#############################################################################
# Augmented DisjointSets
#############################################################################



cdef class GiniDisjointSets(DisjointSets):
    """
    Augmented disjoint sets (Union-Find) over {0,1,...,n-1}
    Allow to compute the normalized Gini index for the
    subset sizes distribution, i.e.,
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

    n : int
        The cardinality of the set being partitioned.
    """
    cdef np.int_t* cnt      # cnt[find(x)] is the size of the relevant subset
    cdef np.int_t* tab      # tab[i] gives the number of subsets of size i
    cdef np.int_t* tab_next # an array-based...
    cdef np.int_t* tab_prev # ...doubly-linked list...
    cdef np.int_t  tab_head # ...for quickly accessing and iterating thru...
    cdef np.int_t  tab_tail # ...self.tab data
    cdef np.double_t gini   # the Gini index of the current subset sizes

    def __cinit__(self, np.int_t n):
        # Cython manual: "the __cinit__() method of the base type
        # is automatically called before your __cinit__() method is called"
        cdef np.int_t i

        self.cnt = <np.int_t*>PyMem_Malloc(self.n*sizeof(np.int_t))
        for i in range(self.n):
            self.cnt[i] = 1

        self.tab = <np.int_t*>PyMem_Malloc((self.n+1)*sizeof(np.int_t))
        for i in range(0, self.n+1):
            self.tab[i] = 0
        self.tab[1] = self.n

        self.tab_next = <np.int_t*>PyMem_Malloc((self.n+1)*sizeof(np.int_t))
        self.tab_prev = <np.int_t*>PyMem_Malloc((self.n+1)*sizeof(np.int_t))
        self.tab_head = 1
        self.tab_tail = 1

        self.gini = 0.0 # = gini([1,1,...,1])


    def __dealloc__(self):
        if self.cnt != NULL:
            PyMem_Free(self.cnt)
            self.cnt = <np.int_t*>NULL
        if self.tab != NULL:
            PyMem_Free(self.tab)
            self.tab = <np.int_t*>NULL
        if self.tab_prev != NULL:
            PyMem_Free(self.tab_prev)
            self.tab_prev = <np.int_t*>NULL
        if self.tab_next != NULL:
            PyMem_Free(self.tab_next)
            self.tab_next = <np.int_t*>NULL
        # Cython manual: "The __dealloc__() method of the superclass
        # will always be called, even if it is overridden."


    cpdef np.int_t union(self, np.int_t x, np.int_t y):
        """
        Merges the sets containing given x and y.
        Let px be the parent id of x, and py be the parent id of y.
        If px < py, then the new parent id of py will be set to py.
        Otherwise, px will have py as its parent.

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
        cdef np.int_t i, size1, size2, v, w

        x = self.find(x)
        y = self.find(y)
        if self.k == 1: raise Exception("no more subsets to merge")
        if x == y: raise Exception("find(x) == find(y)")
        if y < x: x,y = y,x
        self.par[y] = x
        self.k -= 1

        size1 = self.cnt[x]
        size2 = self.cnt[y]

#        self.gini *= self.n*(self.k-1.0)
#
#        for i in range(self.n):
#            if i == self.par[i]:
#                self.gini -= fabs(self.cnt[i]-size1)
#                self.gini -= fabs(self.cnt[i]-size2)
#                self.gini += fabs(self.cnt[i]-size1-size2)
#
#        self.gini += fabs(size2-size1)
#        self.gini -= fabs(size2-size1-size2)
#        self.gini -= fabs(size1-size1-size2)
#        self.gini /= <np.double_t>(self.n*(self.k-2.0))


        self.cnt[x] += self.cnt[y]
        self.cnt[y] = 0

        self.tab[size1] -= 1
        self.tab[size2] -= 1
        self.tab[size1+size2] += 1

        if self.tab_tail < size1+size2: # new tail
            self.tab_prev[size1+size2] = self.tab_tail
            self.tab_next[self.tab_tail] = size1+size2
            self.tab_tail = size1+size2
        elif self.tab[size1+size2] == 1: # new elem in the 'middle'
            w = self.tab_tail
            while w > size1+size2: w = self.tab_prev[w]
            v = self.tab_next[w]
            self.tab_next[w] = size1+size2
            self.tab_prev[v] = size1+size2
            self.tab_next[size1+size2] = v
            self.tab_prev[size1+size2] = w

        if size2 < size1: size1, size2 = size2, size1
        if self.tab[size1] == 0:
            if self.tab_head == size1:
                self.tab_head = self.tab_next[self.tab_head]
            else: # remove in the 'middle'
                self.tab_next[self.tab_prev[size1]] = self.tab_next[size1]
                self.tab_prev[self.tab_next[size1]] = self.tab_prev[size1]

        if self.tab[size2] == 0 and size1 != size2: # i.e., size2>size1
            if self.tab_head == size2:
                self.tab_head = self.tab_next[self.tab_head]
            else: # remove in the 'middle'
                self.tab_next[self.tab_prev[size2]] = self.tab_next[size2]
                self.tab_prev[self.tab_next[size2]] = self.tab_prev[size2]

        if self.tab[self.tab_head] <= 0: raise Exception("ASSERT FAIL: self.tab[self.tab_head] > 0")
        if self.tab[self.tab_tail] <= 0: raise Exception("ASSERT FAIL: self.tab[self.tab_tail] > 0")

        self.gini = 0.0
        if self.tab_head != self.tab_tail:
            v = self.tab_head
            i = 0
            while v != self.tab_tail:
                w = v
                v = self.tab_next[v]
                i += self.tab[w]
                # delta_i = (v-w)
                self.gini += (v-w)*i*(self.k-i)
            self.gini /= <np.double_t>(self.n*(self.k-1.0))
            if self.gini > 1.0: self.gini = 1.0 # account for round-off errors
            if self.gini < 0.0: self.gini = 0.0

        return x


    cpdef np.int_t get_count(self, np.int_t x):
        """
        Get the size of the set with x in it.

        Run time: the cost of finding the parent of x.


        Parameters:
        ----------

        x : int
            An integer in {0,...,n-1}, representing an element to find.


        Returns:
        -------

        count : int
            The size of the set including x.
        """
        return self.cnt[self.find(x)]


    cpdef np.ndarray[np.int_t] get_counts(self):
        """
        Generate a list of set sizes.
        The vector is ordered nondecreasingly.


        Returns:
        -------

        counts : ndarray
             Gives the cardinality of each set in the partition.
        """
        cdef np.int_t i = 0, j
        cdef np.ndarray[np.int_t] out = np.empty(self.k, dtype=np.int_)
        v = self.tab_head
        while True:
            for j in range(self.tab[v]):
                out[i] = v
                i += 1
            if v == self.tab_tail: break
            else: v = self.tab_next[v]
        return out


#    cpdef np.ndarray[np.int_t] get_parents(self):
#        """
#        Get vector with all elements v s.t. v == find(v)
#        """
#        pass
#    # if we ever need this, an array-based doubly-linked list of pars should be added


    cpdef np.double_t get_gini(self):
        """
        Get the Gini index for inequity of subset size distribution

        Run time: O(1), as the Gini index is updated during a call
        to self.union().


        Returns:
        -------

        g : float
            The Gini index of self.get_counts()
        """
        return self.gini


    cpdef np.int_t get_smallest_count(self):
        """
        Get the size of the smallest set.

        Run time: O(1)


        Returns:
        -------

        size : float
            The cardinality of the smallest set in the current partition.
        """
        return self.tab_head


    def __repr__(self):
        """
        Calls self.to_lists()
        """
        return "GiniDisjointSets("+repr(self.to_lists())+")"





#############################################################################
# HDBSCAN* Clustering Algorithm - auxiliary functions
#############################################################################


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



#############################################################################
# The Genie+ Clustering Algorithm (internal)
#############################################################################

cpdef np.ndarray[np.int_t] genie_from_mst(tuple mst,
                     np.int_t n_clusters=2,
                     np.double_t gini_threshold=0.3,
                     bint noise_leaves=False):
    """
    Compute a k-partition based on a precomputed MST.

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

    mst = genieclust.mst.MST_pair(
        scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(X, "euclidean")),


    If gini_threshold==1.0 and noise_leaves==False, then basically this
    is the single linkage algorithm. Set gini_threshold==1.0 and
    noise_leaves==True to get a HDBSCAN-like behavior (and make sure
    the MST is computed w.r.t. the mutual reachability distance).


    Parameters:
    ----------

    mst : tuple
        See genieclust.mst.MST_pair()

    n_clusters : int, default=2
        Number of clusters the data is split into.

    gini_threshold : float, default=0.3
        The threshold for the Genie correction

    noise_leaves : bool
        Mark leaves as noise

    Returns:
    -------

    labels_ : ndarray, shape (n,)
        Predicted labels, representing a partition of X.
        labels_[i] gives the cluster id of the i-th input point.
        If noise_leaves==True, then label -1 denotes a noise point.
    """
    cdef np.int_t n, i, j, curidx, m, i1, i2, lastm, lastidx, previdx
    cdef noise_count
    cdef np.ndarray[np.int_t] res
    cdef np.int_t* next_edge
    cdef np.int_t* prev_edge
    cdef np.int_t* denoise_index
    cdef np.int_t* denoise_index_rev
    cdef np.int_t* res_cluster_id
    cdef np.ndarray[np.int_t,ndim=2] mst_i = mst[0]
    cdef np.ndarray[np.int_t] deg = get_tree_node_degrees(mst_i)
    n = mst_i.shape[0]+1

    denoise_index     = <np.int_t*>PyMem_Malloc(n*sizeof(np.int_t))
    denoise_index_rev = <np.int_t*>PyMem_Malloc(n*sizeof(np.int_t))
    # Create the non-noise points' translation table (for GiniDisjointSets)
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
        PyMem_Free(denoise_index)
        PyMem_Free(denoise_index_rev)
        raise Exception("The requested number of clusters is too large \
            with this many detected noise points")

    # When the Genie correction is on, some MST edges will be chosen
    # in a non-consecutive order. An array-based skiplist will speed up
    # searching within the not-yet-consumed edges.
    next_edge = <np.int_t*>PyMem_Malloc(n*sizeof(np.int_t))
    prev_edge = <np.int_t*>PyMem_Malloc(n*sizeof(np.int_t))
    if noise_leaves:
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
        curidx  = 0
        lastidx = 0
        for i in range(n-1):
            next_edge[i] = i+1
            prev_edge[i] = i-1

    cdef GiniDisjointSets ds = GiniDisjointSets(n-noise_count)

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

        ds.union(denoise_index_rev[i1], denoise_index_rev[i2])



    res = np.empty(n, dtype=np.int_)
    res_cluster_id = <np.int_t*>PyMem_Malloc(n*sizeof(np.int_t))
    for i in range(n): res_cluster_id[i] = -1
    cdef np.int_t c = 0
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

    PyMem_Free(res_cluster_id)
    PyMem_Free(denoise_index)
    PyMem_Free(denoise_index_rev)
    PyMem_Free(prev_edge)
    PyMem_Free(next_edge)

    return res

