#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
The Genie Clustering Algorithm
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
#from sklearn.base import BaseEstimator, ClusterMixin
import scipy.spatial.distance

include "disjoint_sets.pyx"
include "mst.pyx"

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


cdef class Genie(): # (BaseEstimator, ClusterMixin):
    """
    The Genie Clustering Algorithm

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
    quality while retaining the single linkage speed,


    This is a new implementation of an O(n sqrt(n))-time algorithm
    (provided that the MST /minimum spanning tree of the
    pairwise distance graph/ has already been computed).


    Parameters:
    ----------

    n_clusters : int, default=2
        Number of clusters the data is split into.

    gini_threshold : float, default=0.3
        The threshold for the Genie correction

    metric : str or function, default="euclidean"
        See scipy.spatial.distance.pdist()


    Attributes:
    --------

    labels_ : ndarray, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit():
        an integer vector c with c[i] denoting the cluster id
        (in {0, ..., n_clusters-1}) of the i-th object.
    """

    cdef np.int_t n_clusters
    cdef np.double_t gini_threshold
    cdef str metric
    cdef object labels_

    def __cinit__(self,
                  np.int_t n_clusters=2,
                  np.double_t gini_threshold=0.3,
                  metric="euclidean"):
        self.n_clusters = n_clusters
        self.gini_threshold = gini_threshold
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
        and return the predicted labels.


        The MST may, for example, be determined as follows:

        mst = genieclust.mst.MST_pair(
            scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(X, "euclidean")),
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
        # Yup, computing the whole distance matrix here
        # @TODO@: change this
        mst = MST_pair(
            scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(X, self.metric)),
        )
        self.fit_from_mst(mst)


    cpdef fit_from_mst(self, tuple mst):
        """
        Compute a k-partition based on a precomputed MST.

        The MST may, for example, be determined as follows:

        mst = genieclust.mst.MST_pair(
            scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(X, "euclidean")),



        Parameters:
        ----------

        mst : tuple
            See genieclust.mst.MST_pair()
        """
        cdef np.int_t n, i, curidx, m, i1, i2, lastm, lastidx, previdx

        cdef np.ndarray[np.int_t,ndim=2] mst_i = mst[0]
        #cdef np.ndarray[np.double_t] mst_d = mst[1] -- not needed
        n = mst_i.shape[0]+1

        cdef np.int_t* next_edge = <np.int_t*>PyMem_Malloc(n*sizeof(np.int_t))
        cdef np.int_t* prev_edge = <np.int_t*>PyMem_Malloc(n*sizeof(np.int_t))
        for i in range(n-1):
            next_edge[i] = i+1
            prev_edge[i] = i-1

        cdef GiniDisjointSets ds = GiniDisjointSets(n)

        curidx = 0
        lastidx = 0
        lastm = 0
        for i in range(n-self.n_clusters):
            if ds.get_gini() > self.gini_threshold:
                m = ds.get_smallest_count()
                if m != lastm or lastidx < curidx:
                    lastidx = curidx
                #assert lastidx < n-1
                #assert lastidx >= 0

                while ds.get_count(mst_i[lastidx,0]) != m and ds.get_count(mst_i[lastidx,1]) != m:
                    lastidx = next_edge[lastidx]
                    #assert lastidx < n-1
                    #assert lastidx >= 0

                i1, i2 = mst_i[lastidx,0], mst_i[lastidx,1]

                if lastidx == curidx:
                    curidx = next_edge[curidx]
                    lastidx = curidx
                else:
                    previdx = prev_edge[lastidx]
                    lastidx = next_edge[lastidx]
                    #assert previdx < lastidx
                    #assert previdx >= 0
                    #assert lastidx < n
                    next_edge[previdx] = lastidx
                    prev_edge[lastidx] = previdx
                lastm = m


            else: # single linkage-like
                i1, i2 = mst_i[curidx,0], mst_i[curidx,1]
                curidx = next_edge[curidx]


            ds.union(i1, i2)


        PyMem_Free(prev_edge)
        PyMem_Free(next_edge)

        self.labels_ = ds.to_list_normalized()
