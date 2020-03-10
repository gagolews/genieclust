# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
#############################################################################
# Disjoint Sets (Union-Find)
# A Python class to represent partitions of the set {0,1,...,n-1} for any n
#############################################################################

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


from . cimport c_gini_disjoint_sets
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector



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
        cdef vector[ssize_t] counts = self.ds.get_counts()
        cdef ssize_t k = counts.size(), i
        cdef np.ndarray[ssize_t] out = np.empty(k, dtype=np.intp)
        for i in range(k):
            out[i] = counts[i]
        return out


    def __repr__(self):
        """
        Calls self.to_lists()
        """
        return "GiniDisjointSets("+repr(self.to_lists())+")"
