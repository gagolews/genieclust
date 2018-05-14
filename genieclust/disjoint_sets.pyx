#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
Disjoint Sets (Union-Find)
A Python class to represent partitions of the set {0,1,...,n-1} for any n

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
from libc.math cimport fabs


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




