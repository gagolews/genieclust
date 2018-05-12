#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
Disjoint Sets (Union-Find)
A Python class to represent partitions of the set {0,1,...,n-1} for any n
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


cdef class DisjointSets:
    """
    Disjoint Sets over {0,1,...,n-1}
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
        returns number of subsets-1
        a.k.a. how many calls to union() can we still perform
        """
        return self.k-1


    cpdef np.int_t find(self, np.int_t x):
        """ finds the subset id for a given x """
        if x < 0 or x >= self.n: raise Exception("elem not in {0,1,...,n-1}")
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]


    cpdef np.int_t union(self, np.int_t x, np.int_t y):
        """ merges subsets containing given x and y  """
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
        Returns a list m such that m[i] denotes the parent id of i.
        """
        cdef np.int_t i
        cdef np.ndarray[np.int_t] m = np.empty(self.n, dtype=np.int_)
        for i in range(self.n):
            m[i] = self.find(i)
        return m


    cpdef np.ndarray[np.int_t] to_list_normalized(self):
        """
        Returns a list m such that m[i] denotes the normalized parent id of i.
        The resulting values are in {0,1,...,k-1}, where k is the current
        number of subsets.
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




