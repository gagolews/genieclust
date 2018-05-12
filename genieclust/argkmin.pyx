#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
Find the index of the k-th smallest element in an array
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
import warnings


ctypedef fused arrayT:
    np.ndarray[np.double_t]
    np.ndarray[np.int_t]


cpdef np.int_t argkmin(arrayT x, np.int_t k):
    """
    Returns the index of the (k-1)-th smallest value in an array x,
    where argkmin(x, 0) == argmin(x), or, more generally,
    argkmin(x, k) == np.argsort(x)[k].

    Arguments:
    * x - an 1D-array x
    * k in {0,...,len(x)-1}, preferably small

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
