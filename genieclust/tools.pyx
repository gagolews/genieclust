# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



## TODO: (good first issue) Implement argkmax

## TODO: (good first issue) Implement ksmallest, klargest


"""
Auxiliary functions
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


cimport libc.math
from libcpp cimport bool
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





cdef extern from "../src/c_argfuns.h":
    # void Cargsort[T](Py_ssize_t* ret, T* x, Py_ssize_t n, bint stable)
    Py_ssize_t Cargkmin[T](T* x, Py_ssize_t n, Py_ssize_t k, Py_ssize_t* buf)
    # Py_ssize_t Cargkmin[T](T* x, Py_ssize_t n, Py_ssize_t k, Py_ssize_t* buf)  TODO




cpdef np.ndarray[Py_ssize_t] cummin(T[:] x):
    """
    genieclust.tools.cummin(x)

    Cumulative minimum


    Parameters
    ----------

    x : ndarray
        A vector with *n* elements of type ``int``, ``float``,
        or ``double``.


    Returns
    -------

    ndarray
        A vector of length *n* whose ``i``-th element
        is equal to ``min(x[:i])``.


    See Also
    --------

    genieclust.tools.cummax : Cumulative maximum


    Examples
    --------

    >>> genieclust.tools.cummin(np.r_[3, 4, 2, 1, 5, 6])
    array([3, 3, 2, 1, 1, 1])

    """
    cdef Py_ssize_t n = x.shape[0], i
    cdef np.ndarray[T] ret = np.empty_like(x)
    ret[0] = x[0]
    for i in range(1, n):
        if x[i] <= ret[i-1]:
            ret[i] = x[i]
        else:
            ret[i] = ret[i-1]

    return ret


cpdef np.ndarray[Py_ssize_t] cummax(T[:] x):
    """
    genieclust.tools.cummax(x)

    Cumulative maximum


    Parameters
    ----------

    x : ndarray
        A vector with *n* elements of type ``int``, ``float``,
        or ``double``.


    Returns
    -------

    ndarray
        A vector of length *n* whose ``i``-th element
        is equal to ``max(x[:i])``.


    See Also
    --------

    genieclust.tools.cummin : Cumulative minimum


    Examples
    --------

    >>> genieclust.tools.cummax(np.r_[3, 2, 1, 4, 6, 5])
    array([3, 3, 3, 4, 6, 6])

    """
    cdef Py_ssize_t n = x.shape[0], i
    cdef np.ndarray[T] ret = np.empty_like(x)
    ret[0] = x[0]
    for i in range(1, n):
        if x[i] >= ret[i-1]:
            ret[i] = x[i]
        else:
            ret[i] = ret[i-1]

    return ret





cpdef Py_ssize_t argkmin(np.ndarray[T] x, int k):
    """
    genieclust.tools.argkmin(x, k)

    Finds the position of an order statistic.
    It holds ``argkmin(x, 0) == argmin(x)``, or, more generally,
    ``argkmin(x, k) == np.argsort(x)[k]``.


    Run time is :math:`O(nk)` and working memory is :math:`O(k)`.
    An insertion sort-like scheme is used to locate the order statistic.
    In practice, the function is very fast for small `k` and randomly
    ordered or almost sorted (increasingly) data.


    Parameters
    ----------

    x : ndarray
        A vector with *n* elements of type ``int``, ``float``,
        or ``double``.

    k : int
        An integer between 0 and *n* - 1, preferably small.


    Returns
    -------

    int
        The index where the (`k`-1)-th smallest value in `x` is located.


    Examples
    --------

    >>> x = np.r_[2, 3, 6, 5, 1, 4]
    >>> genieclust.tools.argkmin(x, 0) # index of the smallest value
    4
    >>> genieclust.tools.argkmin(x, 1) # index of the 2nd smallest value
    0
    """
    x = np.asarray(x, dtype=x.dtype, order="C")  # ensure c_contiguity
    cdef Py_ssize_t n = x.shape[0]
    # TODO: allow returning the k indexes
    # TODO: argkmax
    return Cargkmin(&x[0], n, k, <Py_ssize_t*>0)



# cpdef np.ndarray[Py_ssize_t] _argsort(np.ndarray[T] x, bint stable=True):
#     """
#     (provided for testing only)
#
#     Finds the/an ordering permutation
#
#
#     Parameters
#     ----------
#
#     x : ndarray
#         A vector with *n* elements of type ``int``, ``float``,
#         or ``double``.
#
#     stable : bool
#         Should a stable (a bit slower) sorting algorithm be used?
#
#
#     Returns
#     -------
#
#     ndarray
#         An ordering permutation of `x`, an integer vector `o` of length *n*
#         with elements between 0 and *n* - 1
#         such that ``x[o[0]] <= x[o[1]] <= ... <= x[o[n-1]]``.
#
#
#     Notes
#     -----
#
#     Finds the/an ordering permutation of a ``c_contiguous`` array `x`
#     The ordering permutation is uniquely defined provided that
#     `stable` is ``True``. Otherwise *an* ordering permutation
#     will be generated.
#
#     """
#     x = np.asarray(x, dtype=x.dtype, order="C")  # ensure c_contiguity
#     cdef Py_ssize_t n = x.shape[0]
#     cdef np.ndarray[Py_ssize_t] ret = np.empty(n, dtype=np.intp)
#     Cargsort(&ret[0], &x[0], n, stable)
#     return ret
