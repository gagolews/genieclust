# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3




# Functions a Person Might Find Useful
#
# Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License
# Version 3, 19 November 2007, published by the Free Software Foundation.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License Version 3 for more details.
# You should have received a copy of the License along with this program.
# If not, see <https://www.gnu.org/licenses/>.






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



from . cimport c_argfuns


################################################################################
# cummin and cummax
################################################################################


cpdef np.ndarray[ssize_t] cummin(T[:] x):
    """
    cummin(x)

    Cumulative minimum


    Parameters
    ----------

    x : array_like
        A c_contiguous vector of ints, floats, or doubles of length n.


    Returns
    -------

    ndarray, shape (n,)
        It holds ``ret[i] = min(x[:i])`` for all ``i``.
    """
    cdef ssize_t n = x.shape[0], i
    cdef np.ndarray[T] ret = np.empty_like(x)
    ret[0] = x[0]
    for i in range(1, n):
        if x[i] <= ret[i-1]:
            ret[i] = x[i]
        else:
            ret[i] = ret[i-1]

    return ret


cpdef np.ndarray[ssize_t] cummax(T[:] x):
    """
    cummax(x)

    Cumulative maximum


    Parameters
    ----------

    x : array_like
        A c_contiguous vector of ints, floats, or doubles of length n.

    Returns
    -------

    ndarray, shape (n,)
        It holds ``ret[i] = max(x[:i])`` for all ``i``.
    """
    cdef ssize_t n = x.shape[0], i
    cdef np.ndarray[T] ret = np.empty_like(x)
    ret[0] = x[0]
    for i in range(1, n):
        if x[i] >= ret[i-1]:
            ret[i] = x[i]
        else:
            ret[i] = ret[i-1]

    return ret



cpdef np.ndarray[ssize_t] argsort(T[::1] x, bint stable=True):
    """
    argsort(x, stable=True)

    Finds the[1]_ ordering permutation of a c_contiguous array x


    Parameters
    ----------

    x : array_like
        A c_contiguous vector of ints, floats, or doubles of length n.

    stable : bool
        Should a stable (a bit slower) sorting algorithm be used?


    Returns
    -------

    ndarray
        The ordering permutation.


    Notes
    -----

    .. [1] The ordering permutation is uniquely defined provided that
       `stable` is True, otherwise it's *an* ordering permutation.

    """
    cdef ssize_t n = x.shape[0]
    cdef np.ndarray[ssize_t] ret = np.empty(n, dtype=np.intp)
    c_argfuns.Cargsort(&ret[0], &x[0], n, stable)
    return ret




cpdef ssize_t argkmin(T[::1] x, int k):
    """
    argkmin(x, k)

    Returns the index of the (k-1)-th smallest value in an array x


    Parameters
    ----------

    x : array_like
        A c_contiguous vector of ints, floats, or doubles of length n.

    k : int
        an integer in {0,...,n-1}, preferably small


    Returns
    -------

    int
        The index where the (k-1)-th smallest value in x is located.


    Notes
    -----

    It holds ``argkmin(x, 0) == argmin(x)``, or, more generally,
    ``argkmin(x, k) == np.argsort(x)[k]``.

    Run time is O(nk) and working mem is O(k). An insertion sort-like
    scheme is used to locate the order statistics.
    In practice, it's very fast for small k and randomly ordered
    or almost sorted (increasingly) data.

    ================================== =============== ==================
    Example timings                    `argkmin(x, k)` `np.argsort(x)[k]`
    ================================== =============== ==================
    (ascending)  n= 100000000, k=   1:      0.060s       4.388s
    (descending)                            0.168s       7.329s
    (random)                                0.073s      26.673s
    (ascending)  n= 100000000, k=   5:      0.060s       4.403s
    (descending)                            0.505s       7.414s
    (random)                                0.072s      26.447s
    (ascending)  n= 100000000, k= 100:      0.061s       4.390s
    (descending)                            8.007s       7.639s
    (random)                                0.075s      27.299s
    ================================== =============== ==================
    """
    cdef ssize_t n = x.shape[0]
    return c_argfuns.Cargkmin(&x[0], n, k, <ssize_t*>0)


