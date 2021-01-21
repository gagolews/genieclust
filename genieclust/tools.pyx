# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



## TODO: (good first issue) Implement argkmax


"""
Functions a person might find useful, but not necessarily
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2021, Marek Gagolewski <https://www.gagolewski.com>      #
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

    numpy.cumsum : Cumulative sum

    numpy.cumprod : Cumulative product


    Examples
    --------

    >>> genieclust.tools.cummin(np.r_[3, 4, 2, 1, 5, 6])
    array([3, 3, 2, 1, 1, 1])

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

    numpy.cumsum : Cumulative sum

    numpy.cumprod : Cumulative product


    Examples
    --------

    >>> genieclust.tools.cummax(np.r_[3, 2, 1, 4, 6, 5])
    array([3, 3, 3, 4, 6, 6])

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





cpdef ssize_t argkmin(np.ndarray[T] x, int k):
    """
    genieclust.tools.argkmin(x, k)

    Finds the position of an order statistic


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


    Notes
    -----

    It holds ``argkmin(x, 0) == argmin(x)``, or, more generally,
    ``argkmin(x, k) == np.argsort(x)[k]``.

    Run time is :math:`O(nk)` and working memory is :math:`O(k)`.
    An insertion sort-like scheme is used to locate the order statistic.
    In practice, the function is very fast for small `k` and randomly ordered
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


    Examples
    --------

    >>> x = np.r_[2, 3, 6, 5, 1, 4]
    >>> genieclust.tools.argkmin(x, 0) # index of the smallest value
    4
    >>> genieclust.tools.argkmin(x, 1) # index of the 2nd smallest value
    0
    """
    x = np.array(x, dtype=x.dtype, copy=False, order="C") # assure c_contiguity
    cdef ssize_t n = x.shape[0]
    return c_argfuns.Cargkmin(&x[0], n, k, <ssize_t*>0)



cpdef np.ndarray[ssize_t] _argsort(np.ndarray[T] x, bint stable=True):
    """
    genieclust.tools.argsort(x, stable=True)

    Finds the/an ordering permutation (provided for testing only)


    Parameters
    ----------

    x : ndarray
        A vector with *n* elements of type ``int``, ``float``,
        or ``double``.

    stable : bool
        Should a stable (a bit slower) sorting algorithm be used?


    Returns
    -------

    ndarray
        An ordering permutation of `x`, an integer vector `o` of length *n*
        with elements between 0 and *n* - 1
        such that ``x[o[0]] <= x[o[1]] <= ... <= x[o[n-1]]``.


    Notes
    -----

    Finds the/an ordering permutation of a ``c_contiguous`` array `x`
    The ordering permutation is uniquely defined provided that
    `stable` is ``True``. Otherwise *an* ordering permutation
    will be generated.

    """
    x = np.array(x, dtype=x.dtype, copy=False, order="C") # assure c_contiguity
    cdef ssize_t n = x.shape[0]
    cdef np.ndarray[ssize_t] ret = np.empty(n, dtype=np.intp)
    c_argfuns.Cargsort(&ret[0], &x[0], n, stable)
    return ret







cpdef np.ndarray[floatT] _core_distance(np.ndarray[floatT,ndim=2] dist, int M):
    """
    (provided for testing only)


    Given a pairwise distance matrix, computes the "core distance", i.e.,
    the distance of each point to its M-th nearest neighbour.
    Note that M==1 always yields all the distances equal to 0.0.
    The core distances are needed when computing the mutual reachability
    distance in the HDBSCAN* algorithm.

    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi: 10.1145/2733381.

    The input distance matrix for a given point cloud X may be computed,
    e.g., via a call to
    `scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))`.


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
        neighbour. The i-th point's 1st nearest neighbour is the i-th point itself.
    """
    cdef ssize_t n = dist.shape[0], i, j
    cdef floatT v
    cdef np.ndarray[floatT] d_core = np.zeros(n,
        dtype=np.float32 if floatT is float else np.float64)
    cdef floatT[::1] row

    if M < 1: raise ValueError("M < 1")
    if dist.shape[1] != n: raise ValueError("not a square matrix")
    if M >= n: raise ValueError("M >= matrix size")

    if M == 1: return d_core # zeros

    cdef vector[ssize_t] buf = vector[ssize_t](M)
    for i in range(n):
        row = dist[i,:]
        j = c_argfuns.Cargkmin(&row[0], row.shape[0], M-1, buf.data())
        d_core[i] = dist[i, j]

    return d_core


cpdef np.ndarray[floatT,ndim=2] _mutual_reachability_distance(
        np.ndarray[floatT,ndim=2] dist,
        np.ndarray[floatT] d_core):
    """
    (provided for testing only)


    Given a pairwise distance matrix,
    computes the mutual reachability distance w.r.t. the given
    core distance vector, see genieclust.internal.core_distance().

    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi: 10.1145/2733381.

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
    cdef ssize_t n = dist.shape[0], i, j
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



