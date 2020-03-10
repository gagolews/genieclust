# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Adjusted- and Nonadjusted Rand Score,
Adjusted- and Nonadjusted Fowlkes-Mallows Score
(for vectors of `small' ints)

See Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


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


cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np
import numpy as np
from libc.math cimport fabs, sqrt
from numpy.math cimport INFINITY
import scipy.spatial.distance
import warnings

ctypedef fused intT:
    int
    long
    long long
    ssize_t



cdef struct RandResult:
    double ar
    double r
    double fm
    double afm


cpdef np.ndarray[ssize_t,ndim=2] normalize_confusion_matrix(ssize_t[:,:] C):
    """
    Applies pivoting to a given confusion matrix.
    Nice if C actually summarises clustering results,
    where actual labels do not matter.

    Parameters:
    ----------

    C : ndarray, shape (kx,ky)
        a confusion matrix


    Returns:
    -------

    C_normalized: ndarray, shape(kx,ky)
    """
    cdef np.ndarray[ssize_t,ndim=2] C2 = np.array(C, dtype=np.intp)
    cdef ssize_t xc = C2.shape[0], yc = C2.shape[1]
    cdef ssize_t i, j, w

    for i in range(xc-1):
        w = i
        for j in range(i+1, yc): # find w = argmax C[i,w], w=i,i+1,...yc-1
            if C2[i,w] < C2[i,j]: w = j
        for j in range(xc): # swap columns i and w
            C2[j,i], C2[j,w] = C2[j,w], C2[j,i]

    return C2


cpdef np.ndarray[ssize_t,ndim=2] confusion_matrix(intT[:] x, intT[:] y):
    """
    Computes the confusion matrix (as a dense matrix)


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths


    Returns:
    -------

    C : ndarray, shape (kx, ky)
        a confusion matrix
    """
    cdef ssize_t n = x.shape[0], i
    if n != y.shape[0]: raise ValueError("incompatible lengths")

    cdef intT xmin = x[0], ymin = y[0]
    cdef intT xmax = x[0], ymax = y[0]
    for i in range(1, n):
        if   x[i] < xmin: xmin = x[i]
        elif x[i] > xmax: xmax = x[i]

        if   y[i] < ymin: ymin = y[i]
        elif y[i] > ymax: ymax = y[i]

    cdef ssize_t xc = (xmax-xmin+1)
    cdef ssize_t yc = (ymax-ymin+1)

    # if xc == yc == 1 or xc == yc == 0 or xc == yc == n: return 1.0

    if xc*yc > 10000:
        raise ValueError("max_size of the confusion matrix exceeded")

    cdef np.ndarray[ssize_t,ndim=2] C = np.zeros((xc, yc), dtype=np.intp)
    for i in range(n):
        C[x[i]-xmin, y[i]-ymin] += 1

    return C


cpdef np.ndarray[ssize_t,ndim=2] normalized_confusion_matrix(intT[:] x, intT[:] y):
    """
    Computes the confusion matrix between x and y
    and applies pivoting. Nice for summarising clustering results,
    where actual labels do not matter.


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two k-partitions of the same set


    Returns:
    -------

    C : ndarray, shape (kx, ky)
        a confusion matrix
    """
    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    return normalize_confusion_matrix(C)



cpdef RandResult compare_partitions(ssize_t[:,:] C):
    """
    Computes the adjusted and nonadjusted Rand- and FM scores
    based on a given confusion matrix.

    See Hubert L., Arabie P., Comparing Partitions,
       Journal of Classification 2(1), 1985, 193-218, esp. Eqs. (2) and (4)


    Parameters:
    ----------

    C : ndarray, shape (kx,ky)
        a confusion matrix

    Returns:
    -------

    scores : dict
        a dictionary with keys 'ar', 'r', 'afm', 'fm', giving
        the adjusted Rand, Rand, adjusted Fowlkes-Mallows, and
        Fowlkes-Mallows scores, respectively.
    """
    cdef ssize_t xc = C.shape[0], yc = C.shape[1], i, j
    cdef ssize_t n = 0
    for i in range(xc):
        for j in range(yc):
            n += C[i, j]

    cdef double sum_comb_x = 0.0, sum_comb = 0.0, sum_comb_y = 0.0
    cdef double t, prod_comb, mean_comb, e_fm
    for i in range(xc):
        t = 0.0
        for j in range(yc):
            t += C[i, j]
            sum_comb += C[i, j]*(C[i, j]-1.0)*0.5
        sum_comb_x += t*(t-1.0)*0.5 # comb2(t)

    for j in range(yc):
        t = 0.0
        for i in range(xc):
            t += C[i, j]
        sum_comb_y += t*(t-1.0)*0.5 # comb2(t)


    prod_comb = (sum_comb_x*sum_comb_y)/n/(n-1.0)*2.0 # expected sum_comb,
                                        # see Eq.(2) in (Hubert, Arabie, 1985)
    mean_comb = (sum_comb_x+sum_comb_y)*0.5
    e_fm = prod_comb/sqrt(sum_comb_x*sum_comb_y) # expected FM (variant)

    cdef RandResult res
    res.ar  = (sum_comb-prod_comb)/(mean_comb-prod_comb)
    res.r   = 1.0 + (2.0*sum_comb - (sum_comb_x+sum_comb_y))/n/(n-1.0)*2.0
    res.fm  = sum_comb/sqrt(sum_comb_x*sum_comb_y)
    res.afm = (res.fm - e_fm)/(1.0 - e_fm) # Eq.(4) in (Hubert, Arabie, 1985)

    return res


cpdef RandResult compare_partitions2(intT[:] x, intT[:] y):
    """
    Calls compare_partitions(confusion_matrix(x, y)).


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two k-partitions of the same set

    Returns:
    -------

    scores : dict
        a dictionary with keys 'ar', 'r', 'afm', 'fm', giving
        the adjusted Rand, Rand, adjusted Fowlkes-Mallows, and
        Fowlkes-Mallows scores, respectively.
    """
    return compare_partitions(confusion_matrix(x, y))


cpdef double adjusted_rand_score(intT[:] x, intT[:] y):
    """
    The Rand index adjusted for chance.

    See Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two k-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    return compare_partitions(confusion_matrix(x, y)).ar


cpdef double rand_score(intT[:] x, intT[:] y):
    """
    The original Rand index (not adjusted for chance),
    yielding the `probability' of agreement between the two partitions

    See Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two k-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    return compare_partitions(confusion_matrix(x, y)).r


cpdef double adjusted_fm_score(intT[:] x, intT[:] y):
    """
    The Fowlkes-Mallows index adjusted for chance,

    See Eqs. (2) and (4)  in Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218

    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two k-partitions of the same set


    Returns:

    score : double
        partition similarity measure
    """

    return compare_partitions(confusion_matrix(x, y)).afm


cpdef double fm_score(intT[:] x, intT[:] y):
    """
    The original Fowlkes-Mallows index (not adjusted for chance)

    See Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218

    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two k-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    return compare_partitions(confusion_matrix(x, y)).fm
