# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Adjusted- and Nonadjusted Rand Score,
Adjusted- and Nonadjusted Fowlkes-Mallows Score,
Adjusted-, Normalised and Nonadjusted Mutual Information Score,
Normalised Purity, Pair Sets Index
(for vectors of "small" ints)


References
==========

Hubert L., Arabie P., Comparing Partitions,
Journal of Classification 2(1), 1985, pp. 193-218, esp. Eqs. (2) and (4)

Vinh N.X., Epps J., Bailey J.,
Information theoretic measures for clusterings comparison:
Variants, properties, normalization and correction for chance,
Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

Rezaei M., Franti P., Set matching measures for external cluster validity,
IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
doi:10.1109/TKDE.2016.2551240



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


cimport numpy as np
import numpy as np
from . cimport c_compare_partitions


cpdef np.ndarray[ssize_t,ndim=2] normalize_confusion_matrix(ssize_t[:, ::1] C):
    """
    Applies pivoting to a given confusion matrix.
    Nice if C actually summarises clustering results,
    where actual labels do not matter.

    Parameters:
    ----------

    C : ndarray, shape (xc,yc)
        a c_contiguous confusion matrix


    Returns:
    -------

    C_normalized: ndarray, shape(xc,yc)
    """
    cdef np.ndarray[ssize_t,ndim=2] C_normalized = np.array(C, dtype=np.intp)
    cdef ssize_t xc = C_normalized.shape[0]
    cdef ssize_t yc = C_normalized.shape[1]
    c_compare_partitions.Capply_pivoting(&C_normalized[0,0], xc, yc)
    return C_normalized



cpdef np.ndarray[ssize_t,ndim=2] confusion_matrix(x, y):
    """
    Computes the confusion matrix (as a dense matrix)


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths


    Returns:
    -------

    C : ndarray, shape (xc, yc)
        a confusion matrix
    """
    cdef np.ndarray[ssize_t] _x = np.array(x, dtype=np.intp)
    cdef ssize_t n = _x.shape[0]
    cdef ssize_t xmin, xmax
    c_compare_partitions.Cminmax(<ssize_t*>(&_x[0]), n, <ssize_t*>(&xmin), <ssize_t*>(&xmax))
    cdef ssize_t xc = (xmax-xmin+1)

    cdef np.ndarray[ssize_t] _y = np.array(y, dtype=np.intp)
    if n != y.shape[0]: raise ValueError("incompatible lengths")
    cdef ssize_t ymin, ymax
    c_compare_partitions.Cminmax(<ssize_t*>(&_y[0]), n, <ssize_t*>(&ymin), <ssize_t*>(&ymax))
    cdef ssize_t yc = (ymax-ymin+1)

    cdef ssize_t CONFUSION_MATRIX_MAXSIZE = 10000
    if xc*yc > CONFUSION_MATRIX_MAXSIZE:
        raise ValueError("CONFUSION_MATRIX_MAXSIZE exceeded")

    cdef np.ndarray[ssize_t,ndim=2] C = np.empty((xc, yc), dtype=np.intp)
    c_compare_partitions.Ccontingency_table(&C[0,0], xc, yc, xmin, ymin, &_x[0], &_y[0], n)
    return C



cpdef np.ndarray[ssize_t,ndim=2] normalized_confusion_matrix(x, y):
    """
    Computes the confusion matrix between x and y
    and applies pivoting. Nice for summarising clustering results,
    where actual labels do not matter.


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    C : ndarray, shape (xc, yc)
        a confusion matrix
    """
    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    return normalize_confusion_matrix(C)



cpdef dict compare_partitions(ssize_t[:,::1] C):
    """
    Computes the adjusted and nonadjusted Rand- and FM scores,
    nonadjusted, normalised and adjusted mutual information scores,
    normalised purity and pair sets index.


    References
    ==========

    Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, pp. 193-218, esp. Eqs. (2) and (4)

    Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

    Rezaei M., Franti P., Set matching measures for external cluster validity,
    IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
    doi:10.1109/TKDE.2016.2551240


    Parameters:
    ----------

    C : ndarray, shape (xc, yc)
        a confusion matrix


    Returns:
    -------

    scores : dict
        a dictionary with keys 'ar', 'r', 'afm', 'fm', 'mi', 'nmi', 'ami', 'npur' and 'psi'
        giving the following scores: adjusted Rand, Rand,
        adjusted Fowlkes-Mallows, Fowlkes-Mallows
        mutual information, normalised mutual information (NMI_sum),
        adjusted mutual information (AMI_sum),
        normalised purity and pair sets index, respectively.
    """
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    if xc > yc:
        raise ValueError("number of rows in the confusion matrix \
            must be less than or equal to the number of columns")
    cdef dict res1 = c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc)
    cdef dict res2 = c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc)
    cdef dict res3 = c_compare_partitions.Ccompare_partitions_match(&C[0,0], xc, yc)
    return {**res1, **res2, **res3}


cpdef dict compare_partitions2(x, y):
    """
    Calls compare_partitions(confusion_matrix(x, y)).


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set

    Returns:
    -------

    scores : dict
        see compare_partitions().
    """
    return compare_partitions(confusion_matrix(x, y))





cpdef double adjusted_rand_score(x, y):
    """
    The Rand index adjusted for chance.

    See Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure;
        by definition of this index, returned values might be negative.
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).ar


cpdef double rand_score(x, y):
    """
    The original Rand index (not adjusted for chance),
    yielding the `probability' of agreement between the two partitions

    See Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).r


cpdef double adjusted_fm_score(x, y):
    """
    The Fowlkes-Mallows index adjusted for chance,

    See Eqs. (2) and (4)  in Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218

    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:

    score : double
        partition similarity measure;
        by definition of this index, returned values might be negative.
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).afm


cpdef double fm_score(x, y):
    """
    The original Fowlkes-Mallows index (not adjusted for chance)

    See Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218

    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).fm


cpdef double mi_score(x, y):
    """
    Mutual information score

    See: Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).mi



cpdef double normalized_mi_score(x, y):
    """
    Normalised mutual information score (NMI_sum)

    See: Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).nmi


cpdef double adjusted_mi_score(x, y):
    """
    Adjusted mutual information score (AMI_sum)

    See: Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure;
        by definition of this index, returned values might be negative.
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).ami



cpdef double normalised_purity(x, y):
    """
    Normalised Purity: (Purity-1/K)/(1-1/K).

    Purity is similar to Accuracy, however it permutes the columns of the
    confusion matrix based on the solution to the Linear Sum Assignment Problem.


    See, e.g.: Rezaei M., Franti P., Set matching measures for external cluster validity,
    IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
    doi:10.1109/TKDE.2016.2551240


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    if xc > yc:
        raise ValueError("number of rows in the confusion matrix \
            must be less than or equal to the number of columns")
    return c_compare_partitions.Ccompare_partitions_match(&C[0,0], xc, yc).npur




cpdef double pair_sets_index(x, y):
    """
    Pair Sets Index (PSI) adjusted for chance

    Pairing is based on the solution to the Linear Sum Assignment Problem
    of the confusion matrix.


    See, e.g.: Rezaei M., Franti P., Set matching measures for external cluster validity,
    IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
    doi:10.1109/TKDE.2016.2551240


    Parameters:
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two K-partitions of the same set


    Returns:
    -------

    score : double
        partition similarity measure
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    if xc > yc:
        raise ValueError("number of rows in the confusion matrix \
            must be less than or equal to the number of columns")
    return c_compare_partitions.Ccompare_partitions_match(&C[0,0], xc, yc).psi

