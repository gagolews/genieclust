# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
External cluster validity measures (partition similarity scores)
"""


# Adjusted- and Nonadjusted Rand Score,
# Adjusted- and Nonadjusted Fowlkes-Mallows Score,
# Adjusted-, Normalised and Nonadjusted Mutual Information Score,
# Normalised Accuracy, Pair Sets Index
# (for vectors of "small" ints)
#
#
# References
# ==========
#
# Hubert L., Arabie P., Comparing Partitions,
# Journal of Classification 2(1), 1985, pp. 193-218, esp. Eqs. (2) and (4)
#
# Vinh N.X., Epps J., Bailey J.,
# Information theoretic measures for clusterings comparison:
# Variants, properties, normalization and correction for chance,
# Journal of Machine Learning Research 11, 2010, pp. 2837-2854.
#
# Rezaei M., Franti P., Set matching measures for external cluster validity,
# IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
# doi:10.1109/TKDE.2016.2551240
#
#
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


cimport numpy as np
import numpy as np
from . cimport c_compare_partitions


cpdef np.ndarray[ssize_t,ndim=2] normalize_confusion_matrix(ssize_t[:, ::1] C):
    """
    Applies pivoting to a given confusion matrix


    Parameters
    ----------

    C : ndarray
        A ``c_contiguous`` confusion matrix (contingency table).


    Returns
    -------

    ndarray
        A normalised confusion matrix of the same shape as `C`.


    Notes
    -----

    This function permutes the columns of `C` to get the largest
    elements in each row on the main diagonal.

    This comes in handy whenever C actually summarises the results generated
    by clustering algorithms, where actual label values do not matter
    (e.g., (1, 2, 0) can be remapped to (0, 2, 1) with no change in meaning.


    Examples
    --------

    >>> ...
    """
    cdef np.ndarray[ssize_t,ndim=2] C_normalized = np.array(C, dtype=np.intp)
    cdef ssize_t xc = C_normalized.shape[0]
    cdef ssize_t yc = C_normalized.shape[1]
    c_compare_partitions.Capply_pivoting(&C_normalized[0,0], xc, yc)
    return C_normalized



cpdef np.ndarray[ssize_t,ndim=2] confusion_matrix(x, y):
    """
    Computes the confusion matrix (as a dense matrix)


    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths


    Returns
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
    if n != _y.shape[0]: raise ValueError("incompatible lengths")
    cdef ssize_t ymin, ymax
    c_compare_partitions.Cminmax(<ssize_t*>(&_y[0]), n, <ssize_t*>(&ymin), <ssize_t*>(&ymax))
    cdef ssize_t yc = (ymax-ymin+1)

    cdef ssize_t CONFUSION_MATRIX_MAXSIZE = 10000
    if xc*yc > CONFUSION_MATRIX_MAXSIZE:
        raise ValueError("CONFUSION_MATRIX_MAXSIZE exceeded")

    cdef np.ndarray[ssize_t,ndim=2] C = np.empty((xc, yc), dtype=np.intp)
    c_compare_partitions.Ccontingency_table(&C[0,0], xc, yc,
        xmin, ymin, &_x[0], &_y[0], n)
    return C



cpdef np.ndarray[ssize_t,ndim=2] normalized_confusion_matrix(x, y):
    """
    Computes the confusion matrix between x and y
    and applies pivoting. Nice for summarising clustering results,
    where actual labels do not matter.


    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
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
    normalised accuracy and pair sets index.

    Let `x` and `y` represent two partitions of a set of n
    elements into K and L, respectively,
    nonempty and pairwise disjoint subsets,
    e.g., two clusterings of a dataset with n observations
    represented as label vectors. Moreover, let C be the confusion
    matrix (with K rows and L columns, K<=L)
    corresponding to `x` and `y`, see also `confusion_matrix()`.
    This function implements scores that quantify the similarity between `x`
    and `y`. They can be used as external cluster
    validity measures, i.e., in the presence of reference (ground-truth)
    partitions.

    Every index except `mi_score()` (which computes the mutual
    information score) outputs 1 given two identical partitions.
    Note that partitions are always defined up to a bijection of the set of
    possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4)
    represent the same 2-partition.

    `rand_score()` gives the Rand score (the `probability' of agreement
    between the two partitions) and `adjusted_rand_score()` is its version
    corrected for chance, its expected value is 0.0 for two independent
    partitions. Due to the adjustment, the resulting index might also
    be negative for some inputs.

    Similarly, `fm_score()` gives the Fowlkes-Mallows (FM) score
    and `adjusted_fm_score()` is its adjusted-for-chance version.

    Note that both the (unadjusted) Rand and FM scores are bounded from below
    by $1/(K+1)$, where K is the number of clusters (unique labels
    in `x` and `y`), hence their adjusted versions are preferred.

    `mi_score()`, `adjusted_mi_score()` and `normalized_mi_score()` are
    information-theoretic scores, based on mutual information,
    see the definition of $AMI_{sum}$ and $NMI_{sum}$
    in (Vinh et al., 2010).

    `normalized_accuracy()` is defined as $(Accuracy(C_\sigma)-1/L)/(1-1/L)$,
    where $C_sigma$ is a version of the confusion matrix for given `x` and `y`,
    K<=L, with columns permuted based on the solution to the
    Maximal Linear Sum Assignment Problem.
    $Accuracy(C[sigma])$ is sometimes referred to as Purity,
    e.g., in (Rendon et al. 2011).

    `pair_sets_index()` gives the Pair Sets Index (PSI)
    adjusted for chance (Rezaei, Franti, 2016), K<=L.
    Pairing is based on the solution to the Linear Sum Assignment Problem
    of a transformed version of the confusion matrix.



    References
    ----------

    Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, pp. 193-218, esp. Eqs. (2) and (4)

    Rendon E., Abundez I., Arizmendi A., Quiroz E.M.,
    Internal versus external cluster validation indexes,
    International Journal of Computers and Communications 5(1), 2011, pp. 27-34.

    Rezaei M., Franti P., Set matching measures for external cluster validity,
    IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
    doi:10.1109/TKDE.2016.2551240

    Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.



    Parameters
    ----------

    C : ndarray, shape (xc, yc)
        a confusion matrix, xc <= yc


    Returns
    -------

    scores : dict
        a dictionary with keys 'ar', 'r', 'afm', 'fm', 'mi', 'nmi', 'ami',
        'nacc' and 'psi',
        giving the following scores: adjusted Rand, Rand,
        adjusted Fowlkes-Mallows, Fowlkes-Mallows
        mutual information, normalised mutual information (NMI_sum),
        adjusted mutual information (AMI_sum),
        normalised accuracy and pair sets index, respectively.
    """
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    if xc > yc:
        raise ValueError("number of rows in the confusion matrix \
            must be less than or equal to the number of columns")
    cdef dict res1 = c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc)
    cdef dict res2 = c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc)
    cdef double nacc = c_compare_partitions.Ccompare_partitions_nacc(&C[0,0], xc, yc)
    cdef double psi = c_compare_partitions.Ccompare_partitions_psi(&C[0,0], xc, yc)
    return {**res1,
            **res2,
            "nacc": nacc,
            "psi": psi}


cpdef dict compare_partitions2(x, y):
    """
    Calls compare_partitions(confusion_matrix(x, y)).


    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
    -------

    scores : dict
        see compare_partitions().
    """
    return compare_partitions(confusion_matrix(x, y))





cpdef double adjusted_rand_score(x, y):
    """
    The Rand index adjusted for chance.

    For more details, see compare_partitions().


    References
    ----------

    Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
    -------

    score : double
        partition similarity measure;
        by the very definition of this index,
        returned values might be negative.
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).ar


cpdef double rand_score(x, y):
    """
    The original Rand index (not adjusted for chance),
    that yields the `probability' of agreement between the two partitions.

    The index is bounded from below 1/(K+1), where K is the number of clusters
    (unique labels in `x` and `y`), hence its adjusted version are preferred,
    see `adjusted_rand_score()`.

    For more details, see compare_partitions().



    References
    ----------

    Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
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

    See Eqs. (2) and (4)  in (Hubert, Arabie, 1985).

    For more details, see compare_partitions().




    References
    ----------

    Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218


    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
    --------

    score : double
        partition similarity measure;
        by the very definition of this index,
        returned values might be negative.
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).afm


cpdef double fm_score(x, y):
    """
    The original Fowlkes-Mallows index (not adjusted for chance)

    The index is bounded from below 1/(K+1), where K is the number of clusters
    (unique labels in `x` and `y`), hence its adjusted version are preferred,
    see `adjusted_fm_score()`.

    For more details, see compare_partitions().





    References
    ----------

    Hubert L., Arabie P., Comparing Partitions,
    Journal of Classification 2(1), 1985, 193-218

    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
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

    For more details, see compare_partitions().




    References
    ----------

    Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
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

    For more details, see compare_partitions().




    References
    ----------

    Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
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

    For more details, see compare_partitions().




    References
    ----------

    Vinh N.X., Epps J., Bailey J.,
    Information theoretic measures for clusterings comparison:
    Variants, properties, normalization and correction for chance,
    Journal of Machine Learning Research 11, 2010, pp. 2837-2854.

    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
    -------

    score : double
        partition similarity measure;
        by definition of this index, returned values might be negative.
    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).ami



cpdef double normalized_accuracy(x, y):
    """
    Normalised Accuracy: (Accuracy(C[sigma])-1/L)/(1-1/L).

    C[sigma] is a version of the confusion matrix for given x and y
    with columns permuted based on the solution to the
    Maximal Linear Sum Assignment Problem.

    Accuracy(C[sigma]) is sometimes referred to as Purity,
    e.g., in (Rendon et al. 2011).

    It is assumed that y represents an L-partition
    and x represents a K-partition and that K<=L.

    For more details, see compare_partitions().




    References
    ----------

    Rendon E., Abundez I., Arizmendi A., Quiroz E.M.,
    Internal versus external cluster validation indexes,
    International Journal of Computers and Communications 5(1), 2011, pp. 27-34.

    Rezaei M., Franti P., Set matching measures for external cluster validity,
    IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
    doi:10.1109/TKDE.2016.2551240


    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
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
    return c_compare_partitions.Ccompare_partitions_nacc(&C[0,0], xc, yc)




cpdef double pair_sets_index(x, y):
    """
    Pair Sets Index (PSI) adjusted for chance

    Pairing is based on the solution to the Linear Sum Assignment Problem
    of a transformed version of the confusion matrix.

    It is assumed that y represents an L-partition
    and x represents a K-partition and that K<=L.

    For more details, see compare_partitions().




    References
    ----------

    Rezaei M., Franti P., Set matching measures for external cluster validity,
    IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
    doi:10.1109/TKDE.2016.2551240



    Parameters
    ----------

    x, y : ndarray, shape (n,)
        two small-int vectors of the same lengths, representing
        two partitions of the same set


    Returns
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
    return c_compare_partitions.Ccompare_partitions_psi(&C[0,0], xc, yc)

