# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
External cluster validity measures and partition similarity scores

These indices can be used for comparing the outputs of clustering algorithms
with reference (ground truth) labels.

For more details, see the
`Framework for Benchmarking Clustering Algorithms
<https://clustering-benchmarks.gagolewski.com>`_.
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2023, Marek Gagolewski <https://www.gagolewski.com>      #
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


cimport numpy as np
import numpy as np
from . cimport c_compare_partitions


cpdef np.ndarray[Py_ssize_t,ndim=1] normalizing_permutation(C):
    """
    genieclust.compare_partitions.normalizing_permutation(C)

    Determines the permutation of columns of a confusion matrix
    so that the sum of the elements on the main diagonal is the largest
    possible (by solving the maximal assignment problem)


    Parameters
    ----------

    C : ndarray
        A confusion matrix (contingency table),
        whose row count is not greater than the column count


    Returns
    -------

    ndarray
        A vector of indexes


    See also
    --------

    genieclust.compare_partitions.confusion_matrix :
        Determines the confusion matrix

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible


    Notes
    -----

    This function comes in handy when `C` summarises the results generated
    by clustering algorithms, where the actual label values do not matter
    (e.g., (1, 2, 0) can be remapped to (0, 2, 1) with no change in meaning).


    Examples
    --------

    >>> x = np.r_[1, 2, 1, 2, 2, 2, 3, 1, 2, 1, 2, 1, 2, 2]
    >>> y = np.r_[3, 3, 3, 3, 2, 2, 3, 1, 2, 3, 2, 3, 2, 2]
    >>> C = genieclust.compare_partitions.confusion_matrix(x, y)
    >>> C
    array([[1, 0, 4],
           [0, 6, 2],
           [0, 0, 1]])
    >>> I = genieclust.compare_partitions.normalizing_permutation(C)
    >>> I
    array([2, 1, 0])
    >>> C[:, I]
    array([[4, 0, 1],
           [2, 6, 0],
           [1, 0, 0]])
    """
    cdef np.ndarray[double,ndim=2] _C = np.array(C, dtype=np.double)
    cdef Py_ssize_t xc = _C.shape[0]
    cdef Py_ssize_t yc = _C.shape[1]
    cdef np.ndarray[Py_ssize_t,ndim=1] perm = np.empty(yc, dtype=np.intp)
    if xc > yc:
        raise ValueError("number of rows cannot be greater than the number of columns")

    c_compare_partitions.Cnormalizing_permutation(&_C[0,0], xc, yc, &perm[0])

    return perm


cpdef np.ndarray[Py_ssize_t,ndim=2] normalize_confusion_matrix(C):
    """
    genieclust.compare_partitions.normalize_confusion_matrix(C)

    Permutes the rows and columns of a confusion matrix
    so that the sum of the elements
    on the main diagonal is the largest possible (by solving
    the maximal assignment problem)


    Parameters
    ----------

    C : ndarray
        A confusion matrix (contingency table),
        whose row count is not greater than the column count



    Returns
    -------

    ndarray
        A normalised confusion matrix of the same shape as `C`.


    See also
    --------

    genieclust.compare_partitions.confusion_matrix :
        Determines the confusion matrix

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible

    genieclust.compare_partitions.normalizing_permutation :
        The underlying function to determine the ordering permutation
        of the columns


    Notes
    -----

    This function comes in handy when `C` summarises the results generated
    by clustering algorithms, where the actual label values do not matter
    (e.g., (1, 2, 0) can be remapped to (0, 2, 1) with no change in meaning).


    Examples
    --------

    >>> x = np.r_[1, 2, 1, 2, 2, 2, 3, 1, 2, 1, 2, 1, 2, 2]
    >>> y = np.r_[3, 3, 3, 3, 2, 2, 3, 1, 2, 3, 2, 3, 2, 2]
    >>> C = genieclust.compare_partitions.confusion_matrix(x, y)
    >>> C
    array([[1, 0, 4],
           [0, 6, 2],
           [0, 0, 1]])
    >>> genieclust.compare_partitions.normalize_confusion_matrix(C)
    array([[4, 0, 1],
           [2, 6, 0],
           [1, 0, 0]])
    """
    cdef np.ndarray[Py_ssize_t,ndim=2] _C = np.array(C, dtype=np.intp)
    cdef np.ndarray[Py_ssize_t,ndim=2] C_normalized = np.array(_C, dtype=np.intp)
    cdef Py_ssize_t xc = C_normalized.shape[0]
    cdef Py_ssize_t yc = C_normalized.shape[1]
    if xc > yc:
        raise ValueError("number of rows cannot be greater than the number of columns")
    c_compare_partitions.Capply_pivoting(&_C[0,0], xc, yc, &C_normalized[0,0])
    return C_normalized


cpdef np.ndarray[Py_ssize_t,ndim=2] confusion_matrix(x, y):
    """
    genieclust.compare_partitions.confusion_matrix(x, y)

    Computes the confusion matrix for two label vectors


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths.


    Returns
    -------

    C : ndarray
        A (dense) confusion matrix (contingency table)
        with ``max(x)-min(x)+1`` rows
        and  ``max(y)-min(y)+1`` columns.


    See also
    --------

    genieclust.compare_partitions.normalize_confusion_matrix :
        Permutes the rows and columns of a confusion matrix so that
        the sum of the elements of the main diagonal is the largest possible


    Examples
    --------

    >>> x = np.r_[1, 2, 1, 2, 2, 2, 3, 1, 2, 1, 2, 1, 2, 2]
    >>> y = np.r_[3, 3, 3, 3, 2, 2, 3, 1, 2, 3, 2, 3, 2, 2]
    >>> C = genieclust.compare_partitions.confusion_matrix(x, y)
    >>> C
    array([[1, 0, 4],
           [0, 6, 2],
           [0, 0, 1]])

    """
    cdef np.ndarray[Py_ssize_t] _x = np.array(x, dtype=np.intp)
    cdef Py_ssize_t n = _x.shape[0]
    cdef Py_ssize_t xmin, xmax
    c_compare_partitions.Cminmax(<Py_ssize_t*>(&_x[0]), n, <Py_ssize_t*>(&xmin), <Py_ssize_t*>(&xmax))
    cdef Py_ssize_t xc = (xmax-xmin+1)

    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)
    if n != _y.shape[0]: raise ValueError("incompatible lengths")
    cdef Py_ssize_t ymin, ymax
    c_compare_partitions.Cminmax(<Py_ssize_t*>(&_y[0]), n, <Py_ssize_t*>(&ymin), <Py_ssize_t*>(&ymax))
    cdef Py_ssize_t yc = (ymax-ymin+1)

    cdef Py_ssize_t CONFUSION_MATRIX_MAXSIZE = 1000000
    if xc*yc > CONFUSION_MATRIX_MAXSIZE:
        raise ValueError("CONFUSION_MATRIX_MAXSIZE exceeded")

    cdef np.ndarray[Py_ssize_t,ndim=2] C = np.empty((xc, yc), dtype=np.intp)
    c_compare_partitions.Ccontingency_table(&C[0,0], xc, yc,
        xmin, ymin, &_x[0], &_y[0], n)
    return C



cpdef np.ndarray[Py_ssize_t,ndim=2] normalized_confusion_matrix(x, y):
    """
    genieclust.compare_partitions.normalized_confusion_matrix(x, y)

    Computes the confusion matrix for two label vectors and
    permutes its rows and columns so that the sum of the elements
    of the main diagonal is the largest possible (by solving
    the maximal assignment problem)


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths.

    use_sum : bool
        Whether the pivoting should be based on

    Returns
    -------

    C : ndarray
        A (dense) confusion matrix  (contingency table)
        with ``max(x)-min(x)+1`` rows
        and  ``max(y)-min(y)+1`` columns.


    See also
    --------

    genieclust.compare_partitions.normalizing_permutation :
        The underlying function to determine the ordering
        permutation of the columns

    genieclust.compare_partitions.confusion_matrix :
        Determines the confusion matrix



    Examples
    --------

    >>> x = np.r_[1, 2, 1, 2, 2, 2, 3, 1, 2, 1, 2, 1, 2, 2]
    >>> y = np.r_[3, 3, 3, 3, 2, 2, 3, 1, 2, 3, 2, 3, 2, 2]
    >>> genieclust.compare_partitions.normalized_confusion_matrix(x, y)
    array([[4, 0, 1],
           [2, 6, 0],
           [1, 0, 0]])

    """
    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    return normalize_confusion_matrix(C)



cpdef dict compare_partitions(Py_ssize_t[:,::1] C):
    """
    genieclust.compare_partitions.compare_partitions(C)

    Computes a series of external cluster validity measures


    Parameters
    ----------

    C : ndarray
        A ``c_contiguous`` confusion matrix (contingency table)
        with :math:`K` rows and :math:`L` columns.


    Returns
    -------

    scores : dict
        A dictionary with the following keys:

        ``'ar'``
            Adjusted Rand index
        ``'r'``
            Rand index  (unadjusted for chance)
        ``'afm'``
            Adjusted Fowlkes-Mallows index
        ``'fm'``
            Fowlkes-Mallows index (unadjusted for chance)
        ``'mi'``
            Mutual information score
        ``'nmi'``
            Normalised mutual information :math:`(\\mathrm{NMI}_\\mathrm{sum})`
        ``'ami'``
            Adjusted mutual information   :math:`(\\mathrm{AMI}_\\mathrm{sum})`
        ``'nacc'``
            Normalised (set-matching) accuracy
        ``'psi'``
            Pair sets index
        ``'spsi'``
            Simplified pair sets index
        ``'aaa'``
            Adjusted asymmetric accuracy;
            it is assumed that rows in `C` represent the ground-truth
            partition


    See also
    --------

    genieclust.compare_partitions.confusion_matrix :
        Computes a confusion matrix

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible

    genieclust.compare_partitions.compare_partitions2 :
        A wrapper around this function that accepts two label vectors on input


    genieclust.compare_partitions.adjusted_asymmetric_accuracy
    genieclust.compare_partitions.normalized_accuracy
    genieclust.compare_partitions.pair_sets_index
    genieclust.compare_partitions.adjusted_rand_score
    genieclust.compare_partitions.rand_score
    genieclust.compare_partitions.adjusted_fm_score
    genieclust.compare_partitions.fm_score
    genieclust.compare_partitions.mi_score
    genieclust.compare_partitions.normalized_mi_score
    genieclust.compare_partitions.adjusted_mi_score




    Notes
    -----

    Let `x` and `y` represent two partitions of the same set with :math:`n`
    elements into, respectively, :math:`K` and :math:`L`
    nonempty and pairwise disjoint subsets.
    For instance, these can be two clusterings of a dataset with :math:`n`
    observations specified as vectors of labels. Moreover, let `C` be the
    confusion matrix with :math:`K` rows and :math:`L` columns,
    corresponding to `x` and `y`; see also
    :func:`confusion_matrix`.

    This function implements a few scores that aim to quantify
    the similarity between `x` and `y`.

    These functions can be used as external cluster
    validity measures, where we assume that `x` is
    the reference (ground-truth) partition; compare [5]_.

    Each index except `adjusted_asymmetric_accuracy`
    can act as a pairwise partition similarity score: it is symmetric,
    i.e., ``index(x, y) == index(y, x)``.

    Each index except `mi_score` (which computes the mutual information score)
    outputs the value of 1.0 if two identical partitions are given.
    Note that partitions are always defined up to a bijection of the set of
    possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4)
    represent the same 2-partition.

    `adjusted_asymmetric_accuracy` [2]_ is an external cluster validity measure
    which assumes that the label vector `x` (or rows in the confusion
    matrix) represents the reference (ground truth) partition.
    It is a corrected-for-chance summary of the proportion of correctly
    classified points in each cluster (with cluster matching based on the
    solution to the maximal linear sum assignment problem; see
    :func:`normalize_confusion_matrix`), given by:
    :math:`(\\max_\\sigma \\sum_{i=1}^K (c_{i, \sigma(i)}/(c_{i, 1}+...+c_{i, K})) - 1)/(K - 1)`,
    where :math:`C` is the confusion matrix.

    `normalized_accuracy` is a measure defined as
    :math:`(\\mathrm{Accuracy}(C_\\sigma)-1/\\max(K,L))/(1-1/\\max(K,L))`,
    where :math:`C_\\sigma` is a version of the confusion matrix
    for given `x` and `y` with columns permuted
    based on the solution to the maximal linear sum assignment problem.
    Note that the :math:`\\mathrm{Accuracy}(C_\\sigma)` part
    is sometimes referred to as set-matching classification
    rate or pivoted accuracy.

    `pair_sets_index` gives the Pair Sets Index (PSI)
    adjusted for chance [3]_.
    Pairing is based on the solution to the linear sum assignment problem
    of a transformed version of the confusion matrix.
    Its simplified version assumes E=1 in the definition of the index,
    i.e., uses Eq. (20) instead of (18); see [3]_.

    `rand_score` gives the Rand score (the "probability" of agreement
    between the two partitions) and `adjusted_rand_score` is its version
    corrected for chance [1]_ (especially Eqs. (2) and (4) therein):
    its expected value is 0.0 for two independent
    partitions. Due to the adjustment, the resulting index might also
    be negative for some inputs.

    Similarly, `fm_score` gives the Fowlkes-Mallows (FM) score
    and `adjusted_fm_score` is its adjusted-for-chance version [1]_.

    Note that both the (unadjusted) Rand and FM scores are bounded from below
    by :math:`1/(K+1)` if :math:`K = L`, hence their adjusted versions
    are preferred.

    `mi_score`, `adjusted_mi_score` and `normalized_mi_score` are
    information-theoretic indices based on mutual information,
    see the definition of :math:`\\mathrm{AMI}_\\mathrm{sum}`
    and :math:`\\mathrm{NMI}_\\mathrm{sum}` in [4]_.



    References
    ----------

    .. [1]
        Hubert L., Arabie P., Comparing Partitions,
        *Journal of Classification* 2(1), 1985, 193-218.

    .. [2]
        Gagolewski M., Adjusted asymmetric accuracy: A well-behaving external
        cluster validity measure, 2022, under review (preprint).
        https://doi.org/10.48550/arXiv.2209.02935.

    .. [3]
        Rezaei M., Franti P., Set matching measures for external cluster validity,
        *IEEE Transactions on Knowledge and Data Mining* 28(8), 2016,
        2173-2186. https://doi.org/10.1109/TKDE.2016.2551240.

    .. [4]
        Vinh N.X., Epps J., Bailey J.,
        Information theoretic measures for clusterings comparison:
        Variants, properties, normalization and correction for chance,
        *Journal of Machine Learning Research* 11, 2010, 2837-2854.

    .. [5]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com


    Examples
    --------

    >>> x = np.r_[1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2]
    >>> y = np.r_[2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1]
    >>> C = genieclust.compare_partitions.confusion_matrix(x, y)
    >>> C
    array([[ 1, 10],
           [ 8,  2]])
    >>> {k : round(v, 2) for k, v in
    ...      genieclust.compare_partitions.compare_partitions(C).items()}
    {'ar': 0.49, 'r': 0.74, 'fm': 0.73, 'afm': 0.49, 'mi': 0.29, 'nmi': 0.41, 'ami': 0.39, 'nacc': 0.71, 'psi': 0.65, 'spsi': 0.63, 'aaa': 0.71}
    >>> {k : round(v, 2) for k, v in
    ...      genieclust.compare_partitions.compare_partitions2(x,y).items()}
    {'ar': 0.49, 'r': 0.74, 'fm': 0.73, 'afm': 0.49, 'mi': 0.29, 'nmi': 0.41, 'ami': 0.39, 'nacc': 0.71, 'psi': 0.65, 'spsi': 0.63, 'aaa': 0.71}
    >>> round(genieclust.compare_partitions.adjusted_asymmetric_accuracy(x, y), 2)
    0.71

    """
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]

    cdef dict res1 = c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc)

    cdef dict res2 = c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc)

    cdef double nacc = c_compare_partitions.Ccompare_partitions_nacc(&C[0,0], xc, yc)

    cdef dict res3 = c_compare_partitions.Ccompare_partitions_psi(&C[0,0], xc, yc)
    cdef dict res3_clipped = {
        "psi": res3["psi_unclipped"],
        "spsi": res3["spsi_unclipped"],
    }

    if res3_clipped["psi"]  < 0.0: res3_clipped["psi"]  = 0.0
    if res3_clipped["spsi"] < 0.0: res3_clipped["spsi"] = 0.0

    cdef double aaa = np.nan
    if xc == yc:
        aaa = c_compare_partitions.Ccompare_partitions_aaa(&C[0,0], xc, yc)

    return {
        **res1,
        **res2,
        "nacc": nacc,
        **res3_clipped,
        "aaa": aaa,
    }




cpdef dict compare_partitions2(x, y):
    """
    genieclust.compare_partitions.compare_partitions2(x, y)

    Computes a series of partition similarity scores


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    scores : dict
        See `genieclust.compare_partitions.compare_partitions`.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        The underlying function

    genieclust.compare_partitions.confusion_matrix :
        Determines the contingency table

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible


    Notes
    -----

    Calls :func:`compare_partitions` on the result of
    returned by :func:`confusion_matrix`.


    """
    return compare_partitions(confusion_matrix(x, y))





cpdef double adjusted_rand_score(x, y):
    """
    genieclust.compare_partitions.adjusted_rand_score(x, y)

    The Rand index adjusted for chance


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.

    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).ar


cpdef double rand_score(x, y):
    """
    genieclust.compare_partitions.rand_score(x, y)

    The original Rand index not adjusted for chance


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).r


cpdef double adjusted_fm_score(x, y):
    """
    genieclust.compare_partitions.adjusted_fm_index(x, y)

    The Fowlkes-Mallows index adjusted for chance


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).afm


cpdef double fm_score(x, y):
    """
    genieclust.compare_partitions.fm_score(x, y)

    The original Fowlkes-Mallows index (not adjusted for chance)


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_pairs(&C[0,0], xc, yc).fm


cpdef double mi_score(x, y):
    """
    genieclust.compare_partitions.mi_score(x, y)

    Mutual information score


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).mi



cpdef double normalized_mi_score(x, y):
    """
    genieclust.compare_partitions.normalised_mi_score(x, y)

    Normalised mutual information score :math:`(\\mathrm{NMI}_\\mathrm{sum})`


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).nmi


cpdef double adjusted_mi_score(x, y):
    """
    genieclust.compare_partitions.adjusted_mi_score(x, y)

    Adjusted mutual information score :math:`(\\mathrm{AMI}_\\mathrm{sum})`


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).ami



cpdef double normalized_accuracy(x, y):
    """
    genieclust.compare_partitions.normalized_accuracy(x, y)

    Normalised accuracy


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix

    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible


    Notes
    -----

    See :func:`compare_partitions` for more details.

    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_nacc(&C[0,0], xc, yc)



cpdef double adjusted_asymmetric_accuracy(x, y):
    """
    genieclust.compare_partitions.adjusted_asymmetric_accuracy(x, y)

    Adjusted asymmetric accuracy (AAA) [1]_.



    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.
        `x` is the set of ground truth (reference) labels
        and `y` is a partition whose quality we would like to asses


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix

    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible


    Notes
    -----

    Let :math:`C` be a confusion matrix with :math:`K` rows
    and :math:`L` columns.
    AAA is an external cluster validity measure.
    It is a corrected-for-chance summary of the proportion of correctly
    classified points in each cluster (with cluster matching based on the
    solution to the maximal linear sum assignment problem; see
    :func:`normalize_confusion_matrix`), given by:
    :math:`(\\max_\\sigma \\sum_{i=1}^K (c_{i, \sigma(i)}/(c_{i, 1}+...+c_{i, K})) - 1)/(K - 1)`.
    Missing columns are treated as if they were filled with 0s.

    Note that this measure is not symmetric, i.e., ``index(x, y)`` does not
    have to be equal to ``index(y, x)``.

    See [1]_ for more details and :func:`compare_partitions` for more functions.


    References
    ----------

    .. [1]
        Gagolewski M., Adjusted asymmetric accuracy: A well-behaving external
        cluster validity measure, 2022, under review (preprint).
        https://doi.org/10.48550/arXiv.2209.02935

    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_aaa(&C[0,0], xc, yc)


cpdef double pair_sets_index(x, y, bint simplified=False):
    """
    genieclust.compare_partitions.pair_sets_index(x, y)

    Pair Sets Index (PSI) adjusted for chance


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths,
        representing two partitions of the same set.

    simplified : bool
        Whether to assume E=1 in the definition of the index,
        i.e., use Eq. (20) instead of (18); see [1]_.

    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores based on a confusion matrix

    genieclust.compare_partitions.compare_partitions2 :
        Computes multiple similarity scores based on two label vectors

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible


    Notes
    -----

    See :func:`compare_partitions` for more details.


    References
    ----------

    .. [1]
        Rezaei M., Franti P., Set matching measures for external cluster validity,
        *IEEE Transactions on Knowledge and Data Mining* 28(8), 2016,
        2173-2186. https://doi.org/10.1109/TKDE.2016.2551240.

    """

    cdef np.ndarray[Py_ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    cdef double res

    if simplified:
        res = c_compare_partitions.Ccompare_partitions_psi(&C[0,0], xc, yc).spsi_unclipped
    else:
        res = c_compare_partitions.Ccompare_partitions_psi(&C[0,0], xc, yc).psi_unclipped

    if res < 0.0: res = 0.0

    return res
