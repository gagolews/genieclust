# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Partition similarity scores
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


cimport numpy as np
import numpy as np
from . cimport c_compare_partitions


cpdef np.ndarray[ssize_t,ndim=2] normalize_confusion_matrix(ssize_t[:, ::1] C):
    """
    genieclust.compare_partitions.normalize_confusion_matrix(C)

    Applies pivoting to a given confusion matrix


    Parameters
    ----------

    C : ndarray
        A ``c_contiguous`` confusion matrix (contingency table).


    Returns
    -------

    ndarray
        A normalised confusion matrix of the same shape as `C`.


    See also
    --------

    genieclust.compare_partitions.confusion_matrix :
        Determines the confusion matrix


    Notes
    -----

    This function permutes the columns of `C` so as to relocate the largest
    elements in each row onto the main diagonal.

    It may come in handy whenever `C` summarises the results generated
    by clustering algorithms, where actual label values do not matter
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
    cdef np.ndarray[ssize_t,ndim=2] C_normalized = np.array(C, dtype=np.intp)
    cdef ssize_t xc = C_normalized.shape[0]
    cdef ssize_t yc = C_normalized.shape[1]
    c_compare_partitions.Capply_pivoting(&C_normalized[0,0], xc, yc)
    return C_normalized




cpdef np.ndarray[ssize_t,ndim=2] confusion_matrix(x, y):
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
        Applies pivoting


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

    cdef ssize_t CONFUSION_MATRIX_MAXSIZE = 1000000
    if xc*yc > CONFUSION_MATRIX_MAXSIZE:
        raise ValueError("CONFUSION_MATRIX_MAXSIZE exceeded")

    cdef np.ndarray[ssize_t,ndim=2] C = np.empty((xc, yc), dtype=np.intp)
    c_compare_partitions.Ccontingency_table(&C[0,0], xc, yc,
        xmin, ymin, &_x[0], &_y[0], n)
    return C



cpdef np.ndarray[ssize_t,ndim=2] normalized_confusion_matrix(x, y):
    """
    genieclust.compare_partitions.normalized_confusion_matrix(x, y)

    Computes the confusion matrix for two label vectors and applies pivoting


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths.


    Returns
    -------

    C : ndarray
        A (dense) confusion matrix  (contingency table)
        with ``max(x)-min(x)+1`` rows
        and  ``max(y)-min(y)+1`` columns.


    See also
    --------

    genieclust.compare_partitions.normalize_confusion_matrix :
        Applies pivoting

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
    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    return normalize_confusion_matrix(C)



cpdef dict compare_partitions(ssize_t[:,::1] C):
    """
    genieclust.compare_partitions.compare_partitions(C)

    Computes a series of partition similarity scores



    Parameters
    ----------

    C : ndarray
        A ``c_contiguous`` confusion matrix (contingency table)
        with :math:`K` rows and :math:`L` columns, where :math:`K \\le L`.


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
            Normalised accuracy (purity)
        ``'psi'``
            Pair sets index


    See also
    --------

    genieclust.compare_partitions.confusion_matrix :
        Computes a confusion matrix
    genieclust.compare_partitions.compare_partitions2 :
        A wrapper around this function that accepts two label vectors on input
    genieclust.compare_partitions.adjusted_rand_score
    genieclust.compare_partitions.rand_score
    genieclust.compare_partitions.adjusted_fm_score
    genieclust.compare_partitions.fm_score
    genieclust.compare_partitions.mi_score
    genieclust.compare_partitions.normalized_mi_score
    genieclust.compare_partitions.adjusted_mi_score
    genieclust.compare_partitions.normalized_accuracy
    genieclust.compare_partitions.pair_sets_index



    Notes
    -----

    Let `x` and `y` represent two partitions of the same set with :math:`n`
    elements into, respectively, :math:`K` and :math:`L`
    nonempty and pairwise disjoint subsets.
    For instance, these can be two clusterings of a dataset with :math:`n`
    observations specified as vectors of labels. Moreover, let `C` be the
    confusion matrix (with :math:`K` rows and :math:`L` columns, :math:`K \\leq L`)
    corresponding to `x` and `y`, see also
    `genieclust.compare_partitions.confusion_matrix`.

    This function implements a few scores that aim to quantify
    the similarity between `x` and `y`.
    Partition similarity scores can be used as external cluster validity
    measures — for comparing the outputs of clustering algorithms
    with reference (ground truth) labels, see, e.g.,
    https://github.com/gagolews/clustering_benchmarks_v1
    for a suite of benchmark datasets.

    Every index except `mi_score` (which computes the mutual
    information score) outputs 1 given two identical partitions.
    Note that partitions are always defined up to a bijection of the set of
    possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4)
    represent the same 2-partition.

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

    `normalized_accuracy` is defined as
    :math:`(\\mathrm{Accuracy}(C_\\sigma)-1/L)/(1-1/L)`,
    where :math:`C_\\sigma` is a version of the confusion matrix
    for given `x` and `y`, :math:`K \\leq L`, with columns permuted
    based on the solution to the Maximal Linear Sum Assignment Problem.
    :math:`\\mathrm{Accuracy}(C_\\sigma)` is sometimes referred to as Purity,
    e.g., in [2]_.

    `pair_sets_index` gives the Pair Sets Index (PSI)
    adjusted for chance [3]_, :math:`K \\leq L`.
    Pairing is based on the solution to the Linear Sum Assignment Problem
    of a transformed version of the confusion matrix.



    References
    ----------

    .. [1]
        Hubert L., Arabie P., Comparing Partitions,
        *Journal of Classification* 2(1), 1985, 193-218.

    .. [2]
        Rendon E., Abundez I., Arizmendi A., Quiroz E.M.,
        Internal versus external cluster validation indexes,
        *International Journal of Computers and Communications* 5(1), 2011,
        27-34.

    .. [3]
        Rezaei M., Franti P., Set matching measures for external cluster validity,
        *IEEE Transactions on Knowledge and Data Mining* 28(8), 2016,
        2173-2186. doi:10.1109/TKDE.2016.2551240.

    .. [4]
        Vinh N.X., Epps J., Bailey J.,
        Information theoretic measures for clusterings comparison:
        Variants, properties, normalization and correction for chance,
        *Journal of Machine Learning Research* 11, 2010, 2837-2854.




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
    {'ar': 0.49, 'r': 0.74, 'fm': 0.73, 'afm': 0.49, 'mi': 0.29, 'nmi': 0.41, 'ami': 0.39, 'nacc': 0.71, 'psi': 0.65}
    >>> {k : round(v, 2) for k, v in
    ...      genieclust.compare_partitions.compare_partitions2(x,y).items()}
    {'ar': 0.49, 'r': 0.74, 'fm': 0.73, 'afm': 0.49, 'mi': 0.29, 'nmi': 0.41, 'ami': 0.39, 'nacc': 0.71, 'psi': 0.65}
    >>> round(genieclust.compare_partitions.adjusted_rand_score(x, y), 2)
    0.49

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


    Notes
    -----

    Calls ``genieclust.compare_partitions.compare_partitions(C)``,
    where ``C = genieclust.compare_partitions.confusion_matrix(x, y)``.


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

    See `genieclust.compare_partitions.compare_partitions` for more details.

    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
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

    See `genieclust.compare_partitions.compare_partitions` for more details.


    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
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

    See `genieclust.compare_partitions.compare_partitions` for more details.


    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
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

    See `genieclust.compare_partitions.compare_partitions` for more details.


    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
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

    See `genieclust.compare_partitions.compare_partitions` for more details.


    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
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

    See `genieclust.compare_partitions.compare_partitions` for more details.


    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
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

    See `genieclust.compare_partitions.compare_partitions` for more details.


    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    return c_compare_partitions.Ccompare_partitions_info(&C[0,0], xc, yc).ami



cpdef double normalized_accuracy(x, y):
    """
    genieclust.compare_partitions.normalized accuracy(x, y)

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


    Notes
    -----

    See `genieclust.compare_partitions.compare_partitions` for more details.

    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    if xc > yc:
        raise ValueError("Number of rows in the confusion matrix "
            "must be less than or equal to the number of columns.")
    return c_compare_partitions.Ccompare_partitions_nacc(&C[0,0], xc, yc)




cpdef double pair_sets_index(x, y):
    """
    genieclust.compare_partitions.pair_sets_index(x, y)

    Pair Sets Index (PSI) adjusted for chance


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

    See `genieclust.compare_partitions.compare_partitions` for more details.

    """

    cdef np.ndarray[ssize_t,ndim=2] C = confusion_matrix(x, y)
    cdef ssize_t xc = C.shape[0]
    cdef ssize_t yc = C.shape[1]
    if xc > yc:
        raise ValueError("Number of rows in the confusion matrix "
            "must be less than or equal to the number of columns.")
    return c_compare_partitions.Ccompare_partitions_psi(&C[0,0], xc, yc)

