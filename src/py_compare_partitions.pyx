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


cdef extern from "../src/c_compare_partitions.h":
    cdef struct CComparePartitionsPairsResult:
        double ar
        double r
        double fm
        double afm

    cdef struct CComparePartitionsInfoResult:
        double mi
        double nmi
        double ami

    cdef struct CCompareSetMatchingResult:
        double psi_unclipped
        double spsi_unclipped

    void Cminmax[T](const T* x, Py_ssize_t n, T* xmin, T* xmax)

    void Ccontingency_table[T](T* Cout, Py_ssize_t xc, Py_ssize_t yc,
        Py_ssize_t xmin, Py_ssize_t ymin,
        Py_ssize_t* x, Py_ssize_t* y, Py_ssize_t n) except+

    void Cnormalizing_permutation[T](const T* C, Py_ssize_t xc, Py_ssize_t yc,
        Py_ssize_t* Iout) except+

    void Capply_pivoting[T](const T* C, Py_ssize_t xc, Py_ssize_t yc, T* Cout) except+

    CComparePartitionsPairsResult Ccompare_partitions_pairs[T](const T* C,
        Py_ssize_t xc, Py_ssize_t yc) except+

    CComparePartitionsInfoResult Ccompare_partitions_info[T](const T* C,
        Py_ssize_t xc, Py_ssize_t yc) except+

    double Ccompare_partitions_npa[T](const T* C, Py_ssize_t xc, Py_ssize_t yc) except+

    double Ccompare_partitions_nca[T](const T* C, Py_ssize_t xc, Py_ssize_t yc) except+

    CCompareSetMatchingResult Ccompare_partitions_psi[T](const T* C,
        Py_ssize_t xc, Py_ssize_t yc) except+


cdef np.ndarray _get_confusion_matrix(x, y=None, bint force_double=False):
    """
    Internal function.

    Parameters
    ----------

    x : array-like
        a confusion matrix (if y is None) or a label vector

    y : array-like or None
        another label vector or None

    force_double : bool


    Returns
    -------

    ndarray
        A confusion matrix of the dtype intp or double
        (unless force_double is True, then it will be the latter)
    """
    cdef np.ndarray _C
    if y is None:
        _C = np.array(x)
        if _C.ndim != 2 or _C.shape[0] < 2 or _C.shape[1] < 2:
            raise ValueError("if `y` is None, `x` should be a confusion matrix")
        if (not force_double) and np.can_cast(_C, np.intp):
            return np.array(_C, dtype=np.intp)
        else:
            return np.array(_C, dtype=np.double)
    else:
        return confusion_matrix(x, y, force_double=force_double)


cpdef np.ndarray[Py_ssize_t,ndim=1] normalizing_permutation(C):
    """
    genieclust.compare_partitions.normalizing_permutation(C)

    Determines the permutation of columns of a confusion matrix
    so that the sum of the elements on the main diagonal is the largest
    possible (by solving the maximal assignment problem).


    Parameters
    ----------

    C : ndarray
        A confusion matrix (contingency table),
        whose row count is not greater than the column count;
        can be a matrix of elements of the double type


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

    cdef np.ndarray[double, ndim=2] _C = _get_confusion_matrix(C, None, force_double=True)
    cdef Py_ssize_t xc = _C.shape[0]
    cdef Py_ssize_t yc = _C.shape[1]
    cdef np.ndarray[Py_ssize_t,ndim=1] perm = np.empty(yc, dtype=np.intp)
    if xc > yc:
        raise ValueError("the number of rows cannot be greater than the number of columns")

    Cnormalizing_permutation(&_C[0,0], xc, yc, &perm[0])

    return perm


cpdef np.ndarray normalize_confusion_matrix(C):
    """
    genieclust.compare_partitions.normalize_confusion_matrix(C)

    Permutes the rows and columns of a confusion matrix so that the sum
    of the elements on the main diagonal is the largest possible
    (by solving the maximal assignment problem).


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
    cdef np.ndarray _C_intp_or_double = _get_confusion_matrix(C, None, force_double=False)
    cdef Py_ssize_t xc = _C_intp_or_double.shape[0]
    cdef Py_ssize_t yc = _C_intp_or_double.shape[1]
    if xc > yc:
        raise ValueError("the number of rows cannot be greater than the number of columns")

    cdef np.ndarray[Py_ssize_t,ndim=2] _Ci
    cdef np.ndarray[Py_ssize_t,ndim=2] _Di
    cdef np.ndarray[double,ndim=2] _Cd
    cdef np.ndarray[double,ndim=2] _Dd
    if np.can_cast(_C_intp_or_double, np.intp):
        _Ci = np.array(_C_intp_or_double, dtype=np.intp)
        _Di = np.zeros_like(_Ci)
        Capply_pivoting(&_Ci[0,0], xc, yc, &_Di[0,0])
        return _Di
    else:
        _Cd = np.array(_C_intp_or_double, dtype=np.double)
        _Dd = np.zeros_like(_Cd)
        Capply_pivoting(&_Cd[0,0], xc, yc, &_Dd[0,0])
        return _Dd


cpdef np.ndarray confusion_matrix(x, y, bint force_double=False):
    """
    genieclust.compare_partitions.confusion_matrix(x, y, force_double=False)

    Computes the confusion matrix for two label vectors


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths.

    force_double : bool
        If the return dtype should be 'double'.


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
    Cminmax(<Py_ssize_t*>(&_x[0]), n, <Py_ssize_t*>(&xmin), <Py_ssize_t*>(&xmax))
    cdef Py_ssize_t xc = (xmax-xmin+1)

    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)
    if n != _y.shape[0]: raise ValueError("incompatible lengths")
    cdef Py_ssize_t ymin, ymax
    Cminmax(<Py_ssize_t*>(&_y[0]), n, <Py_ssize_t*>(&ymin), <Py_ssize_t*>(&ymax))
    cdef Py_ssize_t yc = (ymax-ymin+1)

    cdef Py_ssize_t CONFUSION_MATRIX_MAXSIZE = 100_000_000
    if xc*yc > CONFUSION_MATRIX_MAXSIZE:
        raise ValueError("CONFUSION_MATRIX_MAXSIZE exceeded")

    cdef np.ndarray[Py_ssize_t,ndim=2] Ci
    cdef np.ndarray[double,ndim=2] Cd

    if not force_double:
        Ci = np.empty((xc, yc), dtype=np.intp)
        Ccontingency_table(&Ci[0,0], xc, yc,
            xmin, ymin, &_x[0], &_y[0], n)
        return Ci
    else:
        Cd = np.empty((xc, yc), dtype=np.double)
        Ccontingency_table(&Cd[0,0], xc, yc,
            xmin, ymin, &_x[0], &_y[0], n)
        return Cd



cpdef np.ndarray normalized_confusion_matrix(x, y, bint force_double=False):
    """
    genieclust.compare_partitions.normalized_confusion_matrix(x, y, force_double=False)

    Computes the confusion matrix for two label vectors and
    permutes its rows and columns so that the sum of the elements
    of the main diagonal is the largest possible (by solving
    the maximal assignment problem).


    Parameters
    ----------

    x, y : array_like
        Two vectors of "small" integers of identical lengths.

    force_double : bool
        If the return dtype should be 'double'.

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
    return normalize_confusion_matrix(confusion_matrix(x, y, force_double=force_double))



cpdef dict compare_partitions(x, y=None, bint psi_clipped=True):
    """
    genieclust.compare_partitions.compare_partitions(x, y=None, psi_clipped=True)

    Computes a series of external cluster validity measures


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None

    psi_clipped : bool
        Whether pair sets index should be clipped to 0


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
        ``'npa'``
            Normalised pivoted accuracy
        ``'psi'``
            Pair sets index
        ``'spsi'``
            Simplified pair sets index
        ``'nca'``
            Normalised clustering accuracy;
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

    genieclust.compare_partitions.normalized_clustering_accuracy
    genieclust.compare_partitions.normalized_pivoted_accuracy
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
    They can be used as external cluster
    validity measures, where we assume that `x` is
    the reference (ground-truth) partition
    whilst `y` is the vector of predicted cluster memberships; compare [5]_.

    All indices except `normalized_clustering_accuracy`
    can act as a pairwise partition similarity score: they are symmetric,
    i.e., ``index(x, y) == index(y, x)``.

    Each index except `mi_score` (which computes the mutual information score)
    outputs the value of 1.0 if two identical partitions are given.
    Note that partitions are always defined up to a bijection of the set of
    possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4)
    represent the same 2-partition.

    `normalized_clustering_accuracy` [2]_ is an asymmetric external cluster
    validity measure which assumes that the label vector `x` (or rows in the
    confusion matrix) represents the reference (ground truth) partition.
    It is an average proportion of correctly classified points in each cluster
    above the worst case scenario of uniform membership assignment,
    with cluster ID matching based on the solution to the maximal linear
    sum assignment problem; see :func:`normalized_confusion_matrix`).
    It is given by:
    :math:`(\\max_\\sigma \\frac{1}{K} \\sum_{j=1}^K \\frac{c_{\\sigma(j), j}-c_{\\sigma(j),\\cdot}/K}{c_{\\sigma(j),\\cdot}-c_{\\sigma(j),\\cdot}/K})`,
    where :math:`C` is a confusion matrix with :math:`K` rows and columns,
    :math:`\\sigma` is a permutation of the set :math:`\\{1,\\dots,K\\}`, and
    and :math:`c_{i, \\cdot}=c_{i, 1}+...+c_{i, K}` is the `i`-th row sum,
    under the assumption that :math:`0/0=0`.

    `normalized_pivoted_accuracy` is defined as
    :math:`(\\max_\\sigma \\sum_{j=1}^{K} \\frac{c_{\\sigma(j),j}/n-1/K}{1-1/K}`,
    where :math:`\\sigma` is a permutation of the set :math:`\\{1,\\dots,K\\}`,
    and :math:`n` is the sum of all elements in :math:`C`.

    `pair_sets_index` (PSI) was introduced in [3]_.
    The simplified PSI assumes E=1 in the definition of the index,
    i.e., uses Eq. (20) in [3]_ instead of Eq. (18).
    For non-square matrices, missing rows/columns are assumed
    to be filled with 0s.

    `rand_score` gives the Rand score (the "probability" of agreement
    between the two partitions) and `adjusted_rand_score` is its version
    corrected for chance [1]_ (especially Eqs. (2) and (4) therein):
    its expected value is 0.0 for two independent
    partitions. Due to the adjustment, the resulting index may
    be negative for some inputs.

    Similarly, `fm_score` gives the Fowlkes-Mallows (FM) score
    and `adjusted_fm_score` is its adjusted-for-chance version [1]_.

    `mi_score`, `adjusted_mi_score` and `normalized_mi_score` are
    information-theoretic indices based on mutual information,
    see the definition of :math:`\\mathrm{AMI}_\\mathrm{sum}`
    and :math:`\\mathrm{NMI}_\\mathrm{sum}` in [4]_.



    References
    ----------

    .. [1]
        Hubert, L., Arabie, P., Comparing partitions,
        *Journal of Classification* 2(1), 1985, 193-218.

    .. [2]
        Gagolewski, M., Normalised clustering accuracy: An asymmetric external
        cluster validity measure, *Journal of Classification* 42, 2025, 2-30.
        https://doi.org/10.1007/s00357-024-09482-2.

    .. [3]
        Rezaei, M., Franti, P., Set matching measures for external cluster validity,
        *IEEE Transactions on Knowledge and Data Mining* 28(8), 2016,
        2173-2186. https://doi.org/10.1109/TKDE.2016.2551240.

    .. [4]
        Vinh, N.X., Epps, J., Bailey, J.,
        Information theoretic measures for clusterings comparison:
        Variants, properties, normalization and correction for chance,
        *Journal of Machine Learning Research* 11, 2010, 2837-2854.

    .. [5]
        Gagolewski, M., *A Framework for Benchmarking Clustering Algorithms*,
        *SoftwareX* 20, 2022, 101270.
        https://clustering-benchmarks.gagolewski.com.


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
    {'ar': 0.49, 'r': 0.74, 'fm': 0.73, 'afm': 0.49, 'mi': 0.29, 'nmi': 0.41, 'ami': 0.39, 'npa': 0.71, 'psi': 0.65, 'spsi': 0.63, 'nca': 0.71}
    >>> {k : round(v, 2) for k, v in
    ...      genieclust.compare_partitions.compare_partitions(x,y).items()}
    {'ar': 0.49, 'r': 0.74, 'fm': 0.73, 'afm': 0.49, 'mi': 0.29, 'nmi': 0.41, 'ami': 0.39, 'npa': 0.71, 'psi': 0.65, 'spsi': 0.63, 'nca': 0.71}
    >>> round(genieclust.compare_partitions.normalized_clustering_accuracy(x, y), 2)
    0.71

    """
    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)

    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]

    cdef dict res1 = Ccompare_partitions_pairs(&C[0,0], xc, yc)

    cdef dict res2 = Ccompare_partitions_info(&C[0,0], xc, yc)

    cdef double npa = Ccompare_partitions_npa(&C[0,0], xc, yc)

    cdef dict res3 = Ccompare_partitions_psi(&C[0,0], xc, yc)
    cdef dict res3_clipped = {
        "psi": res3["psi_unclipped"],
        "spsi": res3["spsi_unclipped"],
    }

    if psi_clipped:
        if res3_clipped["psi"]  < 0.0: res3_clipped["psi"]  = 0.0
        if res3_clipped["spsi"] < 0.0: res3_clipped["spsi"] = 0.0

    cdef double nca = np.nan
    if xc == yc:
        nca = Ccompare_partitions_nca(&C[0,0], xc, yc)

    return {
        **res1,
        **res2,
        "npa": npa,
        **res3_clipped,
        "nca": nca,
    }




cpdef double adjusted_rand_score(x, y=None):
    """
    genieclust.compare_partitions.adjusted_rand_score(x, y=None)

    The Rand index adjusted for chance


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores


    Notes
    -----

    See :func:`compare_partitions` for more details.

    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_pairs(&C[0,0], xc, yc).ar


cpdef double rand_score(x, y=None):
    """
    genieclust.compare_partitions.rand_score(x, y=None)

    The original Rand index not adjusted for chance


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_pairs(&C[0,0], xc, yc).r


cpdef double adjusted_fm_score(x, y=None):
    """
    genieclust.compare_partitions.adjusted_fm_index(x, y=None)

    The Fowlkes-Mallows index adjusted for chance


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores
        Computes multiple similarity scores based on two label vectors


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_pairs(&C[0,0], xc, yc).afm


cpdef double fm_score(x, y=None):
    """
    genieclust.compare_partitions.fm_score(x, y=None)

    The original Fowlkes-Mallows index (not adjusted for chance)


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_pairs(&C[0,0], xc, yc).fm


cpdef double mi_score(x, y=None):
    """
    genieclust.compare_partitions.mi_score(x, y=None)

    Mutual information score


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_info(&C[0,0], xc, yc).mi



cpdef double normalized_mi_score(x, y=None):
    """
    genieclust.compare_partitions.normalised_mi_score(x, y=None)

    Normalised mutual information score :math:`(\\mathrm{NMI}_\\mathrm{sum})`


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_info(&C[0,0], xc, yc).nmi


cpdef double adjusted_mi_score(x, y=None):
    """
    genieclust.compare_partitions.adjusted_mi_score(x, y=None)

    Adjusted mutual information score :math:`(\\mathrm{AMI}_\\mathrm{sum})`


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores


    Notes
    -----

    See :func:`compare_partitions` for more details.


    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_info(&C[0,0], xc, yc).ami



cpdef double normalized_pivoted_accuracy(x, y=None):
    """
    genieclust.compare_partitions.normalized_pivoted_accuracy(x, y=None)

    Normalised pivoted accuracy (NPA)


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores

    genieclust.compare_partitions.normalized_confusion_matrix :
        Determines the confusion matrix and permutes the rows and columns
        so that the sum of the elements of the main diagonal is the largest
        possible


    Notes
    -----

    See :func:`compare_partitions` for more details.

    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_npa(&C[0,0], xc, yc)



cpdef double normalized_clustering_accuracy(x, y=None):
    """
    genieclust.compare_partitions.normalized_clustering_accuracy(x, y=None)

    Normalised clustering accuracy (NCA) [1]_.

    This measure is asymmetric – it is assumed that `x`
    represents the ground truth labels, whilst `y` give predicted cluster IDs.


    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None


    Returns
    -------

    double
        Similarity score.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores

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
        Gagolewski, M., Normalised clustering accuracy: An asymmetric external
        cluster validity measure, *Journal of Classification* 42, 2025, 2-30.
        https://doi.org/10.1007/s00357-024-09482-2.

    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    return Ccompare_partitions_nca(&C[0,0], xc, yc)


cpdef double pair_sets_index(x, y=None, bint simplified=False, bint clipped=True):
    """
    genieclust.compare_partitions.pair_sets_index(x, y=None, simplified=False, clipped=True)

    Pair sets index (PSI) [1]_

    For non-square confusion matrices, missing rows/columns
    are assumed to be filled with 0s.

    Parameters
    ----------

    x : ndarray
        A confusion matrix (contingency table)
        or a vector of labels (if y is not None)

    y : None or ndarray
        a vector of labels or None

    simplified : bool
        Whether to assume E=1 in the definition of the index,
        i.e.,use Eq. (20) in [1]_ instead of Eq. (18).

    clipped : bool
        Whether the result should be clipped to the unit interval.


    Returns
    -------

    double
        Partition similarity measure.


    See also
    --------

    genieclust.compare_partitions.compare_partitions :
        Computes multiple similarity scores

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
        Rezaei, M., Franti, P., Set matching measures for external cluster validity,
        *IEEE Transactions on Knowledge and Data Mining* 28(8), 2016,
        2173-2186. https://doi.org/10.1109/TKDE.2016.2551240.

    """

    cdef np.ndarray[double, ndim=2] C = _get_confusion_matrix(x, y, force_double=True)
    cdef Py_ssize_t xc = C.shape[0]
    cdef Py_ssize_t yc = C.shape[1]
    cdef double res

    if simplified:
        res = Ccompare_partitions_psi(&C[0,0], xc, yc).spsi_unclipped
    else:
        res = Ccompare_partitions_psi(&C[0,0], xc, yc).psi_unclipped

    if clipped:
        if res < 0.0: res = 0.0

    return res
