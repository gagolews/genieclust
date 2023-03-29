# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
So-called internal cluster validity indices

The greater the index value, the more *valid* (whatever that means)
the assessed partition. For consistency, the Ball-Hall and
Davies-Bouldin indexes take negative values.

These measures were critically reviewed in (Gagolewski, Bartoszuk, Cena, 2022;
https://doi.org/10.1016/j.ins.2021.10.004;
`preprint <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_).
See Section 2 therein for the respective definitions.

For even more details, see the
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
from . cimport c_cvi


cdef Py_ssize_t _get_K(np.ndarray[Py_ssize_t] _y, size_t n, size_t d):
    """
    internal function for CVIs - check correctness of shapes and get clust.num.
    """

    if n != _y.shape[0]:
        raise ValueError(
            "Number of elements in y does not match the number of rows in X."
        )

    cdef Py_ssize_t ymin, ymax
    c_compare_partitions.Cminmax(<Py_ssize_t*>(&_y[0]), n, <Py_ssize_t*>(&ymin), <Py_ssize_t*>(&ymax))
    cdef Py_ssize_t K = ymax-ymin+1
    if ymin != 0:
        raise ValueError("min(y) != 0.")

    if K <= 1 or K > 100000:
        raise ValueError("Incorrect y.")

    return K


cpdef double calinski_harabasz_index(X, y):
    """
    genieclust.cluster_validity.calinski_harabasz_index(X, y)

    Computes the value of the Caliński-Harabasz index [3]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    .. [3]
        Calinski T., Harabasz J., A dendrite method for cluster analysis,
        *Communications in Statistics* 3(1), 1974, 1–27,
        https://doi.org/10.1080/03610927408827101.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_calinski_harabasz_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K
    )

    return res


cpdef double negated_ball_hall_index(X, y):
    """
    genieclust.cluster_validity.negated_ball_hall_index(X, y)

    Computes the value of the negated Ball-Hall index [3]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    .. [3]
        Ball G.H., Hall D.J.,
        *ISODATA: A novel method of data analysis and pattern classification*,
        Technical report No. AD699616, Stanford Research Institute, 1965.
    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_negated_ball_hall_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K
    )

    return res


cpdef double negated_davies_bouldin_index(X, y):
    """
    genieclust.cluster_validity.negated_davies_bouldin_index(X, y)

    Computes the value of the Davies-Bouldin index [3]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    .. [3]
        Davies D.L., Bouldin D.W., A Cluster Separation Measure,
        *IEEE Transactions on Pattern Analysis and Machine Intelligence*
        PAMI-1 (2), 1979, 224-227, https://doi.org/10.1109/TPAMI.1979.4766909.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_negated_davies_bouldin_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K
    )

    return res


cpdef double negated_wcss_index(X, y):
    """
    genieclust.cluster_validity.negated_wcss_index(X, y)

    Computes the value of the negated within-cluster sum of squares
    (used as the objective function in the k-means and Ward algorithm)

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_negated_wcss_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K
    )

    return res


cpdef double silhouette_index(X, y):
    """
    genieclust.cluster_validity.silhouette_index(X, y)

    Computes the value of the The Silhouette index
    (average silhouette score) [3]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    .. [3]
        Rousseeuw P.J., Silhouettes: A Graphical Aid to the Interpretation and
        Validation of Cluster Analysis, *Computational and Applied Mathematics*
        20, 1987, 53-65, https://doi.org/10.1016/0377-0427(87)90125-7.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_silhouette_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K
    )

    return res


cpdef double silhouette_w_index(X, y):
    """
    genieclust.cluster_validity.silhouette_w_index(X, y)

    Computes the value of the The Silhouette W index
    (mean of the cluster average silhouette widths) [3]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    .. [3]
        Rousseeuw P.J., Silhouettes: A Graphical Aid to the Interpretation and
        Validation of Cluster Analysis, *Computational and Applied Mathematics*
        20, 1987, 53-65, https://doi.org/10.1016/0377-0427(87)90125-7.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_silhouette_w_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K
    )

    return res


cpdef double wcnn_index(X, y, int M=25):
    """
    genieclust.cluster_validity.wcnn_index(X, y, M=25)

    Computes the within-cluster near-neighbours index [2]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.

    M : int
        number of nearest neighbours

    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef size_t _M = M

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_wcnn_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K, _M
    )

    return res



cpdef double dunnowa_index(
    X, y, int M=25, str owa_numerator="SMin:5", str owa_denominator="Const"
):
    """
    genieclust.cluster_validity.dunnowa_index(X, y,
        M=25, owa_numerator="SMin:5", owa_denominator="Const")

    Computes the generalised Dunn indices based on near-neighbours and
    OWA operators [2]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.

    M : int
        number of nearest neighbours

    owa_numerator, owa_denominator : str
        specifies the OWA operators to use in the definition of the DuNN index;
        one of: ``"Mean"``, ``"Min"``, ``"Max"``, ``"Const"``,
        ``"SMin:D"``, ``"SMax:D"``, where \code{D} is an integer
        defining the degree of smoothness


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef size_t _M = M

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_dunnowa_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K, _M,
            unicode(owa_numerator).encode('utf8'),
            unicode(owa_denominator).encode('utf8')
    )

    return res


cpdef double generalised_dunn_index(X, y, int lowercase_d=1, int uppercase_d=2):
    """
    genieclust.cluster_validity.generalised_dunn_index(X, y,
        lowercase_d=1, uppercase_d=2)

    Computes the generalised Dunn indices (by Bezdek and Pal) [3]_.

    See [1]_ and [2]_ for the definition and discussion.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n, d)
        `n` data points in a feature space of dimensionality `d`

    y : array_like
        A vector of "small" integers representing a partition of the
        `n` input points; `y[i]` is the cluster ID of the `i`-th point,
        where `0 <= y[i] < K` and `K` is the number of clusters.

    M : int
        number of nearest neighbours

    lowercase_d : int
        an integer between 1 and 5, denoting
        :math:`d_1`, ..., :math:`d_5` in the definition
        of the generalised Dunn index (numerator:
        min, max, and mean pairwise intracluster distance,
        distance between cluster centroids,
        weighted point-centroid distance, respectively)

    uppercase_d : int
        an integer between 1 and 3, denoting
        :math:`D_1`, ..., :math:`D_3` in the definition
        of the generalised Dunn index (denominator:
        max and min pairwise intracluster distance, average point-centroid
        distance, respectively)


    Returns
    -------

    index : float
        Computed index value.
        The greater the index value, the more *valid* (whatever that means)
        the assessed partition.


    See also
    --------

    genieclust.cluster_validity.calinski_harabasz_index :
        The Caliński-Harabasz index

    genieclust.cluster_validity.dunnowa_index :
        Generalised Dunn indices based on near-neighbours and
        OWA operators (by Gagolewski)

    genieclust.cluster_validity.generalised_dunn_index :
        Generalised Dunn indices (by Bezdek and Pal)

    genieclust.cluster_validity.negated_ball_hall_index :
        The Ball-Hall index (negated)

    genieclust.cluster_validity.negated_davies_bouldin_index :
        The Davies-Bouldin index (negated)

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index



    References
    ----------

    .. [1]
        Gagolewski M., *A Framework for Benchmarking Clustering Algorithms*,
        https://clustering-benchmarks.gagolewski.com

    .. [2]
        Gagolewski M., Bartoszuk M., Cena A.,
        Are cluster validity measures (in)valid?, *Information Sciences* **581**,
        620–636, 2021, https://doi.org/10.1016/j.ins.2021.10.004
        `(preprint) <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_.

    .. [3]
        Bezdek J., Pal N., Some new indexes of cluster validity,
        *IEEE Transactions on Systems, Man, and Cybernetics, Part B* 28,
        1998, 301-315, https://doi.org/10.1109/3477.678624/.

    """
    cdef np.ndarray[Py_ssize_t] _y = np.array(y, dtype=np.intp)

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t n = _X.shape[0]
    cdef size_t d = _X.shape[1]

    cdef Py_ssize_t K = _get_K(_y, n, d)

    return c_cvi.c_generalised_dunn_index(
        <double*>(&_X[0, 0]), <Py_ssize_t*>(&_y[0]), n, d, K,
            lowercase_d, uppercase_d
    )

    return res
