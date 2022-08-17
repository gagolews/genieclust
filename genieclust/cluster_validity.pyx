# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Implementation of a number of the so-called internal cluster validity indices
critically reviewed in (Gagolewski, Bartoszuk, Cena, 2022;
https://doi.org/10.1016/j.ins.2021.10.004;
`preprint <https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf>`_).
See Section 2 therein for the respective definitions.

The greater the index value, the more *valid* (whatever that means)
the assessed partition. For consistency, the Ball-Hall and
Davies-Bouldin indexes take negative values.

For more details, see the
`Framework for Benchmarking Clustering Algorithms
<https://clustering-benchmarks.gagolewski.com>`_.
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2022, Marek Gagolewski <https://www.gagolewski.com>      #
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

    genieclust.cluster_validity.silhouette_index :
        The Silhouette index (average silhouette score)

    genieclust.cluster_validity.silhouette_w_index :
        The Silhouette W index (mean of the cluster average silhouette widths)

    genieclust.cluster_validity.wcnn_index :
        The within-cluster near-neighbours index

    genieclust.cluster_validity.negated_wcss_index :
        Within-cluster sum of squares (used as the objective function
        in the k-means and Ward algorithm) (negated)



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
    cdef np.ndarray[ssize_t] _y = np.array(y, dtype=np.intp)
    cdef size_t n = _y.shape[0]

    cdef np.ndarray[double, ndim=2] _X = np.array(X, dtype=np.double, ndmin=2, order="C")
    cdef size_t d = _X.shape[1]

    if n != _X.shape[0]:
        raise ValueError("number of elements in y does not match the number of rows in X")

    cdef ssize_t ymin, ymax
    c_compare_partitions.Cminmax(<ssize_t*>(&_y[0]), n, <ssize_t*>(&ymin), <ssize_t*>(&ymax))
    cdef ssize_t K = ymax-ymin+1
    if ymin != 0:
        raise ValueError("min(y) != 0")

    if K <= 1 or K > 100000:
        raise ValueError("incorrect y")


    return c_cvi.c_calinski_harabasz_index(<double*>(&_X[0, 0]), <ssize_t*>(&_y[0]), n, d, K)

    return res

