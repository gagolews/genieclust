"""
Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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
import scipy.spatial
import genieclust
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt


def _lumbermark(
    X,
    n_clusters,
    min_cluster_size,
    min_cluster_factor,
    M,
    skip_leaves,
    verbose=False
):
    assert n_clusters > 0
    assert min_cluster_size >= 0
    assert min_cluster_factor >= 0

    n = X.shape[0]


    if M <= 2:
        mst_w, mst_e = genieclust.internal.mst_from_distance(X, "euclidean")
    else:
        if M-1 >= X.shape[0]:
            raise ValueError("`M` is too large")

        nn_dist, nn_ind = genieclust.internal.knn_from_distance(
            X, k=M-1, metric="euclidean", verbose=verbose
        )

        # d_core = nn_dist.mean(axis=1).astype(X.dtype, order="C") NOTE: not better ;)
        d_core = genieclust.internal.get_d_core(nn_dist, nn_ind, M) # i.e., d_core = nn_dist[:, M-2].astype(X.dtype, order="C")

        _o = np.argsort(d_core)[::-1]  # order wrt decreasing "core" size
        mst_w, mst_e = genieclust.internal.mst_from_distance(
            np.ascontiguousarray(X[_o, :]),
            metric="euclidean",
            d_core=np.ascontiguousarray(d_core[_o]),
            verbose=verbose
        )
        mst_e = np.c_[ _o[mst_e[:, 0]], _o[mst_e[:, 1]] ]


    _o = np.argsort(mst_w)
    mst_w = mst_w[_o]
    mst_e = mst_e[_o, :]

    l = genieclust.internal.lumbermark_from_mst(mst_w, mst_e, n_clusters, min_cluster_size, min_cluster_factor, skip_leaves)

    return (
        l["labels"],
        mst_w,
        mst_e,
        l["n_clusters"],
        l["links"],
        l["is_noise"]
    )



class Lumbermark(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        *,
        n_clusters,
        min_cluster_size=10,
        min_cluster_factor=0.25,  # try also 0.1 or less for clusters of imbalanced sizes
        M=2,  # try also M=6 or M=4 or M=11  (mutual reachability distance); M=2 is M=1 but with skip_leaves
        skip_leaves=None,  # by default, True if M>1
        verbose=False
    ):
        self.n_clusters              = n_clusters
        self.min_cluster_size        = min_cluster_size
        self.min_cluster_factor      = min_cluster_factor
        self.M                       = M
        self.skip_leaves             = skip_leaves
        self.verbose                 = verbose


    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns.

        y : None
            Ignored.

        Returns
        -------

        self : genieclust.Genie
            The object that the method was called on.
        """

        self.X = np.ascontiguousarray(X)
        self.n_samples_ = self.X.shape[0]

        if self.skip_leaves is None:
            self.skip_leaves_ = (self.M>1)
        else:
            self.skip_leaves_ = self.skip_leaves

        (
            self.labels_,
            self._tree_w,
            self._tree_e,
            self._n_clusters,
            self._tree_cutlist,
            self._is_noise
        ) = _lumbermark(
            self.X,
            n_clusters=self.n_clusters,
            min_cluster_size=self.min_cluster_size,
            min_cluster_factor=self.min_cluster_factor,
            M=self.M,
            skip_leaves=self.skip_leaves_,
            verbose=self.verbose
        )

        return self



    def fit_predict(self, X, y=None):
        """
        Perform cluster analysis of a dataset and return the predicted labels.


        Parameters
        ----------

        X : object
            See `genieclust.Genie.fit`.

        y : None
            See `genieclust.Genie.fit`.


        Returns
        -------

        labels_ : ndarray
            `self.labels_` attribute.


        See also
        --------

        genieclust.Genie.fit

        """
        self.fit(X, y)
        return self.labels_
