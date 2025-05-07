"""
The Silentmark clustering algorithm:

Detect noise points based on how many points have a point amongst
    their nearest neighbours,
clusterise a dataset with noise points omitted using RSL (better) or Genie (worse),
assign points to "nearest" clusters.

Overall, clustering with RSL wrt a mutual reachability distance is better.
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


import genieclust
import numpy as np
import scipy.spatial
import robust_single_linkage
import scipy.spatial.distance
from sklearn.base import BaseEstimator, ClusterMixin
from numba import jit


@jit
def _classify_noise_via_spanning_trees(X, y_pred, is_noise):
    y_pred = y_pred.copy()

    n = X.shape[0]
    X_noise = X[is_noise, :]
    X_core = X[~is_noise, :]
    which_noise = np.flatnonzero(y_pred <= 0)
    which_core  = np.flatnonzero(y_pred > 0)
    n_noise = len(which_noise)

    # we'll be extending the current tree spanning the non-noise points;
    # we'll be adding n_noise edges to the tree
    # this will be a Prim-like algorithm

    i = len(which_core)
    ind_left = np.concatenate((which_core, which_noise))

    # ind_nn[j] will the vertex from the current tree closest to vertex j
    dist_nn = np.repeat(np.inf, n)
    idx_nn = -np.ones(n, dtype=np.intp)
    for j in range(n_noise):
        #_d = scipy.spatial.distance.cdist(X_noise[j,:].reshape(1,-1), X_core).ravel()
        _d = np.sqrt(np.sum((X_noise[j, :]-X_core)**2, axis=1))
        assert _d.shape[0] == X_core.shape[0]
        _i = np.argmin(_d)
        idx_nn[which_noise[j]] = which_core[_i]
        dist_nn[which_noise[j]] = _d[_i]

    add_e = np.empty((n_noise, 2), dtype=np.intp)
    add_w = np.empty(n_noise, dtype=np.intp)

    while i < n:  # because i==n is possible at the start
        # ind_left[:i] - points in the tree
        # ind_left[i:] - points not yet in the tree

        # find the shortest edge connecting points_left to the tree
        which_min = i
        for j in range(i+1, n):
            if dist_nn[ind_left[j]] < dist_nn[ind_left[which_min]]:
                which_min = j

        # add ind_left[which_min] to the tree; swap i<->which_min
        ind_left[which_min], ind_left[i] = ind_left[i], ind_left[which_min]

        # (ind_left[i], idx_nn[ind_left[i]]) is the connecting edge
        assert y_pred[ind_left[i]] <= 0
        assert y_pred[idx_nn[ind_left[i]]] > 0

        y_pred[ind_left[i]] = y_pred[idx_nn[ind_left[i]]]

        add_w[i-(n-n_noise)] = dist_nn[ind_left[i]]
        add_e[i-(n-n_noise), :] = (idx_nn[ind_left[i]], ind_left[i])

        if i == n-1: break

        # update idx_nn and dist_nn
        for j in range(i+1, n):
            _d = np.sqrt(np.sum((X[ind_left[i], :]-X[ind_left[j], :])**2))
            if _d < dist_nn[ind_left[j]]:
                dist_nn[ind_left[j]] = _d
                idx_nn[ind_left[j]] = ind_left[i]

        i += 1

    return y_pred, add_e, add_w


def _clusterise_without_noise_points(X, L, n_neighbors, noise_threshold, noise_postprocess="tree"):
    n_clusters = L.n_clusters

    n = X.shape[0]

    if n_neighbors > 0:
        kd = scipy.spatial.KDTree(X)
        nn_w, nn_a = kd.query(X, n_neighbors+1)
        assert np.all(nn_a[:, 0] == np.arange(n))
        nn_w = np.array(nn_w)[:, 1:]  # exclude self
        nn_a = np.array(nn_a)[:, 1:]

        how_many = np.bincount(nn_a.ravel(), minlength=n)
        is_noise = (how_many<=noise_threshold)
    else:
        is_noise = np.repeat(False, n)

    X_core = X[~is_noise, :]
    y_pred_unadj = L.fit_predict(X_core)+1  # 0-based -> 1-based

    y_pred = np.zeros(n, dtype=y_pred_unadj.dtype)
    y_pred[~is_noise] = y_pred_unadj

    # classify noise points:

    if noise_postprocess == "closest":
        # # variant A based on closest non-noise point:
        q = np.flatnonzero(y_pred <= 0)
        if len(q) > 0:
            kd2 = scipy.spatial.KDTree(X_core)
            nn_w2, nn_a2 = kd2.query(X[q, :], 1)  # find closest points
            y_pred[q] = y_pred[~is_noise][nn_a2]
        add_e = None
        add_w = None
    #else
        # # variant A that guarantees nothing:
        # q = np.flatnonzero(y_pred <= 0)
        # j = 0
        # while len(q) > 0 and j < n_neighbors:
        #     q_prev = q
        #     q = []
        #     for i in q_prev:
        #         if y_pred[nn_a[i, j]] > 0:
        #             y_pred[i] = y_pred[nn_a[i, j]]
        #         else:
        #             q.append(i)
        #     j += 1
        #
        # assert np.all(y_pred > 0)  # not necessarily true...
    elif noise_postprocess == "tree":
        # # variant based on spanning trees
        y_pred, add_e, add_w = _classify_noise_via_spanning_trees(X, y_pred, is_noise)
    else:
        raise Exception("incorrect `noise_postprocess`")

    tree_skiplist = L._tree_skiplist
    tree_w = L._tree_w
    tree_e = L._tree_e
    which_core  = np.flatnonzero(~is_noise)
    tree_e = which_core[tree_e]

    if add_w is not None: tree_w = np.concatenate((tree_w, add_w))
    if add_e is not None: tree_e = np.concatenate((tree_e, add_e))


    assert np.all(y_pred > 0)
    y_pred = y_pred - 1   # 1-based to 0-based !!!


    return y_pred, tree_w, tree_e, tree_skiplist, is_noise





class Silentmark(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        *,
        n_clusters,
        n_neighbors=None,
        min_cluster_size=5,       # RobustSingleLinkage
        min_cluster_factor=0.15,  # RobustSingleLinkage
        skip_leaves=True,         # RobustSingleLinkage
        gini_threshold=0.3,       # Genie
        noise_threshold=None,
        noise_postprocess="tree",
        verbose=False
    ):
        """
        ...
        """

        self.n_clusters          = n_clusters
        self.n_neighbors         = n_neighbors
        self.noise_threshold     = noise_threshold
        self.min_cluster_size    = min_cluster_size    # RobustSingleLinkage
        self.min_cluster_factor  = min_cluster_factor  # RobustSingleLinkage
        self.skip_leaves         = skip_leaves         # RobustSingleLinkage
        self.gini_threshold      = gini_threshold      # Genie
        self.noise_postprocess   = noise_postprocess
        self.verbose             = verbose


    def fit(self, X, y=None, mst_skiplist=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns.

        y : None
            Ignored.

        mst_skiplist : None


        Returns
        -------

        self : genieclust.Silentmark
            The object that the method was called on.
        """

        self.X = np.ascontiguousarray(X)
        self.n_samples_ = self.X.shape[0]

        # L = robust_single_linkage.RobustSingleLinkageClustering(
        #     n_clusters=self.n_clusters,
        #     M=1,  # ordinary distance, not the mutual reachability one
        #     min_cluster_factor=self.min_cluster_factor,
        #     min_cluster_size=self.min_cluster_size,
        #     skip_leaves=self.skip_leaves
        # )

        L = genieclust.Genie(n_clusters=self.n_clusters, gini_threshold=self.gini_threshold)


        self.n_neighbors_ = self.n_neighbors
        if self.n_neighbors_ is None: # TODO: default
            self.n_neighbors_ = max(5, int(np.sqrt(self.n_samples_/self.n_clusters*self.min_cluster_factor)))

        self.noise_threshold_ = self.noise_threshold
        if self.noise_threshold_ is None: # TODO: default
            self.noise_threshold_ = self.n_neighbors_ - 1
        elif self.noise_threshold_ in [str(i) for i in range(-self.n_neighbors_, self.n_neighbors_+1)]:
            self.noise_threshold_ = self.n_neighbors_ - int(self.noise_threshold_)
        elif self.noise_threshold_ == "half":
            self.noise_threshold_ = np.floor(self.n_neighbors_ / 2)
        elif self.noise_threshold_ == "uhalf":
            self.noise_threshold_ = np.ceil(self.n_neighbors_ / 2)
        elif type(self.noise_threshold_) is int:
            pass
        else:
            raise Exception("incorrect `noise_threshold`")

        (
            self.labels_,
            self._tree_w,
            self._tree_e,
            self._tree_skiplist,
            self._is_noise,
        ) = _clusterise_without_noise_points(
            self.X,
            L,
            self.n_neighbors_,
            self.noise_threshold_,
            self.noise_postprocess
        )

        return self



    def fit_predict(self, X, y=None, mst_skiplist=None):
        """
        Perform cluster analysis of a dataset and return the predicted labels.


        Parameters
        ----------

        X : object
            See `genieclust.Silentmark.fit`.

        y : None
            See `genieclust.Silentmark.fit`.

        mst_skiplist : None


        Returns
        -------

        labels_ : ndarray
            `self.labels_` attribute.


        See also
        --------

        genieclust.Silentmark.fit

        """
        self.fit(X, y, mst_skiplist)
        return self.labels_


