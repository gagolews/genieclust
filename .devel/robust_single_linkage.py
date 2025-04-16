"""
A Robust Single Linkage Clustering Algorithm
(Divisive; aka Robust Single Divide);
Honours n_clusters
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

SKIP = np.iinfo(int).min  # must be negative
UNSET = -1 # must be negative
NOISE = 0  # must be 0


# TODO: 0-based -> 1-based!!!



def _robust_single_linkage_clustering(
    X,
    n_clusters,
    min_cluster_size=10,
    min_cluster_factor=0.2,
    mst_skiplist=None,      # default = empty
    M=1,  # TODO: mutual reachability distance
    skip_leaves=False,
    verbose=False
):
    assert n_clusters > 0
    assert min_cluster_size >= 0
    assert min_cluster_factor >= 0

    n = X.shape[0]

    if mst_skiplist is None:
        mst_skiplist = []

    if M <= 1:
        mst_w, mst_e = genieclust.internal.mst_from_distance(X, "euclidean")
    else:
        if M-1 >= X.shape[0]:
            raise ValueError("`M` is too large")

        nn_dist, nn_ind = genieclust.internal.knn_from_distance(
            X, k=M-1, metric="euclidean", verbose=verbose
        )

        d_core = genieclust.internal.get_d_core(nn_dist, nn_ind, M)

        _o = np.argsort(d_core)[::-1]  # order wrt decreasing "core" size
        mst_w, mst_e = genieclust.internal.mst_from_distance(
            np.ascontiguousarray(X[_o, :]),
            metric="euclidean",
            d_core=np.ascontiguousarray(d_core[_o]),
            verbose=verbose
        )
        mst_e = np.c_[ _o[mst_e[:, 0]], _o[mst_e[:, 1]] ]


    _o = np.argsort(mst_w)[::-1]
    mst_w = mst_w[_o]  # weights sorted decreasingly
    mst_e = mst_e[_o, :]


    # TODO: store as two arrays: indices, indptr (compressed sparse row format)
    mst_a = [ [] for i in range(n) ]
    for i in range(n-1):
        mst_a[mst_e[i, 0]].append(i)
        mst_a[mst_e[i, 1]].append(i)
    n_leaves = 0
    for i in range(n):
        mst_a[i] = np.array(mst_a[i])
        if len(mst_a[i]) == 1:
            n_leaves += 1
    mst_a = np.array(mst_a, dtype="object")

    if skip_leaves:
        min_cluster_size = max(min_cluster_size, min_cluster_factor*(n-n_leaves)/n_clusters)
    else:
        min_cluster_size = max(min_cluster_size, min_cluster_factor*n/n_clusters)

    def visit(v, e, c):  # v->w  where mst_e[e,:]={v,w}
        if mst_labels[e] == SKIP or (c != NOISE and mst_labels[e] == NOISE):
            return 0
        iv = int(mst_e[e, 1] == v)
        w = mst_e[e, 1-iv]
        tot = 1
        for e2 in mst_a[w]:
            if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
                tot += visit(w, e2, c)
        mst_s[e, iv] = tot if c != NOISE else -1
        mst_s[e, 1-iv] = 0

        labels[w] = c
        mst_labels[e] = c

        return tot


    def mark_clusters():
        c = 0
        for v in range(n):
            if labels[v] == UNSET:
                c += 1
                labels[v] = c
                for e in mst_a[v]:
                    visit(v, e, c)

        # c is the number of clusters
        counts = np.bincount(labels)  # cluster counts, counts[0] == number of noise points

        # fix mst_s now that we know the cluster sizes
        for i in range(n-1):
            if mst_labels[i] > 0:
                j = int(mst_s[i, 1] == 0)
                mst_s[i, j] = counts[np.abs(mst_labels[i])]-mst_s[i, 1-j]
        min_mst_s = np.min(mst_s, axis=1)

        return c, counts, min_mst_s

    c = len(mst_skiplist)
    while True:
        last_iteration = (len(mst_skiplist)+1 >= n_clusters)

        mst_s = np.zeros((n-1, 2), dtype=int)  # sizes: mst_s[e, 0] is the size of the cluster that appears when we .... TODO
        labels = np.repeat(UNSET, n)
        mst_labels = np.repeat(UNSET, n-1)
        for s in mst_skiplist:
            mst_labels[s] = SKIP  # mst_skiplist

        if skip_leaves and not last_iteration:
            for i in range(n):
                if len(mst_a[i]) == 1:  # a leaf
                    labels[i] = NOISE
                    mst_labels[mst_a[i][0]] = NOISE

        c, counts, min_mst_s = mark_clusters()
        assert c == len(mst_skiplist)+1


        # the longest node that yields not too small a cluster
        is_limb = (mst_labels > 0)

        is_limb &= (min_mst_s >= min(min_cluster_size, np.max(min_mst_s[is_limb])))

        # alternatively - worse
        # is_limb &= (min_mst_s >= 10)
        # is_limb &= (np.min(mst_s, axis=1)/np.max(mst_s, axis=1)) >= 0.25

        #_ee = min_mst_s[(mst_labels > 0)]/min_cluster_size
        #print(np.round(_ee[_ee>0.49], 2))

        which_limbs = np.flatnonzero(is_limb)  # TODO: we just need the first non-zero here....
        cutting_e = which_limbs[0]
        cutting_w = mst_w[cutting_e]


        if last_iteration:
            assert c == n_clusters
            return (
                labels, mst_w, mst_e, mst_labels, mst_s,
                mst_skiplist, cutting_e
            )
        else:
            mst_skiplist.append(cutting_e)



class RobustSingleLinkageClustering(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        *,
        n_clusters,
        min_cluster_size=10,
        min_cluster_factor=0.2,
        M=1,
        skip_leaves=False,
        verbose=False
    ):
        self.n_clusters              = n_clusters
        self.min_cluster_size        = min_cluster_size
        self.min_cluster_factor      = min_cluster_factor
        self.M                       = M
        self.skip_leaves             = skip_leaves
        self.verbose                 = verbose


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

        self : genieclust.Genie
            The object that the method was called on.
        """

        self.X = np.ascontiguousarray(X)
        self.n_samples_ = self.X.shape[0]

        (
            self.labels_, self._mst_w, self._mst_e, self._mst_labels,
            self._mst_s, self._mst_skiplist, self._mst_cutting
        ) = _robust_single_linkage_clustering(
            self.X,
            n_clusters=self.n_clusters,
            min_cluster_size=self.min_cluster_size,
            min_cluster_factor=self.min_cluster_factor,
            M=self.M,
            skip_leaves=self.skip_leaves,
            verbose=self.verbose,
            mst_skiplist=mst_skiplist
        )

        return self



    def fit_predict(self, X, y=None, mst_skiplist=None):
        """
        Perform cluster analysis of a dataset and return the predicted labels.


        Parameters
        ----------

        X : object
            See `genieclust.Genie.fit`.

        y : None
            See `genieclust.Genie.fit`.

        mst_skiplist : None


        Returns
        -------

        labels_ : ndarray
            `self.labels_` attribute.


        See also
        --------

        genieclust.Genie.fit

        """
        self.fit(X, y, mst_skiplist)
        return self.labels_
