"""
A remix of the Genie clustering algorithm - premerge limbs (min_cluster_size)
or postmerge them. Not so good.
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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


def _eugeniusz(
    X,
    n_clusters,
    gini_threshold=0.3,
    min_cluster_size=10,
    M=1,
    verbose=False
):
    assert n_clusters > 0
    assert min_cluster_size >= 0

    n = X.shape[0]

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

    # NOTE/TODO ???
    #min_cluster_size = max(min_cluster_size, min_cluster_factor*n/n_clusters)


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


    mst_s = np.zeros((n-1, 2), dtype=int)  # sizes: mst_s[e, 0] is the size of the cluster that appears when we .... TODO
    labels = np.repeat(UNSET, n)
    mst_labels = np.repeat(UNSET, n-1)

    c, counts, min_mst_s = mark_clusters()
    assert c == 1


    ds = genieclust.internal.GiniDisjointSets(n)

    if False:
        peripheral_edges = (min_mst_s < min_cluster_size)
        for i in np.flatnonzero(peripheral_edges):
            ds.union(mst_e[i, 0], mst_e[i, 1])

        left_edges = np.flatnonzero(~peripheral_edges)
        while ds.get_k() > n_clusters:
            if ds.get_gini() < gini_threshold:
                # edges are sorted decreasingly wrt their weights
                i = left_edges[0]
                left_edges = left_edges[1:]  # cheap
                ds.union(mst_e[i, 0], mst_e[i, 1])
            else:
                s = ds.get_smallest_count()
                i = 0
                while min(ds.get_count(mst_e[left_edges[i], 0]), ds.get_count(mst_e[left_edges[i], 1])) > s:
                    i += 1
                ds.union(mst_e[left_edges[i], 0], mst_e[left_edges[i], 1])
                left_edges = np.r_[left_edges[:i], left_edges[(i+1):]]  # costly
    else:
        peripheral_edges = (min_mst_s < min_cluster_size)
        peripheral_nodes = mst_e[mst_s < min_cluster_size]

        left_edges = np.flatnonzero(~peripheral_edges)
        while len(left_edges)+1 > n_clusters:
            if ds.get_gini() < gini_threshold:
                # edges are sorted decreasingly wrt their weights
                i = left_edges[0]
                left_edges = left_edges[1:]  # cheap
                ds.union(mst_e[i, 0], mst_e[i, 1])
            else:
                s = np.inf
                i = -1
                j = 0
                while j < len(left_edges):
                    t = min(ds.get_count(mst_e[left_edges[j], 0]), ds.get_count(mst_e[left_edges[j], 1]))
                    if t < s:
                        i = j
                        s = t
                    j += 1
                ds.union(mst_e[left_edges[i], 0], mst_e[left_edges[i], 1])
                left_edges = np.r_[left_edges[:i], left_edges[(i+1):]]  # costly

        for i in np.flatnonzero(peripheral_edges):
            ds.union(mst_e[i, 0], mst_e[i, 1])

    labels = ds.to_list_normalized() + 1  # 0 = noise cluster
    mst_labels = None

    return (
        labels, mst_w, mst_e, mst_labels, mst_s, left_edges
    )



class Eugeniusz(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        *,
        n_clusters,
        gini_threshold=0.3,
        min_cluster_size=10,
        M=1,
        verbose=False
    ):
        self.n_clusters              = n_clusters
        self.gini_threshold          = gini_threshold
        self.min_cluster_size        = min_cluster_size
        self.M                       = M
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

        (
            self.labels_,
            self._tree_w,
            self._tree_e,
            self._tree_labels,
            self._tree_s,
            self._tree_cutlist
        ) = _eugeniusz(
            self.X,
            n_clusters=self.n_clusters,
            gini_threshold=self.gini_threshold,
            min_cluster_size=self.min_cluster_size,
            M=self.M,
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
