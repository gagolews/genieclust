"""
...
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
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import os.path
import scipy.spatial
import sys
from importlib import reload
import lumbermark
lumbermark = reload(lumbermark)
import robust_single_linkage
robust_single_linkage = reload(robust_single_linkage)
import mst_examples
mst_examples = reload(mst_examples)
sys.setrecursionlimit(100000)
import scipy.spatial.distance
from numba import jit

data_path = os.path.join("~", "Projects", "clustering-data-v1")

# noise point detection

@jit
def classify_noise_via_spanning_trees(X, y_pred, is_noise):
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

    while True:
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
        y_pred[ind_left[i]]= y_pred[idx_nn[ind_left[i]]]

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


def clusterise_without_noise_points(L, M=None, noise_threshold=None):
    n_clusters = L.n_clusters

    n_neighbors = M-1

    if noise_threshold is None:
        noise_threshold = n_neighbors-1

    n = X.shape[0]

    kd = scipy.spatial.KDTree(X)
    nn_w, nn_a = kd.query(X, n_neighbors+1)
    assert np.all(nn_a[:, 0] == np.arange(n))
    nn_w = np.array(nn_w)[:, 1:]  # exclude self
    nn_a = np.array(nn_a)[:, 1:]

    how_many = np.bincount(nn_a.ravel(), minlength=n)
    is_noise = (how_many<=noise_threshold)
    X_core = X[~is_noise, :]
    y_pred_unadj = L.fit_predict(X_core)

    y_pred = np.zeros(n, dtype=y_pred_unadj.dtype)
    y_pred[~is_noise] = y_pred_unadj

    # classify noise points:

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

    # # variant B based on closest non-noise point:
    # q = np.flatnonzero(y_pred <= 0)
    # if len(q) > 0:
    #     kd2 = scipy.spatial.KDTree(X_core)
    #     nn_w2, nn_a2 = kd2.query(X[q, :], 1)  # find closest points
    #     y_pred[q] = y_pred[~is_noise][nn_a2]


    # # variant C based on spanning trees
    y_pred, add_e, add_w = classify_noise_via_spanning_trees(X, y_pred, is_noise)

    mst_w = L._mst_w
    mst_e = L._mst_e
    which_core  = np.flatnonzero(~is_noise)
    mst_e = which_core[mst_e]

    st_e = np.concatenate((mst_e, add_e))
    st_w = np.concatenate((mst_w, add_w))

    skiplist = L._mst_skiplist


    assert np.all(y_pred > 0)

    return y_pred, is_noise, st_e, st_w, skiplist





plt.clf()
_i = 0
n_examples = 30
for ex in range(n_examples):
    _i += 1
    plt.subplot(int(np.floor(np.sqrt(n_examples))), int(np.ceil(np.sqrt(n_examples))), _i)
    X, y_true, n_clusters, skiplist, example = mst_examples.get_example(ex, data_path)

    n_clusters = max(y_true)

    L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters, M=1, min_cluster_factor=0.1, skip_leaves=True, min_cluster_size=5)

    M = max(6, int(np.sqrt(X.shape[0]/n_clusters*0.1)))
    print(X.shape[0], M)
    y_pred, is_noise, st_e, st_w, skiplist = clusterise_without_noise_points(L, M)

    genieclust.plots.plot_scatter(X[~is_noise,:], labels=y_pred[~is_noise]-1)
    genieclust.plots.plot_scatter(X, labels=y_pred-1, alpha=0.2)
    genieclust.plots.plot_segments(st_e, X, alpha=0.2)
    genieclust.plots.plot_segments(st_e[skiplist, :], X, color="yellow", alpha=0.2)
    plt.axis("equal")

    # npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
    nca = genieclust.compare_partitions.normalized_clustering_accuracy(y_true[y_true>0], y_pred[y_true>0])

    #s1, s2 = treelhouette.treelhouette_score(L)
    plt.title("%s NCA=%.2f" % (example, nca))

plt.tight_layout()
