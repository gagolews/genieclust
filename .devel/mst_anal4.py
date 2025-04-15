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


data_path = os.path.join("~", "Projects", "clustering-data-v1")

# noise point detection


def clusterise_without_noise_points(L, M, noise_threshold=None):

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
    X2 = X[~is_noise, :].copy()
    y_pred_unadj = L.fit_predict(X2)

    y_pred = np.zeros(n, dtype=y_pred_unadj.dtype)
    y_pred[~is_noise] = y_pred_unadj

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


    q = np.flatnonzero(y_pred <= 0)
    if len(q) > 0:
        kd2 = scipy.spatial.KDTree(X2)
        nn_w2, nn_a2 = kd2.query(X[q, :], 1)  # find closest points
        y_pred[q] = y_pred[~is_noise][nn_a2]

    return y_pred, is_noise





plt.clf()
_i = 0
for ex in range(16):
    _i += 1
    plt.subplot(4, 4, _i)
    X, y_true, n_clusters, skiplist, example = mst_examples.get_example(ex, data_path)

    n_clusters = max(y_true)

    L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters, M=1, min_cluster_factor=0.1, skip_leaves=False, min_cluster_size=5)


    y_pred, is_noise = clusterise_without_noise_points(L, 25)

    genieclust.plots.plot_scatter(X[~is_noise,:], labels=y_pred[~is_noise]-1)
    genieclust.plots.plot_scatter(X, labels=y_pred-1, alpha=0.2)
    plt.axis("equal")

    # npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
    nca = genieclust.compare_partitions.normalized_clustering_accuracy(y_true[y_true>0], y_pred[y_true>0])

    #s1, s2 = treelhouette.treelhouette_score(L)
    plt.title("%s NCA=%.2f" % (example, nca))

plt.tight_layout()

