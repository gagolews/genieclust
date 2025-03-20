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

from generalized_normalized_clustering_accuracy import generalized_normalized_clustering_accuracy as GNCA
from generalized_normalized_clustering_accuracy import generalized_normalized_pivoted_accuracy as GNPA

data_path = os.path.join("~", "Projects", "clustering-data-v1")



plt.clf()
X, y_true, n_clusters, skiplist, example = mst_examples.get_example(18, data_path)

n_clusters = max(y_true)

# L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters)
L = lumbermark.Lumbermark(n_clusters=n_clusters, verbose=False, n_neighbors=5, outlier_factor=1.5, noise_cluster=True)

y_pred = L.fit_predict(X, mst_skiplist=skiplist)  # TODO: 0-based -> 1-based!!!

mst_examples.plot_mst_2d(L, mst_draw_edge_labels=False, alpha=0.7)
npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
nca = GNCA(y_true[y_true>0], y_pred[y_true>0])

plt.title("%s NA=%.2f NCA=%.2f" % (example, npa, nca))
plt.tight_layout()

n = X.shape[0]

if False:
    assert n <= 10_000
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))
    for i in range(5):  # ?????
        mst_w, mst_e = genieclust.internal.mst_from_complete(D)
        genieclust.plots.plot_segments(mst_e, X, color="gray", linestyle="--", alpha=0.2)
        D[mst_e[:,1],mst_e[:,0]] = np.inf
        D[mst_e[:,0],mst_e[:,1]] = np.inf
else:
    n_neighbors = 10
    kd = scipy.spatial.KDTree(X)
    nn_w, nn_a = kd.query(X, n_neighbors+1)
    nn_w = np.array(nn_w)[:, 1:]  # exclude self
    nn_a = np.array(nn_a)[:, 1:]

    nn_w_agg = mst_examples.aggregate(nn_w[y_pred>0, -1], y_pred[y_pred>0], np.mean)[0]
    #nn_w_agg = mst_examples.aggregate(nn_w[y_pred>0, -1], y_pred[y_pred>0], np.max)[0]

    mst_labels = L._mst_labels
    mst_w = L._mst_w

    _howmanynn = np.sum(nn_w <= nn_w_agg[y_pred-1].reshape(-1, 1), axis=1)
    for i in range(n_neighbors):
        _wh = np.c_[np.arange(n), nn_a[:, i]]
        genieclust.plots.plot_segments(_wh[i<_howmanynn,:], X, color="gray", linestyle="--", alpha=0.4)

    for i in range(n_neighbors+1):
        for j in np.flatnonzero(_howmanynn == i):
            plt.text(X[j, 0], X[j, 1], i)
