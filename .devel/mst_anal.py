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

from generalized_normalized_clustering_accuracy import generalized_normalized_clustering_accuracy as GNCA
from generalized_normalized_clustering_accuracy import generalized_normalized_pivoted_accuracy as GNPA

data_path = os.path.join("~", "Projects", "clustering-data-v1")



plt.clf()
_i = 0
for ex in range(12):
    _i += 1
    plt.subplot(3, 4, _i)
    X, y_true, n_clusters, skiplist, example = mst_examples.get_example(ex, data_path)

    n_clusters = max(y_true)

    #L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters*3)
    L = lumbermark.Lumbermark(n_clusters=n_clusters, verbose=False, n_neighbors=5, cluster_size_factor=0.2, outlier_factor=1.5, noise_cluster=False)

    y_pred = L.fit_predict(X, mst_skiplist=skiplist)  # TODO: 0-based -> 1-based!!!


    mst_examples.plot_mst_2d(L)
    npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
    nca = GNCA(y_true[y_true>0], y_pred[y_true>0])

    plt.title("%s NA=%.2f NCA=%.2f" % (example, npa, nca))
plt.tight_layout()


stop()



mst_draw_edge_labels = False

mst_e = L._mst_e
mst_w = L._mst_w
mst_s = L._mst_s
min_mst_s = np.min(mst_s, axis=1)
mst_labels = L._mst_labels
n = X.shape[0]
skiplist = L._mst_skiplist
cutting = L._mst_cutting
mst_internodes = L.__dict__.get("_mst_internodes", [])

#
plt.clf()
#
# MST
plt.subplot(3, 2, (2, 6))
# nn_threshold = np.mean(nn_w[:, -1])
# nn_w[:, -1] > nn_threshold
# y_pred[nn_w[:, -1] > nn_threshold] = 0
genieclust.plots.plot_scatter(X[:,:2], labels=y_pred-1)
plt.axis("equal")
if mst_draw_edge_labels:
    for i in range(n-1):
        plt.text(
            (X[mst_e[i,0],0]+X[mst_e[i,1],0])/2,
            (X[mst_e[i,0],1]+X[mst_e[i,1],1])/2,
            "%d (%d-%d)" % (i, mst_s[i, 0], mst_s[i, 1]),
            color="gray" if mst_labels[i] == lumbermark.SKIP else genieclust.plots.col[mst_labels[i]-1],
            va='top'
        )
for i in range(n_clusters+1):
    genieclust.plots.plot_segments(mst_e[mst_labels == i, :], X, color=genieclust.plots.col[i-1],
        alpha=0.2, linestyle="-" if i>0 else ":")
genieclust.plots.plot_segments(mst_e[mst_labels<0,:], X, color="yellow", linestyle="-", linewidth=3)
genieclust.plots.plot_segments(mst_e[mst_internodes,:], X, color="orange", linestyle="-", linewidth=3)
if cutting is not None:
    genieclust.plots.plot_segments(mst_e[[cutting],:], X, color="blue", linestyle="--", alpha=1.0, linewidth=3)
stop()
#
#
# Edge weights + cluster sizes
ax1 = plt.subplot(3, 2, 1)
op = np.flatnonzero(mst_labels>0)
ax1.plot(mst_w[op], color='blue')
ax2 = ax1.twinx()
ax2.plot(np.arange(len(op)), min_mst_s[op], c='blue', alpha=0.2)
if cutting is not None:
    idx = np.sum(op < cutting)
    plt.text(idx, min_mst_s[cutting], cutting, ha='center', va='bottom', color=genieclust.plots.col[mst_labels[cutting]-1])
#
# MST edges per cluster
ax1 = plt.subplot(3, 2, 3)
ax2 = ax1.twinx()
last = 0
for i in range(1, n_clusters+1):
    op = np.flatnonzero(mst_labels == i)
    len_op = len(op)
    ax1.plot(np.arange(last, last+len_op), mst_w[op],
        color=genieclust.plots.col[i-1])
    ax2.plot(np.arange(last, last+len_op), min_mst_s[op], c=genieclust.plots.col[i-1], alpha=0.2)
    if cutting is not None and mst_labels[cutting] == i:
        idx = np.sum(op < cutting)
        plt.text(last+idx, min_mst_s[cutting], cutting, ha='center', va='bottom', color=genieclust.plots.col[mst_labels[cutting]-1])
    # ce = (np.arange(1, n)*internodes)[op]   # 1-shift
    # for j in np.flatnonzero(ce):
        # idx = ce[j]-1 # unshift
        # plt.text(j, min_mst_s[idx], idx, ha='center', color=genieclust.plots.col[mst_labels[idx]-1])

    last += len_op
#
#
# treelhouette
plt.subplot(3, 2, 5)
cluster_distances = np.ones((n_clusters, n_clusters))*np.inf
for e in skiplist:
    i, j = mst_e[e, :]
    assert y_pred[i] > 0
    assert y_pred[j] > 0
    cluster_distances[y_pred[i]-1, y_pred[j]-1] = mst_w[e]
    cluster_distances[y_pred[j]-1, y_pred[i]-1] = mst_w[e]
# leave the diagonal to inf
min_intercluster_distances = np.min(cluster_distances, axis=0)
#
# Variant 1) per-vertex:
# a = np.zeros(n)
# for i in range(n):
    # a[i] = np.min(mst_w[mst_a[i][mst_labels[mst_a[i]] == y_pred[i]]])
# l = y_pred
#
# Variant 2) per-edge:
a = mst_w[mst_labels > 0]
l = mst_labels[mst_labels > 0]
#
b = min_intercluster_distances[l - 1]
s = np.where(a<b, 1.0 - a/b, b/a - 1.0)
o1 = np.argsort(s)[::-1]
o2 = np.argsort(l[o1], kind='stable')
plt.bar(np.arange(len(s)), s[o1][o2], width=1.0, color=np.array(genieclust.plots.col)[l[o1]-1][o2])
treelhouette_score = np.mean(s)
print("treelhouette_score=%.3f" % treelhouette_score)
plt.axhline(treelhouette_score, color='gray')



# DBSCAN is non-adaptive - cannot detect clusters of different densities well
#plt.violinplot([ mst_w[mst_labels==i] for i in range(1, c+1) ])


# r = np.median(mst_w[mst_labels == 3])
# def visit2(v, e, r):

