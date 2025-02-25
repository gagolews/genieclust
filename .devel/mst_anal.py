import genieclust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import os.path
import scipy.spatial



data_path = os.path.join("~", "Projects", "clustering-data-v1")
np.random.seed(123)



examples = [
    ["new", "blobs4a", [], 4],
    ["new", "blobs4b", [], 4],
    ["wut", "mk4", [], 3],
    ["wut", "mk3", [], 3],
    ["sipu", "pathbased", [], 3],  # :/
    ["wut", "mk2", [], 2],  # :/
    ["wut", "graph", [], 10],
    ["graves", "fuzzyx", [], 4],
    ["fcps", "engytime", [], 2],
    ["new", "blobs3a", [], 3],
    ["graves", "zigzag_outliers", [], 3],
    ["fcps", "target", [], 2],
    ["sipu", "spiral", [], 3],
    ["sipu", "jain", [], 2],
    ["sipu", "unbalance", [], 8],
    ["sipu", "aggregation", [], 7],
    ["wut", "z2", [], 5],
    ["wut", "isolation", [], 3],
    ["wut", "x3", [], 3],
    ["sipu", "compound", [], 5],
    ["graves", "parabolic", [], 2],
    ["sipu", "flame", [], 2],
    ["graves", "dense", [], 2],
    ["wut", "circles", [], 1],
    ["wut", "twosplashes", [], 2],
    ["fcps", "twodiamonds", [], 2],
    ["other", "chameleon_t8_8k", [], 8],
    ["other", "chameleon_t7_10k", [], 9],
    ["other", "chameleon_t5_8k", [], 6],
    ["other", "chameleon_t4_8k", [], 6],
    ["wut", "labirynth", [], 6],
    ["wut", "z3", [], 4],
    ["other", "hdbscan", [], 6],
    ["sipu", "s1", [], 15],
    ["fcps", "wingnut", [], 2],
]


battery, dataset, skiplist, n_clusters = examples[0]

if battery != "new":
    b = clustbench.load_dataset(battery, dataset, path=data_path)
    X, y_true = b.data, b.labels[0]
else:
    from sklearn.datasets import make_blobs
    if dataset == "blobs4a":
        X, y_true = make_blobs(
            n_samples=[500, 500, 100, 100],
            cluster_std=[0.05, 0.2, 0.2, 0.2],
            centers=[[1,1], [1,-1], [-1,-1], [-1,1]],
            random_state=42
        )
        xapp = np.array([
            [0,0],
            [-0.1,0.25],
            [-0.1,-0.1],
            [0.1,0.25],
        ])
        X = np.append(X, xapp, axis=0)
        y_true = np.r_[y_true+1, np.repeat(0, xapp.shape[0])]
    elif dataset == "blobs4b":
        X, y_true = make_blobs(
            n_samples=[800, 800, 100, 100],
            cluster_std=[0.05, 0.2, 0.3, 0.2],
            centers=[[1,1], [1,-1], [-1,-1], [-1,1]],
            random_state=42
        )
        xapp = np.array([
            [0,0],
            [-0.1,0.25],
            [-0.1,-0.1],
            [0.1,0.25],
        ])
        X = np.append(X, xapp, axis=0)
        y_true = np.r_[y_true+1, np.repeat(0, xapp.shape[0])]
    elif dataset == "blobs3a":
        # see https://github.com/gagolews/genieclust/issues/91
        X, y_true = make_blobs(
            n_samples=[1000, 100, 100],
            cluster_std=1,
            random_state=42
        )

from importlib import reload
import lumbermark
lumbermark = reload(lumbermark)

import sys
sys.setrecursionlimit(100000)


L = lumbermark.Lumbermark(n_clusters=n_clusters, verbose=True, n_neighbors=5)
y_pred = L.fit_predict(X, mst_skiplist=skiplist)
# TODO: 0-based -> 1-based!!!



print(np.bincount(y_pred))

## TODO: n_neighbors -- robust single linkage?



mst_draw_edge_labels = False

mst_e = L._mst_e
mst_w = L._mst_w
mst_s = L._mst_s
min_mst_s = np.min(mst_s, axis=1)
mst_labels = L._mst_labels
n = X.shape[0]
skiplist = L._mst_skiplist
cutting = L._mst_cutting
mst_internodes = L._mst_internodes

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

