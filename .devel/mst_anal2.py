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
import lumbermark2
lumbermark2 = reload(lumbermark2)
import robust_single_linkage
robust_single_linkage = reload(robust_single_linkage)
import mst_examples
mst_examples = reload(mst_examples)
sys.setrecursionlimit(100000)

from generalized_normalized_clustering_accuracy import generalized_normalized_clustering_accuracy as GNCA
from generalized_normalized_clustering_accuracy import generalized_normalized_pivoted_accuracy as GNPA

data_path = os.path.join("~", "Projects", "clustering-data-v1")



plt.clf()
X, y_true, n_clusters, skiplist, example = mst_examples.get_example(0, data_path)

n_clusters = max(y_true)

# L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters, min_cluster_factor=0.2)
# L = lumbermark.Lumbermark(n_clusters=n_clusters, verbose=False, n_neighbors=5, outlier_factor=1.5, noise_cluster=True)
L = lumbermark2.Lumbermark2(n_clusters=n_clusters, verbose=False, n_neighbors=5, outlier_factor=1.5, noise_cluster=True)

y_pred = L.fit_predict(X, mst_skiplist=skiplist)  # TODO: 0-based -> 1-based!!!


mst_examples.plot_mst_2d(L, mst_draw_edge_labels=True)
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
def get_intercluster_distances():
    mst_a = [ [] for i in range(n) ]
    for i in range(n-1):
        mst_a[mst_e[i, 0]].append(i)
        mst_a[mst_e[i, 1]].append(i)
    for i in range(n):
        mst_a[i] = np.array(mst_a[i])
    mst_a = np.array(mst_a, dtype="object")

    def visit(v, e):  # from v along e
        iv = int(mst_e[e, 1] == v)
        w = mst_e[e, 1-iv]

        if y_pred[w] > 0:
            # reached a coloured vertex - stop
            return [(y_pred[w], 0.0)]
        if len(mst_a[w]) == 1:
            # reached a leaf - stop
            return []

        res = []
        for e2 in mst_a[w]:
            if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
                res += [(l, w+mst_w[e2]) for (l, w) in visit(w, e2)]

        return res

    D = np.ones((n_clusters, n_clusters))*np.inf
    for e in skiplist:
        res = []
        v, w = mst_e[e, :]
        res_v = visit(v, e)
        res_w = visit(w, e)
        for (lv, dv) in res_v:
            for (lw, dw) in res_w:
                D[lv-1, lw-1] = np.minimum(D[lv-1, lw-1], dv+dw+mst_w[e])
                D[lw-1, lv-1] = D[lv-1, lw-1]
    return D
#
cluster_distances = get_intercluster_distances()
# print(np.round(cluster_distances, 2))
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
weighted_treelhouette_score = np.mean(mst_examples.aggregate(s, mst_labels[mst_labels>0], np.mean)[0])
print("treelhouette_score=%.3f, weighted_treelhouette_score=%.3f" % (treelhouette_score, weighted_treelhouette_score))
plt.axhline(treelhouette_score, color='gray')
plt.axhline(weighted_treelhouette_score, color='lightgray')


# DBSCAN is non-adaptive - cannot detect clusters of different densities well
#plt.violinplot([ mst_w[mst_labels==i] for i in range(1, c+1) ])


# r = np.median(mst_w[mst_labels == 3])
# def visit2(v, e, r):

