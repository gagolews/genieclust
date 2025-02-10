import genieclust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import os.path
import scipy.spatial

# from sklearn.datasets import make_blobs
# X, labels = make_blobs(
#     n_samples=[1000, 100, 100],
#     cluster_std=1,
#     random_state=42
# )
# skiplist = [1197, 1198]


examples = [
    ["fcps", "wingnut", [1014]],
    ["sipu", "spiral", [310, 309]],
    ["wut", "z2", [895]],
    ["sipu", "aggregation", [785, 784, 786, 783]],
    ["graves", "parabolic", [956]],
    ["sipu", "pathbased", [293, 271]],
    ["sipu", "compound", [397, 396, 328, 347]],
    ["other", "hdbscan", [2051, 2047, 2034, 1975, 1942]],
    ["fcps", "engytime", [3597]],
    ["wut", "z2", [897, 895, 893, 892, 683]],
    ["wut", "z2", [894, 895, 893, 889]],
    ["wut", "z2", [897, 895, 893]],
    ["wut", "labirynth", [3544, 3543, 3541, 3533]],
]

example = examples[0]
data_path = os.path.join("~", "Projects", "clustering-data-v1")
np.random.seed(123)
b = clustbench.load_dataset(example[0], example[1], path=data_path)
X, labels = b.data, b.labels[0]
skiplist = example[2]

SKIPEDGE = np.iinfo(int).min


n = X.shape[0]
min_cluster_size = n/(len(skiplist)+1)/4
noise_size = 3
gini_threshold = 0.5

mst = genieclust.internal.mst_from_distance(X, "euclidean")
mst_w, mst_e = mst
adj_list = [ [] for i in range(n) ]
for i in range(n-1):
    adj_list[mst_e[i, 0]].append(i)
    adj_list[mst_e[i, 1]].append(i)
for i in range(n):
    adj_list[i] = np.array(adj_list[i])

def visit(v, e, c):  # v->w  where mst_e[e,:]={v,w}
    if c > 0 and (mst_labels[e] == SKIPEDGE or mst_labels[e] == 0):
        return 0
    iv = int(mst_e[e, 1] == v)
    w = mst_e[e, 1-iv]
    tot = 1
    for e2 in adj_list[w]:
        if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
            tot += visit(w, e2, c)
    mst_s[e, iv] = tot
    mst_s[e, 1-iv] = 0

    if c > 0 or tot < noise_size:
        labels[w] = c
        mst_labels[e] = c

    return tot


mst_s = np.zeros((n-1, 2), dtype=int)
labels = -np.ones(n, dtype=int)
mst_labels = -np.ones(n-1, dtype=int)
for s in skiplist:
    mst_labels[s] = SKIPEDGE  # skiplist
#
# preprocess – noise points:
c = 0
# v = 0  # anything
# for e in adj_list[v]:
#     visit(v, e, c)
#
# mark clusters
c = 0
for v in range(n):
    if labels[v] == -1:
        c += 1
        labels[v] = c
        for e in adj_list[v]:
            visit(v, e, c)
counts = np.bincount(labels)
for i in range(n-1):
    if mst_labels[i] != SKIPEDGE:
        j = int(mst_s[i, 1] == 0)
        mst_s[i, j] = counts[np.abs(mst_labels[i])]-mst_s[i, 1-j]
min_mst_s = np.min(mst_s, axis=1)


nn_k = 5
kd = scipy.spatial.KDTree(X)
nn_w, nn_e = kd.query(X, nn_k+1)
nn_w = np.array(nn_w)[:, 1:]  # exclude self
nn_e = np.array(nn_e)[:, 1:]

nn_threshold = np.mean(nn_w[:, -1])
nn_w[:, -1] > nn_threshold

plt.clf()
genieclust.plots.plot_scatter(X)
plt.axis("equal")
for i in range(nn_k):
    genieclust.plots.plot_segments(np.c_[np.arange(n), nn_e[:,i]], X)

raise Exception("")

#
plt.clf()
#
# Edge weights + cluster sizes
ax1 = plt.subplot(3, 2, 1)
op = np.argsort(mst_w)[::-1]
op = op[mst_labels[op]>0]
#(mst_w[op])[min_mst_s[op]>min_cluster_size]
ax1.plot(mst_w[op])
ax2 = ax1.twinx()
ax2.plot(np.arange(len(op)), min_mst_s[op], c='orange', alpha=0.3)
# Variant 1) mark candidate cut edges based on cut sizes
which_cut = np.flatnonzero(min_mst_s[op]>min_cluster_size)
for i in which_cut:
   plt.text(i, min_mst_s[op[i]], op[i], ha='center', color=genieclust.plots.col[mst_labels[op[i]]-1])
# Variant 2) mark candidate cut edges based on Gini indices of cluster sizes
# which_cut = []
# for i in range(len(op)):
#     assert mst_labels[op[i]] > 0
#     #g = genieclust.inequality.gini_index(np.r_[np.delete(counts, [0, mst_labels[op[i]]]), mst_s[op[i], :]])  # doesn't work well
#     g = genieclust.inequality.gini_index(mst_s[op[i], :])
#     if g < gini_threshold and min_mst_s[op[i]] > min_cluster_size:
#         plt.text(i, min_mst_s[op[i]], op[i], ha="center")
#        which_cut.append(i)
#
print(op[which_cut])
#
# MST edges per cluster
ax1 = plt.subplot(3, 2, 3)
ax2 = ax1.twinx()
last = 0
for i in range(1, c+1):
    _w = mst_w[mst_labels==i]
    _o = np.argsort(_w)[::-1]
    ax1.plot(
        np.arange(last, last+counts[i]-1),
        _w[_o],
        color=genieclust.plots.col[i-1]
    )
    ax2.plot(
        np.arange(last, last+counts[i]-1),
        min_mst_s[mst_labels==i][_o],
        color=genieclust.plots.col[i-1],
        alpha=0.2
    )
    last += counts[i] - 1
#
#
# treelhouette
plt.subplot(3, 2, 5)
cluster_distances = np.ones((c, c))*np.inf
for e in skiplist:
    i, j = mst_e[e, :]
    assert labels[i] > 0
    assert labels[j] > 0
    cluster_distances[labels[i]-1, labels[j]-1] = mst_w[e]
    cluster_distances[labels[j]-1, labels[i]-1] = mst_w[e]
# leave the diagonal to inf
min_intercluster_distances = np.min(cluster_distances, axis=0)
#
# Variant 1) per-vertex:
# a = np.zeros(n)
# for i in range(n):
    # a[i] = np.min(mst_w[adj_list[i][mst_labels[adj_list[i]] == labels[i]]])
# l = labels
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
print(treelhouette_score)
plt.axhline(treelhouette_score, color='gray')
#
# MST
plt.subplot(3, 2, (2, 6))
labels[nn_w[:, -1] > nn_threshold] = 0
genieclust.plots.plot_scatter(X, labels=labels-1)
genieclust.plots.plot_segments(mst_e, X, style="b-", alpha=0.1)
#genieclust.plots.plot_segments(mst_e[min_mst_s>min_cluster_size,:], X, alpha=0.5)
#genieclust.plots.plot_segments(mst_e[mst_labels<0,:], X, style="w-")
plt.axis("equal")
for i in range(n-1):
    plt.text(
        (X[mst_e[i,0],0]+X[mst_e[i,1],0])/2,
        (X[mst_e[i,0],1]+X[mst_e[i,1],1])/2,
        "%d (%d)" % (i, min(mst_s[i, 0], mst_s[i, 1])),
        color="gray" if mst_labels[i] == SKIPEDGE else genieclust.plots.col[mst_labels[i]-1],
    )



# DBSCAN is non-adaptive - cannot detect clusters of different densities well
#plt.violinplot([ mst_w[mst_labels==i] for i in range(1, c+1) ])


# r = np.median(mst_w[mst_labels == 3])
# def visit2(v, e, r):

