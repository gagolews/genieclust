import genieclust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench

import os.path
data_path = os.path.join("~", "Projects", "clustering-data-v1")
b = clustbench.load_dataset("sipu", "pathbased", path=data_path)
X = b.data
#labels = b.labels[0]


mst = genieclust.internal.mst_from_distance(X, "euclidean")
mst_w, mst_e = mst
n = len(mst_w)+1
adj_list = [ [] for i in range(n) ]
for i in range(n-1):
    adj_list[mst_e[i, 0]].append(i)
    adj_list[mst_e[i, 1]].append(i)


def visit(v, e, c):  # v->w  where mst_e[e,:]={v,w}
    if mst_labels[e] < 0:  # skiplist
        return 0
    assert mst_labels[e] == 0
    iv = int(mst_e[e, 1] == v)
    w = mst_e[e, 1-iv]
    labels[w] = c
    mst_labels[e] = c
    tot = 1
    for e2 in adj_list[w]:
        if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
            tot += visit(w, e2, c)
    mst_s[e, iv] = tot
    mst_s[e, 1-iv] = 0
    return tot



min_cluster_size = n/20
# TODO: store mst_s in each iteration
# TODO: restart from the vertices adjacent to the skipped edge
skiplist = [293, 271]
mst_s = np.zeros((n-1, 2), dtype=int)
labels = np.zeros(n, dtype=int)
mst_labels = np.zeros(n-1, dtype=int)
for s in skiplist:
    mst_labels[s] = -1  # skiplist
c = 0
while True:
    v = np.argmin(labels)
    print(v, labels[v])
    if labels[v] > 0: break
    c += 1
    labels[v] = c
    for e in adj_list[v]:
        visit(v, e, c)
    print(labels)
counts = np.bincount(labels)
for i in range(n-1):
    j = int(mst_s[i, 1] == 0)
    mst_s[i, j] = counts[mst_labels[i]]-mst_s[i, 1-j]
min_mst_s = np.min(mst_s, axis=1)
op = np.argsort(mst_w)[::-1]
#(mst_w[op])[min_mst_s[op]>min_cluster_size]
which_cut = op[np.nonzero(min_mst_s[op]>min_cluster_size)]
print(which_cut)
#
plt.clf()
#
ax1 = plt.subplot(2, 2, 1)
ax1.plot(mst_w[op])
ax2 = ax1.twinx()
ax2.plot(np.arange(n-1), min_mst_s[op], c='orange', alpha=0.3)
for i in which_cut:
    plt.text(op[i], min_mst_s[i], i, ha='center')
#
plt.subplot(2, 2, (2, 4))
genieclust.plots.plot_scatter(X, labels=labels)
genieclust.plots.plot_segments(mst_e, X, style="b-", alpha=0.1)
genieclust.plots.plot_segments(mst_e[min_mst_s>min_cluster_size,:], X, style="r-")
genieclust.plots.plot_segments(mst_e[mst_labels<0,:], X, style="w-")
plt.axis("equal")
for i in range(n-1):
    plt.text(
        (X[mst_e[i,0],0]+X[mst_e[i,1],0])/2,
        (X[mst_e[i,0],1]+X[mst_e[i,1],1])/2,
        "%d (%d)" % (i, min(mst_s[i, 0], mst_s[i, 1]))
    )
#
ax1 = plt.subplot(2, 2, 3)
ax2 = ax1.twinx()
last = 0
for i in range(1, c+1):
    _w = mst_w[mst_labels==i]
    _o = np.argsort(_w)[::-1]
    ax1.plot(
        np.arange(last, last+counts[i]-1),
        _w[_o],
        color=genieclust.plots.col[i]
    )
    ax2.plot(
        np.arange(last, last+counts[i]-1),
        min_mst_s[mst_labels==i][_o],
        color=genieclust.plots.col[i],
        alpha=0.2
    )
    last += counts[i] - 1


# DBSCAN is non-adaptive - cannot detect clusters of different densities well
plt.violinplot([ mst_w[mst_labels==i] for i in range(1, c+1) ])
