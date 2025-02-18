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
    ["other", "hdbscan", [256,260,273,404,332]],
    ["other", "hdbscan", []],
    ["fcps", "wingnut", []],
    ["sipu", "pathbased", [4,5,7,19,27]],
    ["wut", "labirynth", [0,1,3,11, 18,72,77,120]],
    ["wut", "labirynth", []],
    ["sipu", "compound", [0, 1, 54, 153]],
    ["sipu", "aggregation", []],
    ["fcps", "engytime", []],
    ["wut", "z2", [1, 3, 5, 6]],
    ["sipu", "aggregation", [0, 1, 2, 536, 3, 24]],
    ["wut", "z2", []],
    ["sipu", "compound", []],
    ["sipu", "pathbased", []],
    ["graves", "parabolic", []],
    ["sipu", "spiral", []],
]

example = examples[0]
data_path = os.path.join("~", "Projects", "clustering-data-v1")
np.random.seed(123)
b = clustbench.load_dataset(example[0], example[1], path=data_path)
X, labels = b.data, b.labels[0]
skiplist = example[2]

n = X.shape[0]

SKIP = np.iinfo(int).min  # must be negative
UNSET = -1 # must be negative
NOISE = 0  # must be 0


min_cluster_size = n/20  # this should be a function of the desired number of clusters
max_noise_size = n/100
nn_k = 5
# gini_threshold = 0.5

mst = genieclust.internal.mst_from_distance(X, "euclidean")
mst_w, mst_e = mst
op = np.argsort(mst_w)[::-1]
mst_w = mst_w[op]  # sorted decreasingly
mst_e = mst_e[op, :]

adj_list = [ [] for i in range(n) ]
for i in range(n-1):
    adj_list[mst_e[i, 0]].append(i)
    adj_list[mst_e[i, 1]].append(i)
for i in range(n):
    adj_list[i] = np.array(adj_list[i])




kd = scipy.spatial.KDTree(X)
nn_w, nn_e = kd.query(X, nn_k+1)
nn_w = np.array(nn_w)[:, 1:]  # exclude self
nn_e = np.array(nn_e)[:, 1:]




def visit(v, e, c):  # v->w  where mst_e[e,:]={v,w}
    if mst_labels[e] == SKIP or (c != NOISE and mst_labels[e] == NOISE):
        return 0
    iv = int(mst_e[e, 1] == v)
    w = mst_e[e, 1-iv]
    tot = 1
    for e2 in adj_list[w]:
        if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
            tot += visit(w, e2, c)
    mst_s[e, iv] = tot if c != NOISE else -1
    mst_s[e, 1-iv] = 0

    labels[w] = c
    mst_labels[e] = c

    return tot


mst_s = np.zeros((n-1, 2), dtype=int)
labels = np.repeat(UNSET, n)
mst_labels = np.repeat(UNSET, n-1)
for s in skiplist:
    mst_labels[s] = SKIP  # skiplist



def mark_clusters():
    c = 0
    for v in range(n):
        if labels[v] == UNSET:
            c += 1
            labels[v] = c
            for e in adj_list[v]:
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


c, counts, min_mst_s = mark_clusters()


# a point is an outlier if its k-th nearest neighbour is just too far away -
# - relative to the "typical" distances to k-nearest neighbours within the point's cluster
# (each cluster can be of different density, so we take this into account)
# or may be considered outliers after a cluster breaks down to smaller ones
noise_k = -1  # which nearest neighbour do we take into account?
Q13 = np.array([np.percentile(nn_w[labels==j+1, noise_k], [25, 75]) for j in range(c)])
bnd = (Q13[:, 1]+1.5*(Q13[:, 1]-Q13[:, 0]))[labels-1]
outliers = (nn_w[:, noise_k]>bnd)

# a noise edge is incident to a noise point,
# provided that its removal leads to too small a cluster
noise_edges  = (outliers[mst_e[:,0]] | outliers[mst_e[:,1]])
noise_edges &= (min_mst_s <= max_noise_size)

# now all descendants of non-outliers in the direction of noise edges
# should be marked as noise


for i in [0, 1]:
    wh = np.flatnonzero(noise_edges & (~outliers[mst_e[:, i]]))
    for e in wh:
        if mst_labels[e] == SKIP or mst_labels[e] == NOISE: continue

        v = mst_e[e, i]
        assert not outliers[v]

        if mst_s[e, i] <= max_noise_size:
            visit(v, e, NOISE)

mst_s[:,:] = 0
labels[labels != NOISE] = UNSET
mst_labels[(mst_labels != SKIP) & (mst_labels != NOISE)] = UNSET
c, counts, min_mst_s = mark_clusters()


#
# a cut edge is incident to a noise point, its removal leads to a new cluster of "considerable" size
# cut_k = -1
cut_edges  = (outliers[mst_e[:,0]] | outliers[mst_e[:,1]])
cut_edges &= (min_mst_s >= min_cluster_size)
which_cut_edges = np.flatnonzero(cut_edges)
print("which_cut_edges=%s" % (which_cut_edges, ))


x  = pd.DataFrame(np.c_[np.arange(n-1), mst_w, np.sort(mst_s, axis=1), mst_labels][mst_labels>0, :], columns=["i", "w", "s0", "s1", "label"])
print(x.loc[x.s0 >= min_cluster_size, :].head(10))



# TODO mark candidate cut edges based on cut sizes
# which_cut_edges = np.flatnonzero(min_mst_s[op]>min_cluster_size)  # TODO.....

#
plt.clf()
#
# MST
plt.subplot(3, 2, (2, 6))
# nn_threshold = np.mean(nn_w[:, -1])
# nn_w[:, -1] > nn_threshold
# labels[nn_w[:, -1] > nn_threshold] = 0
genieclust.plots.plot_scatter(X, labels=labels-1)
plt.axis("equal")
for i in range(n-1):
    plt.text(
        (X[mst_e[i,0],0]+X[mst_e[i,1],0])/2,
        (X[mst_e[i,0],1]+X[mst_e[i,1],1])/2,
        "%d (%d-%d)" % (i, mst_s[i, 0], mst_s[i, 1]),
        color="gray" if mst_labels[i] == SKIP else genieclust.plots.col[mst_labels[i]-1],
        va='top'
    )
for i in range(c+1):
    genieclust.plots.plot_segments(mst_e[mst_labels == i, :], X, color=genieclust.plots.col[i-1],
        alpha=0.2, linestyle="-" if i>0 else ":")
genieclust.plots.plot_segments(mst_e[mst_labels<0,:], X, color="yellow", linestyle="-", linewidth=3)
genieclust.plots.plot_segments(mst_e[cut_edges,:], X, color="blue", linestyle="--", alpha=0.7, linewidth=3)
#
#
# Edge weights + cluster sizes
ax1 = plt.subplot(3, 2, 1)
op = np.flatnonzero(mst_labels>0)
ax1.plot(mst_w[op], color='blue')
ax2 = ax1.twinx()
ax2.plot(np.arange(len(op)), min_mst_s[op], c='blue', alpha=0.2)
ce = (np.arange(1, n)*cut_edges)[op]   # 1-shift
for j in np.flatnonzero(ce):  # TODO this is wrong - numbering...
    idx = ce[j]-1 # unshift
    plt.text(j, min_mst_s[idx], idx, ha='center', color=genieclust.plots.col[mst_labels[idx]-1])
#
# MST edges per cluster
ax1 = plt.subplot(3, 2, 3)
ax2 = ax1.twinx()
last = 0
for i in range(1, c+1):
    op = np.flatnonzero(mst_labels == i)
    len_op = len(op)
    ax1.plot(np.arange(last, last+len_op), mst_w[op],
        color=genieclust.plots.col[i-1])
    ax2.plot(np.arange(last, last+len_op), min_mst_s[op], c=genieclust.plots.col[i-1], alpha=0.2)
    ce = (np.arange(1, n)*cut_edges)[op]   # 1-shift
    for j in np.flatnonzero(ce):  # TODO this is wrong - numbering...
        idx = ce[j]-1 # unshift
        plt.text(j, min_mst_s[idx], idx, ha='center', color=genieclust.plots.col[mst_labels[idx]-1])

    last += len_op
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



# DBSCAN is non-adaptive - cannot detect clusters of different densities well
#plt.violinplot([ mst_w[mst_labels==i] for i in range(1, c+1) ])


# r = np.median(mst_w[mst_labels == 3])
# def visit2(v, e, r):

