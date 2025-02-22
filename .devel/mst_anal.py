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


data_path = os.path.join("~", "Projects", "clustering-data-v1")
np.random.seed(123)


mst_draw_edge_labels = False


examples = [
    ["fcps", "engytime", [], 2],
    ["wut", "x3", [], 3],
    ["graves", "dense", [], 2],
    ["sipu", "compound", [], 5],  # :/
    ["wut", "z2", [], 5],
    ["graves", "parabolic", [], 2],
    ["new", "blobs4b", [0, 1, 3], 4],
    ["new", "blobs4a", [0, 1, 2], 4],
    ["fcps", "twodiamonds", [3], 2],
    ["sipu", "unbalance", [2,0,3,1,4,5,7], 8],
    ["sipu", "s1", [2, 1, 3, 4, 6, 15, 46, 54, 70, 84, 85, 94, 9, 87]],
    ["sipu", "pathbased", [4,5,7,19,27]],
    ["other", "hdbscan", []],
    ["fcps", "wingnut", [0]],
    ["other", "hdbscan", [256,260,273,404,332]],
    ["wut", "labirynth", [0,1,3,11, 18,72,77,120]],
    ["wut", "labirynth", []],
    ["sipu", "compound", [0, 1, 54, 153]],
    ["sipu", "aggregation", []],
    ["wut", "z2", [1, 3, 5, 6]],
    ["sipu", "aggregation", [0, 1, 2, 536, 3, 24]],
    ["sipu", "pathbased", []],
    ["sipu", "spiral", []],
]


example = examples[0]
battery = example[0]
dataset = example[1]
skiplist = example[2]
n_clusters = example[3]

if battery != "new":
    b = clustbench.load_dataset(battery, dataset, path=data_path)
    X, labels = b.data, b.labels[0]
else:
    from sklearn.datasets import make_blobs
    if dataset == "blobs4a":
        X, labels = make_blobs(
            n_samples=[1000, 1000, 100, 100],
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
        labels = np.r_[labels+1, np.repeat(0, xapp.shape[0])]
    if dataset == "blobs4b":
        X, labels = make_blobs(
            n_samples=[1000, 1000, 100, 100],
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
        labels = np.r_[labels+1, np.repeat(0, xapp.shape[0])]


n = X.shape[0]

SKIP = np.iinfo(int).min  # must be negative
UNSET = -1 # must be negative
NOISE = 0  # must be 0


min_cluster_size = max(10, 0.1*n/n_clusters)  # this should be a function of the desired number of clusters
max_noise_size = max(5, 0.01*n/n_clusters)
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
adj_list = np.array(adj_list, dtype="object")



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
# or may be considered as an outlier after a cluster breaks down to smaller ones
noise_k = -1  # which nearest neighbour do we take into account?
Q13 = np.array([np.percentile(nn_w[labels==j+1, noise_k], [25, 75]) for j in range(c)])
bnd = (Q13[:, 1]+1.5*(Q13[:, 1]-Q13[:, 0]))[labels-1]
is_outlier = (nn_w[:, noise_k]>bnd)



# cut out all "small" branches (<max_noise_size) that consist of outliers,
# starting from leaves

# leaves are vertices of degree 1 (in the forest with the SKIP edges removed!)
leaves = mst_e[:,::-1][mst_s == 1]
is_leaf = np.repeat(False, n)
is_leaf[leaves] = True


# a noise edge is incident to a noise point,
# provided that its removal leads to too small a cluster
outlier_edges  = (is_outlier[mst_e[:,0]] | is_outlier[mst_e[:,1]])
outlier_edges &= (min_mst_s <= max_noise_size)
outlier_edges &= (mst_labels != SKIP)


def mark_noise(v):
    if not is_outlier[v]: return

    e = -1
    for e2 in adj_list[v]:
        if mst_labels[e2] > 0:
            if e < 0:
                e = e2  # first non-NOISE, non-SKIP edge
            else:
                return  # we want a vertex that's incident of only one such edge

    if e == -1:
        return

    if min_mst_s[e] > max_noise_size:
        return

    labels[v] = NOISE
    mst_labels[e] = NOISE
    print(v, e, mst_e[e,:])

    iv = int(mst_e[e, 1] == v)
    v = mst_e[e, 1-iv]
    mark_noise(v)  # tail call


for v in np.flatnonzero(is_leaf):
    mark_noise(v)


# # All descendants of non-outliers in the direction of noise edges should be marked as noise
# # TODO This is incorrect
# noise_edges  = (is_outlier[mst_e[:,0]] | is_outlier[mst_e[:,1]])
# noise_edges &= (min_mst_s <= max_noise_size)
# noise_edges &= (mst_labels != SKIP)
# for i in [0, 1]:
#     wh = np.flatnonzero(noise_edges & (~is_outlier[mst_e[:, i]]))
#     for e in wh:
#         if mst_labels[e] == SKIP or mst_labels[e] == NOISE: continue
#
#         v = mst_e[e, i]
#         assert not is_outlier[v]
#
#         if mst_s[e, i] <= max_noise_size:
#             visit(v, e, NOISE)



mst_s[:,:] = 0
labels[labels != NOISE] = UNSET
mst_labels[(mst_labels != SKIP) & (mst_labels != NOISE)] = UNSET
c, counts, min_mst_s = mark_clusters()



#
# a cut edge is incident to a noise point, its removal leads to a new cluster of "considerable" size
# cut_k = -1
cut_edges  = (is_outlier[mst_e[:,0]] | is_outlier[mst_e[:,1]])
cut_edges &= (min_mst_s >= min_cluster_size)
cut_edges &= (mst_labels != SKIP)
which_cut_edges = np.flatnonzero(cut_edges)[:5]
print("which_cut_edges=%s" % (which_cut_edges, ))



# Q13 = np.array([np.percentile(mst_w[mst_labels==j+1], [25, 75]) for j in range(c)])
# bnd = np.r_[np.nan, (Q13[:, 1]+1.5*(Q13[:, 1]-Q13[:, 0]))]
# outlier_edges = (mst_w>bnd)  # x > np.nan == False
# Q13 = np.array([np.percentile(min_mst_s[mst_labels==j+1], [25, 75]) for j in range(c)])
# bnd = np.r_[np.nan, (Q13[:, 1]+1.5*(Q13[:, 1]-Q13[:, 0]))]
bnd = np.r_[np.nan, np.maximum(min_cluster_size, 0.1*counts[1:])]
bnd = bnd[np.where(mst_labels > 0, mst_labels, 0)]
outlier_edges = (min_mst_s>bnd)  # x > np.nan == False
outlier_edges &= (min_mst_s >= min_cluster_size)
outlier_edges &= (mst_labels != SKIP)
outlier_edges &= ~cut_edges
which_outlier_edges = np.flatnonzero(outlier_edges)[:5]
print("which_outlier_edges=%s" % (which_outlier_edges, ))

_x = []
for _w in [which_cut_edges, which_outlier_edges]:
    _x.append(pd.DataFrame(np.c_[_w, mst_w[_w], np.sort(mst_s, axis=1)[_w,:], mst_labels[_w]], columns=["i", "w", "s0", "s1", "label"]))
print(pd.concat(_x, keys=["cut", "outlier"]))


candidate_edge = which_cut_edges[0] if len(which_cut_edges) else which_outlier_edges[0]
print("candidate_edge=%d" % candidate_edge)

#
plt.clf()
#
# MST
plt.subplot(3, 2, (2, 6))
# nn_threshold = np.mean(nn_w[:, -1])
# nn_w[:, -1] > nn_threshold
# labels[nn_w[:, -1] > nn_threshold] = 0
genieclust.plots.plot_scatter(X, labels=labels-1)
if mst_draw_edge_labels:
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
genieclust.plots.plot_segments(mst_e[[candidate_edge],:], X, color="blue", linestyle="--", alpha=0.7, linewidth=3)
plt.axis("equal")
#
#
# Edge weights + cluster sizes
ax1 = plt.subplot(3, 2, 1)
op = np.flatnonzero(mst_labels>0)
ax1.plot(mst_w[op], color='blue')
ax2 = ax1.twinx()
ax2.plot(np.arange(len(op)), min_mst_s[op], c='blue', alpha=0.2)
idx = np.sum(op < candidate_edge)
plt.text(idx, min_mst_s[candidate_edge], candidate_edge, ha='center', va='bottom', color=genieclust.plots.col[mst_labels[candidate_edge]-1])
# ce = (np.arange(1, n)*cut_edges)[op]   # 1-shift
# for j in np.flatnonzero(ce):
    # idx = ce[j]-1 # unshift
    # plt.text(j, min_mst_s[idx], idx, ha='center', color=genieclust.plots.col[mst_labels[idx]-1])
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
    if mst_labels[candidate_edge] == i:
        idx = np.sum(op < candidate_edge)
        plt.text(last+idx, min_mst_s[candidate_edge], candidate_edge, ha='center', va='bottom', color=genieclust.plots.col[mst_labels[candidate_edge]-1])
    # ce = (np.arange(1, n)*cut_edges)[op]   # 1-shift
    # for j in np.flatnonzero(ce):
        # idx = ce[j]-1 # unshift
        # plt.text(j, min_mst_s[idx], idx, ha='center', color=genieclust.plots.col[mst_labels[idx]-1])

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
print("treelhouette_score=%.3f" % treelhouette_score)
plt.axhline(treelhouette_score, color='gray')



# DBSCAN is non-adaptive - cannot detect clusters of different densities well
#plt.violinplot([ mst_w[mst_labels==i] for i in range(1, c+1) ])


# r = np.median(mst_w[mst_labels == 3])
# def visit2(v, e, r):

