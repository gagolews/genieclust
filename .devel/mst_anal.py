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
    ["sipu", "jain", [], 2],
    ["sipu", "flame", [], 2],
    ["fcps", "twodiamonds", [], 2],
    ["other", "chameleon_t8_8k", [], 8],
    ["other", "chameleon_t7_10k", [], 9],
    ["other", "chameleon_t5_8k", [], 6],
    ["other", "chameleon_t4_8k", [], 6],
    ["sipu", "spiral", [], 3],
    ["wut", "labirynth", [], 6],
    ["wut", "z3", [], 4],
    ["sipu", "compound", [], 5],
    ["other", "hdbscan", [], 6],
    ["sipu", "pathbased", [], 3],  # :/
    ["sipu", "s1", [], 15],
    ["sipu", "unbalance", [], 8],
    ["wut", "z2", [], 5],
    ["graves", "dense", [], 2],
    ["fcps", "engytime", [], 2],
    ["fcps", "wingnut", [], 2],
    ["wut", "x3", [], 3],
    ["graves", "parabolic", [], 2],
    ["sipu", "aggregation", [], 7],
    ["new", "blobs4a", [], 4],
    ["new", "blobs4b", [], 4],
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
            n_samples=[900, 900, 100, 100],
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
            n_samples=[900, 900, 100, 100],
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


def lumbermark(
    X, n_clusters, min_cluster_size=None, max_twig_size=None, noise_cluster=False,
    n_neighbors=5, cut_internodes=True, skiplist=[]
):
    """
    set n_neighbors to 0 to disable noise point detection

    twig = any small, leafless branch of a woody plant
    internode =  the part of a plant stem between two nodes
    limb = one of the larger branches of a tree
    """
    X = np.ascontiguousarray(X)
    n = X.shape[0]

    assert n_clusters > 0
    assert n_neighbors >= 0

    if n_neighbors == 0:
        cut_internodes = False
        mark_noise = False

    if min_cluster_size is None:
        min_cluster_size = max(15, 0.1*n/n_clusters)

    if max_twig_size is None:
        max_twig_size = max(5, 0.01*n/n_clusters)


    SKIP = np.iinfo(int).min  # must be negative
    UNSET = -1 # must be negative
    NOISE = 0  # must be 0


    if n_neighbors > 0:
        kd = scipy.spatial.KDTree(X)
        nn_w, nn_a = kd.query(X, n_neighbors+1)
        nn_w = np.array(nn_w)[:, 1:]  # exclude self
        nn_a = np.array(nn_a)[:, 1:]


    mst_w, mst_e = genieclust.internal.mst_from_distance(X, "euclidean")
    _o = np.argsort(mst_w)[::-1]
    mst_w = mst_w[_o]  # weights sorted decreasingly
    mst_e = mst_e[_o, :]


    # TODO: store as two arrays: indices, indptr (compressed sparse row format)
    mst_a = [ [] for i in range(n) ]
    for i in range(n-1):
        mst_a[mst_e[i, 0]].append(i)
        mst_a[mst_e[i, 1]].append(i)
    for i in range(n):
        mst_a[i] = np.array(mst_a[i])
    mst_a = np.array(mst_a, dtype="object")



    def visit(v, e, c):  # v->w  where mst_e[e,:]={v,w}
        if mst_labels[e] == SKIP or (c != NOISE and mst_labels[e] == NOISE):
            return 0
        iv = int(mst_e[e, 1] == v)
        w = mst_e[e, 1-iv]
        tot = 1
        for e2 in mst_a[w]:
            if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
                tot += visit(w, e2, c)
        mst_s[e, iv] = tot if c != NOISE else -1
        mst_s[e, 1-iv] = 0

        labels[w] = c
        mst_labels[e] = c

        return tot


    def mark_clusters():
        c = 0
        for v in range(n):
            if labels[v] == UNSET:
                c += 1
                labels[v] = c
                for e in mst_a[v]:
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



    def mark_noise(v):
        if not is_outlier[v]: return

        e = -1
        for e2 in mst_a[v]:
            if mst_labels[e2] > 0:
                if e < 0:
                    e = e2  # first non-NOISE, non-SKIP edge
                else:
                    return  # we want a vertex that's incident of only one such edge

        if e == -1:
            return

        if min_mst_s[e] > max_twig_size:
            return

        labels[v] = NOISE
        mst_labels[e] = NOISE
        # print(v, e, mst_e[e,:])

        iv = int(mst_e[e, 1] == v)
        v = mst_e[e, 1-iv]
        mark_noise(v)  # tail call


    while True:
        mst_s = np.zeros((n-1, 2), dtype=int)  # sizes: mst_s[e, 0] is the size of the cluster that appears when
        labels = np.repeat(UNSET, n)
        mst_labels = np.repeat(UNSET, n-1)
        for s in skiplist:
            mst_labels[s] = SKIP  # skiplist


        c, counts, min_mst_s = mark_clusters()

        if c >= n_clusters and not noise_cluster: return labels


        if n_neighbors == 0:
            is_outlier = np.repeat(False, n)
        else:
            # a point is an outlier if its k-th nearest neighbour is just too far away -
            # - relative to the "typical" distances to k-nearest neighbours within the point's cluster
            # (each cluster can be of different density, so we take this into account)
            # or may be considered as an outlier after a cluster breaks down to smaller ones
            outlier_k = -1  # which nearest neighbour do we take into account?
            Q13 = np.array([np.percentile(nn_w[labels==j+1, outlier_k], [25, 75]) for j in range(c)])
            bnd_outlier = np.r_[np.nan, (Q13[:, 1]+1.5*(Q13[:, 1]-Q13[:, 0]))]
            is_outlier = (nn_w[:, outlier_k] > bnd_outlier[labels])


            # cut out all twigs ("small" branches of size <= max_twig_size)
            # that solely consist of outliers, starting from leaves


            # leaves are vertices of degree 1 (in the forest with the SKIP edges removed!)
            leaves = mst_e[:,::-1][mst_s == 1]
            #is_leaf = np.repeat(False, n)
            #is_leaf[leaves] = True

            for v in leaves:  #np.flatnonzero(is_leaf):
                mark_noise(v)


            mst_s[:,:] = 0
            labels[labels != NOISE] = UNSET
            mst_labels[(mst_labels != SKIP) & (mst_labels != NOISE)] = UNSET
            c, counts, min_mst_s = mark_clusters()


        if c >= n_clusters: return labels


        if not cut_internodes:
            internodes = np.repeat(False, n-1)
        else:
            # an internode is an edge incident to an outlier, whose removal leads to a new cluster of "considerable" size
            internodes  = (is_outlier[mst_e[:,0]] | is_outlier[mst_e[:,1]])
            internodes &= (min_mst_s >= min_cluster_size)
            # internodes &= (min_mst_s >= min_cluster_size_frac*np.sum(mst_s, axis=1))
            internodes &= (mst_labels != SKIP)

        which_internodes = np.flatnonzero(internodes)
        #print("which_internodes=%s" % (which_internodes, ))


        limbs = (mst_labels > 0)
        limbs &= (min_mst_s >= min_cluster_size)
        # limbs &= (min_mst_s >= min_cluster_size_frac*np.sum(mst_s, axis=1))
        limbs &= (mst_labels != SKIP)
        limbs &= ~internodes

        which_limbs = np.flatnonzero(limbs)
        #print("which_limbs=%s" % (which_limbs, ))

        # _x = []
        # for _w in [which_internodes, which_limbs]:
        #     _x.append(pd.DataFrame(np.c_[_w, mst_w[_w], np.sort(mst_s, axis=1)[_w,:], mst_labels[_w]], columns=["i", "w", "s0", "s1", "label"]))
        # _x = pd.concat(_x, keys=["cut", "outlier"])#.reset_index(level=0)#.groupby(["level_0", "label"]).head(1)
        # print(_x)


        cutting = which_internodes[0] if len(which_internodes) > 0 else which_limbs[0]
        #print("cutting=%d" % cutting)

        skiplist.append(cutting)




labels = lumbermark(X, n_clusters)


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
if mst_draw_edge_labels:
    for i in range(n-1):
        plt.text(
            (X[mst_e[i,0],0]+X[mst_e[i,1],0])/2,
            (X[mst_e[i,0],1]+X[mst_e[i,1],1])/2,
            "%d (%d-%d)" % (i, mst_s[i, 0], mst_s[i, 1]),
            color="gray" if mst_labels[i] == SKIP else genieclust.plots.col[mst_labels[i]-1],
            va='top'
        )
for i in range(n_clusters+1):
    genieclust.plots.plot_segments(mst_e[mst_labels == i, :], X, color=genieclust.plots.col[i-1],
        alpha=0.2, linestyle="-" if i>0 else ":")
genieclust.plots.plot_segments(mst_e[mst_labels<0,:], X, color="yellow", linestyle="-", linewidth=3)
genieclust.plots.plot_segments(mst_e[[cutting],:], X, color="blue", linestyle="--", alpha=0.7, linewidth=3)
#
#
# Edge weights + cluster sizes
ax1 = plt.subplot(3, 2, 1)
op = np.flatnonzero(mst_labels>0)
ax1.plot(mst_w[op], color='blue')
ax2 = ax1.twinx()
ax2.plot(np.arange(len(op)), min_mst_s[op], c='blue', alpha=0.2)
idx = np.sum(op < cutting)
plt.text(idx, min_mst_s[cutting], cutting, ha='center', va='bottom', color=genieclust.plots.col[mst_labels[cutting]-1])
# ce = (np.arange(1, n)*internodes)[op]   # 1-shift
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
    if mst_labels[cutting] == i:
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
    # a[i] = np.min(mst_w[mst_a[i][mst_labels[mst_a[i]] == labels[i]]])
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

