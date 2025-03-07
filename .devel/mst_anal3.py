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

# split to too many clusters, then merge based on e-radius or NNs


plt.clf()
X, y_true, n_clusters, skiplist, example = mst_examples.get_example(3, data_path)

n_clusters = max(y_true)

L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters*2)
#L = lumbermark.Lumbermark(n_clusters=n_clusters, verbose=False, n_neighbors=5, cluster_size_factor=0.2, outlier_factor=1.5, noise_cluster=True)

y_pred = L.fit_predict(X, mst_skiplist=skiplist)  # TODO: 0-based -> 1-based!!!

mst_examples.plot_mst_2d(L, mst_draw_edge_labels=True)
npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
nca = GNCA(y_true[y_true>0], y_pred[y_true>0])

plt.title("%s NA=%.2f NCA=%.2f" % (example, npa, nca))
plt.tight_layout()


n_neighbors = 10
kd = scipy.spatial.KDTree(X)
nn_w, nn_a = kd.query(X, n_neighbors+1)
nn_w = np.array(nn_w)[:, 1:]  # exclude self
nn_a = np.array(nn_a)[:, 1:]

mst_labels = L._mst_labels
mst_w = L._mst_w



while True:
    k = np.max(y_pred)

    agg_dists, _f = mst_examples.aggregate(mst_w, mst_labels, lambda x: np.mean(x)+2*np.std(x))
    agg_dists = agg_dists[_f>0]
    print(agg_dists)

    if False:
        mst_a = kd.query_ball_point(X, agg_dists[y_pred-1])  # includes self
        n = X.shape[0]
        k = np.max(y_pred)
        C = np.zeros((k, k))
        for i in range(n):
            for v in mst_a[i]:
                C[y_pred[i]-1, y_pred[v]-1] += 1
    else:
        C = np.zeros((k, k))
        for i in range(n_neighbors):
            C += genieclust.compare_partitions.confusion_matrix(y_pred[y_pred>0], y_pred[nn_a[y_pred>0, i]])
        #C /= k

        n = X.shape[0]
        for i in range(n_neighbors):
            genieclust.plots.plot_segments(np.c_[np.arange(n), nn_a[:, i]], X, color="gray", linestyle="--", alpha=0.2)


    C = (C+C.T)/2
    print(np.round(C, 2))

    CU = np.triu(C, 1)
    im, jm = np.unravel_index(np.argmax(CU, keepdims=True), CU.shape)
    im, jm = im[0], jm[0]
    print(im, jm, CU[im, jm])
    assert CU[im, jm] > 0

    stop()

    mst_labels[mst_labels == jm+1] = im+1
    mst_labels[mst_labels == k] = jm+1

    y_pred[y_pred == jm+1] = im+1
    y_pred[y_pred == k] = jm+1

    plt.clf()
    genieclust.plots.plot_scatter(X, labels=y_pred)
    plt.axis("equal")
    plt.show()

    k -= 1
    if k == n_clusters: break




