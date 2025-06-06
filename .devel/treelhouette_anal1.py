"""
...
Treelhouettes 2025-05-15 presentation 1
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import genieclust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import os.path
import scipy.spatial
import sys
from importlib import reload
import test_algo_202505_3
test_algo_202505_3 = reload(test_algo_202505_3)
import treelhouette
treelhouette = reload(treelhouette)

import mst_examples
mst_examples = reload(mst_examples)
sys.setrecursionlimit(100000)

#from generalized_normalized_clustering_accuracy import generalized_normalized_clustering_accuracy as GNCA
#from generalized_normalized_clustering_accuracy import generalized_normalized_pivoted_accuracy as GNPA

data_path = os.path.join("~", "Projects", "clustering-data-v1")



plt.clf()
X, y_true, n_clusters, skiplist, example = mst_examples.get_example(4, data_path)

n_clusters = max(y_true)

# L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters, M=1, min_cluster_factor=0.25, skip_leaves=False, min_cluster_size=10)
L = test_algo_202505_3.Lumbermark(n_clusters=n_clusters, M=1, min_cluster_size=15)
# L = lumbermark2.Lumbermark2(n_clusters=n_clusters, verbose=False, n_neighbors=5, outlier_factor=1.5, noise_cluster=True)

y_pred = L.fit_predict(X)+1


# mst_examples.plot_mst_2d(L, mst_draw_edge_labels=True)
# npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
# nca = GNCA(y_true[y_true>0], y_pred[y_true>0])
# plt.title("%s NA=%.2f NCA=%.2f" % (example, npa, nca))
# plt.tight_layout()
#
# stop()



mst_draw_edge_labels = False
mst_e = L._tree_e
mst_w = L._tree_w
# mst_s = L._mst_s
# min_mst_s = np.min(mst_s, axis=1)
mst_labels = L._tree_labels
n = X.shape[0]
skiplist = L._tree_cutlist
cutting = None
mst_internodes = L.__dict__.get("_mst_internodes", [])

# y_pred[mst_e[(mst_s[:,0] <= 1) & (mst_labels > 0), 1]] = 0
# y_pred[mst_e[(mst_s[:,1] <= 1) & (mst_labels > 0), 0]] = 0
# mst_labels[   (min_mst_s <= 1) & (mst_labels > 0)] = 0


# nn_dist, nn_ind = genieclust.internal.knn_from_distance(X, k=M-1, metric="euclidean")
# nn_tick = np.zeros(n, dtype=int)
# for v in nn_ind.ravel():
#     nn_tick[v] += 1
# plt.clf()
# plt.axis("equal")
# genieclust.plots.plot_scatter(X[:,:2], labels=nn_tick>3)
# genieclust.plots.plot_segments(mst_e, X)
# #genieclust.plots.plot_scatter(X[:,:2], labels=y_pred-1)


plt.rcParams.update({  # further graphical parameters
    "font.size":         15,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Alegreya Sans", "Alegreya"],
    "figure.autolayout": True,
    "figure.dpi":        120,
    "figure.figsize":    (7, 4),
})

#
plt.clf()
#
# MST
plt.subplot(1, 2, 1)
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

# treelhouette
plt.subplot(1, 2, 2)

#
cluster_distances = treelhouette.get_intercluster_distances(L)
print(np.round(cluster_distances, 2))
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
plt.barh(np.arange(len(s))[::-1]+1, s[o1][o2], height=1.0, color=np.array(genieclust.plots.col)[l[o1]-1][o2])
# plt.xlim(-1, 1)
treelhouette_score = np.mean(s)
weighted_treelhouette_score = np.mean(mst_examples.aggregate(s, mst_labels[mst_labels>0], np.mean)[0])
print("treelhouette_score=%.3f, weighted_treelhouette_score=%.3f" % (treelhouette_score, weighted_treelhouette_score))
plt.axvline(treelhouette_score, color='gray')
# plt.axhline(weighted_treelhouette_score, color='lightgray')
plt.title("treelhouette_score=%.3f" % (treelhouette_score,))
plt.tight_layout()
plt.savefig(example.replace("/", "_")+".pdf")

# DBSCAN is non-adaptive - cannot detect clusters of different densities well
#plt.violinplot([ mst_w[mst_labels==i] for i in range(1, c+1) ])


# r = np.median(mst_w[mst_labels == 3])
# def visit2(v, e, r):

