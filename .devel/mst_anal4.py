"""
2025-04-17: Test Lumbermark
Updated 2025-07-28 (genieclust 1.2.0, Lumbermark+quitefastmst)
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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os.path
import scipy.spatial
import sys
import scipy.spatial.distance

from importlib import reload

import genieclust
genieclust = reload(genieclust)
import clustbench

import lumbermark
lumbermark = reload(lumbermark)

import mst_examples
mst_examples = reload(mst_examples)

sys.setrecursionlimit(100000)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# np.random.seed(123)
# X = np.random.randn(100, 2)
# mst_w, mst_e = genieclust.internal.mst_from_distance(X, "euclidean")
# n = X.shape[0]
# n_clusters = 3
# min_cluster_size = 10
# min_cluster_factor = 0.25
# skip_leaves = True
# l = genieclust.internal.lumbermark_from_mst(mst_w, mst_e, n_clusters, min_cluster_size, min_cluster_factor, skip_leaves)
#
# print(l)
#
# y_pred = l["labels"]+1
# is_noise = l["is_noise"]
# genieclust.plots.plot_segments(mst_e, X, alpha=0.2)
# genieclust.plots.plot_scatter(X[~is_noise,:], labels=y_pred[~is_noise]-1)
# genieclust.plots.plot_scatter(X, labels=y_pred-1, alpha=0.2)
# plt.axis("equal")
# plt.show()
# stop()
# # TODO...




# np.random.seed(123)
# X = np.random.rand(100000, 8)
#
# import hdbscan
# import timeit
# timeit.timeit("hdbscan.hdbscan_._hdbscan_boruvka_kdtree(X, 1, gen_min_span_tree=False)", globals=dict(hdbscan=hdbscan, X=X), number=3)
#
# timeit.timeit("genieclust.Genie(M=1).fit_predict(X)", globals=dict(genieclust=genieclust, X=X), number=3)


data_path = os.path.join("~", "Projects", "clustering-data-v1")

plt.rcParams.update({  # further graphical parameters
    "font.size":         5,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Alegreya Sans", "Alegreya"],
    "figure.autolayout": True,
    "figure.dpi":        150,
    "figure.figsize":    (10,10),
})

plt.clf()
_i = 0
n_examples = 47
ncas = []
_nrow, _ncol = int(np.floor(np.sqrt(n_examples))), int(np.ceil(np.sqrt(n_examples)))
if _nrow*_ncol < n_examples:
    _ncol += 1
    assert _nrow*_ncol >= n_examples
total_time = 0.0
for ex in range(n_examples):
    _i += 1
    plt.subplot(_nrow, _ncol, _i)
    X, y_true, n_clusters, example = mst_examples.get_example(ex, data_path)

    # if X.shape[1] != 2:
        # continue

    t0 = time.time()
    algo = ["Lumbermark", "Genie"][0]
    eps = 0.00000011920928955078125
    if algo == "Lumbermark":
        algo_params = dict(M=6, min_cluster_factor=0.15, postprocess="all", quitefastmst_params=dict(mutreach_adj=-1-eps))
        L = lumbermark.Lumbermark(n_clusters=n_clusters, **algo_params)
        y_pred = L.fit_predict(X)+1  # 0-based -> 1-based
    elif algo == "Genie":
        algo_params = dict(M=6, gini_threshold=0.4, postprocess="all", quitefastmst_params=dict(mutreach_adj=-1-eps))
        L = genieclust.Genie(n_clusters=n_clusters, **algo_params)
        y_pred = L.fit_predict(X)+1  # 0-based -> 1-based
    else:
        stop("incorrect 'algo'")
    total_time += time.time()-t0

    is_noise = L._is_noise  # np.repeat(False, X.shape[0])#
    tree_e = L._tree_e
    tree_w = L._tree_w
    tree_skiplist = L._tree_cutlist

    # else:
    #     # Can't force hdbscan to always output a given number of clusters
    #     if algo == 1:
    #         s1, s2 = 2, int(X.shape[0])
    #
    #         while 1 < s1 <= s2:
    #             s = int((s1+s2)/2)
    #             h = hdbscan.HDBSCAN(min_cluster_size=s, min_samples=7)  # min_samples = mutreach dist param
    #
    #             y_pred = h.fit_predict(X) + 1
    #             k = max(y_pred)
    #
    #             print(s, k, n_clusters)
    #
    #             if k == n_clusters: break
    #             elif k < n_clusters: s2 = s-1
    #             else: s1 = s+1
    #     else:
    #         s1, s2 = 0, 100
    #
    #         maxit = 10
    #         while maxit > 0:
    #             maxit -= 1
    #             s = (s1+s2)/2
    #             y_pred, _ = hdbscan.robust_single_linkage(X, s, k=5)
    #
    #
    #             y_pred = y_pred + 1
    #             k = max(y_pred)
    #
    #             print(s, k, n_clusters)
    #
    #             if k == n_clusters: break
    #             elif k < n_clusters: s2 = s
    #             else: s1 = s
    #
    #    if k != n_clusters:
    #        plt.text(X[:,0].mean(), X[:,1].mean(), "!!!!")
    #
    #    is_noise = (y_pred == 0)
    #
    #    assert sum(~is_noise) > 0
    #
    #    if sum(is_noise) > 0:
    #        kd2 = scipy.spatial.KDTree(X[~is_noise,:])
    #        nn_w2, nn_a2 = kd2.query(X[is_noise, :], 1)  # find closest points
    #        y_pred[is_noise] = y_pred[~is_noise][nn_a2]
    #
    #    tree_e = None
    #    tree_w = None
    #    tree_skiplist = None



    if is_noise is not None:
        genieclust.plots.plot_scatter(X[~is_noise,:2], labels=y_pred[~is_noise]-1)
    else:
        genieclust.plots.plot_scatter(X[:,:2], labels=y_pred[:]-1)

    genieclust.plots.plot_scatter(X[:,:2], labels=y_pred-1, alpha=0.2)

    # if tree_e is not None:
        # genieclust.plots.plot_segments(tree_e, X, alpha=0.2)

    # if tree_skiplist is not None:
        # genieclust.plots.plot_segments(tree_e[tree_skiplist, :], X, color="yellow", alpha=0.2)

    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])

    # npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
    nca = genieclust.compare_partitions.normalized_clustering_accuracy(y_true[y_true>0], y_pred[y_true>0])
    if nca < 0: stop()

    #s1, s2 = treelhouette.treelhouette_score(L)
    plt.title("%s NCA=%.2f" % (example, nca), loc="left")
    ncas.append(nca)

plt.tight_layout()
plt.show()

ncas = np.array(ncas)
print("%s %r\ntime: %.3f" % (algo, algo_params, total_time))
print("NCA: %10s %10s %10s %10s %10s" % ("Min", "Median", "Mean", "#<.8", "#≥.95"))
print("NCA: %10.3f %10.3f %10.3f %10d %10d" % (np.min(ncas), np.median(ncas), np.mean(ncas), np.sum(ncas<0.8), np.sum(ncas>0.95)))
