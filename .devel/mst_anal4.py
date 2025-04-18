"""
2025-04-17: Test Lumbermark - Robust SL with noise point detection
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
import scipy.spatial.distance

from importlib import reload

import lumbermark
lumbermark = reload(lumbermark)

import robust_single_linkage
robust_single_linkage = reload(robust_single_linkage)

import mst_examples
mst_examples = reload(mst_examples)

sys.setrecursionlimit(100000)




data_path = os.path.join("~", "Projects", "clustering-data-v1")


plt.clf()
_i = 0
n_examples = 30
ncas = []
for ex in range(n_examples):
    _i += 1
    plt.subplot(int(np.floor(np.sqrt(n_examples))), int(np.ceil(np.sqrt(n_examples))), _i)
    X, y_true, n_clusters, skiplist, example = mst_examples.get_example(ex, data_path)

    n_clusters = max(y_true)

    #L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters, M=1, min_cluster_factor=0.1, skip_leaves=True, min_cluster_size=5)
    L = lumbermark.Lumbermark(n_clusters=n_clusters, noise_postprocess="tree", n_neighbors=15, noise_threshold="half")

    y_pred = L.fit_predict(X)


    is_noise = L._is_noise
    tree_e = L._tree_e
    tree_w = L._tree_w
    tree_skiplist = L._tree_skiplist


    genieclust.plots.plot_scatter(X[~is_noise,:], labels=y_pred[~is_noise]-1)
    genieclust.plots.plot_scatter(X, labels=y_pred-1, alpha=0.2)
    genieclust.plots.plot_segments(tree_e, X, alpha=0.2)
    genieclust.plots.plot_segments(tree_e[tree_skiplist, :], X, color="yellow", alpha=0.2)
    plt.axis("equal")

    # npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
    nca = genieclust.compare_partitions.normalized_clustering_accuracy(y_true[y_true>0], y_pred[y_true>0])

    #s1, s2 = treelhouette.treelhouette_score(L)
    plt.title("%s NCA=%.2f" % (example, nca))
    ncas.append(nca)

plt.tight_layout()
print("Average NCA=%.2f" % np.mean(ncas))
