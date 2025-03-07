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



plt.clf()
_i = 0
for ex in range(12):
    _i += 1
    plt.subplot(3, 4, _i)
    X, y_true, n_clusters, skiplist, example = mst_examples.get_example(ex, data_path)

    n_clusters = max(y_true)

    L = robust_single_linkage.RobustSingleLinkageClustering(n_clusters=n_clusters)
    #L = lumbermark.Lumbermark(n_clusters=n_clusters, verbose=False, n_neighbors=5, cluster_size_factor=0.2, outlier_factor=1.5, noise_cluster=False)

    y_pred = L.fit_predict(X, mst_skiplist=skiplist)  # TODO: 0-based -> 1-based!!!


    mst_examples.plot_mst_2d(L)
    npa = GNPA(y_true[y_true>0], y_pred[y_true>0])
    nca = GNCA(y_true[y_true>0], y_pred[y_true>0])

    plt.title("%s NPA=%.2f NCA=%.2f" % (example, npa, nca))
plt.tight_layout()
