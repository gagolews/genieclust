#%cd /home/gagolews/Python/genieclust/devel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import genieclust

np.set_printoptions(precision=5, threshold=10, edgeitems=5)
plt.style.use('seaborn-whitegrid')
#plt.rcParams["figure.figsize"] = (8,4)

path = os.path.join("..", "benchmark_data")
dataset = "jain"
X = np.loadtxt(os.path.join(path, "%s.data.gz" % dataset), ndmin=2)
X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1))
X = X.astype(np.float32, order="C", copy=False)
labels_true = np.loadtxt(os.path.join(path, "%s.labels0.gz" % dataset), dtype=np.intp)-1
n_clusters = int(len(np.unique(labels_true))-(np.min(labels_true)==-1))
gini_threshold = 0.3


mst_dist, mst_ind = genieclust.internal.mst_from_distance(X)
res = genieclust.internal.genie_from_mst(mst_dist, mst_ind,
            n_clusters=n_clusters,
            gini_threshold=gini_threshold,
            noise_leaves=False)
labels = res["labels"]
print(labels)

plt.rcParams["figure.figsize"] = (8,4)
plt.subplot("121")
genieclust.plots.plot_scatter(X, labels_true)
plt.title("%s (n=%d, true n_clusters=%d)"%(dataset, X.shape[0], n_clusters))
plt.axis("equal")
plt.subplot("122")
genieclust.plots.plot_scatter(X, labels)
plt.title("%s Genie g=%g"%(dataset, gini_threshold))
plt.axis("equal")
plt.show()


