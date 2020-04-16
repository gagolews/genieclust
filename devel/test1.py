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
labels_true = np.loadtxt(os.path.join(path, "%s.labels0.gz" % dataset),
    dtype=np.intp)-1
n_clusters = int(len(np.unique(labels_true))-(np.min(labels_true)==-1))
gini_threshold = 0.3


g = genieclust.Genie(n_clusters=n_clusters,
            gini_threshold=gini_threshold)
labels = g.fit_predict(X)
print(labels)

plt.rcParams["figure.figsize"] = (8,4)
plt.subplot("121")
genieclust.plots.plot_scatter(X, labels=labels_true)
plt.title("%s (n=%d, true n_clusters=%d)"%(dataset, X.shape[0], n_clusters))
plt.axis("equal")
plt.subplot("122")
genieclust.plots.plot_scatter(X, labels=labels)
plt.title("%s Genie g=%g"%(dataset, gini_threshold))
plt.axis("equal")
plt.show()


# create the linkage matrix, see scipy.cluster.hierarchy.linkage
Z = np.column_stack((g.children_, g.distances_, g.counts_))
# correct for possible departures from ultrametricity:
Z[:,2] = genieclust.tools.cummin(Z[::-1,2])[::-1]
import scipy.cluster.hierarchy
scipy.cluster.hierarchy.dendrogram(Z)
plt.show()


X = np.array(
    [[0, 0], [0, 1], [1, 0],
     [0, 4], [0, 3], [1, 4],
     [4, 0], [3, 0], [4, 1],
     [4, 4], [3, 4], [4, 3]])
g = genieclust.Genie(n_clusters=n_clusters,
            gini_threshold=gini_threshold)
labels = g.fit_predict(X)
# create the linkage matrix, see scipy.cluster.hierarchy.linkage
Z = np.column_stack((g.children_, g.distances_, g.counts_))
print(Z)


import scipy.cluster.hierarchy
plt.subplot(121)
genieclust.plots.plot_scatter(X)
plt.axis("equal")
plt.subplot(122)
scipy.cluster.hierarchy.dendrogram(Z)
plt.show()


