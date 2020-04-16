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
dataset = "flame"
X = np.loadtxt(os.path.join(path, "%s.data.gz" % dataset), ndmin=2)
X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1))
X = X.astype(np.float32, order="C", copy=False)
labels_true = np.loadtxt(os.path.join(path, "%s.labels0.gz" % dataset),
    dtype=np.intp)-1
n_clusters = int(len(np.unique(labels_true))-(np.min(labels_true)==-1))


g = genieclust.Genie(n_clusters=n_clusters,
            gini_threshold=0.3)
labels_g = g.fit_predict(X)
print(labels_g)

gic = genieclust.GIc(n_clusters=n_clusters,
            gini_thresholds=[0.3, 0.3])
labels_gic = gic.fit_predict(X)
print(labels_gic)


plt.rcParams["figure.figsize"] = (12,4)
plt.subplot("131")
genieclust.plots.plot_scatter(X, labels=labels_true)
plt.title("%s (n=%d, true n_clusters=%d)"%(dataset, X.shape[0], n_clusters))
plt.axis("equal")
plt.subplot("132")
genieclust.plots.plot_scatter(X, labels=labels_g)
plt.title("%s Genie g=%g"%(dataset, g.gini_threshold))
plt.axis("equal")
plt.subplot("133")
genieclust.plots.plot_scatter(X, labels=labels_gic)
plt.title("%s GIc g=%r"%(dataset, gic.gini_thresholds))
plt.axis("equal")
plt.show()


