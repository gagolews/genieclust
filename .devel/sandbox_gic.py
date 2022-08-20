#%%silent
#%%restart
#%%cd @

import sys
# "https://github.com/gagolews/clustering_benchmarks_v1"
benchmarks_path = "/home/gagolews/Projects/clustering_benchmarks_v1"
sys.path.append(benchmarks_path)
from load_dataset import load_dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import genieclust
np.set_printoptions(precision=5, threshold=10, edgeitems=5)
pd.set_option("min_rows", 20)
plt.style.use('seaborn-whitegrid')
#plt.rcParams["figure.figsize"] = (8,4)

X, labels_true, dataset = load_dataset("sipu/aggregation", benchmarks_path)
X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1))
X = X.astype(np.float32, order="C", copy=False)
labels_true = [l-1 for l in labels_true] # noise class==-1

labels_true = labels_true[0]
n_clusters = int(len(np.unique(labels_true))-(np.min(labels_true)==-1))


g = genieclust.Genie(n_clusters=n_clusters,
            gini_threshold=0.3,
            M=1)
labels_g = g.fit_predict(X)
print(genieclust.compare_partitions.compare_partitions2(labels_true, labels_g))
print(labels_g)

gic = genieclust.GIc(n_clusters=n_clusters,
            gini_thresholds=[0.1, 0.3, 0.5, 0.7],
            add_clusters=10,
            M=1)
labels_gic = gic.fit_predict(X)
print(labels_gic)
print(genieclust.compare_partitions.compare_partitions2(labels_gic, labels_g))

#%%eof

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


