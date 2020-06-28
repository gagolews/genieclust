#%%silent
#%%restart
#%%cd @

import sys
# "https://github.com/gagolews/clustering_benchmarks_v1"
benchmarks_path = "/home/gagolews/Projects/clustering_benchmarks_v1"
sys.path.append(benchmarks_path)
from load_dataset import load_dataset
from generate_hKmg import generate_hKmg
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import genieclust
np.set_printoptions(precision=5, threshold=10, edgeitems=5)
pd.set_option("min_rows", 20)
plt.style.use('seaborn-whitegrid')
#plt.rcParams["figure.figsize"] = (8,4)

#X, labels_true, dataset = load_dataset("h2mg/", benchmarks_path)
#labels_true = [l-1 for l in labels_true] # noise class==-1

np.random.seed(123)
n = 10_000
d = 2
s_cor = 100*d
#K = 2
#X = np.random.normal(size=(n,d))
#labels_true = np.random.choice(np.r_[1:(K+1)], n)
#dataset = "random.normal"
X, labels0, labels1 = generate_hKmg(d,
                np.r_[n//2, n//2],
                np.array([ [500]*d, [600]*d ]),
                np.r_[s_cor, s_cor], random_state=123)
labels_true = [labels0, labels1]
dataset = "hKmg"
X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1))
X = X.astype(np.float32, order="C", copy=False)
labels_true = labels_true[0]
n_clusters = int(len(np.unique(labels_true))-(np.min(labels_true)==-1))
import nmslib
num_neighbours = 3#max(256, int(math.sqrt(X.shape[0])))
num_threads = 4 # set to OMP_***....
verbose = True
index = nmslib.init(method='hnsw', space='l2')
index.addDataPointBatch(X)
index.createIndex({'post': 2}, print_progress=verbose)
nns = index.knnQueryBatch(X, k=num_neighbours+1, num_threads=num_threads)
nn_d = np.ones((X.shape[0], num_neighbours+1), dtype=X.dtype)*np.inf
nn_i = np.ones((X.shape[0], num_neighbours+1), dtype=np.intp)*(-1)
for i in range(X.shape[0]):
    nn_d[i, :len(nns[i][1])] = nns[i][1]
    nn_i[i, :len(nns[i][0])] = nns[i][0]
#nn_d = np.stack([np.r_[nn[1], [np.inf]*(num_neighbours+1-nn[1].shape[0])] for nn in nns])
#nn_i = np.stack([np.r_[nn[0], [-1]*(num_neighbours+1-nn[0].shape[0])] for nn in nns])
#nn_i = nn_i.astype(np.intp, order="C")
#nn_d = nn_d.astype(X.dtype, order="C")
# mst_from_nn ignores self-loops
mst_d, mst_i = genieclust.internal.mst_from_nn(nn_d, nn_i,
    stop_disconnected=False, stop_inexact=False, verbose=verbose)

print(np.sum(mst_i[:,0]<0))
out = genieclust.internal.genie_from_mst(mst_d, mst_i, n_clusters=1)


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


