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
import scipy.spatial.distance
from rpy2.robjects.packages import importr
stats = importr("stats")
genie = importr("genie")
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import time
import gc
np.random.seed(12345)
#plt.rcParams["figure.figsize"] = (8,4)

X, labels_true, dataset = load_dataset("mnist/digits", benchmarks_path)
labels_true = [l-1 for l in labels_true] # noise class==-1

print(X.shape)

#n = 1_000_000
#d = 10
#s_cor = 30*d
##K = 2
##X = np.random.normal(size=(n,d))
##labels_true = np.random.choice(np.r_[1:(K+1)], n)
##dataset = "random.normal"
#X, labels0, labels1 = generate_hKmg(d,
                #np.r_[n//5, 4*n//5],
                #np.array([ [500]*d, [600]*d ]),
                #np.r_[s_cor, s_cor], random_state=123)
#labels_true = [labels0, labels1]
#dataset = "hKmg"




X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1))
X = X.astype(np.float32, order="C", copy=False)
labels_true = labels_true[0]
n_clusters = int(len(np.unique(labels_true))-(np.min(labels_true)==-1))
import nmslib
nmslib_n_neighbors = 64
num_neighbours = min(X.shape[0]-1, max(1, int(nmslib_n_neighbors)))
num_threads = 4 # set to OMP_***....
verbose = True

# https://github.com/nmslib/nmslib/blob/master/manual/spaces.md
# all dense-vector spaces require float32 numpy-array input (except squared l2)
# sparse scipy matrices (csr_matrix)
# otherwise, string arrays
#   levenshtein = ASCII strings
#   other = space-separated numbers
t02 = time.time()
# data_type=nmslib.DataType.DENSE_VECTOR|OBJECT_AS_STRING|SPARSE_VECTOR
# dtype = nmslib.DistType.FLOAT|INT use FLOAT except for `leven`
# https://github.com/nmslib/nmslib/blob/master/manual/methods.md
# https://github.com/nmslib/nmslib/blob/master/manual/spaces.md
nmslib_params_init = dict(method='hnsw', space='l2')# data_type, dtype
nmslib_params_index = dict(post=2, indexThreadQty=num_threads)
nmslib_params_query = dict()
index = nmslib.init(**nmslib_params_init)
index.addDataPointBatch(X)
index.createIndex(nmslib_params_index, print_progress=verbose)
index.setQueryTimeParams(nmslib_params_query)
nns = index.knnQueryBatch(X, k=num_neighbours+1, num_threads=num_threads)
index = None
gc.collect()
# TODO: when determining M, note that the actual number NNs might be < M!

#t03 = time.time()
#nn_d = np.ones((X.shape[0], num_neighbours+1), dtype=X.dtype)*np.inf
#nn_i = np.ones((X.shape[0], num_neighbours+1), dtype=np.intp)*(-1)
#for i in range(X.shape[0]):
    #nn_d[i, :len(nns[i][1])] = nns[i][1]
    #nn_i[i, :len(nns[i][0])] = nns[i][0]
#mst_d, mst_i = genieclust.internal.mst_from_nn(nn_d, nn_i,
    #stop_disconnected=False, stop_inexact=False, verbose=verbose)
#print("aaa=%.3f" % (time.time()-t03))

#nn_d = None
#nn_i = None
t03 = time.time()
mst_d, mst_i = genieclust.internal.mst_from_nn_list(nns, k_max=num_neighbours,
    stop_disconnected=False, verbose=verbose)
print("bbb=%.3f" % (time.time()-t03))

nns = None
gc.collect()
#print(np.sum(mst_i[:,0]<0))
out = genieclust.internal.genie_from_mst(mst_d, mst_i,
    n_clusters=n_clusters, new_merge=False)["labels"]
t12 = time.time()
print("t_py_approx=%.3f" % (t12-t02))

print(genieclust.compare_partitions.confusion_matrix(out, labels_true))
print(genieclust.compare_partitions.compare_partitions2(out, labels_true)["ar"])


out = genieclust.internal.genie_from_mst(mst_d, mst_i,
    n_clusters=n_clusters, new_merge=False, gini_threshold=0.1)["labels"]
print("t_py_approx=%.3f" % (t12-t02))

print(genieclust.compare_partitions.confusion_matrix(out, labels_true))
print(genieclust.compare_partitions.compare_partitions2(out, labels_true)["ar"])




#t02 = time.time()
#g = genieclust.Genie(n_clusters=n_clusters, verbose=verbose)
#labels_g = g.fit_predict(X)
#t12 = time.time()
#print("t_py=%.3f" % (t12-t02))
#g = None
#gc.collect()

#print(n_clusters)
#print(genieclust.compare_partitions.confusion_matrix(out, labels_g))
#print(genieclust.compare_partitions.compare_partitions2(out, labels_g)["ar"])
#print(genieclust.compare_partitions.compare_partitions2(labels_g, labels_true)["ar"])
##print(labels_g)


#t02 = time.time()
#res2 = stats.cutree(genie.hclust2(objects=X), n_clusters)
#t12 = time.time()
#print("t_r=%.3f" % (t12-t02))
#res2 = np.array(res2, np.intp)
#print(genieclust.compare_partitions.confusion_matrix(res2, labels_g))
#print(genieclust.compare_partitions.compare_partitions2(res2, labels_g)["ar"])

#gic = genieclust.GIc(n_clusters=n_clusters,
            #gini_thresholds=[0.1, 0.3, 0.5, 0.7],
            #add_clusters=10,
            #M=1)
#labels_gic = gic.fit_predict(X)
#print(labels_gic)
#print(genieclust.compare_partitions.compare_partitions2(labels_gic, labels_g))

##%%eof

#plt.rcParams["figure.figsize"] = (12,4)
#plt.subplot("131")
#genieclust.plots.plot_scatter(X, labels=labels_true)
#plt.title("%s (n=%d, true n_clusters=%d)"%(dataset, X.shape[0], n_clusters))
#plt.axis("equal")
#plt.subplot("132")
#genieclust.plots.plot_scatter(X, labels=labels_g)
#plt.title("%s Genie g=%g"%(dataset, g.gini_threshold))
#plt.axis("equal")
#plt.subplot("133")
#genieclust.plots.plot_scatter(X, labels=labels_gic)
#plt.title("%s GIc g=%r"%(dataset, gic.gini_thresholds))
#plt.axis("equal")
#plt.show()


