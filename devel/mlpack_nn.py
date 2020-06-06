#%%silent
#%%restart
#%%cd @

DATASET = ["sipu/worms_2", "sipu/worms_64",
           "other/chameleon_t4_8k", "mnist/fashion"][-1]

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


#X, labels_true, dataset = load_dataset(DATASET, benchmarks_path)
#X = X[:, X.var(axis=0) > 0] # remove all columns of 0 variance
## add a tiny bit of white noise:
#np.random.seed(123)
#X += np.random.normal(0.0, X.std(ddof=1)*1e-6, size=X.shape)
#X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1))
#X = X.astype(np.float32, order="C", copy=False)
#labels_true = [l-1 for l in labels_true] # noise class==-1

#labels_true = labels_true[0]
#n_clusters = int(len(np.unique(labels_true))-(np.min(labels_true)==-1))



import mlpack
import time
import faiss



n = 25_000
res = []
for n in (10_000, 20_000, 40_000):
    for d in range(2, 10):
        DATASET = "random.norm(n=%d, d=%d)" % (n, d)
        np.random.seed(123)
        X = np.random.normal(size=(n,d))
        print(DATASET)
        t02 = time.time()
        tree2 = genieclust.internal.mst_from_distance(X)
        t12 = time.time()
        print("genieclust.from_distance: %.3f" % (t12-t02))
        t01 = time.time()
        tree1 = mlpack.emst(input=X)
        t11 = time.time()
        print("mlpack.emst: %.3f" % (t11-t01))
        res.append(dict(n=n, d=d, mlpack_emst=t11-t01, genieclust_from_distance=t12-t02))

import pandas as pd
res = pd.DataFrame(res)
print(res)

res = res.set_index(["n", "d"]).stack().reset_index()
res.columns = ["n", "d", "method", "time"]


import seaborn as sns
sns.relplot(data=res, x="d", y="time", hue="method", style="n", kind="line")
plt.title("np.random.normal(size=(n,d))")
plt.show()



### sipu/worms_2
#genieclust.from_distance [4 threads]: 12.274
#mlpack.emst: 0.563

### mnist/fashion
#genieclust.from_distance [4 threads]: 552.031
#mlpack.emst: 3827.782


#print(DATASET)
#k = 10
#t0 = time.time()
#nn1 = mlpack.krann(reference=X, k=k)
#t1 = time.time()
#print("mlpack.krann: %.3f" % (t1-t0))
#t0 = time.time()
#nn2 = mlpack.lsh(reference=X, k=k)
#t1 = time.time()
#print("mlpack.lsh: %.3f" % (t1-t0))
#t0 = time.time()
#nn = faiss.IndexFlatL2(X.shape[1])
#nn.add(X)
#nn3 = nn.search(X, k)
#t1 = time.time()
#print("faiss: %.3f" % (t1-t0))

##sipu/worms_2
##mlpack.krann: 7.993
##mlpack.lsh: 4.729
##faiss: 30.729

##mnist/fashion
##mlpack.krann: 17.518
##mlpack.lsh: 13.563
##faiss: 131.342
