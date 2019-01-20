import numpy as np
from genieclust.genie import *
from genieclust.inequity import *
from genieclust.compare_partitions import *
import genieclust.mst
import genieclust.internal
import time
import gc


import scipy.spatial.distance
from rpy2.robjects.packages import importr
stats = importr("stats")
genie = importr("genie")
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
path = "benchmark_data"

def test_gic():
    for dataset in ["jain", "Aggregation", "unbalance", "h2mg_64_50"]:#, "h2mg_1024_50", "t4_8k", "bigger"]:
        if dataset == "bigger":
            np.random.seed(123)
            n = 100_000
            X = np.random.normal(size=(n,32))
            labels = np.random.choice(np.r_[1,2], n)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
            labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intc)-1

        K = len(np.unique(labels[labels>=0]))
        # center X + scale (NOT: standardize!)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)

        mst_d, mst_i = genieclust.mst.mst_from_distance(X)

        print("%-20s n=%7d d=%4d"%(dataset,X.shape[0],X.shape[1]))
        for g in [ np.r_[0.1],  np.r_[0.2],  np.r_[0.3], np.r_[0.4], np.r_[0.5], np.r_[0.6], np.r_[0.7], np.arange(1, 8)/10, np.empty(0)]:
            print(g, end="\t")
            gc.collect()

            t01 = time.time()
            labels_gic = genieclust.internal.gic_from_mst(mst_d, mst_i,
                n_clusters=K, n_features=X.shape[1], gini_thresholds=g,
                noise_leaves=False)
            t11 = time.time()
            print("t_py=%.3f" % (t11-t01), end="\t")

            assert len(np.unique(labels_gic[labels_gic>=0])) == K
            print()


if __name__ == "__main__":
    test_gic()
