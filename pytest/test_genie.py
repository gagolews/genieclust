import numpy as np
from genieclust.genie import *
from genieclust.inequity import *
from genieclust.compare_partitions import *
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

def test_genie():
    for dataset in ["pathbased", "s1", "a1", "Aggregation", "WUT_Smile", "unbalance"]: #, "bigger"
        if dataset == "bigger":
            np.random.seed(123)
            n = 25000
            X = np.random.normal(size=(n,2))
            labels = np.random.choice(np.r_[1,2], n)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
            labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intc)
        label_counts = np.unique(labels,return_counts=True)[1]
        k = len(label_counts)

        for g in [0.01, 0.3, 0.5, 0.7, 1.0]:
            gc.collect()

            #D = scipy.spatial.distance.pdist(X)
            #D = scipy.spatial.distance.squareform(D)

            print("%-20s g=%.2f"%(dataset,g), end="\t")

            #t0 = time.time()
            res1 = Genie(k, g).fit_predict(X)+1

            assert len(np.unique(res1)) == k
            #print("%-20s g=%.2f\t t_py=%.3f" % (dataset, g, time.time()-t0), end="\t")

            D = stats.dist(X)
            #t0 = time.time()
            res2 = stats.cutree(genie.hclust2(D, thresholdGini=g), k)
            #print("t_r=%.3f" % (time.time()-t0))
            res2 = np.array(res2, np.intc)
            assert len(np.unique(res2)) == k

            ari = adjusted_rand_score(res1, res2)
            print("%g" % ari)
            assert ari>1.0-1e-12

            res1, res2, D = None, None, None


if __name__ == "__main__":
    test_genie()
