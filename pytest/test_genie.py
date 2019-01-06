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

def test_genie(metric='euclidean'):
    for dataset in ["pathbased", "s1", "a1", "Aggregation", "WUT_Smile", "unbalance"]:
        if dataset == "bigger":
            np.random.seed(123)
            n = 1_000
            X = np.random.normal(size=(n,1000))
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

            t0 = time.time()
            res1 = Genie(k, g, exact=True, nn_params=dict(metric=metric)).fit_predict(X)+1
            print("t_py=%.3f" % (time.time()-t0), end="\t")

            assert len(np.unique(res1)) == k

            t0 = time.time()
            res2 = stats.cutree(genie.hclust2(objects=X, d=metric, thresholdGini=g), k)
            print("t_r=%.3f" % (time.time()-t0), end="\t")
            res2 = np.array(res2, np.intc)
            assert len(np.unique(res2)) == k

            ari = adjusted_rand_score(res1, res2)
            print("ARI=%.3f" % ari)
            assert ari>1.0-1e-12

            res1, res2 = None, None


if __name__ == "__main__":
    test_genie('euclidean')
    test_genie('manhattan')
