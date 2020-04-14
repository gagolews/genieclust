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
    for dataset in ["t4_8k", "h2mg_64_50", "h2mg_1024_50"]:#[, "bigger""s1", "Aggregation", "unbalance", "h2mg_64_50"]:#, "h2mg_1024_50", "t4_8k", "bigger"]:
        if dataset == "bigger":
            np.random.seed(123)
            n = 25_000
            X = np.random.normal(size=(n,32))
            labels = np.random.choice(np.r_[1,2,3,4,5,6,7,8], n)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
            labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intp)-1

        k = len(np.unique(labels[labels>=0]))
        # center X + scale (NOT: standardize!)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)

        for g in [0.01, 0.3, 1.0]:
            gc.collect()

            #D = scipy.spatial.distance.pdist(X)
            #D = scipy.spatial.distance.squareform(D)

            print("%-20s g=%.2f n=%7d d=%4d"%(dataset,g,X.shape[0],X.shape[1]), end="\t")

            t01 = time.time()
            res1 = Genie(k, g, exact=True, nn_params=dict(metric=metric)).fit_predict(X)+1
            t11 = time.time()
            print("t_py=%.3f" % (t11-t01), end="\t")

            assert len(np.unique(res1)) == k

            t02 = time.time()
            res2 = stats.cutree(genie.hclust2(objects=X, d=metric, thresholdGini=g), k)
            t12 = time.time()
            print("t_r=%.3f" % (t12-t02), end="\t")
            res2 = np.array(res2, np.intp)
            assert len(np.unique(res2)) == k

            ari = adjusted_rand_score(res1, res2)
            print("ARI=%.3f" % ari, end="\t")
            assert ari>1.0-1e-12

            print("t_rel=%.3f" % ((t11-t01)/(t12-t02),), end="\t")

            t03 = time.time()
            res3 = Genie(k, g, exact=False, compute_full_tree=False, n_neighbors=32, nn_params=dict(metric=metric)).fit_predict(X)+1
            t13 = time.time()
            print("t_py2=%.3f" % (t13-t03), end="\t")
            print("t_rel=%.3f" % ((t03-t13)/(t01-t11),), end="\t")

            ari = adjusted_rand_score(res1, res3)
            print("ARI2=%.3f" % ari, end="\t")
            assert ari>1.0-1e-12

            res1, res2 = None, None
            print("")


if __name__ == "__main__":
    print("TODO: Is R::genie using OMP? Is Py::genieclust using OMP?")
    print("**Euclidean**")
    test_genie('euclidean')
    #print("**Manhattan**")
    #test_genie('manhattan')
