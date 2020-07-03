import numpy as np
import genieclust
import time
import gc


import scipy.spatial.distance
from rpy2.robjects.packages import importr
stats = importr("stats")
genie = importr("genie")
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import os
if os.path.exists("devel/benchmark_data"):
    path = "devel/benchmark_data"
elif os.path.exists("benchmark_data"):
    path = "benchmark_data"
else:
    path = "../benchmark_data"

# TODO test  -1 <= labels < n_clusters

# TODO: compute_full_tree and test cutree


def test_genie(metric='euclidean'):
    for dataset in ["bigger", "s1", "Aggregation", "unbalance", "h2mg_64_50"]:#, "h2mg_1024_50", "t4_8k", "bigger"]:
        if dataset == "bigger":
            np.random.seed(123)
            n = 10000
            d = 2
            K = 2
            X = np.random.normal(size=(n,d))
            labels = np.random.choice(np.r_[0:K], n)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
            labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intp)-1

        k = len(np.unique(labels[labels>=0]))

        # center X + scale (NOT: standardize!)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)

        #t01 = time.time()
        #hdbscan.RobustSingleLinkage().fit_predict(X)
        #t11 = time.time()
        #print("t_robustsl=%.3f" % (t11-t01), end="\t")

        #t01 = time.time()
        #hdbscan.HDBSCAN().fit_predict(X)
        #t11 = time.time()
        #print("t_hdbscan=%.3f" % (t11-t01), end="\t")


        for g in [0.01, 0.3, 0.5, 0.7, 1.0]:
            gc.collect()

            #D = scipy.spatial.distance.pdist(X)
            #D = scipy.spatial.distance.squareform(D)

            print("%-20s g=%.2f n=%5d d=%2d"%(dataset,g,X.shape[0],X.shape[1]), end="\t")

            t01 = time.time()
            _res1 = genieclust.Genie(
                k, gini_threshold=g, exact=True, affinity=metric, compute_full_tree=True)
            res1 = _res1.fit_predict(X)+1
            t11 = time.time()
            print("t_py=%.3f" % (t11-t01), end="\t")

            assert np.all(np.diff(_res1.distances_)>= 0.0)
            assert len(np.unique(res1)) == k

            t02 = time.time()
            res2 = stats.cutree(genie.hclust2(objects=X, d=metric, thresholdGini=g), k)
            t12 = time.time()
            print("t_r=%.3f" % (t12-t02), end="\t")
            res2 = np.array(res2, np.intp)
            assert len(np.unique(res2)) == k

            ari = genieclust.compare_partitions.adjusted_rand_score(res1, res2)
            print("ARI=%.3f" % ari, end="\t")
            assert ari>1.0-1e-12

            print("t_rel=%.3f" % ((t11-t01)/(t12-t02),), end="\t")


            res1, res2 = None, None
            print("")



def test_genie_precomputed():
    for dataset in ["x1", "s1", "Aggregation"]:#, "h2mg_1024_50", "t4_8k", "bigger"]:
        if dataset == "bigger":
            np.random.seed(123)
            n = 10000
            X = np.random.normal(size=(n,2))
            labels = np.random.choice(np.r_[1,2], n)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
            labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intp)-1

        k = len(np.unique(labels[labels>=0]))

        # center X + scale (NOT: standardize!)
        X = X+np.random.normal(0, 0.0001, X.shape)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X = X.astype("float32")

        D = scipy.spatial.distance.pdist(X)
        if np.random.rand(1) > 0.5:
            D = scipy.spatial.distance.squareform(D)

        for g in [0.01, 0.3, 0.5, 0.7, 1.0]:
            gc.collect()

            print("%-20s g=%.2f n=%5d d=%2d"%(dataset,g,X.shape[0],X.shape[1]), end="\t")

            _res1 = genieclust.Genie(
                k, gini_threshold=g, exact=True, affinity="precomputed")
            res1 = _res1.fit_predict(D)+1

            _res2 = genieclust.Genie(
                k, gini_threshold=g, exact=True, compute_full_tree=True,
                affinity="euclidean", mlpack_enabled=True)
            res2 = _res2.fit_predict(X)+1
            ari = genieclust.compare_partitions.adjusted_rand_score(res1, res2)
            print("ARI=%.3f" % ari, end="\t")
            assert ari>1.0-1e-12
            assert np.all(np.diff(_res2.distances_)>= 0.0)

            _res2 = genieclust.Genie(
                k, gini_threshold=g, exact=True,
                affinity="euclidean", compute_full_tree=True, mlpack_enabled=False)
            res2 = _res2.fit_predict(X)+1
            ari = genieclust.compare_partitions.adjusted_rand_score(res1, res2)
            print("ARI=%.3f" % ari, end="\t")
            assert ari>1.0-1e-12
            assert np.all(np.diff(_res2.distances_)>= 0.0)

            _res1, _res2 = None, None
            res1, res2 = None, None
            print("")


        # test compute_all_cuts
        K = 16
        g = 0.1
        res1 = genieclust.Genie(K, gini_threshold=g, exact=True, affinity="euclidean",
            compute_all_cuts=True, M=2).fit_predict(X)
        assert res1.shape[1] == X.shape[0]
        # assert res1.shape[0] == K+1   #  that's not necessarily true!
        for k in range(1, res1.shape[0]):
            res2 = genieclust.Genie(k, gini_threshold=g, exact=True, affinity="euclidean",
                M=2).fit_predict(X)
            assert np.all(res2 == res1[k,:])

        # test compute_all_cuts
        K = 16
        g = 0.1
        res1 = genieclust.Genie(K, gini_threshold=g, exact=True, affinity="euclidean",
            compute_all_cuts=True, M=25).fit_predict(X)
        assert res1.shape[1] == X.shape[0]
        # assert res1.shape[0] == K+1   #  that's not necessarily true!
        for k in range(1, res1.shape[0]):
            res2 = genieclust.Genie(k, gini_threshold=g, exact=True, affinity="euclidean",
                M=25).fit_predict(X)
            assert np.all(res2 == res1[k,:])



if __name__ == "__main__":
    print("**Precomputed**")
    test_genie_precomputed()

    print("**Euclidean**")
    test_genie('euclidean')

    print("**Manhattan**")
    test_genie('manhattan')
