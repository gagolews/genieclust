import numpy as np
import genieclust
import time
import gc
import pytest
import scipy.spatial.distance
import numpy as np

try:
    import rpy2
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    stats = importr("stats")
    genie = importr("genie")
except:
    rpy2 = None
    stats = None
    genie = None


try:
    import mlpack
except:
    mlpack = None

try:
    import nmslib
except:
    nmslib = None


import os
if os.path.exists("devel/benchmark_data"):
    path = "devel/benchmark_data"
elif os.path.exists("benchmark_data"):
    path = "benchmark_data"
else:
    path = "../benchmark_data"




def test_cache(metric='euclidean'):
    np.random.seed(123)
    n = 1_000
    d = 10
    K = 2
    X = np.random.normal(size=(n,d))
    labels = np.random.choice(np.r_[0:K], n)

    k = len(np.unique(labels[labels>=0]))

    # center X + scale (NOT: standardize!)
    X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
    X += np.random.normal(0, 0.0001, X.shape)


    for C in [genieclust.Genie, genieclust.GIc]:
        for exact in [True, False]:
            if not exact and (not nmslib or not mlpack):
                continue

            for M in [10,2,1]:
                os.environ["OMP_NUM_THREADS"] = '1'
                t01 = time.time()
                g = C(2, affinity=metric, M=M, exact=exact)
                y1 = g.fit_predict(X)+1
                t11 = time.time()
                print("(1 thread ) t_py=%.3f" % (t11-t01))

                del os.environ["OMP_NUM_THREADS"]
                t01 = time.time()
                g = C(2, affinity=metric, M=M, exact=exact)
                y2 = g.fit_predict(X)+1
                t11 = time.time()
                print("(max threads) t_py=%.3f" % (t11-t01))
                ari = genieclust.compare_partitions.adjusted_rand_score(y1, y2)
                assert ari>1.0-1e-12 or not exact

                t01 = time.time()
                g.gini_threshold = 0.1
                g.n_clusters = 100
                g.M = 20
                g.fit_predict(X)+1
                t11 = time.time()
                print("(reuse-x   ) t_py=%.3f" % (t11-t01))

                t01 = time.time()
                g.postprocess="none"
                g.fit_predict(X)+1
                t11 = time.time()
                print("(reuse-x   ) t_py=%.3f" % (t11-t01))

                t01 = time.time()
                g.gini_threshold = 0.3
                g.postprocess="boundary"
                g.n_clusters = 2
                g.M = M
                y2 = g.fit_predict(X)+1
                t11 = time.time()
                print("(reuse     ) t_py=%.3f" % (t11-t01))
                ari = genieclust.compare_partitions.adjusted_rand_score(y1, y2)
                assert ari>1.0-1e-12 or not exact



if __name__ == "__main__":
    print("**Euclidean**")
    test_cache('euclidean')

    print("**Manhattan**")
    test_cache('manhattan')


