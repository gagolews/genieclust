import numpy as np
import genieclust
import time
import gc
import warnings

import scipy.sparse
import scipy.spatial.distance
import numpy as np



try:
    import rpy2
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    #rpy2.robjects.numpy2ri.activate()
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
if os.path.exists(".devel/benchmark_data"):
    path = ".devel/benchmark_data"
elif os.path.exists("benchmark_data"):
    path = "benchmark_data"
else:
    path = "../benchmark_data"


verbose = True  # codecov!

# TODO test  -1 <= labels < n_clusters
# TODO  different M


def __test_genie_approx(metric='euclidean'):
    if nmslib is None: return
    if mlpack is None: return

    for dataset in ["t4_8k", "h2mg_64_50"]:#, "bigger"]:#[, "bigger""s1", "Aggregation", "unbalance", "h2mg_64_50"]:#, "h2mg_1024_50", "t4_8k", "bigger"]:
        if dataset == "bigger":
            np.random.seed(123)
            n = 10_000
            d = 10
            X = np.random.normal(size=(n,d))
            labels = np.random.choice(np.r_[1,2,3,4,5,6,7,8], n)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
            labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intp)-1

        k = len(np.unique(labels[labels>=0]))
        # center X + scale (NOT: standardize!)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)




        for M in [1, 2, 25]:
            for g in [0.01, 0.3, 0.7]:
                gc.collect()

                #D = scipy.spatial.distance.pdist(X)
                #D = scipy.spatial.distance.squareform(D)

                print("%-20s M=%2d g=%.2f n=%7d d=%4d"%(dataset,M,g,X.shape[0],X.shape[1]), end="\t")

                if not (metric in ['maximum']):
                    t01 = time.time()
                    res1 = genieclust.Genie(k, gini_threshold=g, exact=True, affinity=metric, verbose=verbose, M=M).fit_predict(X)+1
                    t11 = time.time()
                    print("t_py=%.3f" % (t11-t01), end="\t")
                else:
                    res1 = None

                #assert len(np.unique(res1[res1>=0])) == k

                if stats is not None and genie is not None and M == 1 and not (metric in ['cosine', 'maximum']):
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



                t03 = time.time()
                res3 = genieclust.Genie(k, gini_threshold=g, exact=False, affinity=metric, verbose=verbose, M=M).fit_predict(X)+1
                t13 = time.time()
                print("t_py2=%.3f" % (t13-t03), end="\t")

                if res1 is not None:
                    print("t_rel=%.3f" % ((t03-t13)/(t01-t11),), end="\t")

                    ari = genieclust.compare_partitions.adjusted_rand_score(res1, res3)
                    print("ARI2=%.3f" % ari, end="\t")
                    if ari < 1.0-1e-12:
                        warnings.warn("(exact=False) ARI=%.3f for dataset=%s, g=%.2f, affinity=%s" %(
                            ari, dataset, g, metric
                            ))

                    res1, res2 = None, None

                print("")





def __test_sparse(affinity='euclidean_sparse'):
    if nmslib is None: return
    if mlpack is None: return

    np.random.seed(123)
    X = np.random.choice(np.arange(-2.0, 3.0), 1000).reshape(100,-1)
    X *= np.random.rand(*X.shape)

    X = scipy.sparse.csr_matrix(X)

    genieclust.Genie(affinity=affinity, exact=False).fit(X)





def __test_string(affinity='leven'):
    if nmslib is None: return
    if mlpack is None: return

    np.random.seed(123)
    X = []
    for i in range(1, 100):
        X.append("a"*i)

    genieclust.Genie(affinity=affinity, exact=False, cast_float32=False).fit(X)



def test_genie_approx():
    __test_genie_approx('euclidean')
    __test_genie_approx('manhattan')
    __test_genie_approx('cosine')
    __test_genie_approx('maximum')


def test_sparse():
    __test_sparse('euclidean_sparse')
    __test_sparse('manhattan_sparse')
    __test_sparse('cosine_sparse')
    __test_sparse('cosine_sparse_fast')
    __test_sparse('chebyshev_sparse')

def test_string():
    __test_string('leven')



if __name__ == "__main__":
    test_genie_approx()
    test_string()
    test_sparse()
