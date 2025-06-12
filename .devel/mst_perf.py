import os
import numba
import numpy as np
import timeit
import genieclust


n_jobs = 1
os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)
numba.config.THREADING_LAYER = 'omp'




"""
n=100000, d=2, M=1, threads=6
Genie_mlpack:             0.58706      1013.16058
Genie_brute:              6.92215      1013.16058
fast_hdbscan:             0.42798      1013.15980
hdbscan_kdtree:           0.77545      1013.15980
hdbscan_balltree:         5.13781      1013.15980

n=1000000, d=2, M=1, threads=6
Genie_mlpack:             6.78291      3229.30835
Genie_brute:           1039.48722      3229.30835
fast_hdbscan:             7.60030      3229.43496
hdbscan_kdtree:          26.11536      3229.43496


n=10000, d=25, M=1, threads=6
Genie_mlpack:            24.54190     38881.92188
Genie_brute:              0.25670     38881.92188
fast_hdbscan:             1.65068     38881.92565
hdbscan_kdtree:           4.36768     38881.92565

n=25000, d=25, M=1, threads=6
Genie_brute:              1.31742     93263.83594
fast_hdbscan:            11.07844     93263.90168
hdbscan_kdtree:          26.27142     93263.90168

n=25000, d=25, M=10, threads=6
Genie_brute:              3.83056    105238.12500
fast_hdbscan:             2.77448    105238.13837
hdbscan_kdtree:           3.45090    105238.13837

"""



import hdbscan
from sklearn.neighbors import KDTree, BallTree
from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm # BallTreeBoruvkaAlgorithm - much slower
def mst_hdbscan_kdtree(X, M, leaf_size=40, leaf_size_div=3):
    tree = KDTree(X, metric='euclidean', leaf_size=leaf_size)
    alg = KDTreeBoruvkaAlgorithm(
        tree,
        min_samples=M-1,
        metric='euclidean',
        leaf_size=leaf_size // leaf_size_div,  # https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/hdbscan_.py
        approx_min_span_tree=False,
        n_jobs=n_jobs
    )
    _res = alg.spanning_tree()
    tree_w = _res[:,  2].astype(X.dtype, order="C")
    tree_e = _res[:, :2].astype(np.intp, order="C")
    return tree_w, tree_e




import fast_hdbscan
numba.set_num_threads(n_jobs)
def mst_fasthdbscan_kdtree(X, M, leaf_size=40, leaf_size_div=3):
    _res = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(
        X,
        min_samples=M-1
    )
    #print(_res)
    #stop()
    i1 = _res[0][:, 0].astype(np.intp, order="C")
    i2 = _res[0][:, 1].astype(np.intp, order="C")
    dcore = _res[2]
    tree_w = np.maximum(
        np.maximum(dcore[i1], dcore[i2]),
        np.sqrt(
            np.sum((X[i1,:]-X[i2,:])**2, axis=1)
        )
    )
    tree_e = np.c_[i1, i2]
    return tree_w, tree_e





import mlpack
def mst_mlpack(X, M, leaf_size=1):
    if M > 1:
        return None
    _res = mlpack.emst(
        X,
        leaf_size=leaf_size,  # "One-element leaves give the empirically best performance, but at the cost of greater memory requirements."
        naive=False,
        copy_all_inputs=False,
        verbose=False
    )["output"]
    tree_w = _res[:,  2].astype(X.dtype, order="C")
    tree_e = _res[:, :2].astype(np.intp, order="C")
    return tree_w, tree_e


cases = dict(
    hdbscan_kdtree_40_3=lambda X, M: mst_hdbscan_kdtree(X, M, 40, 3),
    fasthdbscan_kdtree=lambda X, M: mst_fasthdbscan_kdtree(X, M),
    mlpack_1=lambda X, M: mst_mlpack(X, M, 1),
    mlpack_4=lambda X, M: mst_mlpack(X, M, 4),
)



for n, d, M in [(100000, 2, 1), (10000, 5, 1), (10000, 2, 10), (10000, 5, 10)] :
    np.random.seed(123)
    X = np.random.randn(n, d)
    print("n=%d, d=%d, M=%d, threads=%d" % (X.shape[0], X.shape[1], M, n_jobs))

    # preflight (e.g., for fast_hdbscan)
    for name, generator in cases.items():
        generator(X[:100,:].copy(), M)

    res = list()
    for case, generator in cases.items():
        t0 = timeit.time.time()
        _res = generator(X, M)
        t1 = timeit.time.time()
        res.append(_res)
        if _res is None: continue
        print("%30s: t=%15.5f Δdist=%15.5f Δind=%10.0f" % (
            case,
            t1-t0,
            np.sum(_res[0])-np.sum(res[0][0]),
            np.sum(_res[1] != res[0][1]),
        ))
