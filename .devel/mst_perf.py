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
hades @ 2025-06-12
n=100000, d=2, M=1, threads=1
              genieclust_brute: t=        8.81032 Δdist=        0.00000 Δind=         0
        genieclust_kdtree_4_16: t=        0.11707 Δdist=        0.00000 Δind=         0
           hdbscan_kdtree_40_3: t=        1.68920 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        1.30274 Δdist=        0.00000 Δind=         0
                      mlpack_1: t=        0.56649 Δdist=        0.00000 Δind=         0
                      mlpack_4: t=        0.39744 Δdist=        0.00000 Δind=         0
n=100000, d=5, M=1, threads=1
              genieclust_brute: t=       13.52134 Δdist=        0.00000 Δind=         0
        genieclust_kdtree_4_16: t=        2.17226 Δdist=        0.00000 Δind=         0
           hdbscan_kdtree_40_3: t=       15.17135 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        4.26879 Δdist=        0.00000 Δind=         0
                      mlpack_1: t=        5.38789 Δdist=        0.00000 Δind=         0
                      mlpack_4: t=        5.69942 Δdist=        0.00000 Δind=         0
n=100000, d=2, M=10, threads=1
              genieclust_brute: t=       19.19615 Δdist=        0.00000 Δind=         0
        genieclust_kdtree_4_16: t=        0.13406 Δdist=        0.00000 Δind=        31
           hdbscan_kdtree_40_3: t=        1.52790 Δdist=       -0.00000 Δind=     13086
            fasthdbscan_kdtree: t=        0.65928 Δdist=       -0.00000 Δind=     13055
n=100000, d=5, M=10, threads=1
              genieclust_brute: t=       29.40015 Δdist=        0.00000 Δind=         0
        genieclust_kdtree_4_16: t=        2.10061 Δdist=        0.00000 Δind=        19
           hdbscan_kdtree_40_3: t=        6.80395 Δdist=       -0.00000 Δind=      7567
            fasthdbscan_kdtree: t=        1.32822 Δdist=       -0.00000 Δind=      7558


"""


def order_tree(tree_w, tree_e):
    i1 = tree_e[:, 0].astype(np.intp, order="C")
    i2 = tree_e[:, 1].astype(np.intp, order="C")
    i1, i2 = np.minimum(i1, i2), np.maximum(i1, i2)
    tree_e = np.c_[i1, i2]

    o = np.argsort(tree_w, kind="stable")
    return tree_w[o], tree_e[o,:]


import hdbscan
from sklearn.neighbors import KDTree
from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
# BallTreeBoruvkaAlgorithm - much slower
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

    return order_tree(_res[:, 2], _res[:, :2])




import fast_hdbscan
def mst_fasthdbscan_kdtree(X, M, leaf_size=40, leaf_size_div=3):
    _res = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(
        X,
        min_samples=M-1
    )

    i1 = _res[0][:, 0].astype(np.intp, order="C")
    i2 = _res[0][:, 1].astype(np.intp, order="C")

    d_core = np.sqrt(
        np.sum((X-X[_res[1][:, -1], :])**2, axis=1)
    )

    tree_w = np.maximum(
        np.maximum(d_core[i1], d_core[i2]),
        np.sqrt(
            np.sum((X[i1,:]-X[i2,:])**2, axis=1)
        )
    )
    tree_e = np.c_[i1, i2]

    tree_w, tree_e = order_tree(tree_w, tree_e)
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


import genieclust
def mst_genieclust_brute(X, M):
    if X.shape[0] > 100_000: return None
    res = genieclust.fastmst.mst_euclid(X, M, use_kdtree=False)
    tree_w, tree_e = res[:2]
    return tree_w, tree_e


import genieclust
def mst_genieclust_kdtree(X, M, max_leaf_size=4, first_pass_max_brute_size=16):
    res = genieclust.fastmst.mst_euclid(X, M, use_kdtree=True, max_leaf_size=max_leaf_size, first_pass_max_brute_size=first_pass_max_brute_size)
    tree_w, tree_e = res[:2]
    return tree_w, tree_e


cases = dict(
    #genieclust_kdtree_2_16=lambda X, M: mst_genieclust_kdtree(X, M, 2, 16),
    #genieclust_kdtree_4_16=lambda X, M: mst_genieclust_kdtree(X, M, 4, 16),
    genieclust_kdtree_4_32=lambda X, M: mst_genieclust_kdtree(X, M, 4, 32),
    #genieclust_kdtree_4_64=lambda X, M: mst_genieclust_kdtree(X, M, 4, 64),
    #genieclust_kdtree_8_32=lambda X, M: mst_genieclust_kdtree(X, M, 8, 32),
    genieclust_brute=lambda X, M: mst_genieclust_brute(X, M),
    mlpack_1=lambda X, M: mst_mlpack(X, M, 1),
    fasthdbscan_kdtree=lambda X, M: mst_fasthdbscan_kdtree(X, M),
    hdbscan_kdtree_40_3=lambda X, M: mst_hdbscan_kdtree(X, M, 40, 3),
    #mlpack_4=lambda X, M: mst_mlpack(X, M, 4),
)


numba.set_num_threads(n_jobs)
n = 250_000
for n, d, M in [(n, 2, 1), (n, 5, 1), (n, 2, 10), (n, 5, 10)]:
    np.random.seed(123)
    X = np.random.randn(n, d)
    print("n=%d, d=%d, M=%d, threads=%d" % (X.shape[0], X.shape[1], M, n_jobs))

    # preflight (e.g., for fast_hdbscan)
    for name, generator in cases.items():
        generator(X[:100, :].copy(), M)

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
