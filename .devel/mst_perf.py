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
# mlpack's source distribution is not available from PyPI
CPPFLAGS="-O3 -march=native" pip3 install hdbscan --force --no-binary="hdbscan" --verbose
CPPFLAGS="-O3 -march=native" pip3 install fast_hdbscan --force --no-binary="fast_hdbscan" --verbose  # relies on numba, which forces -O3 -march=native anyway
CPPFLAGS="-O3 -march=native" pip3 install ~/Python/genieclust --force --verbose
pip3 install numpy==2.2.6  # for numba


hades @ 2025-06-12
n=100000, d=2, M=1, threads=1
              genieclust_brute: t=        8.81032
        genieclust_kdtree_4_16: t=        0.11707
           hdbscan_kdtree_40_3: t=        1.68920
            fasthdbscan_kdtree: t=        1.30274
                      mlpack_1: t=        0.56649
                      mlpack_4: t=        0.39744
n=100000, d=5, M=1, threads=1
              genieclust_brute: t=       13.52134
        genieclust_kdtree_4_16: t=        2.17226
           hdbscan_kdtree_40_3: t=       15.17135
            fasthdbscan_kdtree: t=        4.26879
                      mlpack_1: t=        5.38789
                      mlpack_4: t=        5.69942
n=100000, d=2, M=10, threads=1
              genieclust_brute: t=       19.19615
        genieclust_kdtree_4_16: t=        0.13406
           hdbscan_kdtree_40_3: t=        1.52790
            fasthdbscan_kdtree: t=        0.65928
n=100000, d=5, M=10, threads=1
              genieclust_brute: t=       29.40015
        genieclust_kdtree_4_16: t=        2.10061
           hdbscan_kdtree_40_3: t=        6.80395
            fasthdbscan_kdtree: t=        1.32822



apollo < 2025-06-15                                 : vs 2025-06-15 noon  : vs 2025-06-15 evening -03 -march=native
n=250000, d=2, M=1, threads=1
        genieclust_kdtree_4_32: t=        0.30752   : t=        0.29868  : t=        0.29576 Δdist=        0.00000 Δind=         0
                      mlpack_1: t=        1.48554   : t=        1.48109  : t=        1.47330 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        4.08379   : t=        4.08067  : t=        3.97738 Δdist=        0.00000 Δind=         0
           hdbscan_kdtree_40_3: t=        5.86782   : t=        5.86651  : t=        5.73835 Δdist=        0.00000 Δind=         0
n=250000, d=5, M=1, threads=1
        genieclust_kdtree_4_32: t=        5.78663   : t=        5.81715  : t=        4.59237 Δdist=        0.00000 Δind=         0
                      mlpack_1: t=       14.37002   : t=       14.60630  : t=       14.04031 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=       20.62158   : t=       20.63816  : t=       20.47972 Δdist=        0.00000 Δind=         0
           hdbscan_kdtree_40_3: t=       36.97692   : t=       37.25075  : t=       37.40961 Δdist=        0.00000 Δind=         0
n=250000, d=2, M=10, threads=1
        genieclust_kdtree_4_32: t=        0.36002   : t=        0.49767  : t=        0.48485 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        2.25570   : t=        2.28296  : t=        2.28520 Δdist=        0.00000 Δind=     33794
           hdbscan_kdtree_40_3: t=        6.06295   : t=        6.01803  : t=        5.86769 Δdist=        0.00000 Δind=     33601
n=250000, d=5, M=10, threads=1
        genieclust_kdtree_4_32: t=        5.42228   : t=        3.77974  : t=        3.07490 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        6.56826   : t=        6.41720  : t=        6.58253 Δdist=        0.00000 Δind=     17974
           hdbscan_kdtree_40_3: t=       28.73878   : t=       27.55924  : t=       27.58585 Δdist=        0.00000 Δind=     17897


n=100000, d=2, M=1, threads=1
        genieclust_kdtree_4_32: t=        0.11772  : t=        0.11488    : t=        0.11232 Δdist=        0.00000 Δind=         0
              genieclust_brute: t=        8.71274  : t=        8.70426    : t=        8.98587 Δdist=        0.00000 Δind=         0
                      mlpack_1: t=        0.56442  : t=        0.56202    : t=        0.53602 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        1.30298  : t=        1.30041    : t=        1.36775 Δdist=        0.00000 Δind=         0
           hdbscan_kdtree_40_3: t=        1.67903  : t=        1.67868    : t=        1.57425 Δdist=        0.00000 Δind=         0
n=100000, d=5, M=1, threads=1
        genieclust_kdtree_4_32: t=        2.14303  : t=        2.14580    : t=        1.68788 Δdist=        0.00000 Δind=         0
              genieclust_brute: t=       13.41931  : t=       13.64711    : t=       11.73230 Δdist=        0.00000 Δind=         0
                      mlpack_1: t=        5.29734  : t=        5.32112    : t=        5.09739 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        4.30194  : t=        4.26212    : t=        4.38575 Δdist=        0.00000 Δind=         0
           hdbscan_kdtree_40_3: t=       15.14476  : t=       15.13626    : t=       13.41334 Δdist=        0.00000 Δind=         0
n=100000, d=2, M=10, threads=1
        genieclust_kdtree_4_32: t=        0.13555  : t=        0.16070    : t=        0.15744 Δdist=        0.00000 Δind=         0
              genieclust_brute: t=       18.80316  : t=       19.47750    : t=       20.12245 Δdist=        0.00000 Δind=         0
            fasthdbscan_kdtree: t=        0.65381  : t=        0.66013    : t=        0.65411 Δdist=        0.00000 Δind=     13397
           hdbscan_kdtree_40_3: t=        1.53040  : t=        1.52787    : t=        1.47518 Δdist=        0.00000 Δind=     13245
n=100000, d=5, M=10, threads=1
        genieclust_kdtree_4_32: t=        1.97062  : t=        1.49411    : t=        1.21564 Δdist=        0.00000 Δind=         0
              genieclust_brute: t=       28.00334  : t=       28.54355    : t=       27.26717 Δdist=        0.00000 Δind=        18
            fasthdbscan_kdtree: t=        1.31027  : t=        1.30258    : t=        1.35008 Δdist=       -0.00000 Δind=      6625
           hdbscan_kdtree_40_3: t=        6.76528  : t=        6.71068    : t=        6.67515 Δdist=        0.00000 Δind=      6624


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
