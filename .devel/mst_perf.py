import os
import numba
import numpy as np
import timeit
import genieclust
import pandas as pd


n_jobs = 6
n_trials = 3
seed = 123
n = 100_000
scenarios = [
    # (n, 2, 1,  "pareto(2)"),
    # (n, 5, 1,  "pareto(2)"),
    # (n, 2, 10, "pareto(2)"),
    # (n, 5, 10, "pareto(2)"),
    # (n, 2, 1,  "gumbel(2)+pareto(2)"),
    # (n, 5, 1,  "gumbel(2)+pareto(2)"),
    # (n, 2, 10, "gumbel(2)+pareto(2)"),
    # (n, 5, 10, "gumbel(2)+pareto(2)"),
    (n, 2, 1,  "norm"),
    (n, 5, 1,  "norm"),
    (n, 2, 10, "norm"),
    (n, 5, 10, "norm"),
]



import os
os.environ["COLUMNS"] = "200"  # output width, in characters
np.set_printoptions(
    linewidth=200,   # output width
    legacy="1.25",   # print scalars without type information
)
pd.set_option("display.width", 200)

os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)
numba.config.THREADING_LAYER = 'omp'




"""
# mlpack's source distribution is not available from PyPI
CPPFLAGS="-O3 -march=native" pip3 install hdbscan --force --no-binary="hdbscan" --verbose
CPPFLAGS="-O3 -march=native" pip3 install fast_hdbscan --force --no-binary="fast_hdbscan" --verbose  # relies on numba, which forces -O3 -march=native anyway
CPPFLAGS="-O3 -march=native" pip3 install ~/Python/genieclust --force --verbose
pip3 install numpy==2.2.6  # for numba


hades @ 2025-06-16
                                                                      t                             Δdist                Δind
                                                                    min    median       max           min           max   min   max
n      d M  s                   method                 n_jobs
100000 2 1  gumbel(2)+pareto(2) fasthdbscan_kdtree     1       1.216857  1.218676  1.220487  0.000000e+00  0.000000e+00     0     0
                                genieclust_kdtree_4_32 1       0.111024  0.111162  0.112967  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1       1.289986  1.292468  1.293783  0.000000e+00  0.000000e+00     0     0
                                genieclust_kdtree_4_32 1       0.108799  0.108824  0.112493  0.000000e+00  0.000000e+00     0     0
         10 gumbel(2)+pareto(2) fasthdbscan_kdtree     1       0.619859  0.622164  0.624991  0.000000e+00  0.000000e+00  1380  1380
                                genieclust_kdtree_4_32 1       0.168456  0.168917  0.169724  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1       0.670413  0.671304  0.675058  0.000000e+00  0.000000e+00  1443  1443
                                genieclust_kdtree_4_32 1       0.160311  0.160453  0.160921  0.000000e+00  0.000000e+00     0     0
       5 1  gumbel(2)+pareto(2) fasthdbscan_kdtree     1       2.852075  2.852771  2.857387  0.000000e+00  0.000000e+00     0     0
                                genieclust_kdtree_4_32 1       1.041254  1.042481  1.043532  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1       4.250383  4.254753  4.337570  0.000000e+00  0.000000e+00     0     0
                                genieclust_kdtree_4_32 1       1.686145  1.688906  1.690195  0.000000e+00  0.000000e+00     0     0
         10 gumbel(2)+pareto(2) fasthdbscan_kdtree     1       0.982452  0.982744  0.984426  0.000000e+00  0.000000e+00   457   457
                                genieclust_kdtree_4_32 1       0.773606  0.773924  0.775300  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1       1.289587  1.292168  1.293348 -6.111804e-10 -6.111804e-10   460   460
                                genieclust_kdtree_4_32 1       1.217172  1.217342  1.220304  0.000000e+00  0.000000e+00     0     0

                                                                       t                               Δdist                Δind
                                                                     min     median        max           min           max   min   max
n      d M  s                   method                 n_jobs
250000 2 1  gumbel(2)+pareto(2) fasthdbscan_kdtree     1        3.618479   3.628970   3.638397  5.754373e-08  5.754373e-08    21    21
                                genieclust_kdtree_4_32 1        0.297264   0.297929   0.303596  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1        4.072036   4.091732   4.099964  0.000000e+00  0.000000e+00     0     0
                                genieclust_kdtree_4_32 1        0.281569   0.282983   0.285176  0.000000e+00  0.000000e+00     0     0
         10 gumbel(2)+pareto(2) fasthdbscan_kdtree     1        2.111336   2.116926   2.120986  3.580135e-08  3.580135e-08  3552  3552
                                genieclust_kdtree_4_32 1        0.481247   0.482215   0.482383  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1        2.304079   2.306762   2.314012  0.000000e+00  0.000000e+00  3567  3567
                                genieclust_kdtree_4_32 1        0.493228   0.493790   0.494704  0.000000e+00  0.000000e+00     0     0
       5 1  gumbel(2)+pareto(2) fasthdbscan_kdtree     1       12.615899  12.641380  12.940112  3.664172e-08  3.664172e-08     4     4
                                genieclust_kdtree_4_32 1        3.092354   3.102078   3.110531  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1       19.541083  19.550386  19.800673  0.000000e+00  0.000000e+00     0     0
                                genieclust_kdtree_4_32 1        4.593263   4.604975   4.605837  0.000000e+00  0.000000e+00     0     0
         10 gumbel(2)+pareto(2) fasthdbscan_kdtree     1        4.658862   4.676701   4.689571  0.000000e+00  0.000000e+00  1244  1244
                                genieclust_kdtree_4_32 1        2.563868   2.572786   2.576155  0.000000e+00  0.000000e+00     0     0
            norm                fasthdbscan_kdtree     1        5.366791   5.376925   5.401598  0.000000e+00  0.000000e+00  1220  1220
                                genieclust_kdtree_4_32 1        3.084048   3.092072   3.098270  0.000000e+00  0.000000e+00     0     0

                                                       t                     Δdist       Δind
                                                     min    median       max   min  max   min   max
n      d M  s    method                 n_jobs
250000 2 1  norm fasthdbscan_kdtree     6       1.221030  1.226814  1.294868   0.0  0.0     0     0
                 genieclust_kdtree_4_32 6       0.261044  0.265963  0.269273   0.0  0.0     0     0
         10 norm fasthdbscan_kdtree     6       0.716567  0.733907  0.806738   0.0  0.0  3592  3655
                 genieclust_kdtree_4_32 6       0.410073  0.416372  0.416443   0.0  0.0     0     0
       5 1  norm fasthdbscan_kdtree     6       5.192794  5.227586  5.259531   0.0  0.0     0     0
                 genieclust_kdtree_4_32 6       4.289630  4.289744  4.294656   0.0  0.0     0     0
         10 norm fasthdbscan_kdtree     6       1.711334  1.716666  1.744889   0.0  0.0  1200  1205
                 genieclust_kdtree_4_32 6       2.290573  2.311051  2.318365   0.0  0.0     0     0

                                                        t                               Δdist                Δind
                                                      min     median        max           min           max   min   max
n      d M  s    method                 n_jobs
100000 2 1  norm fasthdbscan_kdtree     6        0.430802   0.430883   0.431582  0.000000e+00  0.000000e+00     0     0
                 genieclust_brute       6        5.757471   5.891397   5.991485  0.000000e+00  0.000000e+00     0     0
                 genieclust_kdtree_4_32 6        0.099180   0.101255   0.101642  0.000000e+00  0.000000e+00     0     0
         10 norm fasthdbscan_kdtree     6        0.236824   0.241025   0.242452  0.000000e+00  0.000000e+00  1453  1477
                 genieclust_brute       6       14.029837  14.097681  14.118144  0.000000e+00  0.000000e+00     0     0
                 genieclust_kdtree_4_32 6        0.135898   0.138323   0.157606  0.000000e+00  0.000000e+00     0     0
       5 1  norm fasthdbscan_kdtree     6        1.304926   1.316546   1.682047  0.000000e+00  0.000000e+00     0     0
                 genieclust_brute       6        7.295061   7.295909   7.526759  0.000000e+00  0.000000e+00     0     0
                 genieclust_kdtree_4_32 6        1.573716   1.574136   1.598902  0.000000e+00  0.000000e+00     0     0
         10 norm fasthdbscan_kdtree     6        0.522237   0.527439   0.539022 -6.111804e-10 -6.111804e-10   466   476
                 genieclust_brute       6       18.267576  18.301994  18.366816  0.000000e+00  0.000000e+00     8     8
                 genieclust_kdtree_4_32 6        0.913481   0.937083   0.941864  0.000000e+00  0.000000e+00     0     0




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


def tree_order(tree_w, tree_e):
    tree_w = tree_w.astype("float", order="C")
    tree_e = tree_e.astype(np.intp, order="C")
    return genieclust.fastmst.tree_order(tree_w, tree_e)


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

    return (_res[:, 2], _res[:, :2])




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

    return (tree_w, tree_e)






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
    # mlpack_1=lambda X, M: mst_mlpack(X, M, 1),
    fasthdbscan_kdtree=lambda X, M: mst_fasthdbscan_kdtree(X, M),
    #hdbscan_kdtree_40_3=lambda X, M: mst_hdbscan_kdtree(X, M, 40, 3),
    #mlpack_4=lambda X, M: mst_mlpack(X, M, 4),
)



import statsmodels
import scipy.stats
from statsmodels.distributions.copula.api import GumbelCopula, CopulaDistribution


results = []
numba.set_num_threads(n_jobs)
for n, d, M, s in scenarios:
    np.random.seed(seed)
    if s == "norm":
        X = np.random.randn(n, d)
    elif s == "unif":
        X = np.random.rand(n, d)
    elif s == "pareto(2)":
        X = scipy.stats.pareto.rvs(2, size=(n, d), random_state=np.random.mtrand._rand)
    elif s == "gumbel(2)+pareto(2)":
        dist = CopulaDistribution(copula=GumbelCopula(theta=2, k_dim=d), marginals=[scipy.stats.pareto(2) for i in range(d)])
        X = dist.rvs(n, random_state=np.random.mtrand._rand)
    else:
        raise Exception("wrong 's'")

    print("n=%d, d=%d, M=%d, s=%s, threads=%d" % (X.shape[0], X.shape[1], M, s, n_jobs))

    # preflight (e.g., for fast_hdbscan)
    for name, generator in cases.items():
        generator(X[:100, :].copy(), M)

    for _trial in range(n_trials):
        _res_ref = None
        for case, generator in cases.items():
            t0 = timeit.time.time()
            _res = generator(X, M)
            t1 = timeit.time.time()
            if _res is None: continue
            if _res_ref is None: _res_ref = _res
            _res = tree_order(*_res)
            print("%30s: t=%15.5f Δdist=%15.5f Δind=%10.0f" % (
                case,
                t1-t0,
                np.sum(_res[0])-np.sum(_res_ref[0]),
                np.sum(_res[1] != _res_ref[1]),
            ))
            results.append(dict(
                method=case, n=n, d=d, M=M, s=s, n_jobs=n_jobs, trial=_trial,
                seed=seed,
                t=t1-t0,
                Δdist=np.sum(_res[0])-np.sum(_res_ref[0]),
                Δind=np.sum(_res[1] != _res_ref[1])
            ))

import pandas as pd
results = pd.DataFrame(results)

aggregates = results.groupby(["n", "d", "M", "s", "method", "n_jobs"]).agg(
    dict(
        t=["min", "median", "max"],
        Δdist=["min", "max"],
        Δind=["min", "max"],
    )
)
print(aggregates)
