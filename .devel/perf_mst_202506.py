n_jobs = 1
n_trials = 3
seed = 123

n = 2**15
scenarios = [
    # (n, 2, 1,  "pareto(2)"),
    # (n, 2, 2,  "pareto(2)"),
    # (n, 2, 10, "pareto(2)"),
    # (n, 5, 1,  "pareto(2)"),
    # (n, 2, 10, "pareto(2)"),
    # (n, 5, 10, "pareto(2)"),
    # (n, 2, 1,  "gumbel(2)+pareto(2)"),
    # (n, 5, 1,  "gumbel(2)+pareto(2)"),
    # (n, 2, 2,  "gumbel(2)+pareto(2)"),
    # (n, 5, 2,  "gumbel(2)+pareto(2)"),
    # (n, 2, 10, "gumbel(2)+pareto(2)"),
    # (n, 5, 10, "gumbel(2)+pareto(2)"),
    # (n, 2, 1, "norm"),
    # (n, 3, 1, "norm"),
    # (n, 5, 1, "norm"),
    # (n, 2, 2, "norm"),
    # (n, 3, 2, "norm"),
    # (n, 5, 2, "norm"),
    # (n, 2, 10, "norm"),
    # (n, 3, 10, "norm"),
    # (n, 5, 10, "norm"),
    # (1208592, -3,  1,  "thermogauss_scan001"),
    # (1208592, -3, 10,  "thermogauss_scan001"),
    # (1208592,  2,  1,  "norm"),
    # (1208592,  2, 10,  "norm"),
    # (1208592,  3,  1,  "norm"),
    # (1208592,  3, 10,  "norm"),
    # (1208592,  5,  1,  "norm"),
    # (1208592,  5, 10,  "norm"),
    # (1208592, 10,  1,  "norm"),
    (1208592, 10, 10,  "norm"),
]

# scenarios = []
# for d in range(2, 11):
#     for log2n in [17]:
#         scenarios.append( (2**log2n, d,  1, "norm") )
#         scenarios.append( (2**log2n, d, 10, "norm") )


# ------------------------------------------------------------------------------

import os
import numba
import numpy as np
import pandas as pd
import timeit
import time

hostname = os.uname()[1]
ofname = "/home/gagolews/Python/genieclust/.devel/perf_mst_202506-%s.csv" % (hostname, )

# import os.path
# if os.path.isfile(ofname): raise Exception("file exists")


if n_jobs > 0:
    os.environ["OMP_NUM_THREADS"] = str(n_jobs)
    os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)

os.environ["COLUMNS"] = "200"  # output width, in characters
np.set_printoptions(
    linewidth=200,   # output width
    legacy="1.25",   # print scalars without type information
)
pd.set_option("display.width", 200)


import hdbscan
from sklearn.neighbors import KDTree
from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
import fast_hdbscan
import genieclust
import mlpack


"""
CPPFLAGS="-O3 -march=native" pip3 install fast_hdbscan --force --no-binary="fast_hdbscan" --verbose  # relies on numba, which forces -O3 -march=native anyway
CPPFLAGS="-O3 -march=native" pip3 install pykdtree --force --no-binary="pykdtree" --verbose
CPPFLAGS="-O3 -march=native" pip3 install numpy==2.2.6  # for numba
CPPFLAGS="-O3 -march=native" pip3 install ~/Python/genieclust --force --verbose
# mlpack's source distribution is not available from PyPI
"""


import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
r_mlpack = importr("mlpack")
r_genieclust = importr("genieclust")



import importlib
modules = [
    'numba', 'cython', 'numpy', 'scipy', 'sklearn', 'pykdtree',
    'genieclust', 'mlpack', 'hdbscan', 'fast_hdbscan', 'rpy2'
]
for m in modules:
    try:
        print("%20s %s" % (m, importlib.import_module(m).__version__))
    except:
        print("%20s ?" %  (m, ))


# ------------------------------------------------------------------------------


def tree_order(tree_w, tree_e):
    tree_w = tree_w.astype("float", order="C")
    tree_e = tree_e.astype(np.intp, order="C")
    return genieclust.fastmst.tree_order(tree_w, tree_e)


def mst_r_mlpack(X, M, leaf_size=1):
    if M > 1 or n_jobs > 1:
        return None

    np_cv_rules = default_converter + numpy2ri.converter
    with np_cv_rules.context():
        _res = r_mlpack.emst(X)

    tree_w = _res[0][:,  2].astype(X.dtype, order="C")
    tree_e = _res[0][:, :2].astype(np.intp, order="C")
    return tree_w, tree_e


def mst_r_quitefast_default(X, M):
    np_cv_rules = default_converter + numpy2ri.converter
    with np_cv_rules.context():
        _res = r_genieclust.mst_euclid(X, M)
    tree_w, tree_e = _res[1], _res[0]-1
    return tree_w, tree_e



# BallTreeBoruvkaAlgorithm - much slower
def mst_hdbscan_kdtree(X, M, leaf_size=40, leaf_size_div=3):
    if X.shape[0] > 300000: return None
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


def mst_mlpack(X, M, leaf_size=1):
    if M > 1 or n_jobs > 1:
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


def mst_quitefast_brute(X, M):
    if X.shape[0] > 300000: return None
    res = genieclust.fastmst.mst_euclid(X, M, algorithm="brute")
    tree_w, tree_e = res[:2]
    return tree_w, tree_e


def mst_quitefast_kdtree_single(X, M, max_leaf_size=32, first_pass_max_brute_size=32, use_dtb=False):
    res = genieclust.fastmst.mst_euclid(
        X, M,
        algorithm="kd_tree_single",
        max_leaf_size=max_leaf_size,
        first_pass_max_brute_size=first_pass_max_brute_size
    )
    tree_w, tree_e = res[:2]
    return tree_w, tree_e


def mst_quitefast_kdtree_dual(X, M, max_leaf_size=8, first_pass_max_brute_size=32, use_dtb=False):
    res = genieclust.fastmst.mst_euclid(
        X, M,
        algorithm="kd_tree_dual",
        max_leaf_size=max_leaf_size,
        first_pass_max_brute_size=first_pass_max_brute_size
    )
    tree_w, tree_e = res[:2]
    return tree_w, tree_e


cases = dict(
    quitefast_kdtree_single   = lambda X, M: mst_quitefast_kdtree_single(X, M),
    quitefast_kdtree_dual     = lambda X, M: mst_quitefast_kdtree_dual(X, M),
    quitefast_brute           = lambda X, M: mst_quitefast_brute(X, M),
    mlpack                    = lambda X, M: mst_mlpack(X, M),
    fasthdbscan_kdtree        = lambda X, M: mst_fasthdbscan_kdtree(X, M),
    hdbscan_kdtree            = lambda X, M: mst_hdbscan_kdtree(X, M),
    r_mlpack                  = lambda X, M: mst_r_mlpack(X, M),
    r_quitefast_default       = lambda X, M: mst_r_quitefast_default(X, M),
)




if n_jobs > 0:
    numba.config.THREADING_LAYER = 'omp'
    numba.set_num_threads(n_jobs)
    genieclust.fastmst.omp_set_num_threads(n_jobs)
else:
    numba.set_num_threads(genieclust.omp_max_treads_original)
    genieclust.fastmst.omp_set_num_threads(genieclust.omp_max_treads_original)

for n, d, M, s in scenarios:
    np.random.seed(seed)
    if s == "norm":
        X = np.random.randn(n, d)
    elif s == "unif":
        X = np.random.rand(n, d)
    elif s == "pareto(2)":
        import scipy.stats
        X = scipy.stats.pareto.rvs(2, size=(n, d), random_state=np.random.mtrand._rand)
    elif s == "gumbel(2)+pareto(2)":
        from statsmodels.distributions.copula.api import GumbelCopula, CopulaDistribution
        dist = CopulaDistribution(copula=GumbelCopula(theta=2, k_dim=d), marginals=[scipy.stats.pareto(2) for i in range(d)])
        X = dist.rvs(n, random_state=np.random.mtrand._rand)
    elif s == "thermogauss_scan001":
        X = np.loadtxt("/home/gagolews/Python/genieclust/.devel/benchmark_data/thermogauss_scan001.3d.gz")
    else:
        raise Exception("wrong 's'")

    print("n=%d, d=%d, M=%d, s=%s, threads=%d" % (X.shape[0], X.shape[1], M, s, n_jobs))

    # preflight (e.g., for fast_hdbscan)
    for name, generator in cases.items():
        generator(X[:100, :].copy(), M)

    for _trial in range(1, n_trials+1):
        results = []
        np.random.seed(_trial)
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
                method=case,
                elapsed=t1-t0,
                Δdist=np.sum(_res[0])-np.sum(_res_ref[0]),
                Σdist=np.sum(_res[0]),
                Δidx=np.sum(_res[1] != _res_ref[1]),
                n=n,
                d=d,
                M=M,
                s=s,
                nthreads=n_jobs,
                trial=_trial,
                seed=seed,
                time=int(time.time()),
                host=hostname,
            ))

        pd.DataFrame(results).to_csv(ofname, index=False, mode="a", header=False)

