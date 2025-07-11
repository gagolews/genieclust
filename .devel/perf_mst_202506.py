n_jobs = 1
n_trials = 1
seed = 123

"""
CPPFLAGS="-O3 -march=native" pip3 install fast_hdbscan --force --no-binary="fast_hdbscan" --verbose  # relies on numba, which forces -O3 -march=native anyway
CPPFLAGS="-O3 -march=native" pip3 install pykdtree --force --no-binary="pykdtree" --verbose
CPPFLAGS="-O3 -march=native" pip3 install numpy==2.2.6  --no-binary="numpy"  --ignore-installed # for numba
CPPFLAGS="-O3 -march=native" pip3 install ~/Python/genieclust --force --verbose
CPPFLAGS="-O3 -march=native" CXX_DEFS="-O3 -march=native" Rscript -e 'install.packages(c("RANN", "Rnanoflann", "dbscan", "nabor", "reticulate", "mlpack"))'
# mlpack's source distribution is not available from PyPI
"""


n = 2**16
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
    # (1208592, -3, 10,  "thermogauss_scan001"),
    # (1208592, 3, 10,  "norm"),
    # (1208592, -3,  1,  "thermogauss_scan001"),
    # (1208592, 3, 1,  "norm"),
    # (n, 2, 1, "norm"),
    (n, 2, 10, "norm"),
    # (n, 5, 1, "norm"),
    (n, 5, 10, "norm"),
    # (1208592,  2,  1,  "norm"),
    # (1208592,  2, 10,  "norm"),
    # (1208592,  3,  1,  "norm"),
    # (1208592,  3, 10,  "norm"),
    # (1208592,  5,  1,  "norm"),
    # (1208592,  5, 10,  "norm"),
    # (1208592,  10,  1,  "norm"),
    # (1208592,  10, 10,  "norm"),
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
import genieclust

start_time = int(time.time())
hostname = os.uname()[1]
ofname = "/home/gagolews/Python/genieclust/.devel/perf_mst_202506-%s.csv" % (hostname, )

# import os.path
# if os.path.isfile(ofname): raise Exception("file exists")


if n_jobs > 0:
    os.environ["OMP_NUM_THREADS"]    = str(n_jobs)
    os.environ["PARLAY_NUM_THREADS"] = str(n_jobs)
    os.environ["NUMBA_NUM_THREADS"]  = str(n_jobs)

os.environ["COLUMNS"] = "200"  # output width, in characters
np.set_printoptions(
    linewidth=200,   # output width
    legacy="1.25",   # print scalars without type information
)
pd.set_option("display.width", 200)






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

import perf_mst_202506_defs as msts

cases = dict(
    quitefast_kdtree_single     = lambda X, M: msts.mst_quitefast_kdtree_single(X, M),
    # quitefast_kdtree_single2    = lambda X, M: msts.mst_quitefast_kdtree_single(X, M, mutreach_adj=-0.00000011920928955078125),
    # quitefast_kdtree_single4    = lambda X, M: msts.mst_quitefast_kdtree_single(X, M, mutreach_adj=+0.00000011920928955078125),
    # quitefast_kdtree_single5    = lambda X, M: msts.mst_quitefast_kdtree_single(X, M, mutreach_adj=+1.00000011920928955078125),
    quitefast_kdtree_dual       = lambda X, M: msts.mst_quitefast_kdtree_dual(X, M),
    quitefast_brute             = lambda X, M: msts.mst_quitefast_brute(X, M),
    # mlpack                     = lambda X, M: msts.mst_mlpack(X, M),
    # wangyiqiu                  = lambda X, M: msts.mst_wangyiqiu(X, M),
    # fasthdbscan_kdtree         = lambda X, M: msts.mst_fasthdbscan_kdtree(X, M),
    # hdbscan_kdtree             = lambda X, M: msts.mst_hdbscan_kdtree(X, M),
    # r_mlpack                   = lambda X, M: msts.mst_r_mlpack(X, M),
    # r_quitefast_default        = lambda X, M: msts.mst_r_quitefast_default(X, M),
)


def tree_order(tree_w, tree_e):
    tree_w = tree_w.astype("float", order="C")
    tree_e = tree_e.astype(np.intp, order="C")
    return genieclust.fastmst.tree_order(tree_w, tree_e)



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
            if _res is None: continue

            if len(_res) == 3:
                t1 = t0 + _res[2]  # own time measurement
                _res = _res[:2]
            else:
                t1 = timeit.time.time()

            if _res_ref is None: _res_ref = _res
            _res = tree_order(*_res)
            nleaves = np.sum(np.unique(_res[1], return_counts=True)[1]==1)
            print("%30s: t=%15.5f Δdist=%15.5f Δind=%10.0f nleaves=%8d" % (
                case,
                t1-t0,
                np.sum(_res[0])-np.sum(_res_ref[0]),
                np.sum(_res[1] != _res_ref[1]),
                nleaves,
            ))
            results.append(dict(
                method=case,
                elapsed=t1-t0,
                Δdist=np.sum(_res[0])-np.sum(_res_ref[0]),
                Σdist=np.sum(_res[0]),
                Δidx=np.sum(_res[1] != _res_ref[1]),
                nleaves=nleaves,
                n=n,
                d=d,
                M=M,
                s=s,
                nthreads=n_jobs,
                trial=_trial,
                seed=seed,
                time=start_time,
                host=hostname,
            ))

        pd.DataFrame(results).to_csv(ofname, index=False, mode="a", header=False)

