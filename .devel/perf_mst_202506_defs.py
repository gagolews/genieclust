import numpy as np
import numba
import hdbscan
from sklearn.neighbors import KDTree
from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
import fast_hdbscan
import genieclust
import mlpack
import subprocess
import re


max_n_slow_methods = 300_000
max_n_medium_methods = 1_000_000
max_n_brute = 300_000

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
r_mlpack = importr("mlpack")
r_genieclust = importr("genieclust")





# # not as fast as ours, lacking Python interface, a newer version does not build
# # see mst_wangyiqiu
# def mst_pargeo(X, M):
#     if M > 1: return None
#     np.savetxt("/tmp/input.numpy", X)
#     subprocess.run([
#         "/home/gagolews/Python/genieclust/.devel/wangyiqiu_pargeo/build/executable/emst", "-o", "/tmp/output.pargeo", "/tmp/input.numpy"], capture_output=True, env=None, check=True)
#     tree_e = np.genfromtxt("/tmp/output.pargeo", dtype=int)
#     tree_w = np.sqrt(
#                 np.sum((X[tree_e[:,0],:]-X[tree_e[:,1],:])**2, axis=1)
#             )
#     return tree_w, tree_e


# forcing this to work required a bit of hackery...
# edit compiler flags in flags.make manually, add -O3 -march=native
def mst_wangyiqiu(X, M):
    np.savetxt("/tmp/input.numpy", X)
    out = subprocess.run([
        "/home/gagolews/Python/genieclust/.devel/wangyiqiu_hdbscan/build/src/hdbscan", "-o", "/tmp/output.wangyiqiu", "-m", str(max(1, M)), "/tmp/input.numpy"], capture_output=True, env=None, check=True)
    t = float(re.search("mst-total-time = (.*)", out.stdout.decode("utf-8")).group(1))
    res = np.loadtxt("/tmp/output.wangyiqiu")
    return (
        res[:, 2].astype("float", order="C"),
        res[:,:2].astype(np.intp, order="C"),
        t
    )


def mst_r_mlpack(X, M, leaf_size=1):
    if M > 1 or n_jobs > 1:
        return None

    if X.shape[0] > max_n_medium_methods:
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
    if X.shape[0] > max_n_slow_methods: return None
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

    if X.shape[0] > max_n_medium_methods:
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


def mst_quitefast_brute(X, M, **kwargs):
    if X.shape[0] > max_n_brute: return None
    res = genieclust.fastmst.mst_euclid(X, M, algorithm="brute", **kwargs)
    tree_w, tree_e = res[:2]
    return tree_w, tree_e


def mst_quitefast_kdtree_single(X, M, **kwargs):
    res = genieclust.fastmst.mst_euclid(
        X, M,
        algorithm="kd_tree_single",
        **kwargs
    )
    tree_w, tree_e = res[:2]
    return tree_w, tree_e


def mst_quitefast_kdtree_dual(X, M, **kwargs):
    res = genieclust.fastmst.mst_euclid(
        X, M,
        algorithm="kd_tree_dual",
        **kwargs
    )
    tree_w, tree_e = res[:2]
    return tree_w, tree_e

