import os
import numba
n_jobs = 6
os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)
numba.config.THREADING_LAYER = 'omp'



import numpy as np
import hdbscan
from sklearn.neighbors import KDTree, BallTree
from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
import timeit
import genieclust
import os
import fast_hdbscan

np.random.seed(123)
X = np.random.randn(1_000_000, 2)
M = 1

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

print("n=%d, d=%d, M=%d, threads=%d" % (X.shape[0], X.shape[1], M, n_jobs))



numba.set_num_threads(n_jobs)
fast_hdbscan.hdbscan.compute_minimum_spanning_tree(X[:100,:], min_samples=M-1)  # numba...
t0 = timeit.time.time()
g = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(X, min_samples=M-1)
t1 = timeit.time.time()
#tot = g[0][:, 2].sum()
i1 = g[0][:, 0].astype(int)
i2 = g[0][:, 1].astype(int)
tot = np.sum(np.maximum(np.maximum(g[2][i1], g[2][i2]), np.sqrt(np.sum((X[i1,:]-X[i2,:])**2, axis=1))))
print("fast_hdbscan:     %15.5f %15.5f" % (t1-t0, tot))


def hdbscan_kdtree(X, M):
    tree = KDTree(X, metric='euclidean', leaf_size=40)
    alg = KDTreeBoruvkaAlgorithm(
        tree,
        min_samples=M-1,
        metric='euclidean',
        leaf_size=40 // 3,
        approx_min_span_tree=False,
        n_jobs=n_jobs
    )
    return alg.spanning_tree()


hdbscan_kdtree(X[:100,:], M)  # numba...
t0 = timeit.time.time()
min_spanning_tree = hdbscan_kdtree(X, M)
t1 = timeit.time.time()
print("hdbscan_kdtree:   %15.5f %15.5f" % (t1-t0, sum(min_spanning_tree.T[2])))


# slower than KD-trees (much slower)
# t0 = timeit.time.time()
# tree = BallTree(X, metric='euclidean', leaf_size=40)
# alg = BallTreeBoruvkaAlgorithm(
#     tree,
#     min_samples=M-1,
#     metric='euclidean',
#     leaf_size=40 // 3,
#     approx_min_span_tree=False,
#     n_jobs=n_jobs
# )
# min_spanning_tree = alg.spanning_tree()
# t1 = timeit.time.time()
# print("hdbscan_balltree: %15.5f %15.5f" % (t1-t0, sum(alg.spanning_tree().T[2])))






if M == 1 and X.shape[1] <= 10:
    t0 = timeit.time.time()
    g = genieclust.Genie(n_clusters=1, gini_threshold=1.0, compute_full_tree=False, mlpack_enabled=True, M=M).fit(X)
    t1 = timeit.time.time()
    print("Genie_mlpack:     %15.5f %15.5f" % (t1-t0, sum(g._tree_w)))


t0 = timeit.time.time()
g = genieclust.Genie(n_clusters=1, gini_threshold=1.0, compute_full_tree=False, mlpack_enabled=False, M=M).fit(X)
t1 = timeit.time.time()
print("Genie_brute:      %15.5f %15.5f" % (t1-t0, sum(g._tree_w)))

