import numpy as np
import hdbscan
from sklearn.neighbors import KDTree, BallTree
from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
import timeit
import genieclust
import os

n_jobs = 6
np.random.seed(123)
X = np.random.randn(100000, 10)
os.environ["OMP_NUM_THREADS"] = str(n_jobs)


t0 = timeit.time.time()
g = genieclust.Genie(n_clusters=1, gini_threshold=1.0, compute_full_tree=True, mlpack_enabled=True).fit(X)
t1 = timeit.time.time()
print(t1-t0, sum(g.distances_))


t0 = timeit.time.time()
g = genieclust.Genie(n_clusters=1, gini_threshold=1.0, compute_full_tree=True, mlpack_enabled=False).fit(X)
t1 = timeit.time.time()
print(t1-t0, sum(g.distances_))




t0 = timeit.time.time()
tree = KDTree(X, metric='euclidean', leaf_size=40)
alg = KDTreeBoruvkaAlgorithm(
    tree,
    min_samples=5,
    metric='euclidean',
    leaf_size=40 // 3,
    approx_min_span_tree=False,
    n_jobs=n_jobs
)
min_spanning_tree = alg.spanning_tree()
t1 = timeit.time.time()
print(t1-t0, sum(alg.spanning_tree().T[2]))


t0 = timeit.time.time()
tree = BallTree(X, metric='euclidean', leaf_size=40)
alg = BallTreeBoruvkaAlgorithm(
    tree,
    min_samples=1,
    metric='euclidean',
    leaf_size=40 // 3,
    approx_min_span_tree=False,
    n_jobs=n_jobs
)
min_spanning_tree = alg.spanning_tree()
t1 = timeit.time.time()
print(t1-t0, sum(alg.spanning_tree().T[2]))

