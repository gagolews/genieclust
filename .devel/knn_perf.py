import os
import numba
n_jobs = 6
os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)
numba.config.THREADING_LAYER = 'omp'



import numpy as np
import hdbscan
import timeit
import genieclust
import os

np.random.seed(123)
X = np.random.randn(100000, 10)
k = 10


# sanity check
nn_dist1, nn_ind1 = genieclust.internal.knn_sqeuclid(X[:100], 3, use_kdtree=True)
nn_dist2, nn_ind2 = genieclust.internal.knn_sqeuclid(X[:100], 3, use_kdtree=False)

assert(np.allclose(nn_dist1, nn_dist2))
assert(np.all(nn_ind1 == nn_ind2))





"""
n=100000, d=2, k=10, threads=6
knn_sqeuclid_kdtree:          0.03269     19309.87145
knn_sqeuclid_brute:           6.82671     19309.87146
knn_from_distance:           12.21293     19309.87146
sklearn kd_tree:              0.24980     19309.87146
fast_hdbscan:                 0.08017     19309.87305

n=1000000, d=2, k=10, threads=6
knn_sqeuclid_kdtree:          0.42556     61526.07995
sklearn kd_tree:              2.58029     61526.08001
fast_hdbscan:                 1.28486     61526.06641

n=100000, d=5, k=10, threads=6
knn_sqeuclid_kdtree:          0.15799    413229.42664
knn_sqeuclid_brute:          10.13204    413229.42665
knn_from_distance:           13.57109    413229.42665
sklearn kd_tree:              1.80433    413229.42665
fast_hdbscan:                 0.36980    413229.37500

n=100000, d=10, k=10, threads=6
knn_sqeuclid_kdtree:          2.84914   1388110.84081
knn_sqeuclid_brute:          12.43359   1388110.84063
knn_from_distance:           24.79300   1388110.84063
sklearn kd_tree:             33.87226   1388110.84063
fast_hdbscan:                 7.24239   1388111.00000


n=100000, d=15, k=10, threads=6
knn_sqeuclid_kdtree:         18.31453   2285870.43865
knn_sqeuclid_brute:          28.91768   2285870.43875
knn_from_distance:           34.49114   2285870.43875
sklearn kd_tree:            159.01136   2285870.43875
fast_hdbscan:                55.34826   2285870.75000


"""

print("n=%d, d=%d, k=%d, threads=%d" % (X.shape[0], X.shape[1], k, n_jobs))


t0 = timeit.time.time()
nn_dist, nn_ind = genieclust.internal.knn_sqeuclid(X, k, use_kdtree=True)
nn_dist = np.sqrt(nn_dist)
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("knn_sqeuclid_kdtree:  %15.5f %15.5f" % (t1-t0, tot))


t0 = timeit.time.time()
nn_dist, nn_ind = genieclust.internal.knn_sqeuclid(X, k, use_kdtree=False)
nn_dist = np.sqrt(nn_dist)
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("knn_sqeuclid_brute:   %15.5f %15.5f" % (t1-t0, tot))


t0 = timeit.time.time()
nn_dist, nn_ind = genieclust.internal.knn_from_distance(X, k)
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("knn_from_distance:    %15.5f %15.5f" % (t1-t0, tot))


import sklearn.neighbors
sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', n_jobs=n_jobs).fit(X[:100,:]).kneighbors(X[:100,:])
t0 = timeit.time.time()
nn_dist, nn_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', n_jobs=n_jobs).fit(X).kneighbors(X)
nn_dist = nn_dist[:, 1:]
nn_ind = nn_ind[:, 1:]
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("sklearn kd_tree:      %15.5f %15.5f" % (t1-t0, tot))




import sklearn.neighbors
import fast_hdbscan
import numba
numba.set_num_threads(n_jobs)
fast_hdbscan.numba_kdtree.parallel_tree_query(fast_hdbscan.hdbscan.kdtree_to_numba(sklearn.neighbors.KDTree(X[:100,:])), X[:100,:], k+1)
t0 = timeit.time.time()
numba_tree = fast_hdbscan.hdbscan.kdtree_to_numba(sklearn.neighbors.KDTree(X))
nn_dist, nn_ind = fast_hdbscan.numba_kdtree.parallel_tree_query(numba_tree, X, k+1)
nn_dist = nn_dist[:, 1:]
nn_ind = nn_ind[:, 1:]
tot = np.sum(nn_dist)
t1 = timeit.time.time()
print("fast_hdbscan:         %15.5f %15.5f" % (t1-t0, tot))

stop()




import sklearn.neighbors
sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(X[:100,:]).kneighbors(X[:100,:])
t0 = timeit.time.time()
nn_dist, nn_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(X).kneighbors(X)
nn_dist = nn_dist[:, 1:]
nn_ind = nn_ind[:, 1:]
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("sklearn ball_tree:    %15.5f %15.5f" % (t1-t0, tot))

import sklearn.neighbors
sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='brute', n_jobs=n_jobs).fit(X[:100,:]).kneighbors(X[:100,:])
t0 = timeit.time.time()
nn_dist, nn_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='brute', n_jobs=n_jobs).fit(X).kneighbors(X)
nn_dist = nn_dist[:, 1:]
nn_ind = nn_ind[:, 1:]
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("sklearn brute:        %15.5f %15.5f" % (t1-t0, tot))

stop()



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

