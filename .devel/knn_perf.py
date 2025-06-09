import os
import numba
import numpy as np
import hdbscan
import timeit
import genieclust
import os





k = 10

n_jobs = 1
os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)
numba.config.THREADING_LAYER = 'omp'






"""
# apollo @ 2025-06-08 14:00:
n=1000000, d=2, k=10, threads=1
knn_sqeuclid_kdtree(32):          0.83525     61526.08001
knn_sqeuclid_picotree(16):        1.77205     61526.08001
fast_hdbscan:                     3.79096     61526.06641
n=1000000, d=5, k=10, threads=1
knn_sqeuclid_kdtree(32):          7.11416   2627958.13058
knn_sqeuclid_picotree(16):       14.97781   2627958.13058
fast_hdbscan:                    39.01639   2627957.25000

# hades @ 2025-06-09 10:30
n=1000000, d=2, k=10, threads=1
knn_sqeuclid_kdtree(32):          0.80502     61526.08001
fast_hdbscan:                     3.52201     61526.06641
n=1000000, d=5, k=10, threads=1
knn_sqeuclid_kdtree(32):          6.96646   2627958.13058
fast_hdbscan:                    32.65678   2627957.25000


# hades < 2025-06-08:
n=1000000, d=2, k=10, threads=1
knn_sqeuclid_kdtree:          0.88863     61526.08001
knn_sqeuclid_picotree:        1.69198     61526.08001
fast_hdbscan:                 1.82479     61526.12109
mlpack:                       1.65162     61526.08001
sklearn kd_tree:              1.94353     61526.08001

n=1000000, d=5, k=10, threads=1
knn_sqeuclid_kdtree:              7.14020   2627958.13058
knn_sqeuclid_picotree:           13.27074   2627958.13058
fast_hdbscan:                    32.59343   2627957.25000



n=100000, d=5, k=10, threads=1
knn_sqeuclid_kdtree:          0.82372    413229.42665
knn_sqeuclid_picotree:        0.53758    413229.42665
mlpack:                       1.64399    413229.42665
sklearn kd_tree:              1.52883    413229.42665
fast_hdbscan:                 0.85333    413229.46875
knn_sqeuclid_brute:          15.71682    413229.42665
knn_from_distance:           23.35285    413229.42665

n=100000, d=10, k=10, threads=1
knn_sqeuclid_kdtree:         20.55747   1388110.84063
knn_sqeuclid_picotree:       10.39866   1388110.84063
mlpack:                      31.62153   1388110.84063
sklearn kd_tree:             25.05803   1388110.84063
fast_hdbscan:                14.29010   1388110.75000
knn_sqeuclid_brute:          26.84659   1388110.84063
knn_from_distance:           39.12568   1388110.84063

n=100000, d=10, k=10, threads=6
knn_sqeuclid_kdtree:          5.27976   1388110.84063
knn_sqeuclid_picotree:        2.89341   1388110.84063
mlpack:                      31.56054   1388110.84063
sklearn kd_tree:             34.20438   1388110.84063
fast_hdbscan:                 4.63993   1388110.75000
knn_sqeuclid_brute:          17.45798   1388110.84063
knn_from_distance:           24.46707   1388110.84063

"""

for d in [2, 5]:
    np.random.seed(123)
    X = np.random.randn(1000000, d)
    print("n=%d, d=%d, k=%d, threads=%d" % (X.shape[0], X.shape[1], k, n_jobs))


    t0 = timeit.time.time()
    nn_dist, nn_ind = genieclust.internal.knn_sqeuclid(X, k, use_kdtree=True)
    nn_dist = np.sqrt(nn_dist)
    t1 = timeit.time.time()
    tot = np.sum(nn_dist)
    print("knn_sqeuclid_kdtree(32):  %15.5f %15.5f" % (t1-t0, tot))



    # t0 = timeit.time.time()
    # nn_dist, nn_ind = genieclust.internal.knn_sqeuclid(X, k, use_kdtree=-1)
    # nn_dist = np.sqrt(nn_dist)
    # t1 = timeit.time.time()
    # tot = np.sum(nn_dist)
    # print("knn_sqeuclid_picotree(16):%15.5f %15.5f" % (t1-t0, tot))



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
    print("fast_hdbscan:             %15.5f %15.5f" % (t1-t0, tot))


    continue


    import mlpack
    t0 = timeit.time.time()
    output = mlpack.knn(reference=X, k=k)
    nn_dist, nn_ind = output['distances'], output['neighbors']
    t1 = timeit.time.time()
    tot = np.sum(nn_dist)
    print("mlpack:                   %15.5f %15.5f" % (t1-t0, tot))




    import sklearn.neighbors
    sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', n_jobs=n_jobs).fit(X[:100,:]).kneighbors(X[:100,:])
    t0 = timeit.time.time()
    nn_dist, nn_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', n_jobs=n_jobs).fit(X).kneighbors(X)
    nn_dist = nn_dist[:, 1:]
    nn_ind = nn_ind[:, 1:]
    t1 = timeit.time.time()
    tot = np.sum(nn_dist)
    print("sklearn kd_tree:          %15.5f %15.5f" % (t1-t0, tot))






    if X.shape[0] > 100_000: continue


    t0 = timeit.time.time()
    nn_dist, nn_ind = genieclust.internal.knn_sqeuclid(X, k, use_kdtree=0)
    nn_dist = np.sqrt(nn_dist)
    t1 = timeit.time.time()
    tot = np.sum(nn_dist)
    print("knn_sqeuclid_brute:       %15.5f %15.5f" % (t1-t0, tot))


    t0 = timeit.time.time()
    nn_dist, nn_ind = genieclust.internal.knn_from_distance(X, k)
    t1 = timeit.time.time()
    tot = np.sum(nn_dist)
    print("knn_from_distance:        %15.5f %15.5f" % (t1-t0, tot))


stop()




import sklearn.neighbors
sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(X[:100,:]).kneighbors(X[:100,:])
t0 = timeit.time.time()
nn_dist, nn_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(X).kneighbors(X)
nn_dist = nn_dist[:, 1:]
nn_ind = nn_ind[:, 1:]
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("sklearn ball_tree:        %15.5f %15.5f" % (t1-t0, tot))

import sklearn.neighbors
sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='brute', n_jobs=n_jobs).fit(X[:100,:]).kneighbors(X[:100,:])
t0 = timeit.time.time()
nn_dist, nn_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='brute', n_jobs=n_jobs).fit(X).kneighbors(X)
nn_dist = nn_dist[:, 1:]
nn_ind = nn_ind[:, 1:]
t1 = timeit.time.time()
tot = np.sum(nn_dist)
print("sklearn brute:            %15.5f %15.5f" % (t1-t0, tot))

stop()



fast_hdbscan.hdbscan.compute_minimum_spanning_tree(X[:100,:], min_samples=M-1)  # numba...
t0 = timeit.time.time()
g = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(X, min_samples=M-1)
t1 = timeit.time.time()
#tot = g[0][:, 2].sum()
i1 = g[0][:, 0].astype(int)
i2 = g[0][:, 1].astype(int)
tot = np.sum(np.maximum(np.maximum(g[2][i1], g[2][i2]), np.sqrt(np.sum((X[i1,:]-X[i2,:])**2, axis=1))))
print("fast_hdbscan:         %15.5f %15.5f" % (t1-t0, tot))


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
print("hdbscan_kdtree:       %15.5f %15.5f" % (t1-t0, sum(min_spanning_tree.T[2])))


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
    print("Genie_mlpack:         %15.5f %15.5f" % (t1-t0, sum(g._tree_w)))


t0 = timeit.time.time()
g = genieclust.Genie(n_clusters=1, gini_threshold=1.0, compute_full_tree=False, mlpack_enabled=False, M=M).fit(X)
t1 = timeit.time.time()
print("Genie_brute:          %15.5f %15.5f" % (t1-t0, sum(g._tree_w)))

