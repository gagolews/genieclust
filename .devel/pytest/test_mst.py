import numpy as np
import sklearn.neighbors
import scipy.spatial.distance
import time
import gc
import genieclust

import os
if os.path.exists(".devel/benchmark_data"):
    path = ".devel/benchmark_data"
elif os.path.exists("benchmark_data"):
    path = "benchmark_data"
else:
    path = "../benchmark_data"



# TODO: test fastmst......

def mst_check(X, metric='euclidean', **kwargs):
    n = X.shape[0]
    d = X.shape[1]

    n_neighbors = n-1

    t0 = time.time()
    dist_complete = scipy.spatial.distance.pdist(X, metric=metric)
    dist_complete = scipy.spatial.distance.squareform(dist_complete)
    mst_d, mst_i = genieclust.oldmst.mst_from_complete(dist_complete)
    print("    precomputed-matrix %10.3fs" % (time.time()-t0,))


    t0 = time.time()
    dist_complete = scipy.spatial.distance.pdist(X, metric=metric)
    mst_d1, mst_i1 = genieclust.oldmst.mst_from_complete(dist_complete.reshape(dist_complete.shape[0],-1))
    print("    precomputed-vector %10.3fs" % (time.time()-t0,))

    assert np.allclose(mst_d.sum(), mst_d1.sum())
    assert np.all(mst_i == mst_i1)
    assert np.allclose(mst_d, mst_d1)

    t0 = time.time()
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    nn.fit(X)
    dist, ind = nn.kneighbors()
    mst_d1, mst_i1 = genieclust.oldmst.mst_from_nn(dist, ind)
    print("    NearestNeighbors %10.3fs" % (time.time()-t0,))


    assert np.allclose(mst_d.sum(), mst_d1.sum())
    assert np.all(mst_i == mst_i1)
    assert np.allclose(mst_d, mst_d1)


    t0 = time.time()
    mst_d2, mst_i2 = genieclust.oldmst.mst_from_distance(X, metric=metric)
    print("    from_distance    %10.3fs" % (time.time()-t0,))

    assert np.allclose(mst_d.sum(), mst_d2.sum())
    assert np.all(mst_i == mst_i2)
    assert np.allclose(mst_d, mst_d2)


    if metric == 'euclidean':
        for algo in ["auto", "brute", "kd_tree_single", "kd_tree_dual"]:
            if d > 20 and algo in ["kd_tree_single", "kd_tree_dual"]:
                continue

            t0 = time.time()
            mst_d2, mst_i2 = genieclust.fastmst.mst_euclid(X, algorithm=algo)
            print("    fastmst_%s    %10.3fs" % (algo, time.time()-t0,))

            assert np.allclose(mst_d.sum(), mst_d2.sum())
            assert np.all(mst_i == mst_i2)
            assert np.allclose(mst_d, mst_d2)

    return True


def mst_mutreach_check(X, metric='euclidean'):
    n = X.shape[0]
    d = X.shape[1]

    D = scipy.spatial.distance.pdist(X, metric=metric)
    D = scipy.spatial.distance.squareform(D)

    for M in [2, 3, 5, n-1]:
        d_core     = genieclust.internal._core_distance(D, M)

        t0 = time.time()
        d_mutreach = genieclust.internal._mutual_reachability_distance(D, d_core)
        mst_d, mst_i = genieclust.oldmst.mst_from_complete(d_mutreach)
        print("    mutreach1-D(%d) %10.3fs" % (M, time.time()-t0,))

        t0 = time.time()
        mst_d2, mst_i2 = genieclust.oldmst.mst_from_distance(X, metric=metric,
            d_core=d_core)
        print("    mutreach2(%d)   %10.3fs" % (M, time.time()-t0,))

        assert np.allclose(mst_d.sum(), mst_d2.sum())
        #assert np.all(mst_i == mst_i2)  # mutreach-many duplicates - ambiguous
        assert np.allclose(mst_d, mst_d2)


        Dsorted = np.argsort(D, axis=1)
        assert np.all(Dsorted[:,0] == np.arange(n))

        nn_i1  = Dsorted[:, 1:M]
        nn_d1 = D[np.repeat(np.arange(n).reshape(-1,1), M-1, axis=1), nn_i1]

        if metric == 'euclidean':
            for algo in ["brute", "auto", "kd_tree_single", "kd_tree_dual"]:
                if d > 20 and algo in ["kd_tree_single", "kd_tree_dual"]:
                    continue

                t0 = time.time()
                mst_d2, mst_i2, nn_d2, nn_i2 = genieclust.fastmst.mst_euclid(X, M=M, algorithm=algo)
                print("    mutreach(%d)-fastmst_%s    %10.3fs" % (M, algo, time.time()-t0,))

                assert np.allclose(d_core, nn_d2[:,-1])
                assert np.allclose(mst_d.sum(), mst_d2.sum())
                #assert np.all(mst_i == mst_i2)  # mutreach-many duplicates - ambiguous
                assert np.allclose(mst_d, mst_d2)
                assert np.allclose(nn_d1, nn_d2)
                assert np.all(nn_i1==nn_i2)

    return True


def test_MST():
    for dataset in ["big_one", "pathbased", "h2mg_64_50"]:
        if dataset == "big_one":
            X =  np.random.rand(1000, 20)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)

        # center X + scale (NOT: standardize!)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)

        print(dataset)
        mst_check(X, metric="euclidean")
        mst_check(X, metric='cityblock')
        mst_check(X, metric='cosine')
        mst_mutreach_check(X, metric='euclidean')
        mst_mutreach_check(X, metric='cosine')
        gc.collect()

if __name__ == "__main__":
    test_MST()
