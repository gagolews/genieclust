import numpy as np
import sklearn.neighbors
import scipy.spatial.distance
import time
import gc
import genieclust.internal
import genieclust.deprecated

def mst_check(X, metric='euclidean', **kwargs):
    n = X.shape[0]
    d = X.shape[1]
    n_neighbors = n-1

    t0 = time.time()
    dist_complete = scipy.spatial.distance.pdist(X, metric=metric)
    dist_complete = scipy.spatial.distance.squareform(dist_complete)
    mst_d, mst_i = genieclust.internal.mst_from_complete(dist_complete)
    print("    precomputed      %10.3fs" % (time.time()-t0,))


    t0 = time.time()
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    nn.fit(X)
    dist, ind = nn.kneighbors()
    mst_d1, mst_i1 = genieclust.internal.mst_from_nn(dist, ind)
    print("    NearestNeighbors %10.3fs" % (time.time()-t0,))

    assert np.allclose(mst_d.sum(), mst_d1.sum())
    assert np.all(mst_i == mst_i1)
    assert np.allclose(mst_d, mst_d1)


    t0 = time.time()
    mst_d2, mst_i2 = genieclust.internal.mst_from_distance(X, metric=metric)
    print("    from_distance    %10.3fs" % (time.time()-t0,))

    assert np.allclose(mst_d.sum(), mst_d2.sum())
    assert np.all(mst_i == mst_i2)
    assert np.allclose(mst_d, mst_d2)

    return True


def mst_mutreach_check(X, metric='euclidean'):
    n = X.shape[0]
    d = X.shape[1]

    D = scipy.spatial.distance.pdist(X, metric=metric)
    D = scipy.spatial.distance.squareform(D)

    for M in [2, 5, 25]:
        d_core     = genieclust.deprecated.core_distance(D, M)

        t0 = time.time()
        d_mutreach = genieclust.deprecated.mutual_reachability_distance(D, d_core)
        mst_d1, mst_i1 = genieclust.internal.mst_from_complete(d_mutreach)
        print("    mutreach1-D %10.3fs" % (time.time()-t0,))

        t0 = time.time()
        mst_d2, mst_i2 = genieclust.internal.mst_from_distance(X, metric=metric,
            metric_params=dict(d_core=d_core))
        print("    mutreach2   %10.3fs" % (time.time()-t0,))

        assert np.allclose(mst_d1.sum(), mst_d2.sum())
        #assert np.all(mst_i1 == mst_i2)   # mutreach dist - many duplicates
        assert np.allclose(mst_d1, mst_d2)

    return True


def test_MST():
    path = "benchmark_data"
    for dataset in ["pathbased", "h2mg_64_50"]:
        if dataset == "big_one":
            X =  np.random.rand(10_000, 1_00)
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)

        # center X + scale (NOT: standardize!)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)

        print(dataset)
        mst_check(X, algorithm='auto')
        mst_check(X, metric='cityblock')
        mst_check(X, metric='cosine')
        mst_mutreach_check(X, metric='cosine')
        gc.collect()

if __name__ == "__main__":
    test_MST()
