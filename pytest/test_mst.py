import numpy as np
from genieclust.mst import *
from genieclust.mst2 import *
import time
import gc
import sklearn.neighbors
import scipy.spatial.distance


def check_MST(X):
    n = X.shape[0]
    n_neighbors = n-1
    kwargs = dict(algorithm='auto')
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, **kwargs)
    nn.fit(X)
    dist, ind = nn.kneighbors()
    mst_i, mst_d = MST_pair2(dist, ind)

    dist_complete = scipy.spatial.distance.pdist(X)
    dist_complete = scipy.spatial.distance.squareform(dist_complete)
    mst_i2, mst_d2 = MST_pair(dist_complete)

    assert np.all(mst_i == mst_i2)
    assert np.allclose(mst_d, mst_d2)
    return True




def test_MST():
    path = "benchmark_data"
    for dataset in ["pathbased", "s1", "a1", "Aggregation", "WUT_Smile", "unbalance"]: #, "bigger"
        if dataset == "bigger":
            np.random.seed(123)
            n = 25000
            X = np.random.normal(size=(n,2))
        else:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)

        # center X + scale (NOT: standardize!)
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)

        print(dataset)
        check_MST(X)


if __name__ == "__main__":
    test_MST()
