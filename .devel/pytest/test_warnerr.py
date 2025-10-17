import numpy as np
import genieclust
import time
import gc
import pytest

import scipy.spatial.distance
import numpy as np

import os
if os.path.exists("devel/benchmark_data"):
    path = "devel/benchmark_data"
elif os.path.exists("benchmark_data"):
    path = "benchmark_data"
else:
    path = "../benchmark_data"




def test_warnerr(metric='euclidean'):
    np.random.seed(123)
    n = 1_000
    d = 10
    K = 2
    X = np.random.normal(size=(n,d))
    labels = np.random.choice(np.r_[0:K], n)

    k = len(np.unique(labels[labels>=0]))

    # center X + scale (NOT: standardize!)
    X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
    X += np.random.normal(0, 0.0001, X.shape)


    with pytest.raises(Exception): genieclust.Genie(n_clusters=-1).fit(X)
    with pytest.raises(Exception): genieclust.Genie(gini_threshold=-1e-12).fit(X)
    with pytest.raises(Exception): genieclust.Genie(gini_threshold=1+1e-12).fit(X)
    with pytest.raises(Exception): genieclust.Genie(metric="euclidianne").fit(X)
    with pytest.raises(Exception): genieclust.Genie(metric="precomputed").fit(X)
    with pytest.raises(Exception): genieclust.Genie(M=-1).fit(X)
    with pytest.raises(Exception): genieclust.Genie(M=n).fit(X)
    with pytest.raises(Exception): genieclust.Genie(postprocess="say what??").fit(X)
    # with pytest.raises(Exception): genieclust.Genie(mlpack_enabled="say what??").fit(X)

    with pytest.raises(Exception): genieclust.GIc(add_clusters=-1).fit(X)
    with pytest.raises(Exception): genieclust.GIc(gini_thresholds=[-1e-12]).fit(X)
    with pytest.raises(Exception): genieclust.GIc(metric="precomputed").fit(scipy.spatial.distance.pdist(X))

    with pytest.warns(Warning): genieclust.Genie(M=1, compute_full_tree=True).fit(X)

    # if mlpack is None:
    #     with pytest.raises(Exception): genieclust.Genie(mlpack_enabled=True).fit(X)
    # else:
    #     with pytest.raises(Exception): genieclust.Genie(mlpack_enabled=True, metric="l1").fit(X)
    #     with pytest.raises(Exception): genieclust.Genie(mlpack_enabled=True, M=2).fit(X)
    #
    # if nmslib is None:
    #     with pytest.raises(Exception): genieclust.Genie(exact=False).fit(X)
    # else:
    #     with pytest.raises(Exception): genieclust.Genie(metric="leven", exact=False, cast_float32=True).fit(X)
    #     with pytest.raises(Exception): genieclust.Genie(metric="precomputed", exact=False).fit(X)
    #     with pytest.raises(Exception): genieclust.Genie(M=10, nmslib_n_neighbors=8, exact=False).fit(X)
    #     with pytest.raises(Exception): genieclust.Genie(nmslib_params_init=[], exact=False).fit(X)
    #     with pytest.raises(Exception): genieclust.Genie(nmslib_params_index=[], exact=False).fit(X)
    #     with pytest.raises(Exception): genieclust.Genie(nmslib_params_query=[], exact=False).fit(X)
    #     #with pytest.warns(Warning): genieclust.Genie(nmslib_params_index=dict(indexThreadQty=3), exact=False).fit(X)
    #     #with pytest.warns(Warning): genieclust.Genie(nmslib_params_init=dict(space="outer"), exact=False).fit(X)


if __name__ == "__main__":
    test_warnerr()
