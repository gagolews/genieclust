#!/usr/bin/env -S python3 -W ignore::FutureWarning
#%%silent
#%%restart
#%%cd @


"""
Apply clustering methods on selected benchmark datasets from the repository
and measure the run-times.

Copyright (C) 2020, Marek Gagolewski, https://www.gagolewski.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""



# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````


import sys
import numpy as np
import pandas as pd
import scipy.stats
import os.path, glob, re, csv, os
from natsort import natsorted
import sklearn, sklearn.metrics
import time
from benchmark_load import *
import sklearn.cluster
import sklearn.mixture
import genieclust
import gc


# ``````````````````````````````````````````````````````````````````````````````
# `````` USER SETTINGS                                                   ```````
# ``````````````````````````````````````````````````````````````````````````````

# https://github.com/gagolews/clustering_benchmarks_v1
benchmarks_path = "/home/gagolews/Projects/clustering_benchmarks_v1"

datasets    = ["wut/trajectories", "sipu/birch1", "sipu/birch2", "sipu/worms_2", "sipu/worms_64", "mnist/digits", "mnist/fashion"]

ofname = "v1-timings.csv"

# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````


def kmeans_with_n_threads(X, n_clusters, n_threads, **kwargs):
    # one of these n_threads setting methods work
    n_threads_old = os.environ["OMP_NUM_THREADS"]
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    k = sklearn.cluster.KMeans(n_clusters=n_clusters, n_jobs=n_threads, **kwargs)
    k._n_threads = n_threads
    k.fit(X)
    os.environ["OMP_NUM_THREADS"] = n_threads_old
    return k


def Genie_with_n_threads(X, n_clusters, n_threads, **kwargs):
    n_threads_old = os.environ["OMP_NUM_THREADS"]
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    g = genieclust.Genie(n_clusters=n_clusters, **kwargs)
    g.fit(X)
    os.environ["OMP_NUM_THREADS"] = n_threads_old
    return g


def register_result(dataset, method, n_clusters, t, labels_pred=None, labels_true=None):
    res = dict(
        time=time.time(),
        dataset=dataset,
        method=method,
        n_clusters=n_clusters,
        t=t
    )
    if labels_true is not None:
        res = {**res, **genieclust.compare_partitions.compare_partitions2(labels_pred, labels_true)}
    return res


def get_timing(dataset):
    input_fname_base = os.path.join(benchmarks_path, dataset)
    X = load_data(input_fname_base+".data.gz", "original")

    res = list()
    n_clusterss = [10, 100, 1000]
    n_threadss = [1, 4, 8]
    gini_thresholds = [0.1, 0.3, 0.5]

    for n_clusters in n_clusterss:
        for n_threads in n_threadss:
            gc.collect()
            t0 = time.time()
            kmeans_with_n_threads(X, n_clusters, n_threads=n_threads)
            t1 = time.time()
            res.append(register_result(dataset, "sklearn_kmeans_t%d"%n_threads, n_clusters, t1-t0))
            print(res[-1])

            t0 = time.time()
            last_g_exact = Genie_with_n_threads(X, n_clusters, n_threads=n_threads)
            t1 = time.time()
            res.append(register_result(dataset, "Genie_0.3_t%d"%n_threads, n_clusters, t1-t0))
            print(res[-1])

            t0 = time.time()
            last_g_approx = Genie_with_n_threads(X, n_clusters, n_threads=n_threads, exact=False)
            t1 = time.time()
            res.append(register_result(dataset, "Genie_0.3_t%d_approx"%n_threads, n_clusters, t1-t0))
            print(res[-1])

    # test the "cached" version of Genie(exact=True):
    for n_clusters in n_clusterss:
        for gini_threshold in gini_thresholds:

            t0 = time.time()
            last_g_exact.gini_threshold = gini_threshold
            last_g_exact.n_clusters     = n_clusters
            labels_true = last_g_exact.fit_predict(X)
            t1 = time.time()
            res.append(register_result(dataset, "Genie_%.1f_tx"%gini_threshold, n_clusters, t1-t0))
            print(res[-1])

    return res



# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````



if __name__ == "__main__":
    print(sklearn.__version__)
    # uname -a

    for dataset in datasets:
        for iter in range(10):
            np.random.seed(iter+1)
            res = get_timing(dataset)
            res_df = pd.DataFrame(res)
            print(res_df)
            res_df.to_csv(ofname, quoting=csv.QUOTE_NONNUMERIC, index=False,
                        header = not os.path.isfile(ofname),
                        mode = "w" if not os.path.isfile(ofname) else "a")
            res, res_df = None, None
            gc.collect()
