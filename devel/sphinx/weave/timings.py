#!/usr/bin/env -S python3 -W ignore::FutureWarning
#%%silent
#%%restart
#%%cd @


"""
Apply clustering methods on selected benchmark datasets from the repository
and measure the run-times.
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020, Marek Gagolewski <https://www.gagolewski.com>           #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````


import sys
import numpy as np
import pandas as pd
import scipy.stats, scipy.cluster
import os.path, glob, re, csv, os
from natsort import natsorted
import sklearn, sklearn.metrics
import time
from benchmark_load import *
import sklearn.cluster
import sklearn.mixture
import genieclust
import gc
import fastcluster


# ``````````````````````````````````````````````````````````````````````````````
# `````` USER SETTINGS                                                   ```````
# ``````````````````````````````````````````````````````````````````````````````

# https://github.com/gagolews/clustering_benchmarks_v1
benchmarks_path = "/home/gagolews/Projects/clustering_benchmarks_v1"

datasets    = [
    "g2mg/g2mg_8_30",
    "g2mg/g2mg_16_30",
    "g2mg/g2mg_32_30",
    "g2mg/g2mg_64_30",
    "g2mg/g2mg_128_30",
    "wut/trajectories",
    "sipu/birch1",
    "sipu/birch2",
    "sipu/worms_2",
    "sipu/worms_64",
    "mnist/digits",
    "mnist/fashion"
    ]

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



def fastcluster_with_linkage(X, n_clusters, linkage, **kwargs):
    linkage_matrix = fastcluster.linkage_vector(X, method=linkage)
    scipy.cluster.hierarchy.cut_tree(linkage_matrix, n_clusters=n_clusters)


def register_result(dataset, method, n_clusters, n_threads, t):
    res = dict(
        timestamp=time.time(),
        dataset=dataset,
        method=method,
        n_clusters=n_clusters,
        n_threads=n_threads,
        elapsed_time=t
    )
    return res


def get_timing(dataset):
    input_fname_base = os.path.join(benchmarks_path, dataset)
    X = load_data(input_fname_base+".data.gz", "original")

    res = list()
    n_clusterss = [10, 100, 1000]
    n_threadss = [1, 3, 6, 12]
    gini_thresholds = [0.1, 0.3, 0.5]

    n_clusters = n_clusterss[0]
    for linkage in ["median", "ward", "centroid"]:
        t0 = time.time()
        fastcluster_with_linkage(X, n_clusters, linkage)
        t1 = time.time()
        res.append(register_result(dataset, "fastcluster_"+linkage, n_clusters, 1, t1-t0))
        print(res[-1])


    for n_threads in n_threadss:
        n_clusters = n_clusterss[0]

        t0 = time.time()
        last_g_exact = Genie_with_n_threads(X, n_clusters, n_threads=n_threads)
        t1 = time.time()
        res.append(register_result(dataset, "Genie_0.3", n_clusters, n_threads, t1-t0))
        print(res[-1])

        t0 = time.time()
        Genie_with_n_threads(X, n_clusters, n_threads=n_threads, exact=False)
        t1 = time.time()
        res.append(register_result(dataset, "Genie_0.3_approx", n_clusters, n_threads, t1-t0))
        print(res[-1])

        for n_clusters in n_clusterss:
            gc.collect()

            t0 = time.time()
            kmeans_with_n_threads(X, n_clusters, n_threads=n_threads)
            t1 = time.time()
            res.append(register_result(dataset, "sklearn_kmeans", n_clusters, n_threads, t1-t0))
            print(res[-1])


    # test the "cached" version of Genie(exact=True):
    for n_clusters in n_clusterss:
        for gini_threshold in gini_thresholds:

            t0 = time.time()
            last_g_exact.gini_threshold = gini_threshold
            last_g_exact.n_clusters     = n_clusters
            labels_true = last_g_exact.fit_predict(X)
            t1 = time.time()
            res.append(register_result(dataset, "Genie_%.1f"%gini_threshold, n_clusters, 0, t1-t0))
            print(res[-1])

    return res



# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````



if __name__ == "__main__":
    print(sklearn.__version__)
    for iter in range(3):
        for dataset in datasets:
            print(dataset, iter)
            np.random.seed(iter+1)
            res = get_timing(dataset)
            res_df = pd.DataFrame(res)
            print(res_df)
            res_df.to_csv(ofname, quoting=csv.QUOTE_NONNUMERIC, index=False,
                        header = not os.path.isfile(ofname),
                        mode = "w" if not os.path.isfile(ofname) else "a")
            res, res_df = None, None
            gc.collect()
