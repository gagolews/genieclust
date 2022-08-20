#!/usr/bin/env -S python3 -W ignore::FutureWarning
#%%silent
#%%restart
#%%cd @


"""
Apply Genie on g2mg-sets of different sizes and measure the run-times.

Copyleft (C) 2020-2021, Marek Gagolewski <https://www.gagolewski.com>

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
np.set_printoptions(precision=3, threshold=50, edgeitems=50)
pd.set_option("min_rows", 200)



# ``````````````````````````````````````````````````````````````````````````````
# `````` USER SETTINGS                                                   ```````
# ``````````````````````````````````````````````````````````````````````````````

os.environ["OMP_NUM_THREADS"] = '6'

ds = [2, 5, 10, 25, 50, 100]
ns = [10_000, 25_000, 50_000, 100_000]
s = 30
mu1 = 500  # cluster1 centre
mu2 = 600  # cluster2 centre

ofname = "v1-g2mg.csv"


niters = 10 # actually for large n this is too much, I have other things to do ;)

# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````



#%%
import numpy as np
import scipy.spatial.distance

#%%
def generate_gKmg(d, n, mu, s, random_state=None):
    """Generates K=len(n) groups of points in R^d together with their
    corresponding labels.

    The i-th group, i=1,...,K, consists of n[i-1] points
    that are sampled from the Gaussian distribution with mean mu[i-1,:]
    and covariance matrix diag(s[i]).
    """
    assert mu.shape[0] == n.shape[0] == s.shape[0]
    assert mu.shape[1] == d
    assert (s>0).all()
    assert (n>0).all()

    K = mu.shape[0] # number of groups

    if random_state is None:
        random_state = np.random.randint(0, 2**32)

    # Each point group is generated separately,
    # with different (yet predictable) random_state,
    # so that changing n[i] generates the same points

    X = []
    for i in range(K):
        rand = np.random.RandomState((random_state+i) % (2**32))
        X.append(rand.randn(n[i], d)*s[i] + mu[i,:])

    X = np.vstack(X)

    labels0 = np.repeat(np.arange(1, K+1), n) #[1,1,...,1,2,...,2,...,K,...,K]

    labels1 = np.argmax(scipy.spatial.distance.cdist(X, mu), axis=1)+1

    return X, labels0, labels1





def register_result(
        random_state,
        dataset,
        n,
        d,
        method,
        n_clusters,
        n_threads,
        t,
        labels_pred,
        labels_true):
    #########################################################
    partsims = [
        genieclust.compare_partitions.compare_partitions2(labels_pred, l)
        for l in labels_true
        ]
    partsims = {
        psm: max([vals[psm] for vals in partsims]) for psm in partsims[0].keys()
    }

    res = dict(
        random_state=random_state,
        timestamp=time.time(),
        dataset=dataset,
        n=n,
        d=d,
        method=method,
        n_clusters=n_clusters,
        n_threads=n_threads,
        elapsed_time=t,
        **partsims
    )
    return res


def get_timing(n, d, s, mu1, mu2, random_state):
    dataset = "g2mg_%d_%s"%(d,s)
    s_cor = s*np.sqrt(d/2)
    assert n % 2 == 0
    X, labels0, labels1 = generate_gKmg(
                d,
                np.r_[n//2, n//2],
                np.array([ [mu1]*d, [mu2]*d ]),
                np.r_[s_cor, s_cor],
                random_state)

    labels_true = [labels0, labels1]

    res = list()
    gini_thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]

    t0 = time.time()
    last_g = genieclust.Genie(n_clusters=2, exact=False)
    labels_pred = last_g.fit_predict(X)
    t1 = time.time()
    res.append(register_result(
        random_state, dataset, n, d,
        "Genie_0.3_approx", 2, os.environ["OMP_NUM_THREADS"], t1-t0,
        labels_pred, labels_true))
    #print(res[-1])

    ## test the "cached" version of Genie(exact=True):
    for gini_threshold in gini_thresholds:
        t0 = time.time()
        last_g.set_params(gini_threshold=gini_threshold)
        labels_pred = last_g.fit_predict(X)
        t1 = time.time()
        res.append(register_result(
            random_state, dataset, n, d,
            "Genie_%.1f_approx"%gini_threshold, 2, 0, t1-t0,
            labels_pred, labels_true))
        #print(res[-1])


    if d <= 10:
        t0 = time.time()
        last_g = genieclust.Genie(n_clusters=2, mlpack_enabled=True)
        labels_pred = last_g.fit_predict(X)
        t1 = time.time()
        res.append(register_result(
            random_state, dataset, n, d,
            "Genie_0.3_mlpack", 2, 1, t1-t0,
            labels_pred, labels_true))
        #print(res[-1])

    t0 = time.time()
    last_g = genieclust.Genie(n_clusters=2, mlpack_enabled=False)
    labels_pred = last_g.fit_predict(X)
    t1 = time.time()
    res.append(register_result(
        random_state, dataset, n, d,
        "Genie_0.3_nomlpack", 2, os.environ["OMP_NUM_THREADS"], t1-t0,
        labels_pred, labels_true))
    #print(res[-1])

    ## test the "cached" version of Genie(exact=True):
    for gini_threshold in gini_thresholds:
        t0 = time.time()
        last_g.set_params(gini_threshold=gini_threshold)
        labels_pred = last_g.fit_predict(X)
        t1 = time.time()
        res.append(register_result(
            random_state, dataset, n, d,
            "Genie_%.1f"%gini_threshold, 2, 0, t1-t0,
            labels_pred, labels_true))
        #print(res[-1])

    return res



# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````
# ``````````````````````````````````````````````````````````````````````````````



if __name__ == "__main__":
    print(os.environ["OMP_NUM_THREADS"])

    for iter in range(niters):
        for d in ds:
            for n in ns:
                np.random.seed(iter+1)
                res = get_timing(n, d, s, mu1, mu2, iter+1)
                res_df = pd.DataFrame(res)
                print(res_df)
                res_df.to_csv(ofname, quoting=csv.QUOTE_NONNUMERIC, index=False,
                        header = not os.path.isfile(ofname),
                        mode = "w" if not os.path.isfile(ofname) else "a")
                res, res_df = None, None
                gc.collect()
