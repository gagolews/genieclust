"""
...
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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


import genieclust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import os.path
import scipy.spatial
from numba import jit


examples = [
    ["sipu", "pathbased", [], 3],  # :/
    ["sipu", "jain", [], 2],
    ["sipu", "aggregation", [], 7], # !!!  Genie fails
    ["sipu", "compound", [], 5],  # Genie fails
    ["wut", "windows", [], 5],  # !!!  Genie fails
    ["sipu", "unbalance", [], 8], # ! !!  Genie fails
    ["new", "blobs4a", [], 4],
    ["wut", "mk2", [], 2],  # :/ !!!
    ["sipu", "spiral", [], 3],
    ["wut", "cross", [], 4],  #  Genie fails :/ !!!
    ["graves", "fuzzyx", [], 4], #  Genie fails
    ["wut", "isolation", [], 3],
    ["wut", "labirynth", [], 6],

    ["wut", "z2", [], 5],
    ["graves", "parabolic", [], 2],
    ["other", "hdbscan", [], 6],
    ["wut", "graph", [], 12],  # Genie fails
    ["wut", "x2", [], 4],
    ["graves", "zigzag_outliers", [], 3],
    ["sipu", "flame", [], 2],
    ["fcps", "engytime", [], 2],
    ["other", "chameleon_t8_8k", [], 8],
    ["other", "chameleon_t7_10k", [], 9],
    ["other", "chameleon_t4_8k", [], 6],
    ["other", "chameleon_t5_8k", [], 6],
    ["wut", "twosplashes", [], 2],
    ["fcps", "twodiamonds", [], 2],
    ["sipu", "s1", [], 15],
    ["fcps", "wingnut", [], 2],
    ["wut", "olympic", [], 3],
    ["wut", "mk3", [], 3],

    ["graves", "dense", [], 2],
    ["wut", "mk4", [], 3],
    ["new", "blobs4b", [], 4],
    ["new", "blobs3a", [], 3],
    ["fcps", "target", [], 2],
    ["wut", "x3", [], 3],
    ["wut", "z3", [], 4],
    ["other", "hdbscan", [], 6],
]



def get_example(i, data_path="."):
    battery, dataset, skiplist, n_clusters = examples[i]

    np.random.seed(123)
    if battery != "new":
        b = clustbench.load_dataset(battery, dataset, path=data_path)
        X, y_true = b.data, b.labels[0]
    else:
        from sklearn.datasets import make_blobs
        if dataset == "blobs4a":
            X, y_true = make_blobs(
                n_samples=[500, 500, 100, 100],
                cluster_std=[0.05, 0.2, 0.2, 0.2],
                centers=[[1,1], [1,-1], [-1,-1], [-1,1]],
                random_state=42
            )
            xapp = np.array([
                [0,0],
                [-0.1,0.25],
                [-0.1,-0.1],
                [0.1,0.25],
            ])
            X = np.append(X, xapp, axis=0)
            y_true = np.r_[y_true+1, np.repeat(0, xapp.shape[0])]
        elif dataset == "blobs4b":
            X, y_true = make_blobs(
                n_samples=[800, 800, 100, 100],
                cluster_std=[0.05, 0.2, 0.3, 0.2],
                centers=[[1,1], [1,-1], [-1,-1], [-1,1]],
                random_state=42
            )
            xapp = np.array([
                [0,0],
                [-0.1,0.25],
                [-0.1,-0.1],
                [0.1,0.25],
            ])
            X = np.append(X, xapp, axis=0)
            y_true = np.r_[y_true+1, np.repeat(0, xapp.shape[0])]
        elif dataset == "blobs3a":
            # see https://github.com/gagolews/genieclust/issues/91
            X, y_true = make_blobs(
                n_samples=[1000, 100, 100],
                cluster_std=1,
                random_state=42
            )

    return X, y_true, n_clusters, skiplist, "%s/%s/%d" % (battery, dataset, n_clusters)




def plot_mst_2d(L, mst_draw_edge_labels=False, alpha=0.2):
    y_pred = L.labels_
    X = L.X
    n_clusters = L.n_clusters
    mst_e = L._mst_e
    mst_w = L._mst_w
    mst_s = L._mst_s
    min_mst_s = np.min(mst_s[:,:], axis=1)
    mst_labels = L._mst_labels
    n = X.shape[0]
    skiplist = L._mst_skiplist
    cutting = L._mst_cutting
    mst_internodes = L.__dict__.get("_mst_internodes", [])
    genieclust.plots.plot_scatter(X[:,:2], labels=y_pred-1)
    plt.axis("equal")
    if mst_draw_edge_labels:
        for i in range(n-1):
            plt.text(
                (X[mst_e[i,0],0]+X[mst_e[i,1],0])/2,
                (X[mst_e[i,0],1]+X[mst_e[i,1],1])/2,
                "%d\n(%d-%d)" % (i, *sorted((mst_s[i, 0], mst_s[i, 1]))),
                color="gray" if mst_labels[i] < -1 else genieclust.plots.col[mst_labels[i]-1],
                va='top'
            )
    for i in range(n_clusters+1):
        genieclust.plots.plot_segments(mst_e[mst_labels == i, :], X, color=genieclust.plots.col[i-1],
            alpha=alpha, linestyle="-" if i>0 else ":")
    genieclust.plots.plot_segments(mst_e[mst_labels<0,:], X, color="yellow", linestyle="-", linewidth=3)
    genieclust.plots.plot_segments(mst_e[mst_internodes,:], X, color="orange", linestyle="-", linewidth=3)
    if cutting is not None:
        genieclust.plots.plot_segments(mst_e[[cutting],:], X, color="blue", linestyle="--", alpha=1.0, linewidth=3)






@jit
def first_nonzero(x, missing=None):
    for i in range(len(x)):
        if x[i]:
            return x[i]
    return missing

@jit
def arg_first_nonzero(x, missing=None):
    for i in range(len(x)):
        if x[i]:
            return i
    return missing



def split(x, f):
    f = np.array(f)
    x = np.array(x)
    assert f.shape[0] == x.shape[0]
    assert f.ndim == 1
    _o = np.argsort(f, kind="stable")
    _u, _s = np.unique(f[_o], return_index=True)
    _v = np.split(x[_o], _s)[1:]
    return np.array(_v, dtype=object), _u


def aggregate(x, f, a):
    _x, _f = split(x, f)
    return np.array([a(gx) for gx in _x]), _f


def reindex(idx, mask, filler=-1):
    """
    Often, we operate on subsets of datasets. When calling functions
    returning index vectors (permutations), their outputs need to be
    adjusted as such indexes are relative to a chosen subset:

    ```
    idx = function_returning_an_index_vector(X[mask])
    idx_corr = reindex(idx, mask)
    ```

    and now `X[idx_corr[idx_corr>=0]]` is equivalent to `X[mask][idx]`.

    Given a permutation `idx` of the set {0, ..., k-1} and
    a Boolean mask vector with k positive values,
    returns a vector `out` such that
    `out[~mask] = -1` and `out[mask] = idx` with elements shifted
    accordingly, taking into account the number of preceding elements
    that were masked out.

    Examples:

    ```
    np.random.seed(123)
    x = np.round(np.random.rand(10), 2)  # example vector
    ## array([0.7 , 0.29, 0.23, 0.55, 0.72, 0.42, 0.98, 0.68, 0.48, 0.39])
    mask = (np.random.rand(len(x)) >= 0.5)  # example mask
    ## array([False,  True, False, False, False,  True, False, False,  True, True])
    idx = np.argsort(x[mask])  # example index vector
    ## array([0, 3, 1, 2])
    idx_corr = reindex(idx, mask)
    ## array([-1,  1, -1, -1, -1,  9, -1, -1,  5,  8])
    x[idx_corr[idx_corr>=0]]
    ## array([0.29, 0.39, 0.42, 0.48])
    np.r_[x, np.nan][idx_corr]
    ## array([ nan, 0.29,  nan,  nan,  nan, 0.39,  nan,  nan, 0.42, 0.48])
    ```
    """
    k = np.sum(mask)
    assert len(idx) == k
    if k == len(mask): return idx
    out = np.repeat(filler, len(mask))
    out[mask] = np.cumsum(~mask)[mask][idx]+idx
    assert np.all(idx == np.argsort(np.argsort(out[out>=0])))
    return out
