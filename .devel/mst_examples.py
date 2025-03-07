import genieclust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import os.path
import scipy.spatial
import lumbermark



examples = [
    ["sipu", "aggregation", [], 7], # !!!  Genie fails
    ["sipu", "compound", [], 5],  # Genie fails
    ["wut", "windows", [], 5],  # !!!  Genie fails
    ["sipu", "pathbased", [], 3],  # :/
    ["sipu", "unbalance", [], 8], # ! !!  Genie fails
    ["new", "blobs4a", [], 4],
    ["wut", "mk2", [], 2],  # :/ !!!
    ["sipu", "spiral", [], 3],
    ["wut", "cross", [], 4],  #  Genie fails :/ !!!
    ["graves", "fuzzyx", [], 4], #  Genie fails
    ["wut", "z2", [], 5],
    ["graves", "parabolic", [], 2],

    ["other", "hdbscan", [], 6],
    ["wut", "graph", [], 12],  # Genie fails
    ["wut", "mk4", [], 3],
    ["graves", "zigzag_outliers", [], 3],
    ["new", "blobs4b", [], 4],
    ["wut", "mk3", [], 3],
    ["fcps", "engytime", [], 2],
    ["new", "blobs3a", [], 3],
    ["wut", "x2", [], 4],
    ["fcps", "target", [], 2],
    ["sipu", "jain", [], 2],
    ["wut", "isolation", [], 3],
    ["wut", "x3", [], 3],
    ["sipu", "flame", [], 2],
    ["graves", "dense", [], 2],
    ["wut", "circles", [], 1],
    ["wut", "twosplashes", [], 2],
    ["fcps", "twodiamonds", [], 2],
    ["other", "chameleon_t8_8k", [], 8],
    ["other", "chameleon_t7_10k", [], 9],
    ["other", "chameleon_t5_8k", [], 6],
    ["other", "chameleon_t4_8k", [], 6],
    ["wut", "labirynth", [], 6],
    ["wut", "z3", [], 4],
    ["other", "hdbscan", [], 6],
    ["sipu", "s1", [], 15],
    ["fcps", "wingnut", [], 2],
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
                color="gray" if mst_labels[i] == lumbermark.SKIP else genieclust.plots.col[mst_labels[i]-1],
                va='top'
            )
    for i in range(n_clusters+1):
        genieclust.plots.plot_segments(mst_e[mst_labels == i, :], X, color=genieclust.plots.col[i-1],
            alpha=alpha, linestyle="-" if i>0 else ":")
    genieclust.plots.plot_segments(mst_e[mst_labels<0,:], X, color="yellow", linestyle="-", linewidth=3)
    genieclust.plots.plot_segments(mst_e[mst_internodes,:], X, color="orange", linestyle="-", linewidth=3)
    if cutting is not None:
        genieclust.plots.plot_segments(mst_e[[cutting],:], X, color="blue", linestyle="--", alpha=1.0, linewidth=3)






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

