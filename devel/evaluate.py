#%%silent
#%%restart
#%%cd @

import sys
# "https://github.com/gagolews/clustering_benchmarks_v1"
benchmarks_path = "/home/gagolews/Projects/clustering_benchmarks_v1"
sys.path.append(benchmarks_path)
from load_dataset import load_dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path, glob, re
from natsort import natsorted
import genieclust
import sklearn.metrics
import seaborn as sns
np.set_printoptions(precision=5, threshold=10, edgeitems=5)
plt.style.use("seaborn-whitegrid")
#plt.rcParams["figure.figsize"] = (8,4)


def get_metrics(labels_true, labels_pred):
    return {**genieclust.compare_partitions.compare_partitions2(labels_true, labels_pred),
            "nm": sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred),
            "am": sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        }


def benchmark(dataset, benchmarks_path, preprocess="none"):
    """
    Processes a single dataset (yup!).
    """
    np.random.seed(123)
    X = np.loadtxt(os.path.join(benchmarks_path, dataset+".data.gz"), ndmin=2)

    if preprocess == "standardise":
        X = (X-X.mean(axis=0))/X.std(axis=0, ddof=1)
    elif preprocess == "standardise_robust":
        X = (X-np.percentile(X, 50, axis=0))/(
            np.percentile(X, 75, axis=0)-np.percentile(X, 25, axis=0)
        )
    else:
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1) # scale all axes proportionally

    X += np.random.normal(0.0, 1e-9, size=X.shape) # add tiny bit of noise
    X = X.astype(np.float32, order="C", copy=False)


    # TODO: robust standardise
    # TODO: add tiny bit of noise
    #X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1))
    #X = X.astype(np.float32, order="C", copy=False)


    print("## %s preprocess=%s (n=%d, d=%d)" %
          (dataset, preprocess, X.shape[0], X.shape[1]))

    label_names = sorted([re.search(r"\.(labels[0-9]+)\.gz", name).group(1)
                        for name in glob.glob(os.path.join(benchmarks_path, dataset+".labels*.gz"))])
    labels = [np.loadtxt(os.path.join(benchmarks_path, "%s.%s.gz" % (dataset,name)), dtype="int")
                        for name in label_names]
    label_counts = [np.bincount(l) for l in labels]
    noise_counts = [c[0] for c in label_counts]
    #have_noise = [bool(c[0]) for c in label_counts]
    label_counts = [c[1:] for c in label_counts]
    true_K = [len(c) for c in label_counts]
    true_G = [genieclust.inequity.gini(c) for c in label_counts]

    ret = []

    genie = genieclust.Genie()
    gic = genieclust.GIc()
    for i in range(len(label_names)):
        #print("#### %s (true_k=%2d, noise=%5d, true_g=%.3f)" % (
        #            label_names[i], true_K[i], noise_counts[i], true_G[i]))

        params = dict(
                dataset=dataset,
                preprocess=preprocess,
                n=X.shape[0],
                d=X.shape[1],
                labels=label_names[i],
                k=true_K[i],
                noise=noise_counts[i],
                g=true_G[i]
        )

        for M in sorted([1, 3, 5, 9])[::-1]:
            # TODO: reuse nn,mst
            for g in np.linspace(0.1, 1.0, 5):
                genie.set_params(n_clusters=true_K[i],
                    gini_threshold=g, M=M, postprocess="all")
                labels_pred = genie.fit_predict(X)
                metrics = genieclust.compare_partitions.compare_partitions2(labels[i], labels_pred)
                ret_cur = {
                    **params,
                    "method": "Genie_M%d_%.1f"%(M,g),
                    **get_metrics(labels[i], labels_pred)
                }
                ret.append(ret_cur)


            for g in [np.r_[0.1, 0.3, 0.5, 0.7], np.linspace(0.0, 1.0, 11)]:
                gic.set_params(n_clusters=true_K[i],
                    gini_thresholds=g, M=M, postprocess="all")
                labels_pred = gic.fit_predict(X)
                metrics = genieclust.compare_partitions.compare_partitions2(labels[i], labels_pred)
                ret_cur = {
                    **params,
                    "method": "GIc_M%d_%d"%(M,len(g)),
                    **get_metrics(labels[i], labels_pred)
                }
                ret.append(ret_cur)

    return ret

res = []

folders = ["sipu"]
for folder in folders:
    fnames = glob.glob(os.path.join(benchmarks_path, folder, "*.data.gz"))
    datasets = natsorted([re.search(r"([^/]*/[^/]*)\.data\.gz", name)[1] for name in fnames])

    for dataset in datasets:
        res += benchmark(dataset, benchmarks_path, preprocess="none")
        res += benchmark(dataset, benchmarks_path, preprocess="standardise")
        res += benchmark(dataset, benchmarks_path, preprocess="standardise_robust")


res = pd.DataFrame(res)

res_max = res.groupby(["dataset", "method", "preprocess"]).max().\
    reset_index().drop(["k", "g", "noise", "labels"], axis=1)


res_summary_ar = res_max.groupby(["method", "preprocess"]).ar.describe().\
    reset_index()

print(res_summary_ar)

#plt.rcParams["figure.figsize"] = (12,4)
#plt.subplot("131")
#sns.boxplot(y="method", x="ar", data=res_max.loc[res_max.preprocess=="none",:], orient="h")
#plt.subplot("132")
#sns.boxplot(y="method", x="ar", data=res_max.loc[res_max.preprocess=="standardise",:], orient="h")
#plt.subplot("133")
#sns.boxplot(y="method", x="ar", data=res_max.loc[res_max.preprocess=="standardise_robust",:], orient="h")

plt.rcParams["figure.figsize"] = (12,16)
sns.boxplot(y="method", x="ar", hue="preprocess", data=res_max, orient="h")


