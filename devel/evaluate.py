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
np.set_printoptions(precision=5, threshold=10, edgeitems=10)
pd.set_option("min_rows", 20)
plt.style.use("seaborn-whitegrid")
#plt.rcParams["figure.figsize"] = (8,4)


def get_metrics(labels_true, labels_pred):

    # disregard noise points from counting
    # noise cluster == 0
    labels_pred = labels_pred[labels_true>0]
    labels_true = labels_true[labels_true>0]

    return {**genieclust.compare_partitions.compare_partitions2(labels_true, labels_pred),
            "nmi": sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred),
            "ami": sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        }


def benchmark(dataset, benchmarks_path, preprocess="none"):
    """
    Processes a single dataset (yup!).
    """
    np.random.seed(123)
    X = np.loadtxt(os.path.join(benchmarks_path, dataset+".data.gz"), ndmin=2)

    if X.shape[0]>=10_000:
        return [] # TODO:        just testing now...............................................

    if preprocess == "standardise":
        X = (X-X.mean(axis=0))/X.std(axis=0, ddof=1)
    elif preprocess == "standardise_robust":
        X = (X-np.percentile(X, 50, axis=0))/(
            np.percentile(X, 75, axis=0)-np.percentile(X, 25, axis=0)
        )
    else:
        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1) # scale all axes proportionally

    X += np.random.normal(0.0, 1e-9, size=X.shape) # add a tiny bit of noise
    X = X.astype(np.float32, order="C", copy=False)


    print("## %s preprocess=%s (n=%d, d=%d)" %
          (dataset, preprocess, X.shape[0], X.shape[1]))

    label_names = sorted([re.search(r"\.(labels[0-9]+)\.gz", name).group(1)
                        for name in glob.glob(os.path.join(benchmarks_path, dataset+".labels*.gz"))])
    labels = [np.loadtxt(os.path.join(benchmarks_path, "%s.%s.gz" % (dataset,name)), dtype="int")
                        for name in label_names]
    label_counts = [np.bincount(l) for l in labels]
    noise_counts = [c[0] for c in label_counts] # noise cluster == 0
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
            for g in [0.1, 0.3, 0.5, 0.7, 1.0]:
                genie.set_params(n_clusters=true_K[i],
                    gini_threshold=g, M=M, postprocess="all")
                labels_pred = genie.fit_predict(X)
                metrics = genieclust.compare_partitions.compare_partitions2(labels[i], labels_pred)
                ret_cur = {
                    **params,
                    "method": "Genie_%.1f"%(g,),
                    "M": M,
                    **get_metrics(labels[i], labels_pred)
                }
                ret.append(ret_cur)

            for add in [5, 1, 0]:
                for g in [np.r_[0.3, 0.5, 0.7], np.linspace(0.0, 1.0, 11), []]:
                    if len(g) == 0 and add > 0: continue

                    gic.set_params(n_clusters=true_K[i],
                        gini_thresholds=g, add_clusters=add, M=M, postprocess="all")
                    labels_pred = gic.fit_predict(X)
                    metrics = genieclust.compare_partitions.compare_partitions2(
                        labels[i], labels_pred)
                    ret_cur = {
                        **params,
                        "method": "GIc_A%d_%d"%(add,len(g)),
                        "M": M,
                        **get_metrics(labels[i], labels_pred)
                    }
                    ret.append(ret_cur)

    return ret

res = []

folders = ["sipu", "wut", "other", "fcps", "graves"]
for folder in folders:
    fnames = glob.glob(os.path.join(benchmarks_path, folder, "*.data.gz"))
    datasets = natsorted([re.search(r"([^/]*/[^/]*)\.data\.gz", name)[1] for name in fnames])

    for dataset in datasets:
        res += benchmark(dataset, benchmarks_path, preprocess="none")
        res += benchmark(dataset, benchmarks_path, preprocess="standardise")
        res += benchmark(dataset, benchmarks_path, preprocess="standardise_robust")


res = pd.DataFrame(res)

res_max = res.loc[(res.preprocess=="none") & res.method.isin(["GIc_A0_3", "Genie_0.3"]) &
                  (~res.dataset.str.contains("2mg")),:].\
    groupby(["dataset", "method", "preprocess", "M"]).max().\
    reset_index().drop(["k", "g", "noise", "labels"], axis=1)


res_summary_ar = res_max.groupby(["method", "preprocess", "M"]).ar.\
    mean().sort_values(ascending=False).rename("mean").\
    reset_index()
print(res_summary_ar)

res_summary_ar = res_max.groupby(["method", "preprocess", "M"]).ar.\
    median().sort_values(ascending=False).rename("median").\
    reset_index()
print(res_summary_ar)


#plt.rcParams["figure.figsize"] = (12,4)
#plt.subplot("131")
#sns.boxplot(y="method", x="ar", data=res_max.loc[res_max.preprocess=="none",:], orient="h")
#plt.subplot("132")
#sns.boxplot(y="method", x="ar", data=res_max.loc[res_max.preprocess=="standardise",:], orient="h")
#plt.subplot("133")
#sns.boxplot(y="method", x="ar", data=res_max.loc[res_max.preprocess=="standardise_robust",:], orient="h")

plt.rcParams["figure.figsize"] = (12,8)
sns.boxplot(y="method", x="ar", hue="M", data=res_max, orient="h")
plt.show()


plt.rcParams["figure.figsize"] = (8,6)
res_max2 = res.copy()
res_max2["preprocess_M"] = res_max2.preprocess+"_"+res_max2.M.astype(str)
res_max2 = res_max2.loc[(~res.dataset.str.contains("2mg")),:].\
    groupby(["dataset", "method", "preprocess_M"]).max().\
    reset_index().drop(["k", "g", "noise", "labels"], axis=1)
res_summary_ar2 = res_max2.groupby(["method", "preprocess_M"]).ar.\
    mean().sort_values(ascending=False).rename("mean").unstack()
sns.heatmap(res_summary_ar2, annot=True, fmt=".2f", vmin=0.5, vmax=1.0)
plt.title("Mean ARI")
plt.show()
res_max2 = res.copy()
res_max2["preprocess_M"] = res_max2.preprocess+"_"+res_max2.M.astype(str)
res_max2 = res_max2.\
    groupby(["dataset", "method", "preprocess_M"]).max().\
    reset_index().drop(["k", "g", "noise", "labels"], axis=1)
res_summary_ar2 = res_max2.groupby(["method", "preprocess_M"]).ar.\
    median().sort_values(ascending=False).rename("median").unstack()
sns.heatmap(res_summary_ar2, annot=True, fmt=".2f", vmin=0.5, vmax=1.0)
plt.title("Median ARI")
plt.show()

#res.to_csv("20200423_results_temp.csv")
