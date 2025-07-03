



# Basics

*Genie* {cite}`genieins` is an agglomerative hierarchical clustering
algorithm. The idea behind *Genie* is beautifully simple. First, it makes each
individual point the sole member of its own cluster. Then, it keeps merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes** a point group of the
smallest size will sometimes be combined with its nearest counterpart.

In the following sections, we will demonstrate that Genie often outperforms
other popular methods in terms of clustering [quality](benchmarks_ar)
and [speed](timings).

Here are a few examples of basic interactions with the Python version
of the `genieclust` {cite}`genieclust` package,
which we can install from [PyPI](https://pypi.org/project/genieclust/), e.g.,
via a call to `pip3 install genieclust` from the command line.




``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import genieclust
```





## Breaking the Ice

Let's load an example benchmark set, `jain` {cite}`jain`, which  comes
with the true corresponding partition (as assigned by experts).


``` python
# see https://github.com/gagolews/genieclust/tree/master/devel/sphinx/weave
dataset = "jain"
# Load an example 2D dataset:
X = np.loadtxt("%s.data.gz" % dataset, ndmin=2)

# Load the corresponding reference labels. The original labels are in {1,2,..,k}.
# We will make them more Python-ish by subtracting 1.
labels_true = np.loadtxt("%s.labels0.gz" % dataset, dtype=np.intp)-1

# The number of unique labels gives the true cluster count:
n_clusters = len(np.unique(labels_true))
```


A scatter plot of the dataset together with the reference labels:


``` python
genieclust.plots.plot_scatter(X, labels=labels_true)
plt.title("%s (n=%d, true n_clusters=%d)" % (dataset, X.shape[0], n_clusters))
plt.axis("equal")
```

``` python
plt.show()
```

(fig:basics-scatter)=
```{figure} basics-figures/basics-scatter-1.*
Reference labels.
```

Let us apply the Genie algorithm (with the default/recommended
`gini_threshold` parameter value). The `genieclust` package's programming
interface is [scikit-learn](https://scikit-learn.org/)-compatible {cite}`sklearn`.
In particular, an object of class `Genie` is equipped with the
`fit` and `fit_predict` methods {cite}`sklearn_api`.




``` python
g = genieclust.Genie(n_clusters=n_clusters)
labels_genie = g.fit_predict(X)
```

For more details, see the documentation of the
[genieclust.Genie](genieclust.Genie) class.



Plotting of the discovered partition:


``` python
genieclust.plots.plot_scatter(X, labels=labels_genie)
plt.title("Genie (gini_threshold=%g)" % g.gini_threshold)
plt.axis("equal")
```

``` python
plt.show()
```

(fig:basics-plot-pred)=
```{figure} basics-figures/basics-plot-pred-3.*
Labels predicted by Genie.
```

Very nice. Great success.

A picture is worth a thousand words, but numbers are worth
millions of pictures. We can compare the resulting clustering with the reference
one by computing, for example, the confusion matrix.



``` python
# Compute the confusion matrix (with pivoting)
genieclust.compare_partitions.normalized_confusion_matrix(labels_true, labels_genie)
## array([[276,   0],
##        [  0,  97]])
```

The above confusion matrix can be summarised by means of partition
similarity measures, such as the adjusted Rand index (`ar`).


``` python
# See also: sklearn.metrics.adjusted_rand_score()
genieclust.compare_partitions.adjusted_rand_score(labels_true, labels_genie)
## 1.0
```

This denotes a perfect match between these two.



## A Comparison with k-means


For the sake of comparison, let us apply the k-means algorithm on the same dataset.



``` python
import sklearn.cluster
km = sklearn.cluster.KMeans(n_clusters=n_clusters)
labels_kmeans = km.fit_predict(X)
genieclust.plots.plot_scatter(X, labels=labels_kmeans)
plt.title("k-means")
plt.axis("equal")
```

``` python
plt.show()
```

(fig:basics-plot-km)=
```{figure} basics-figures/basics-plot-km-5.*
Labels predicted by k-means.
```


It is well known that the k-means algorithm can only split the input space into
convex regions (compare the notion of the
[Voronoi diagrams](https://en.wikipedia.org/wiki/Voronoi_diagram),
so we should not be very surprised with this result.



``` python
# Compute the confusion matrix for the k-means output:
genieclust.compare_partitions.normalized_confusion_matrix(labels_true, labels_kmeans)
## array([[197,  79],
##        [  1,  96]])
```


``` python
# A cluster similarity measure for k-means:
genieclust.compare_partitions.adjusted_rand_score(labels_true, labels_kmeans)
## 0.3241080446115835
```

The adjusted Rand score of $\sim 0.3$ indicates a far-from-perfect fit.



## A Comparison with HDBSCAN\*

Let's also make a comparison against a version of the DBSCAN
{cite}`predbscan,dbscan` algorithm. The original DBSCAN relies on a somewhat
magical `eps` parameter, which might be hard to tune in practice. However,
the [hdbscan](https://github.com/scikit-learn-contrib/hdbscan) package
{cite}`hdbscanpkg` implements its robustified variant
{cite}`hdbscan`, which makes the algorithm much more user-friendly.

Here are the clustering results with the `min_cluster_size` parameter
of 3, 5, 10, and 15:


``` python
import hdbscan
mcs = [3, 5, 10, 15]
for i in range(len(mcs)):
    h = hdbscan.HDBSCAN(min_cluster_size=mcs[i])
    labels_hdbscan = h.fit_predict(X)
    plt.subplot(2, 2, i+1)
    genieclust.plots.plot_scatter(X, labels=labels_hdbscan)
    plt.title("HDBSCAN (min_cluster_size=%d)" % h.min_cluster_size)
    plt.axis("equal")
```

``` python
plt.show()
```

(fig:basics-plot-hdbscan)=
```{figure} basics-figures/basics-plot-hdbscan-7.*
Labels predicted by HDBSCAN\*.
```

**Side note.**
Gray plotting symbols denote "noise" points; we'll get back to this feature
in [another section](noise).




In HDBSCAN\*, `min_cluster_size` affects the "granularity"
of the obtained clusters. Its default value is set to:


``` python
hdbscan.HDBSCAN().min_cluster_size
## 5
```

Unfortunately, we cannot easily guess how many clusters will be generated
by this method. At first glance, it would seem that `min_cluster_size`
should lie somewhere between 10 and 15, but...


``` python
mcs = range(10, 16)
for i in range(len(mcs)):
    h = hdbscan.HDBSCAN(min_cluster_size=mcs[i])
    labels_hdbscan = h.fit_predict(X)
    plt.subplot(3, 2, i+1)
    genieclust.plots.plot_scatter(X, labels=labels_hdbscan)
    plt.title("HDBSCAN (min_cluster_size=%d)"%h.min_cluster_size)
    plt.axis("equal")
```

``` python
plt.show()
```

(fig:basics-plot-hdbscan2)=
```{figure} basics-figures/basics-plot-hdbscan2-9.*
Labels predicted by HDBSCAN\*.
```

Strangely enough, `min_cluster_size` of $11$ generates four clusters,
whereas $11\pm 1$ yields only three point groups.

On the other hand, the Genie algorithm belongs
to the group of *hierarchical agglomerative methods*. By definition,
it can generate a sequence of *nested* partitions, which means that by
increasing `n_clusters`, we split one and only one cluster
into two subgroups. This makes the resulting partitions more stable.


``` python
ncl = range(2, 8)
for i in range(len(ncl)):
    g = genieclust.Genie(n_clusters=ncl[i])
    labels_genie = g.fit_predict(X)
    plt.subplot(3, 2, i+1)
    genieclust.plots.plot_scatter(X, labels=labels_genie)
    plt.title("Genie (n_clusters=%d)"%(g.n_clusters,))
    plt.axis("equal")
```

``` python
plt.show()
```

(fig:basics-plot-genie2)=
```{figure} basics-figures/basics-plot-genie2-11.*
Labels predicted by Genie.
```



## Dendrograms

Dendrogram plotting is possible with `scipy.cluster.hierarchy`:


``` python
import scipy.cluster.hierarchy
g = genieclust.Genie(compute_full_tree=True)
g.fit(X)
```

``` python
linkage_matrix = np.column_stack([g.children_, g.distances_, g.counts_])
scipy.cluster.hierarchy.dendrogram(linkage_matrix,
    show_leaf_counts=False, no_labels=True)
```

``` python
plt.show()
```

(fig:basics-dendrogram-1)=
```{figure} basics-figures/basics-dendrogram-1-13.*
Example dendrogram.
```

Another example:


``` python
scipy.cluster.hierarchy.dendrogram(linkage_matrix,
    truncate_mode="lastp", p=15, orientation="left")
```

``` python
plt.show()
```

(fig:basics-dendrogram-2)=
```{figure} basics-figures/basics-dendrogram-2-15.*
Another example dendrogram.
```

For a list of graphical parameters, refer to the function's manual.



## Further Reading

For more details, refer to the package's API reference
manual: [genieclust.Genie](genieclust.Genie).
To learn more about Python, check out Marek's open-access (free!) textbook
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/)
{cite}`datawranglingpy`.
