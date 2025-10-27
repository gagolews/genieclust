



# TODO: Outlier Detection 🚧

::::{important}
🚧🚧 This chapter is under construction.  Please come back later.
The outlier detection feature will be introduced in version 1.3.0 of the package.
The previously-experimental noise point detector will not be available anymore
as it has now been greatly improved.
🚧🚧
::::



``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import genieclust
```





Let's load an example dataset that can be found
at the [**hdbscan**](https://hdbscan.readthedocs.io)
{cite}`hdbscanpkg` package's project site:


``` python
dataset = "hdbscan"
X = np.loadtxt("%s.data.gz" % dataset, ndmin=2)
labels_true = np.loadtxt("%s.labels0.gz" % dataset, dtype=np.intp) - 1
n_clusters = len(np.unique(labels_true[labels_true>=0]))
```



Here are the reference labels as identified by a manual annotator.
The light grey markers corresponding to the label `-1` designate points
that can be considered outliers.


``` python
genieclust.plots.plot_scatter(X, labels=labels_true, alpha=0.5, markers="o")
plt.title("(n=%d, true n_clusters=%d)" % (X.shape[0], n_clusters))
plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-scatter)=
```{figure} outliers-figures/noise-scatter-1.*
Reference labels
```



## Smoothing Factor

The **genieclust** package allows for clustering with respect
to the mutual reachability distance, $d_M$,
known from the DBSCAN\* algorithm {cite}`hdbscan`.
This metric is parameterised by *a smoothing factor*, $M\ge 1$, which
controls how eagerly points are filtered out from the clustering process.

Namely, instead of the ordinary (usually Euclidean) distance
between two points $i$ and $j$, $d(i,j)$, we take
$d_M(i,j)=\max\{ c_M(i), c_M(j), d(i, j) \}$, where the so-called $M$-*core*
distance $c_M(i)$ is the distance between $i$ and its $M$-th nearest neighbour
(here, not including $i$ itself, unlike in the original paper).

DBSCAN\* and its hierarchical version, HDBSCAN\*, introduces the notion
of noise and core points.  Furthermore, their predecessor, DBSCAN,
also marks certain non-core values as border points.  They all rely
on a specific threshold $\varepsilon$ that is applied onto the points'
core distances.

In **genieclust** we identify anomalies slightly differently.
(🚧🚧 TODO: describe 🚧🚧)


Here are the effects of playing with the $M$ parameter
(we keep the default `gini_threshold`):


``` python
Ms = [10, 25, 50, 100]
for i in range(len(Ms)):
    g = genieclust.Genie(n_clusters=n_clusters, M=Ms[i])
    labels_genie = g.fit_predict(X)
    plt.subplot(2, 2, i+1)
    genieclust.plots.plot_scatter(X, labels=labels_genie, alpha=0.5, markers="o")
    plt.title("(gini_threshold=%g, M=%d)"%(g.gini_threshold, g.M))
    plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-Genie1)=
```{figure} outliers-figures/noise-Genie1-3.*
Labels predicted by Genie with outlier detection
```


Contrary to the HDBSCAN\* method featured in the [**hdbscan**](https://hdbscan.readthedocs.io)
package {cite}`hdbscanpkg`, in our case, we can request a *specific* number of clusters.
Moreover, we can easily switch between partitions of finer or coarser
granularity.  As *Genie* is a hierarchical algorithm, the partitions are properly nested.



``` python
ncs = [4, 5, 6, 7]
for i in range(len(ncs)):
    g = genieclust.Genie(n_clusters=ncs[i], M=100)
    labels_genie = g.fit_predict(X)
    plt.subplot(2, 2, i+1)
    genieclust.plots.plot_scatter(X, labels=labels_genie, alpha=0.5, markers="o")
    plt.title("(n_clusters=%d)"%(g.n_clusters))
    plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-Genie3)=
```{figure} outliers-figures/noise-Genie3-5.*
Labels predicted by Genie when outliers were removed from the dataset – different number of clusters requested
```



## A Comparision with HDBSCAN\*


Here are the results returned by [**hdbscan**](https://hdbscan.readthedocs.io)
with default parameters:


``` python
import hdbscan
```


``` python
h = hdbscan.HDBSCAN()
labels_hdbscan = h.fit_predict(X)
```

``` python
genieclust.plots.plot_scatter(X, labels=labels_hdbscan, alpha=0.5, markers="o")
plt.title("(min_cluster_size=%d, min_samples=%d)" % (
    h.min_cluster_size, h.min_samples or h.min_cluster_size))
plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-HDBSCAN1)=
```{figure} outliers-figures/noise-HDBSCAN1-7.*
Labels predicted by HDBSCAN\*.
```


By tuning up `min_cluster_size` and/or `min_samples` (which corresponds
to our `M`; by the way, `min_samples` defaults to `min_cluster_size`
if not provided explicitly), we can obtain a partition that is even closer
to the reference one:



``` python
mcss = [5, 10, 25]
mss = [5, 10]
for i in range(len(mcss)):
    for j in range(len(mss)):
        h = hdbscan.HDBSCAN(min_cluster_size=mcss[i], min_samples=mss[j])
        labels_hdbscan = h.fit_predict(X)
        plt.subplot(3, 2, i*len(mss)+j+1)
        genieclust.plots.plot_scatter(X, labels=labels_hdbscan, alpha=0.5, markers="o")
        plt.title("(min_cluster_size=%d, min_samples=%d)" % (
            h.min_cluster_size, h.min_samples or h.min_cluster_size))
        plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-HDBSCAN2)=
```{figure} outliers-figures/noise-HDBSCAN2-9.*
Labels predicted by HDBSCAN\* – different settings
```
