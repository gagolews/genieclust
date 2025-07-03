



# Clustering with Noise Point Detection

::::{note}
This section needs to be updated for the new release of *genieclust* ≥ 1.2.0.
::::



``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import genieclust
```





Let's load an example dataset that can be found
the at [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
{cite}`hdbscanpkg` package's project site:


``` python
dataset = "hdbscan"
X = np.loadtxt("%s.data.gz" % dataset, ndmin=2)
labels_true = np.loadtxt("%s.labels0.gz" % dataset, dtype=np.intp) - 1
n_clusters = len(np.unique(labels_true[labels_true>=0]))
```



Here are the "reference" labels as identified by an expert (of course,
each dataset might reveal many different clusterings that a user might
find useful for whatever their goal is).
The `-1` labels denote noise points (light grey markers).


``` python
genieclust.plots.plot_scatter(X, labels=labels_true, alpha=0.5)
plt.title("(n=%d, true n_clusters=%d)" % (X.shape[0], n_clusters))
plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-scatter)=
```{figure} noise-figures/noise-scatter-1.*
Reference labels.
```



## Smoothing Factor


The `genieclust` package allows for clustering with respect
to a mutual reachability distance, $d_M$,
known from the HDBSCAN\* algorithm {cite}`hdbscan`.
It is parameterised by *a smoothing factor*, `M`, which
controls how eagerly we tend to classify points as noise.

Here are the effects of playing with the `M` parameter
(we keep the default `gini_threshold`):


``` python
Ms = [2, 5, 10, 25]
for i in range(len(Ms)):
    g = genieclust.Genie(n_clusters=n_clusters, M=Ms[i])
    labels_genie = g.fit_predict(X)
    plt.subplot(2, 2, i+1)
    genieclust.plots.plot_scatter(X, labels=labels_genie, alpha=0.5)
    plt.title("(gini_threshold=%g, M=%d)"%(g.gini_threshold, g.M))
    plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-Genie1)=
```{figure} noise-figures/noise-Genie1-3.*
Labels predicted by Genie with noise point detection.
```

For a more natural look-and-feel, it can be a good idea to first identify
the noise points with Genie, remove them from the data set (at least temporarily),
and then apply the clustering procedure once again
(did we mention that our algorithm is fast?)
but now with respect to the original distance (here: Euclidean):


``` python
# Step 1: Noise point identification
g1 = genieclust.Genie(n_clusters=n_clusters, M=50)
labels_noise = g1.fit_predict(X)
non_noise = (labels_noise >= 0) # True == non-noise point
# Step 2: Clustering of non-noise points:
g2 = genieclust.Genie(n_clusters=n_clusters)
labels_genie = g2.fit_predict(X[non_noise, :])
# Replace old labels with the new ones:
labels_noise[non_noise] = labels_genie
# Scatter plot:
genieclust.plots.plot_scatter(X, labels=labels_noise, alpha=0.5)
plt.title("(gini_threshold=%g, noise points removed first; M=%d)"%(g2.gini_threshold, g1.M))
plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-Genie2)=
```{figure} noise-figures/noise-Genie2-5.*
Labels predicted by Genie when noise points were removed from the dataset.
```


Contrary to the excellent implementation of HDBSCAN\*
that is featured in the [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
package {cite}`hdbscanpkg` and which also relies on a minimum spanning tree
with respect to $d_M$, here,
we still have the hierarchical Genie {cite}`genieins` algorithm under the hood.
It means that we can request a *specific* number of clusters.
Moreover, we can easily switch between partitions
of finer or coarser granularity.



``` python
ncs = [5, 6, 7, 8, 10, 15]
for i in range(len(ncs)):
    g = genieclust.Genie(n_clusters=ncs[i])
    labels_genie = g.fit_predict(X[non_noise, :])
    plt.subplot(3, 2, i+1)
    labels_noise[non_noise] = labels_genie
    genieclust.plots.plot_scatter(X, labels=labels_noise, alpha=0.5)
    plt.title("(n_clusters=%d)"%(g.n_clusters))
    plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-Genie3)=
```{figure} noise-figures/noise-Genie3-7.*
Labels predicted by Genie when noise points were removed from the dataset – different number of clusters requested.
```



## A Comparision with HDBSCAN\*


Here are the results returned by `hdbscan` with default parameters:


``` python
import hdbscan
```


``` python
h = hdbscan.HDBSCAN()
labels_hdbscan = h.fit_predict(X)
```

``` python
genieclust.plots.plot_scatter(X, labels=labels_hdbscan, alpha=0.5)
plt.title("(min_cluster_size=%d, min_samples=%d)" % (
    h.min_cluster_size, h.min_samples or h.min_cluster_size))
plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-HDBSCAN1)=
```{figure} noise-figures/noise-HDBSCAN1-9.*
Labels predicted by HDBSCAN\*.
```


By tuning up `min_cluster_size` and/or `min_samples` (which corresponds to our `M` parameter;
by the way, `min_samples` defaults to `min_cluster_size` if not provided explicitly),
we can obtain a partition that is even closer to the reference one:



``` python
mcss = [5, 10, 25]
mss = [5, 10]
for i in range(len(mcss)):
    for j in range(len(mss)):
        h = hdbscan.HDBSCAN(min_cluster_size=mcss[i], min_samples=mss[j])
        labels_hdbscan = h.fit_predict(X)
        plt.subplot(3, 2, i*len(mss)+j+1)
        genieclust.plots.plot_scatter(X, labels=labels_hdbscan, alpha=0.5)
        plt.title("(min_cluster_size=%d, min_samples=%d)" % (
            h.min_cluster_size, h.min_samples or h.min_cluster_size))
        plt.axis("equal")
```

``` python
plt.show()
```

(fig:noise-HDBSCAN2)=
```{figure} noise-figures/noise-HDBSCAN2-11.*
Labels predicted by HDBSCAN\* – different settings.
```

Neat.

