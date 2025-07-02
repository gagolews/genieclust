



# Clustering with Noise Point Detection


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













