



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





























