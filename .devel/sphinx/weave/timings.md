



# Timings (How Fast Is It?)

::::{note}
This section needs to be extended.
::::





Thanks to [**quitefastmst**](https://quitefastmst.gagolewski.com/),
the time to perform cluster analysis is pretty low in spaces
of low intrinsic dimensionality.

Let's perform a test on a dataset consisting of 1M points in $\mathbb{R}^2$:


``` python
import time
import numpy as np
import genieclust
np.random.seed(123)
n = 1_000_000
d = 2
X = np.random.randn(n, d)
```

Genie:


``` python
t0 = time.time()
g = genieclust.Genie(n_clusters=2)
g.fit(X)
print("Elapsed time: %.2f secs." % (time.time()-t0))
## Genie(quitefastmst_params={})
## Elapsed time: 2.17 secs.
```

Due to the curse of dimensionality, processing
datasets with high intrinsic dimensionality is slower.

A comparison against k-means:


``` python
import sklearn.cluster
t0 = time.time()
k = sklearn.cluster.KMeans(n_clusters=2)
k.fit(X)
print("Elapsed time: %.2f secs." % (time.time()-t0))
## KMeans(n_clusters=2)
## Elapsed time: 0.16 secs.
```


A comparison against **HDBSCAN**:


``` python
import hdbscan
t0 = time.time()
h = hdbscan.HDBSCAN()
h.fit(X)
print("Elapsed time: %.2f secs." % (time.time()-t0))
## /home/gagolews/.local/lib/python3.13/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
##   warnings.warn(
## /home/gagolews/.local/lib/python3.13/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
##   warnings.warn(
## HDBSCAN()
## Elapsed time: 81.97 secs.
```
