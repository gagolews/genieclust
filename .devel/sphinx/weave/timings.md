



# Timings (How Fast Is It?)

::::{note}
This section needs to be extended.
::::



Time to perform a cluster analysis of a dataset consisting of
1M points in $\mathbb{R}^2$:


``` python
import time
import numpy as np
import genieclust
np.random.seed(123)
n = 1_000_000
d = 2
X = np.random.randn(n, d)
t0 = time.time()
g = genieclust.Genie(n_clusters=2)
g.fit(X)
print("Elapsed time: %.2f secs." % (time.time()-t0))
## Genie()
## Elapsed time: 2.29 secs.
```

Note that due to the curse of dimensionality, processing
data with high intrinsic dimensionality is slower.
