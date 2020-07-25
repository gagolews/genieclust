Benchmarks — Approximate Method
===============================

In one of the :any:`previous sections <timings>` we have demonstrated that the approximate version
of the Genie algorithm (:class:`genieclust.Genie(exact=False, ...) <genieclust.Genie>`), i.e.,
one which relies on `nmslib <https://github.com/nmslib/nmslib/tree/master/python_bindings>`_\ 's
approximate nearest neighbour search, is much faster than the exact one
on large, high-dimensional datasets. In particular, we have noted that
clustering of 1 million points in a 100d Euclidean space
takes less than 5 minutes on a laptop.

As *fast* does not necessarily mean *meaningful* (tl;dr spoiler alert: in our case, it does),
let's again consider  all the datasets
from the `Benchmark Suite for Clustering Algorithms — Version 1 <https://github.com/gagolews/clustering_benchmarks_v1>`_
:cite:`clustering_benchmarks_v1`
(except the ``h2mg`` and ``g2mg`` batteries). Features with variance of 0 were
removed, datasets were centred at **0** and scaled so that they have total
variance of 1. Tiny bit of Gaussian noise was added to each observation.
Clustering is performed with respect to the Euclidean distance.
















On each benchmark dataset ("small" and "large" altogether)
we have fired 10 runs of the approximate Genie method (``exact=False``)
and computed the adjusted Rand (AR) indices to quantify the similarity between the predicted
outputs and the reference ones.

We've computed the differences between each of the 10 AR indices
and the AR index for the exact method. Here is the complete list of datasets
and `gini_threshold`\ s where this discrepancy is seen at least 2 digits of precision:



================  ================  =======  ======  =====  =====  =====  =====  =====  =====
dataset             gini_threshold    count    mean    std    min    25%    50%    75%    max
================  ================  =======  ======  =====  =====  =====  =====  =====  =====
sipu/birch2                    0.7       10   -0.01   0.01  -0.02  -0.02  -0.01  -0.01   0
..                             1         10   -0.35   0.18  -0.44  -0.44  -0.43  -0.43   0
sipu/worms_64                  0.1       10   -0.03   0.01  -0.06  -0.03  -0.02  -0.02  -0.02
..                             0.3       10    0.02   0.01  -0.01   0.02   0.03   0.03   0.03
..                             0.5       10    0.23   0.08   0.11   0.16   0.25   0.29   0.34
wut/trajectories               0.1       10   -0      0.02  -0.05   0      0      0      0
..                             0.3       10   -0      0.02  -0.05   0      0      0      0
..                             0.5       10   -0      0.02  -0.05   0      0      0      0
..                             0.7       10   -0      0.02  -0.05   0      0      0      0
..                             1         10   -0.1    0.32  -1      0      0      0      0
================  ================  =======  ======  =====  =====  =====  =====  =====  ===== 





The only noteworthy  difference is for the ``sipu/birch2`` dataset
where we observe that the approximate method generates worse results
(although recall that `gini_threshold` of 1 corresponds to the single linkage method).
Interestingly, for ``sipu/worms_64``, the in-exact algorithm with `gini_threshold`
of 0.5 yields a much better outcome than the original one.


Here are the descriptive statistics for the AR indices across all the datasets
(for the approximate method we chose the median AR in each of the 10 runs):



================  =======  ======  =====  =====  =====  =====  =====  =====
method              count    mean    std    min    25%    50%    75%    max
================  =======  ======  =====  =====  =====  =====  =====  =====
Genie_0.1              79   0.728  0.307      0  0.516  0.844      1      1
Genie_0.1_approx       79   0.728  0.307      0  0.516  0.844      1      1
Genie_0.3              79   0.755  0.292      0  0.555  0.9        1      1
Genie_0.3_approx       79   0.755  0.292      0  0.568  0.9        1      1
Genie_0.5              79   0.731  0.332      0  0.531  0.844      1      1
Genie_0.5_approx       79   0.734  0.326      0  0.531  0.844      1      1
Genie_0.7              79   0.624  0.376      0  0.264  0.719      1      1
Genie_0.7_approx       79   0.624  0.376      0  0.264  0.719      1      1
Genie_1.0              79   0.415  0.447      0  0      0.174      1      1
Genie_1.0_approx       79   0.409  0.45       0  0      0.148      1      1
================  =======  ======  =====  =====  =====  =====  =====  ===== 





For the recommended ranges of the `gini_threshold` parameter,
i.e., between 0.1 and 0.5, we see that the approximate version of Genie
behaves as good as the original one.
