Timings
=======


**TODO: under construction.**

We consider the "larger" datasets (70000-105600 observations,
see section on Benchmark Results for discussion)
from the
`Benchmark Suite for Clustering Algorithms — Version 1 <https://github.com/gagolews/clustering_benchmarks_v1>`_ [1]_. Features with variance of 0 were removed,
datasets were centred at **0** and scaled so that they have total variance is 1.
Tiny bit of Gaussian noise was added to each observation.
Clustering is performed with respect to the Euclidean distance.

Comparison with k-means from `scikit-learn <https://scikit-learn.org/>`_ version 0.23.1
(`sklearn.cluster.KMeans`)
for different number of threads (default is to use all available threads).
Note that `n_init` defaults to 10.

`mlpack`'s emst is used for low-dimensional spaces.

Timings were performed on a PC running GNU/Linux 5.4.0-40-generic #44-Ubuntu SMP x86_64 kernel with an Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (12M cache, 6 cores, 12 threads)
and total memory of 16242084 kB.







Minimum of 3 run times (**TODO**: except mnist/digits and mnist/fashion):







Results (in seconds) as a function of the number of clusters to detect (6 threads):



=============  ======  ===  ================  ======  ======  =======
dataset        n       d    method                10     100     1000
=============  ======  ===  ================  ======  ======  =======
mnist/digits   70000   719  Genie_0.3         429.79  430.09   429.98
..                          Genie_0.3_approx   42.87   42.92    42.84
..                          sklearn_kmeans     26.3   217.62  1691.68
mnist/fashion  70000   784  Genie_0.3         464.27  464.52   464.37
..                          Genie_0.3_approx   38.18   38.18    38.17
..                          sklearn_kmeans     25.75  225.58  1745.88
sipu/birch1    100000  2    Genie_0.3           0.52    0.52     0.52
..                          Genie_0.3_approx    2.86    2.85     2.9
..                          sklearn_kmeans      0.88    6.86   101.88
sipu/birch2    100000  2    Genie_0.3           0.46    0.46     0.46
..                          Genie_0.3_approx    3.3     3.7      3.41
..                          sklearn_kmeans      0.53    5.01    61.62
sipu/worms_2   105600  2    Genie_0.3           0.57    0.57     0.57
..                          Genie_0.3_approx    3.67    3.7      3.7
..                          sklearn_kmeans      0.86   10.96   111.9
sipu/worms_64  105000  64   Genie_0.3          76.7    76.54    76.64
..                          Genie_0.3_approx    8.26    8.3      8.31
..                          sklearn_kmeans      3.35   37.89   357.84
=============  ======  ===  ================  ======  ======  ======= 






Number of threads (jobs):


.. figure:: figures/timings_timings-plot_1.png
   :width: 15 cm

   Timings [s] as a function of the number of clusters and threads.



**TODO**: for Genie, the number of clusters to extract does not affect
the run-time. Genie itself has :math:`O(n \sqrt{n})` time complexity.

**TODO**: mention cache, show timings — once we determine the MST,
we can play with different `gini_threshold`\ s for "free".


The effect of the curse of dimensionality is clearly visible -- clustering
in very low-dimensional Euclidean spaces is extremely fast.
Then the timings become grow linearly as a function of dimensionality, `d` --
:math:`O(d n^2)` time is needed.

Importantly, the algorithm only needs :math:`O(n)` memory.




Timings as a Function of `n` and `d`
------------------------------------

Synthetic datasets being two Gaussian blobs, each of size `n/2`
(with i.i.d. coordinates), in a `d`-dimensional space.

Medians of 1,3, or 10 timings (depending on the dataset size), in seconds,
on 6 threads:



==================  ===  =======  =======  ========  ========  =========
method                d    10000    50000    100000    500000    1000000
==================  ===  =======  =======  ========  ========  =========
Genie_0.3_approx      2     0.17     0.98      2.12     14.93      33.79
..                    5     0.2      1.3       2.87     22.75      54.66
..                   10     0.25     1.69      3.84     36.18      92.03
..                   25     0.29     1.95      5.46     62.25     158.21
..                   50     0.36     3.15      8.15     83.15     202.32
..                  100     0.48     4.6      12.6     114.68     nan
Genie_0.3_mlpack      2     0.04     0.26      0.55      3.03       6.58
..                    5     0.28     1.96      4.46     28.4       62.75
..                   10     3.08    35.54     92.87    794.71    2014.59
Genie_0.3_nomlpack    2     0.16     2.52      9.87    267.76    1657.86
..                    5     0.14     2.62     11.4     421.46    2997.11
..                   10     0.15     3.21     12.74    719.33    4388.26
..                   25     0.28     6.51     26.65   1627.9     7708.23
..                   50     0.47    11.97     54.52   2646.97   11346.3
..                  100     1       26.07    132.47   4408.07     nan
==================  ===  =======  =======  ========  ========  ========= 




By default, `mlpack_enabled` is ``"auto"``, which translates
to ``True`` if the requested metric is Euclidean, `mlpack` Python package is available
and `d` is not greater than 6.



.. figure:: figures/timings_g2mg-plot_1.png
   :width: 15 cm

   Timings [s] as a function of the dataset size and dimensionality — problem sizes that can be solved within a coffee-break.




Benchmarking Approximate Version
--------------------------------

**TODO:**
On each benchmark dataset ("small" and "large" altogether)
we have fired 10 runs of the approximate Genie method ``exact=False``.






A complete list of datasets and thresholds where the approximate method
yields a difference in AR-indices (AR index of one of 10 runs of the approximate
method runs minus the AR index for the exact method) detectable at 2 digits of precision:



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





The only noteworthy  difference is for the ``sipu/birch`` dataset
where we observe that the approximate method generates worse results
(although recall that `gini_threshold` of 1 corresponds to the single linkage method).
Interestingly, for ``sipu/worms2``, `gini_threshold` of 0.5 behaves much better.



Descriptive statistics for AR-indices (for the approximate method we choose
the median AR in each of the 10 runs):



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






References
----------

.. [1]
    Gagolewski M., Cena A. (Eds.), *Benchmark Suite for Clustering Algorithms — Version 1*,
    2020. https://github.com/gagolews/clustering_benchmarks_v1. doi:10.5281/zenodo.3815066.
