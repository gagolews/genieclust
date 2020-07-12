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



=============  ======  ===  ==============  ======  ======  =======
dataset        n       d    method              10     100     1000
=============  ======  ===  ==============  ======  ======  =======
mnist/digits   70000   719  Genie_0.3       429.79  430.09   429.98
..                          sklearn_kmeans   26.3   217.62  1691.68
mnist/fashion  70000   784  Genie_0.3       464.27  464.52   464.37
..                          sklearn_kmeans   25.75  225.58  1745.88
sipu/birch1    100000  2    Genie_0.3         0.52    0.52     0.52
..                          sklearn_kmeans    0.88    6.86   101.88
sipu/birch2    100000  2    Genie_0.3         0.46    0.46     0.46
..                          sklearn_kmeans    0.53    5.01    61.62
sipu/worms_2   105600  2    Genie_0.3         0.57    0.57     0.57
..                          sklearn_kmeans    0.86   10.96   111.9
sipu/worms_64  105000  64   Genie_0.3        76.7    76.54    76.64
..                          sklearn_kmeans    3.35   37.89   357.84
=============  ======  ===  ==============  ======  ======  ======= 






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

Synthetic datasets being two Guassian blobs, each of size `n/2`
(with i.i.d. coordinates), in a `d`-dimensional space.

Medians of 1,3, or 10 timings (depending on the dataset size), in seconds:



==================  ===  =======  =======  ========  ========  =========
method                d    10000    50000    100000    500000    1000000
==================  ===  =======  =======  ========  ========  =========
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
to ``True`` if the requested metric in Euclidean, `mlpack` Python package is available
and `d` is not greater than 6.



.. figure:: figures/timings_g2mg-plot_1.png
   :width: 15 cm

   Timings [s] as a function of the dataset size and dimensionality. Note the log-log scale.





References
----------

.. [1]
    Gagolewski M., Cena A. (Eds.), *Benchmark Suite for Clustering Algorithms — Version 1*,
    2020. https://github.com/gagolews/clustering_benchmarks_v1. doi:10.5281/zenodo.3815066.
