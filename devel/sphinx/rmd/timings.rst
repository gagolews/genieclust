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

Minimum of 3 run times on a PC running GNU/Linux 5.4.0-40-generic #44-Ubuntu SMP x86_64 kernel with an Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (12M cache, 6 cores, 12 threads)
and total memory of 16242084 kB.












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


.. image:: figures/timings_timings-plot_1.png
   :width: 15 cm



**TODO**: for Genie, the number of clusters to extract does not affect
the run-time. Genie itself has :math:`O(n \sqrt{n})` time complexity.

**TODO**: mention cache, show timings — once we determine the MST,
we can play with different `gini_threshold`\ s for "free".


The effect of the curse of dimensionality is clearly visible -- clustering
in very low-dimensional Euclidean spaces is extremely fast.
Then the timings become grow linearly as a function of dimensionality, `d` --
:math:`O(d n^2)` time is needed.

Importantly, the algorithm only needs :math:`O(n)` memory.





References
----------

.. [1]
    Gagolewski M., Cena A. (Eds.), *Benchmark Suite for Clustering Algorithms — Version 1*,
    2020. https://github.com/gagolews/clustering_benchmarks_v1. doi:10.5281/zenodo.3815066.
