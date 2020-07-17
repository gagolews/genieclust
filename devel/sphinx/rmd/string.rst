Example: String Data and Grouping of DNA
========================================

The *genieclust* package also allows for clustering of character string
data. Let's perform an example grouping based
on `Levenshtein's <https://en.wikipedia.org/wiki/Levenshtein_distance>`_ edit
distance.


.. code:: python

    import numpy as np








We'll use one of the benchmark datasets mentioned in [1]_ as an example:


.. code:: python

    strings = np.loadtxt("actg1.data.gz", dtype=np.str).tolist()
    strings[:5] # preview


.. code::

    ['tataacaaccctgattacatcaagctacgctccggtgcgttgcctcggacgagtgctaatccctccccactgactgtattcatcttgacaata',
    'atgtctccaaagcgtgaccttctagacccgagacgacatatggaggcttggagccgtacctgtgtgaggaaactgtagtacccaaagctattca',
    'gcaattgaagtccagatctaggtatcgtccaagcatattgcctttaagaaatatatttgaccctgtctcttcgtggaggtacacgtcacggaatcgtaagatttccttgg',
    'gacaattatcgcggctttcgccatgcagagtctcgtacaatttgtttcacgcccaatattttccgtgcttcgcgagctaggcagccagggcatttttgga',
    'ttagagcgcttaaccccacaggaaccgagttcccctcatgtggcaaggttctcccgcctcaggtatcacagaaacaaggtatgtagccctaggctacgagc']
    



It comes with a set of reference labels, giving the "true" grouping assigned
by an expert:


.. code:: python

    labels_true = np.loadtxt("actg1.labels0.gz", dtype=np.intp)-1
    n_clusters = len(np.unique(labels_true))
    print(n_clusters)


.. code::

    20
    




Clustering in the string domain requires `nmslib` installed.


.. code:: python

    import genieclust
    g = genieclust.Genie(
        n_clusters=n_clusters,
        exact=False, # use nmslib
        cast_float32=False, # do not convert the string list to a matrix
        nmslib_params_index=dict(post=0), # faster
        affinity="leven")
    labels_pred = g.fit_predict(strings)





The adjusted Rand index can be used as an external cluster validity metric:


.. code:: python

    genieclust.compare_partitions.adjusted_rand_score(labels_true, labels_pred)


.. code::

    0.9352814722212013
    



This indicates a very high degree of similarity between the reference
and the obtained clusterings.



References
----------

.. [1]
    Gagolewski M., Cena A. (Eds.), *Benchmark Suite for Clustering Algorithms — Version 1*,
    2020. https://github.com/gagolews/clustering_benchmarks_v1. doi:10.5281/zenodo.3815066.
