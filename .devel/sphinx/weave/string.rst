Example: String Data and Grouping of DNA
========================================

The *genieclust* package also allows for clustering of character string
data. Let's perform an example grouping based
on `Levenshtein's <https://en.wikipedia.org/wiki/Levenshtein_distance>`_ edit
distance.

We'll use one of the benchmark datasets mentioned in :cite:`genieins`
as an example:









.. code-block:: python

    import numpy as np
    # see https://github.com/gagolews/genieclust/tree/master/devel/sphinx/rmd
    strings = np.loadtxt("actg1.data.gz", dtype=np.str).tolist()
    strings[:5] # preview


::

    ## /tmp/ipykernel_56999/1616393685.py:3: DeprecationWarning: `np.str` is
    ## a deprecated alias for the builtin `str`. To silence this warning, use
    ## `str` by itself. Doing this will not modify any behavior and is safe.
    ## If you specifically wanted the numpy scalar type, use `np.str_` here.
    ## Deprecated in NumPy 1.20; for more details and guidance:
    ## https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
    ##   strings = np.loadtxt("actg1.data.gz", dtype=np.str).tolist()

::

    ## ['tataacaaccctgattacatcaagctacgctccggtgcgttgcctcggacgagtgctaatccctccccactgactgtattcatcttgacaata',
    ## 'atgtctccaaagcgtgaccttctagacccgagacgacatatggaggcttggagccgtacctgtgtgaggaaactgtagtacccaaagctattca',
    ## 'gcaattgaagtccagatctaggtatcgtccaagcatattgcctttaagaaatatatttgaccctgtctcttcgtggaggtacacgtcacggaatcgtaagatttccttgg',
    ## 'gacaattatcgcggctttcgccatgcagagtctcgtacaatttgtttcacgcccaatattttccgtgcttcgcgagctaggcagccagggcatttttgga',
    ## 'ttagagcgcttaaccccacaggaaccgagttcccctcatgtggcaaggttctcccgcctcaggtatcacagaaacaaggtatgtagccctaggctacgagc']



It comes with a set of reference labels, giving the "true" grouping assigned
by an expert:


.. code-block:: python

    labels_true = np.loadtxt("actg1.labels0.gz", dtype=np.intp)-1
    n_clusters = len(np.unique(labels_true))
    print(n_clusters)


::

    ## 20




Clustering in the string domain relies on the
near-neighbour search routines implemented in the `nmslib` package.


.. code-block:: python

    import genieclust
    g = genieclust.Genie(
        n_clusters=n_clusters,
        exact=False, # use nmslib
        cast_float32=False, # do not convert the string list to a matrix
        nmslib_params_index=dict(post=0), # faster
        affinity="leven")
    labels_pred = g.fit_predict(strings)





The adjusted Rand index can be used as an external cluster validity metric:


.. code-block:: python

    genieclust.compare_partitions.adjusted_rand_score(labels_true, labels_pred)


::

    ## 0.9352814722212013



This indicates a very high degree of similarity between the reference
and the obtained clusterings.
