Example: String Data and Grouping of DNA
========================================

The *genieclust* package also allows for clustering of character string
data. Let's perform an example grouping based
on `Levenshtein's <https://en.wikipedia.org/wiki/Levenshtein_distance>`_ edit
distance.

.. important::

    Make sure that the *nmslib* package (an optional dependency) is installed.


We will use one of the benchmark datasets mentioned in :cite:`genieins`
as an example:









.. code-block:: python

    import numpy as np
    # see https://github.com/gagolews/genieclust/tree/master/.devel/sphinx/weave/
    strings = np.loadtxt("actg1.data.gz", dtype=np.str).tolist()
    strings[:5] # preview


::

    ## /tmp/ipykernel_42024/2791853717.py:3: FutureWarning: In the future
    ## `np.str` will be defined as the corresponding NumPy scalar.
    ##   strings = np.loadtxt("actg1.data.gz", dtype=np.str).tolist()

::

    ## ---------------------------------------------------------------------------AttributeError
    ## Traceback (most recent call last)Cell In[1], line 3
    ##       1 import numpy as np
    ##       2 # see
    ## https://github.com/gagolews/genieclust/tree/master/.devel/sphinx/weave/
    ## ----> 3 strings = np.loadtxt("actg1.data.gz", dtype=np.str).tolist()
    ##       4 strings[:5] # preview
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/numpy/__init__.py:319, in __getattr__(attr)
    ##     314     warnings.warn(
    ##     315         f"In the future `np.{attr}` will be defined as the "
    ##     316         "corresponding NumPy scalar.", FutureWarning,
    ## stacklevel=2)
    ##     318 if attr in __former_attrs__:
    ## --> 319     raise AttributeError(__former_attrs__[attr])
    ##     321 if attr == 'testing':
    ##     322     import numpy.testing as testing
    ## AttributeError: module 'numpy' has no attribute 'str'.
    ## `np.str` was a deprecated alias for the builtin `str`. To avoid this
    ## error in existing code, use `str` by itself. Doing this will not
    ## modify any behavior and is safe. If you specifically wanted the numpy
    ## scalar type, use `np.str_` here.
    ## The aliases was originally deprecated in NumPy 1.20; for more details
    ## and guidance see the original release note at:
    ##     https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations



It comes with a set of reference labels, giving the "true" grouping assigned
by an expert:


.. code-block:: python

    labels_true = np.loadtxt("actg1.labels0.gz", dtype=np.intp)-1
    n_clusters = len(np.unique(labels_true))
    print(n_clusters)


::

    ## 20




Clustering in the string domain relies on the
near-neighbour search routines implemented in the *nmslib* package.


.. code-block:: python

    import genieclust
    g = genieclust.Genie(
        n_clusters=n_clusters,
        exact=False, # use nmslib
        cast_float32=False, # do not convert the string list to a matrix
        nmslib_params_index=dict(post=0), # faster
        affinity="leven")
    labels_pred = g.fit_predict(strings)


::

    ## ---------------------------------------------------------------------------NameError
    ## Traceback (most recent call last)Cell In[1], line 8
    ##       1 import genieclust
    ##       2 g = genieclust.Genie(
    ##       3     n_clusters=n_clusters,
    ##       4     exact=False, # use nmslib
    ##       5     cast_float32=False, # do not convert the string list to a
    ## matrix
    ##       6     nmslib_params_index=dict(post=0), # faster
    ##       7     affinity="leven")
    ## ----> 8 labels_pred = g.fit_predict(strings)
    ## NameError: name 'strings' is not defined




The adjusted Rand index can be used as an external cluster validity metric:


.. code-block:: python

    genieclust.compare_partitions.adjusted_rand_score(labels_true, labels_pred)


::

    ## ---------------------------------------------------------------------------NameError
    ## Traceback (most recent call last)Cell In[1], line 1
    ## ----> 1 genieclust.compare_partitions.adjusted_rand_score(labels_true,
    ## labels_pred)
    ## NameError: name 'labels_pred' is not defined



This indicates a very high degree of similarity between the reference
and the obtained clusterings.
