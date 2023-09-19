Example: Sparse Data and Movie Recommendation
=============================================

To illustrate how *genieclust* handles
`sparse data <https://en.wikipedia.org/wiki/Sparse_matrix>`_,
let's perform a simple exercise in movie recommendation based on
`MovieLens <https://grouplens.org/datasets/movielens/latest/>`_ data.

.. important::

    Make sure that the *nmslib* package (an optional dependency) is installed.



.. code-block:: python

    import numpy as np
    import scipy.sparse
    import pandas as pd








First we load the `ratings` data frame
and map the movie IDs to consecutive integers.


.. code-block:: python

    ratings = pd.read_csv("ml-9-2018-small/ratings.csv")
    ratings["movieId"] -= 1
    ratings["userId"] -= 1
    old_movieId_map = np.unique(ratings["movieId"])
    ratings["movieId"] = np.searchsorted(old_movieId_map, ratings["movieId"])
    ratings.head()


::

    ##    userId  movieId  rating  timestamp
    ## 0       0        0     4.0  964982703
    ## 1       0        2     4.0  964981247
    ## 2       0        5     4.0  964982224
    ## 3       0       43     5.0  964983815
    ## 4       0       46     5.0  964982931



Then we read the movie metadata and transform the movie IDs
in the same way:


.. code-block:: python

    movies = pd.read_csv("ml-9-2018-small/movies.csv")
    movies["movieId"] -= 1
    movies = movies.loc[movies.movieId.isin(old_movieId_map), :]
    movies["movieId"] = np.searchsorted(old_movieId_map, movies["movieId"])
    movies.iloc[:, :2].head()


::

    ##    movieId                               title
    ## 0        0                    Toy Story (1995)
    ## 1        1                      Jumanji (1995)
    ## 2        2             Grumpier Old Men (1995)
    ## 3        3            Waiting to Exhale (1995)
    ## 4        4  Father of the Bride Part II (1995)




Conversion of ratings to a CSR-format sparse matrix:


.. code-block:: python

    n = ratings.movieId.max()+1
    d = ratings.userId.max()+1
    X = scipy.sparse.dok_matrix((n,d), dtype=np.float32)
    X[ratings.movieId, ratings.userId] = ratings.rating
    X = X.tocsr()
    print(repr(X))


::

    ## <9724x610 sparse matrix of type '<class 'numpy.float32'>'
    ##         with 100836 stored elements in Compressed Sparse Row format>




First few observations:


.. code-block:: python

    X[:5, :10].todense()


::

    ## matrix([[4. , 0. , 0. , 0. , 4. , 0. , 4.5, 0. , 0. , 0. ],
    ##         [0. , 0. , 0. , 0. , 0. , 4. , 0. , 4. , 0. , 0. ],
    ##         [4. , 0. , 0. , 0. , 0. , 5. , 0. , 0. , 0. , 0. ],
    ##         [0. , 0. , 0. , 0. , 0. , 3. , 0. , 0. , 0. , 0. ],
    ##         [0. , 0. , 0. , 0. , 0. , 5. , 0. , 0. , 0. , 0. ]],
    ## dtype=float32)



Let's extract 200 clusters with Genie with respect to the cosine similarity between films' ratings
as given by users (two movies considered similar if they get similar reviews).
Sparse inputs are supported by the approximate version of the algorithm
which relies on the near-neighbour search routines implemented in the *nmslib* package.



.. code-block:: python

    import genieclust
    g = genieclust.Genie(n_clusters=200, exact=False, affinity="cosinesimil_sparse")
    movies["cluster"] = g.fit_predict(X)


::

    ## ---------------------------------------------------------------------------ValueError
    ## Traceback (most recent call last)Cell In[1], line 3
    ##       1 import genieclust
    ##       2 g = genieclust.Genie(n_clusters=200, exact=False,
    ## affinity="cosinesimil_sparse")
    ## ----> 3 movies["cluster"] = g.fit_predict(X)
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:548, in GenieBase.fit_predict(self, X, y)
    ##     520 def fit_predict(self, X, y=None):
    ##     521     """
    ##     522     Perform cluster analysis of a dataset and return the
    ## predicted labels.
    ##     523
    ##    (...)
    ##     546
    ##     547     """
    ## --> 548     self.fit(X)
    ##     549     return self.labels_
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:1051, in Genie.fit(self, X, y)
    ##     972 """
    ##     973 Perform cluster analysis of a dataset.
    ##     974
    ##    (...)
    ##    1047
    ##    1048 """
    ##    1049 cur_state = self._check_params()  # re-check, they might have
    ## changed
    ## -> 1051 cur_state = self._get_mst(X, cur_state)
    ##    1053 if cur_state["verbose"]:
    ##    1054     print("[genieclust] Determining clusters with Genie++.",
    ## file=sys.stderr)
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:511, in GenieBase._get_mst(self, X,
    ## cur_state)
    ##     509     cur_state = self._get_mst_exact(X, cur_state)
    ##     510 else:
    ## --> 511     cur_state = self._get_mst_approx(X, cur_state)
    ##     513 # this might be an "intrinsic" dimensionality:
    ##     514 self.n_features_  = cur_state["n_features"]
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:380, in GenieBase._get_mst_approx(self,
    ## X, cur_state)
    ##     378 def _get_mst_approx(self, X, cur_state):
    ##     379     if nmslib is None:
    ## --> 380         raise ValueError("Package `nmslib` is not available.")
    ##     382     if cur_state["affinity"] == "precomputed":
    ##     383         raise ValueError(
    ##     384             "`affinity` of 'precomputed' can only be used "
    ##     385             "with `exact` = True.")
    ## ValueError: Package `nmslib` is not available.



Here are the members of an example cluster:


.. code-block:: python

    movies["cluster"] = g.fit_predict(X)
    which_cluster = movies.cluster[movies.title=="Monty Python's The Meaning of Life (1983)"]
    movies.loc[movies.cluster == int(which_cluster)].title.sort_values()


::

    ## ---------------------------------------------------------------------------ValueError
    ## Traceback (most recent call last)Cell In[1], line 1
    ## ----> 1 movies["cluster"] = g.fit_predict(X)
    ##       2 which_cluster = movies.cluster[movies.title=="Monty Python's
    ## The Meaning of Life (1983)"]
    ##       3 movies.loc[movies.cluster ==
    ## int(which_cluster)].title.sort_values()
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:548, in GenieBase.fit_predict(self, X, y)
    ##     520 def fit_predict(self, X, y=None):
    ##     521     """
    ##     522     Perform cluster analysis of a dataset and return the
    ## predicted labels.
    ##     523
    ##    (...)
    ##     546
    ##     547     """
    ## --> 548     self.fit(X)
    ##     549     return self.labels_
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:1051, in Genie.fit(self, X, y)
    ##     972 """
    ##     973 Perform cluster analysis of a dataset.
    ##     974
    ##    (...)
    ##    1047
    ##    1048 """
    ##    1049 cur_state = self._check_params()  # re-check, they might have
    ## changed
    ## -> 1051 cur_state = self._get_mst(X, cur_state)
    ##    1053 if cur_state["verbose"]:
    ##    1054     print("[genieclust] Determining clusters with Genie++.",
    ## file=sys.stderr)
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:511, in GenieBase._get_mst(self, X,
    ## cur_state)
    ##     509     cur_state = self._get_mst_exact(X, cur_state)
    ##     510 else:
    ## --> 511     cur_state = self._get_mst_approx(X, cur_state)
    ##     513 # this might be an "intrinsic" dimensionality:
    ##     514 self.n_features_  = cur_state["n_features"]
    ## File ~/.virtualenvs/python3-default/lib/python3.11/site-
    ## packages/genieclust/genie.py:380, in GenieBase._get_mst_approx(self,
    ## X, cur_state)
    ##     378 def _get_mst_approx(self, X, cur_state):
    ##     379     if nmslib is None:
    ## --> 380         raise ValueError("Package `nmslib` is not available.")
    ##     382     if cur_state["affinity"] == "precomputed":
    ##     383         raise ValueError(
    ##     384             "`affinity` of 'precomputed' can only be used "
    ##     385             "with `exact` = True.")
    ## ValueError: Package `nmslib` is not available.






The above was performed on an abridged version of the MovieLens dataset.
The project's `website <https://grouplens.org/datasets/movielens/latest/>`_
also features a full database that yields a 53,889x283,228 ratings table
(with 27,753,444  non-zero elements) -- such a matrix would definitely
not fit into our RAM if it was in the dense form.
Determining the whole cluster hierarchy takes only 144 seconds.
Here is one of 500 clusters extracted:

.. code::

    ## 13327                       Blackadder Back & Forth (1999)
    ## 13328                  Blackadder's Christmas Carol (1988)
    ## 3341                              Creature Comforts (1989)
    ## 1197       Grand Day Out with Wallace and Gromit, A (1989)
    ## 2778                            Hard Day's Night, A (1964)
    ## 2861                                          Help! (1965)
    ## 2963                              How I Won the War (1967)
    ## 6006        Monty Python Live at the Hollywood Bowl (1982)
    ## 1113                Monty Python and the Holy Grail (1975)
    ## 2703     Monty Python's And Now for Something Completel...
    ## 1058                   Monty Python's Life of Brian (1979)
    ## 6698             Monty Python's The Meaning of Life (1983)
    ## 27284                                  Oliver Twist (1997)
    ## 2216                                 Producers, The (1968)
    ## 4716                                   Quadrophenia (1979)
    ## 6027             Secret Policeman's Other Ball, The (1982)
    ## 27448                                    The Basket (2000)
    ## 2792                                          Tommy (1975)
    ## 10475    Wallace & Gromit in The Curse of the Were-Rabb...
    ## 732                 Wallace & Gromit: A Close Shave (1995)
    ## 708      Wallace & Gromit: The Best of Aardman Animatio...
    ## 1125           Wallace & Gromit: The Wrong Trousers (1993)
    ## 13239    Wallace and Gromit in 'A Matter of Loaf and De...
    ## 2772                               Yellow Submarine (1968)
    ## 1250                             Young Frankenstein (1974)
    ## Name: title, dtype: object



