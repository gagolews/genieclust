What’s New in *genieclust*
==========================

genieclust 0.9.9 (2021-XX-XX)
-----------------------------

-  …

genieclust 0.9.8 (2021-01-08)
-----------------------------

-  [Python] Python >= 3.7 is now required (implied by ``numpy``).

-  [Python] ``nmslib`` is now required.

-  [R] Use ``RcppMLPACK`` directly; remove dependency on ``emstreeR``.

-  [R] Switched to ``tinytest`` for unit testing.

genieclust 0.9.4 (2020-07-31)
-----------------------------

-  [Bugfix] [R] Fix build errors on Solaris.

genieclust 0.9.3 (2020-07-25)
-----------------------------

-  [Bugfix] [Python] Code coverage CI added. Fixed some minor
   inconsistencies. Automated the ``bdist`` build chain.

-  [R] Updated DESCRIPTION to meet the CRAN policies.

genieclust 0.9.2 (2020-07-22)
-----------------------------

-  [BUGFIX] [Python] Fixed broken build script for OS X with no OpenMP.

genieclust 0.9.1 (2020-07-18)
-----------------------------

-  [General] The package has been completely rewritten. The core
   functionality is now implemented in C++ (with OpenMP).

-  [General] Clustering with respect to HDBSCAN*-like mutual
   reachability distances is supported.

-  [General] The parallelised Jarnik-Prim algorithm now supports
   on-the-fly distance computations. Euclidean minimum spanning tree can
   be determined with ``mlpack``, which is much faster in
   low-dimensional spaces.

-  [R] R version is now available.

-  [Python] [Experimental] The GIc algorithm proposed by Anna Cena in
   her 2018 PhD thesis was added.

-  [Python] Approximate version based on nearest neighbour graphs
   produced by ``nmslib`` was added.

genieclust 0.1a2 (2018-05-23)
-----------------------------

-  [Python] Initial PyPI release.
