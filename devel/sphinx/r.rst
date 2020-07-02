R Package *genieclust* Reference
================================

adjusted_rand_score
-------------------

Pairwise Partition Similarity Scores (External Cluster Validity
Measures)

Description
~~~~~~~~~~~

Let ``x`` and ``y`` represent two partitions of a set of ``n`` elements
into ``K`` and ``L`` , respectively, nonempty and pairwise disjoint
subsets, e.g., two clusterings of a dataset with ``n`` observations
represented as label vectors. These functions quantify the similarity
between ``x`` and ``y`` . They can be used as external cluster validity
measures, i.e., in the presence of reference (ground-truth) partitions.

Usage
~~~~~

.. code:: r

   adjusted_rand_score(x, y = NULL)
   rand_score(x, y = NULL)
   adjusted_fm_score(x, y = NULL)
   fm_score(x, y = NULL)
   mi_score(x, y = NULL)
   normalized_mi_score(x, y = NULL)
   adjusted_mi_score(x, y = NULL)
   normalized_accuracy(x, y = NULL)
   pair_sets_index(x, y = NULL)

Arguments
~~~~~~~~~

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``x``                         | an integer vector of length n (or an |
|                               | object coercible to) representing a  |
|                               | K-partition of an n-set, or a        |
|                               | confusion matrix with K rows and L   |
|                               | columns (see ``table(x, y)`` )       |
+-------------------------------+--------------------------------------+
| ``y``                         | an integer vector of length n (or an |
|                               | object coercible to) representing an |
|                               | L-partition of the same set), or     |
|                               | NULL (if x is an K*L confusion       |
|                               | matrix)                              |
+-------------------------------+--------------------------------------+

Details
~~~~~~~

Every index except ``mi_score()`` (which computes the mutual information
score) outputs 1 given two identical partitions. Note that partitions
are always defined up to a bijection of the set of possible labels,
e.g., (1, 1, 2, 1) and (4, 4, 2, 4) represent the same 2-partition.

``rand_score()`` gives the Rand score (the “probability” of agreement
between the two partitions) and ``adjusted_rand_score()`` is its version
corrected for chance, its expected value is 0.0 for two independent
partitions. Due to the adjustment, the resulting index might also be
negative for some inputs.

Similarly, ``fm_score()`` gives the Fowlkes-Mallows (FM) score and
``adjusted_fm_score()`` is its adjusted-for-chance version.

Note that both the (unadjusted) Rand and FM scores are bounded from
below by :math:`1/(K+1)` , where K is the number of clusters (unique
labels in ``x`` and ``y`` ), hence their adjusted versions are
preferred.

``mi_score()`` , ``adjusted_mi_score()`` and ``normalized_mi_score()``
are information-theoretic scores, based on mutual information, see the
definition of :math:`AMI_{sum}` and :math:`NMI_{sum}` in (Vinh et al.,
2010).

``normalized_accuracy()`` is defined as
:math:`(Accuracy(C_\sigma)-1/L)/(1-1/L)` , where :math:`C_\sigma` is a
version of the confusion matrix for given ``x`` and ``y`` ,
:math:`K \leq L` , with columns permuted based on the solution to the
Maximal Linear Sum Assignment Problem. :math:`Accuracy(C_\sigma)` is
sometimes referred to as Purity, e.g., in (Rendon et al. 2011).

``pair_sets_index()`` gives the Pair Sets Index (PSI) adjusted for
chance (Rezaei, Franti, 2016), :math:`K \leq L` . Pairing is based on
the solution to the Linear Sum Assignment Problem of a transformed
version of the confusion matrix.

Value
~~~~~

A single real value giving the similarity score.

References
~~~~~~~~~~

Hubert L., Arabie P., Comparing Partitions, Journal of Classification
2(1), 1985, pp. 193-218, esp. Eqs. (2) and (4)

Rendon E., Abundez I., Arizmendi A., Quiroz E.M., Internal versus
external cluster validation indexes, International Journal of Computers
and Communications 5(1), 2011, pp. 27-34.

Rezaei M., Franti P., Set matching measures for external cluster
validity, IEEE Transactions on Knowledge and Data Mining 28(8), 2016,
pp. 2173-2186, doi:10.1109/TKDE.2016.2551240

Vinh N.X., Epps J., Bailey J., Information theoretic measures for
clusterings comparison: Variants, properties, normalization and
correction for chance, Journal of Machine Learning Research 11, 2010,
pp. 2837-2854.

Examples
~~~~~~~~

.. code:: r

   y_true <- iris[[5]]
   y_pred <- kmeans(as.matrix(iris[1:4]), 3)$cluster
   adjusted_rand_score(y_true, y_pred)
   rand_score(table(y_true, y_pred)) # the same
   adjusted_fm_score(y_true, y_pred)
   fm_score(y_true, y_pred)
   mi_score(y_true, y_pred)
   normalized_mi_score(y_true, y_pred)
   adjusted_mi_score(y_true, y_pred)
   normalized_accuracy(y_true, y_pred)
   pair_sets_index(y_true, y_pred)

emst_mlpack
-----------

Euclidean Minimum Spanning Tree

.. _description-1:

Description
~~~~~~~~~~~

Provides access to an implementation of the Dual-Tree Borůvka algorithm
based on kd-trees. It is fast for (very) low-dimensional Euclidean
spaces. For higher dimensional spaces (say, over 5 features) or other
metrics, use the parallelised Prim-like algorithm implemented in
```mst`` <#mst>`__ .

.. _usage-1:

Usage
~~~~~

.. code:: r

   emst_mlpack(X, verbose = FALSE)

.. _arguments-1:

Arguments
~~~~~~~~~

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``X``                         | a numeric matrix (or an object       |
|                               | coercible to one, e.g., a data frame |
|                               | with numeric-like columns)           |
+-------------------------------+--------------------------------------+
| ``verbose``                   | logical; whether to print diagnostic |
|                               | messages                             |
+-------------------------------+--------------------------------------+

.. _details-1:

Details
~~~~~~~

Calls ``emstreeR::mlpack_mst()`` and converts the result so that it is
compatible with the output of ```mst`` <#mst>`__ .

If the ``emstreeR`` package is not available, an error is generated.

.. _value-1:

Value
~~~~~

An object of class ``mst`` , see ```mst`` <#mst>`__ for details.

.. _references-1:

References
~~~~~~~~~~

March W.B., Ram P., Gray A.G., Fast Euclidean Minimum Spanning Tree:
Algorithm, Analysis, and Applications, Proc. ACM SIGKDD’10 (2010)
603-611, https://mlpack.org/papers/emst.pdf

gclust
------

The Genie++ Hierarchical Clustering Algorithm

.. _description-2:

Description
~~~~~~~~~~~

A reimplementation of Genie - a robust and outlier resistant clustering
algorithm (see Gagolewski, Bartoszuk, Cena, 2016). The Genie algorithm
is based on a minimum spanning tree (MST) of the pairwise distance graph
of a given point set. Just like single linkage, it consumes the edges of
the MST in increasing order of weights. However, it prevents the
formation of clusters of highly imbalanced sizes; once the Gini index
(see ```gini_index`` <#giniindex>`__ ) of the cluster size distribution
raises above ``gini_threshold`` , a forced merge of a point group of the
smallest size is performed. Its appealing simplicity goes hand in hand
with its usability; Genie often outperforms other clustering approaches
on benchmark data, such as
https://github.com/gagolews/clustering_benchmarks_v1 .

The clustering can now also be computed with respect to the mutual
reachability distance (based, e.g., on the Euclidean metric), which is
used in the definition of the HDBSCAN\* algorithm (see Campello et al.,
2015). If ``M`` > 1, then the mutual reachability distance
:math:`m(i,j)` with smoothing factor ``M`` is used instead of the chosen
“raw” distance :math:`d(i,j)` . It holds
:math:`m(i,j)=\max(d(i,j), c(i), c(j))` , where :math:`c(i)` is
:math:`d(i,k)` with :math:`k` being the ( ``M`` -1)-th nearest neighbour
of :math:`i` . This makes “noise” and “boundary” points being “pulled
away” from each other.

The Genie correction together with the smoothing factor ``M`` > 1 (note
that ``M`` = 2 corresponds to the original distance) gives a robustified
version of the HDBSCAN\* algorithm that is able to detect a predefined
number of clusters. Hence it does not dependent on the DBSCAN’s somehow
magical ``eps`` parameter or the HDBSCAN’s ``min_cluster_size`` one.

.. _usage-2:

Usage
~~~~~

.. code:: r

   gclust(d, ...)
   gclust.default(
     d,
     gini_threshold = 0.3,
     distance = c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
     cast_float32 = TRUE,
     verbose = FALSE,
     ...
   )
   gclust.dist(d, gini_threshold = 0.3, verbose = FALSE, ...)
   gclust.mst(d, gini_threshold = 0.3, verbose = FALSE, ...)
   genie(d, ...)
   genie.default(
     d,
     k,
     gini_threshold = 0.3,
     distance = c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
     M = 1L,
     postprocess = c("boundary", "none", "all"),
     detect_noise = M > 1L,
     cast_float32 = TRUE,
     verbose = FALSE,
     ...
   )
   genie.dist(
     d,
     k,
     gini_threshold = 0.3,
     M = 1L,
     postprocess = c("boundary", "none", "all"),
     detect_noise = M > 1L,
     verbose = FALSE,
     ...
   )
   genie.mst(
     d,
     k,
     gini_threshold = 0.3,
     postprocess = c("boundary", "none", "all"),
     detect_noise = FALSE,
     verbose = FALSE,
     ...
   )

.. _arguments-2:

Arguments
~~~~~~~~~

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``d``                         | a numeric matrix (or an object       |
|                               | coercible to one, e.g., a data frame |
|                               | with numeric-like columns) or an     |
|                               | object of class ``dist`` , see       |
|                               | ```dist`` <#dist>`__ or an object of |
|                               | class ``mst`` , see                  |
|                               | ```mst`` <#mst>`__ .                 |
+-------------------------------+--------------------------------------+
| ``...``                       | further arguments passed to other    |
|                               | methods.                             |
+-------------------------------+--------------------------------------+
| ``gini_threshold``            | threshold for the Genie correction,  |
|                               | i.e., the Gini index of the cluster  |
|                               | size distribution; Threshold of 1.0  |
|                               | disables the correction. Low         |
|                               | thresholds highly penalise the       |
|                               | formation of small clusters.         |
+-------------------------------+--------------------------------------+
| ``distance``                  | metric used to compute the linkage,  |
|                               | one of: ``"euclidean"`` (synonym:    |
|                               | ``"l2"`` ), ``"manhattan"`` (a.k.a.  |
|                               | ``"l1"`` and ``"cityblock"`` ),      |
|                               | ``"cosine"`` .                       |
+-------------------------------+--------------------------------------+
| ``cast_float32``              | logical; whether to compute the      |
|                               | distances using 32-bit instead of    |
|                               | 64-bit precision floating-point      |
|                               | arithmetic (up to 2x faster).        |
+-------------------------------+--------------------------------------+
| ``verbose``                   | logical; whether to print diagnostic |
|                               | messages and progress information.   |
+-------------------------------+--------------------------------------+
| ``k``                         | the desired number of clusters to    |
|                               | detect, ``k`` = 1 with ``M`` > 1     |
|                               | acts as a noise point detector.      |
+-------------------------------+--------------------------------------+
| ``M``                         | smoothing factor; ``M`` <= 2 gives   |
|                               | the selected ``distance`` ;          |
|                               | otherwise, the mutual reachability   |
|                               | distance is used.                    |
+-------------------------------+--------------------------------------+
| ``postprocess``               | one of ``"boundary"`` (default),     |
|                               | ``"none"`` or ``"all"`` ; in effect  |
|                               | only if ``M`` > 1. By default, only  |
|                               | “boundary” points are merged with    |
|                               | their nearest “core” points (A point |
|                               | is a boundary point if it is a noise |
|                               | point and it’s amongst its adjacent  |
|                               | vertex’s ``M`` -1 nearest            |
|                               | neighbours). To force a classical    |
|                               | k-partition of a data set (with no   |
|                               | notion of noise), choose “all”.      |
+-------------------------------+--------------------------------------+
| ``detect_noise``              | whether the minimum spanning tree’s  |
|                               | leaves should be marked as noise     |
|                               | points, defaults to ``TRUE`` if      |
|                               | ``M`` > 1 for compatibility with     |
|                               | HDBSCAN\*                            |
+-------------------------------+--------------------------------------+

.. _details-2:

Details
~~~~~~~

Note that as in the case of all the distance-based methods, the
standardisation of the input features is definitely worth giving a try.

If ``d`` is a numeric matrix or an object of class ``dist`` ,
```mst`` <#mst>`__ will be called to compute an MST, which generally
takes at most :math:`O(n^2)` time (the algorithm we provide is
parallelised, environment variable ``OMP_NUM_THREADS`` controls the
number of threads in use). However, see
```emst_mlpack`` <#emstmlpack>`__ for a very fast alternative in the
case of Euclidean spaces of (very) low dimensionality and ``M`` = 1.

Given an minimum spanning tree, the algorithm runs in
:math:`O(n \sqrt{n})` time. Therefore, if you want to test different
``gini_threshold`` s, (or ``k`` s), it is best to explicitly compute the
MST first.

According to the algorithm’s original definition, the resulting
partition tree (dendrogram) might violate the ultrametricity property
(merges might occur at levels that are not increasing w.r.t. a
between-cluster distance). Departures from ultrametricity are corrected
by applying ``height = rev(cummin(rev(height)))`` .

.. _value-2:

Value
~~~~~

``gclust()`` computes the whole clustering hierarchy; it returns a list
of class ``hclust`` , see ```hclust`` <#hclust>`__ . Use
``link{cutree}()`` to obtain an arbitrary k-partition.

``genie()`` returns a ``k`` -partition - a vector with elements in
1,…,k, whose i-th element denotes the i-th input point’s cluster
identifier. Missing values ( ``NA`` ) denote noise points (if
``detect_noise`` is ``TRUE`` ).

Seealso
~~~~~~~

```mst`` <#mst>`__ for the minimum spanning tree routines.

```adjusted_rand_score`` <#adjustedrandscore>`__ (amongst others) for
external cluster validity measures (partition similarity scores).

.. _references-2:

References
~~~~~~~~~~

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and
outlier-resistant hierarchical clustering algorithm, Information
Sciences 363, 2016, pp. 8-23.

Campello R., Moulavi D., Zimek A., Sander J., Hierarchical density
estimates for data clustering, visualization, and outlier detection, ACM
Transactions on Knowledge Discovery from Data 10(1) (2015) 5:1–5:51.

.. _examples-1:

Examples
~~~~~~~~

.. code:: r

   library("datasets")
   data("iris")
   X <- iris[1:4]
   h <- gclust(X)
   y_pred <- cutree(h, 3)
   y_test <- iris[,5]
   plot(iris[,2], iris[,3], col=y_pred,
   pch=as.integer(iris[,5]), asp=1, las=1)
   adjusted_rand_score(y_test, y_pred)
   pair_sets_index(y_test, y_pred)

   ## Fast for low-dimensional Euclidean spaces:
   if (require("emstreeR")) h <- gclust(emst_mlpack(X))

genieclust-package
------------------

The Genie++ Hierarchical Clustering Algorithm (with Extras)

.. _description-3:

Description
~~~~~~~~~~~

See ```genie`` <#genie>`__ for more details.

Author
~~~~~~

Marek Gagolewski

gini_index
----------

Inequity (Inequality) Measures

.. _description-4:

Description
~~~~~~~~~~~

``gini_index()`` gives the normalised Gini index and
``bonferroni_index()`` implements the Bonferroni index.

.. _usage-3:

Usage
~~~~~

.. code:: r

   gini_index(x)
   bonferroni_index(x)

.. _arguments-3:

Arguments
~~~~~~~~~

======== =====================================
Argument Description
======== =====================================
``x``    numeric vector of non-negative values
======== =====================================

.. _details-3:

Details
~~~~~~~

Both indices can be used to quantify the “inequity” of a numeric sample.
They can be perceived as measures of data dispersion. For constant
vectors (perfect equity), the indices yield values of 0. Vectors with
all elements but one equal to 0 (perfect inequity), are assigned scores
of 1. Both indices follow the Pigou-Dalton principle (are Schur-convex):
setting :math:`x_i = x_i - h` and :math:`x_j = x_j + h` with
:math:`h > 0` and x_i - h >= x_j + h (taking from the “rich” and giving
to the “poor”) decreases the inequity.

These indices have applications in economics, amongst others. The Gini
clustering algorithm uses the Gini index as a measure of the inequality
of cluster sizes.

The normalised Gini index is given by:

.. math:: G(x_1,\dots,x_n) = \frac{\sum_{i=1}^{n-1} \sum_{j=i+1}^n |x_i-x_j|}{(n-1) \sum_{i=1}^n x_i}.

The normalised Bonferroni index is given by:

.. math:: B(x_1,\dots,x_n) = \frac{\sum_{i=1}^{n}  (n-\sum_{j=1}^i \frac{n}{n-j+1})x_{\sigma(n-i+1)}}{(n-1) \sum_{i=1}^n x_i}.

Time complexity: :math:`O(n)` for sorted (increasingly) data. Otherwise,
the vector will be sorted.

In particular, for ordered inputs, it holds:

.. math:: G(x_1,\dots,x_n) = \frac{\sum_{i=1}^{n} (n-2i+1) x_{\sigma(n-i+1)}}{(n-1) \sum_{i=1}^n x_i},

where :math:`\sigma` is an ordering permutation of
:math:`(x_1,\dots,x_n)` .

.. _value-3:

Value
~~~~~

The value of the inequity index, a number in :math:`[0, 1]` .

.. _references-3:

References
~~~~~~~~~~

Bonferroni C., Elementi di Statistica Generale, Libreria Seber, Firenze,
1930.

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and
outlier-resistant hierarchical clustering algorithm, Information
Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

Gini C., Variabilita e Mutabilita, Tipografia di Paolo Cuppini, Bologna,
1912.

.. _examples-2:

Examples
~~~~~~~~

.. code:: r

   gini_index(c(2, 2, 2, 2, 2))  # no inequality
   gini_index(c(0, 0, 10, 0, 0)) # one has it all
   gini_index(c(7, 0, 3, 0, 0))  # give to the poor, take away from the rich
   gini_index(c(6, 0, 3, 1, 0))  # (a.k.a. Pigou-Dalton principle)
   bonferroni_index(c(2, 2, 2, 2, 2))
   bonferroni_index(c(0, 0, 10, 0, 0))
   bonferroni_index(c(7, 0, 3, 0, 0))
   bonferroni_index(c(6, 0, 3, 1, 0))

mst
---

Minimum Spanning Tree of the Pairwise Distance Graph

.. _description-5:

Description
~~~~~~~~~~~

An parallelised implementation of a Jarník (Prim/Dijkstra)-like
algorithm for determining a(*) minimum spanning tree (MST) of a complete
undirected graph representing a set of n points with weights given by a
pairwise distance matrix.

(*) Note that there might be multiple minimum trees spanning a given
graph.

.. _usage-4:

Usage
~~~~~

.. code:: r

   mst(d, ...)
   mst.default(
     d,
     distance = c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
     M = 1L,
     cast_float32 = TRUE,
     verbose = FALSE,
     ...
   )
   mst.dist(d, M = 1L, verbose = FALSE, ...)

.. _arguments-4:

Arguments
~~~~~~~~~

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``d``                         | either a numeric matrix (or an       |
|                               | object coercible to one, e.g., a     |
|                               | data frame with numeric-like         |
|                               | columns) or an object of class       |
|                               | ``dist`` , see ```dist`` <#dist>`__  |
|                               | .                                    |
+-------------------------------+--------------------------------------+
| ``...``                       | further arguments passed to or from  |
|                               | other methods.                       |
+-------------------------------+--------------------------------------+
| ``distance``                  | metric used to compute the linkage,  |
|                               | one of: ``"euclidean"`` (synonym:    |
|                               | ``"l2"`` ), ``"manhattan"`` (a.k.a.  |
|                               | ``"l1"`` and ``"cityblock"`` ),      |
|                               | ``"cosine"`` .                       |
+-------------------------------+--------------------------------------+
| ``M``                         | smoothing factor; ``M`` = 1 gives    |
|                               | the selected ``distance`` ;          |
|                               | otherwise, the mutual reachability   |
|                               | distance is used.                    |
+-------------------------------+--------------------------------------+
| ``cast_float32``              | logical; whether to compute the      |
|                               | distances using 32-bit instead of    |
|                               | 64-bit precision floating-point      |
|                               | arithmetic (up to 2x faster).        |
+-------------------------------+--------------------------------------+
| ``verbose``                   | logical; whether to print diagnostic |
|                               | messages and progress information.   |
+-------------------------------+--------------------------------------+

.. _details-4:

Details
~~~~~~~

If ``d`` is a numeric matrix of size :math:`n*p` , the :math:`n*(n-1)/2`
distances are computed on the fly, so that :math:`O(n M)` memory is
used.

The algorithm is parallelised; set the ``OMP_NUM_THREADS`` environment
variable ```Sys.setenv`` <#sys.setenv>`__ to control the number of
threads used.

Time complexity is :math:`O(n^2)` for the method accepting an object of
class ``dist`` and :math:`O(p n^2)` otherwise.

If ``M`` >= 2, then the mutual reachability distance :math:`m(i,j)` with
smoothing factor ``M`` (see Campello et al. 2015) is used instead of the
chosen “raw” distance :math:`d(i,j)` . It holds
:math:`m(i, j)=\max(d(i,j), c(i), c(j))` , where :math:`c(i)` is
:math:`d(i, k)` with :math:`k` being the ( ``M`` -1)-th nearest
neighbour of :math:`i` . This makes “noise” and “boundary” points being
“pulled away” from each other. Genie++ clustering algorithm (see
```gclust`` <#gclust>`__ ) with respect to the mutual reachability
distance gains the ability to identify some observations are noise
points.

Note that the case ``M`` = 2 corresponds to the original distance, but
we are determining the 1-nearest neighbours separately as well, which is
a bit suboptimal; you can file a feature request if this makes your data
analysis tasks too slow.

.. _value-4:

Value
~~~~~

Matrix of class ``mst`` with n-1 rows and 3 columns: ``from`` , ``to``
and ``dist`` . It holds ``from`` < ``to`` . Moreover, ``dist`` is sorted
nondecreasingly. The i-th row gives the i-th edge of the MST.
``(from[i], to[i])`` defines the vertices (in 1,…,n) and ``dist[i]``
gives the weight, i.e., the distance between the corresponding points.

The ``method`` attribute gives the name of the distance used. The
``Labels`` attribute gives the labels of all the input points.

If ``M`` > 1, the ``nn`` attribute gives the indices of the ``M`` -1
nearest neighbours of each point.

.. _seealso-1:

Seealso
~~~~~~~

```emst_mlpack`` <#emstmlpack>`__ for a very fast alternative in case of
(very) low-dimensional Euclidean spaces (and ``M`` = 1).

.. _references-4:

References
~~~~~~~~~~

V. Jarník, O jistém problému minimálním, Práce Moravské Přírodovědecké
Společnosti 6 (1930) 57–63.

C.F. Olson, Parallel algorithms for hierarchical clustering, Parallel
Comput. 21 (1995) 1313–1325.

R. Prim, Shortest connection networks and some generalisations, Bell
Syst. Tech. J. 36 (1957) 1389–1401.

Campello R., Moulavi D., Zimek A., Sander J., Hierarchical density
estimates for data clustering, visualization, and outlier detection, ACM
Transactions on Knowledge Discovery from Data 10(1) (2015) 5:1–5:51.

.. _examples-3:

Examples
~~~~~~~~

.. code:: r

   library("datasets")
   data("iris")
   X <- iris[1:4]
   tree <- mst(X)
