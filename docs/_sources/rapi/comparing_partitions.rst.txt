comparing_partitions: Pairwise Partition Similarity Scores
==========================================================

Description
~~~~~~~~~~~

Let ``x`` and ``y`` represent two partitions of a set of ``n`` elements into *K* and *L*, respectively, nonempty and pairwise disjoint subsets, e.g., two clusterings of a dataset with ``n`` observations represented as label vectors. These functions quantify the similarity between ``x`` and ``y``. They can be used as external cluster validity measures, i.e., in the presence of reference (ground-truth) partitions.

Usage
~~~~~

.. code-block:: r

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

+-------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x`` | an integer vector of length n (or an object coercible to) representing a K-partition of an n-set, or a confusion matrix with K rows and L columns (see ``table(x, y)``) |
+-------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``y`` | an integer vector of length n (or an object coercible to) representing an L-partition of the same set), or NULL (if x is an K*L confusion matrix)                       |
+-------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Details
~~~~~~~

Every index except ``mi_score()`` (which computes the mutual information score) outputs 1 given two identical partitions. Note that partitions are always defined up to a bijection of the set of possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4) represent the same 2-partition.

``rand_score()`` gives the Rand score (the "probability" of agreement between the two partitions) and ``adjusted_rand_score()`` is its version corrected for chance, see (Hubert, Arabie, 1985), its expected value is 0.0 given two independent partitions. Due to the adjustment, the resulting index might also be negative for some inputs.

Similarly, ``fm_score()`` gives the Fowlkes-Mallows (FM) score and ``adjusted_fm_score()`` is its adjusted-for-chance version, see (Hubert, Arabie, 1985).

Note that both the (unadjusted) Rand and FM scores are bounded from below by *1/(K+1)* if *K=L*, hence their adjusted versions are preferred.

``mi_score()``, ``adjusted_mi_score()`` and ``normalized_mi_score()`` are information-theoretic scores, based on mutual information, see the definition of *AMI_{sum}* and *NMI_{sum}* in (Vinh et al., 2010).

``normalized_accuracy()`` is defined as *(Accuracy(C_σ)-1/L)/(1-1/L)*, where *C_σ* is a version of the confusion matrix for given ``x`` and ``y``, *K ≤q L*, with columns permuted based on the solution to the Maximal Linear Sum Assignment Problem. *Accuracy(C_σ)* is sometimes referred to as Purity, e.g., in (Rendon et al. 2011).

``pair_sets_index()`` gives the Pair Sets Index (PSI) adjusted for chance (Rezaei, Franti, 2016), *K ≤q L*. Pairing is based on the solution to the Linear Sum Assignment Problem of a transformed version of the confusion matrix.

Value
~~~~~

A single real value giving the similarity score.

References
~~~~~~~~~~

Hubert L., Arabie P., Comparing Partitions, Journal of Classification 2(1), 1985, 193-218, esp. Eqs. (2) and (4).

Rendon E., Abundez I., Arizmendi A., Quiroz E.M., Internal versus external cluster validation indexes, International Journal of Computers and Communications 5(1), 2011, 27-34.

Rezaei M., Franti P., Set matching measures for external cluster validity, IEEE Transactions on Knowledge and Data Mining 28(8), 2016, 2173-2186.

Vinh N.X., Epps J., Bailey J., Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance, Journal of Machine Learning Research 11, 2010, 2837-2854.

Examples
~~~~~~~~

.. code-block:: r

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
