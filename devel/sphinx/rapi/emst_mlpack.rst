emst_mlpack: Euclidean Minimum Spanning Tree
============================================

Description
~~~~~~~~~~~

Provides access to the implementation of the Dual-Tree Borůvka algorithm from the ``mlpack`` package (if available). It is based on kd-trees and is fast for (very) low-dimensional Euclidean spaces. For higher dimensional spaces (say, over 5 features) or other metrics, use the parallelised Prim-like algorithm implemented in `mst() <mst.html>`__.

Usage
~~~~~

.. code-block:: r

   emst_mlpack(X, leaf_size = 1, naive = FALSE, verbose = FALSE)

Arguments
~~~~~~~~~

+---------------+------------------------------------------------------------------------------------------------+
| ``X``         | a numeric matrix (or an object coercible to one, e.g., a data frame with numeric-like columns) |
+---------------+------------------------------------------------------------------------------------------------+
| ``leaf_size`` | size of leaves in the kd-tree, controls the trade-off between speed and memory consumption     |
+---------------+------------------------------------------------------------------------------------------------+
| ``naive``     | logical; whether to use the naive, quadratic-time algorithm                                    |
+---------------+------------------------------------------------------------------------------------------------+
| ``verbose``   | logical; whether to print diagnostic messages                                                  |
+---------------+------------------------------------------------------------------------------------------------+

Value
~~~~~

An object of class ``mst``, see `mst() <mst.html>`__ for details.

References
~~~~~~~~~~

March W.B., Ram P., Gray A.G., Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications, Proc. ACM SIGKDD'10 (2010) 603-611, https://mlpack.org/papers/emst.pdf.

Curtin R.R., Edel M., Lozhnikov M., Mentekidis Y., Ghaisas S., Zhang S., mlpack 3: A fast, flexible machine learning library, Journal of Open Source Software 3(26), 726, 2018.
