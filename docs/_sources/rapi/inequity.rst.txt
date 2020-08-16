inequity: Inequity (Inequality) Measures
========================================

Description
~~~~~~~~~~~

``gini_index()`` gives the normalised Gini index and ``bonferroni_index()`` implements the Bonferroni index.

Usage
~~~~~

.. code-block:: r

   gini_index(x)

   bonferroni_index(x)

Arguments
~~~~~~~~~

+-------+---------------------------------------+
| ``x`` | numeric vector of non-negative values |
+-------+---------------------------------------+

Details
~~~~~~~

Both indices can be used to quantify the "inequity" of a numeric sample. They can be perceived as measures of data dispersion. For constant vectors (perfect equity), the indices yield values of 0. Vectors with all elements but one equal to 0 (perfect inequity), are assigned scores of 1. Both indices follow the Pigou-Dalton principle (are Schur-convex): setting *x_i = x_i - h* and *x_j = x_j + h* with *h > 0* and *x_i - h ≥q x_j + h* (taking from the "rich" and giving to the "poor") decreases the inequity.

These indices have applications in economics, amongst others. The Gini clustering algorithm uses the Gini index as a measure of the inequality of cluster sizes.

The normalised Gini index is given by:

*G(x_1,…,x_n) = \\frac{ ∑_{i=1}^{n-1} ∑_{j=i+1}^n \|x_i-x_j\| }{ (n-1) ∑_{i=1}^n x_i }.*

The normalised Bonferroni index is given by:

*B(x_1,…,x_n) = \\frac{ ∑_{i=1}^{n} (n-∑_{j=1}^i \\frac{n}{n-j+1}) x_{σ(n-i+1)} }{ (n-1) ∑_{i=1}^n x_i }.*

Time complexity: *O(n)* for sorted (increasingly) data. Otherwise, the vector will be sorted.

In particular, for ordered inputs, it holds:

*G(x_1,…,x_n) = \\frac{ ∑_{i=1}^{n} (n-2i+1) x_{σ(n-i+1)} }{ (n-1) ∑_{i=1}^n x_i },*

where *σ* is an ordering permutation of *(x_1,…,x_n)*.

Value
~~~~~

The value of the inequity index, a number in *[0, 1]*.

References
~~~~~~~~~~

Bonferroni C., Elementi di Statistica Generale, Libreria Seber, Firenze, 1930.

Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm, Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

Gini C., Variabilita e Mutabilita, Tipografia di Paolo Cuppini, Bologna, 1912.

Examples
~~~~~~~~

.. code-block:: r

   gini_index(c(2, 2, 2, 2, 2))  # no inequality
   gini_index(c(0, 0, 10, 0, 0)) # one has it all
   gini_index(c(7, 0, 3, 0, 0))  # give to the poor, take away from the rich
   gini_index(c(6, 0, 3, 1, 0))  # (a.k.a. Pigou-Dalton principle)
   bonferroni_index(c(2, 2, 2, 2, 2))
   bonferroni_index(c(0, 0, 10, 0, 0))
   bonferroni_index(c(7, 0, 3, 0, 0))
   bonferroni_index(c(6, 0, 3, 1, 0))

