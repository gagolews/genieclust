# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Inequality measures
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import numpy as np
cimport numpy as np
np.import_array()

ctypedef fused T:
    int
    long
    long long
    Py_ssize_t
    float
    double

cdef T square(T x):
    return x*x



cdef extern from "../src/c_inequality.h":
    double Cgini_sorted[T](const T* x, Py_ssize_t n) except +
    double Cbonferroni_sorted[T](const T* x, Py_ssize_t n) except +
    double Cdevergottini_sorted[T](const T* x, Py_ssize_t n) except +



cpdef double gini_index(np.ndarray[T] x, bint is_sorted=False):
    """
    genieclust.inequality.gini_index(x, is_sorted=False)

    Computes the normalised Gini index


    Parameters
    ----------

    x : ndarray
        A vector with non-negative elements.

    is_sorted : bool
        Indicates if `x` is already sorted increasingly.


    Returns
    -------

    double
        The value of the inequality index, a number in [0,1].


    See Also
    --------

    genieclust.inequality.bonferroni_index : The normalised Bonferroni index
    genieclust.inequality.devergottini_index : The normalised De Vergottini index


    Notes
    -----

    The normalised Gini [1]_ index is given by:

    .. math::

        G(x_1,\\dots,x_n) = \\frac{
        \\sum_{i=1}^{n-1} \\sum_{j=i+1}^n |x_i-x_j|
        }{
        (n-1) \\sum_{i=1}^n x_i
        }.

    Time complexity is :math:`O(n)` for sorted data; it holds:

    .. math::

        G(x_1,\\dots,x_n) = \\frac{
        \\sum_{i=1}^{n} (n-2i+1) x_{\\sigma(n-i+1)}
        }{
        (n-1) \\sum_{i=1}^n x_i
        },

    where :math:`\\sigma` is an ordering permutation of :math:`(x_1,\\dots,x_n)`.

    In particular, if :math:`n=2`, then
    :math:`G(x_1,x_2) = (x_{(2)}-x_{(1)})/(x_{(2)}+x_{(1)})`.

    The Gini, Bonferroni, and De Vergottini indices can be used to quantify
    the "inequality" of a numeric sample. They can be conceived as normalised
    measures of data dispersion. For constant vectors (perfect equity),
    the indices yield values of 0. Vectors with all elements but one equal to 0
    (perfect inequality), are assigned scores of 1.
    They follow the Pigou-Dalton principle (are Schur-convex):
    setting :math:`x_i = x_i - h` and :math:`x_j = x_j + h` with :math:`h > 0`
    and :math:`x_i - h \\geq  x_j + h` (taking from the "rich" and giving away
    to the "poor") decreases the inequality.

    These indices have applications in economics, amongst others.
    The `Genie` clustering algorithm uses the Gini index as a measure
    of the inequality of cluster sizes.



    References
    ----------

    .. [1]
        C. Gini, *Variabilita e Mutabilita*,
        Tipografia di Paolo Cuppini, Bologna, 1912


    Examples
    --------

    No inequality (perfect equality):

    >>> round(genieclust.inequality.gini_index(np.r_[2, 2,  2, 2, 2]), 2)
    0.0

    One has it all (total inequality):

    >>> round(genieclust.inequality.gini_index(np.r_[0, 0, 10, 0, 0]), 2)
    1.0

    Give to the poor, take away from the rich:

    >>> round(genieclust.inequality.gini_index(np.r_[7, 0,  3, 0, 0]), 2)
    0.85

    Robinhood even more:

    >>> round(genieclust.inequality.gini_index(np.r_[6, 0,  3, 1, 0]), 2)
    0.75
    """

    if not is_sorted: x = np.sort(x)
    else: x = np.asarray(x, dtype=x.dtype, order="C")  # ensure c_contiguity

    return Cgini_sorted(&x[0], x.shape[0])



cpdef double bonferroni_index(np.ndarray[T] x, bint is_sorted=False):
    """
    genieclust.inequality.bonferroni_index(x, is_sorted=False)

    Computes the normalised Bonferroni index


    Parameters
    ----------

    x : ndarray
        A vector with non-negative elements.

    is_sorted : bool
        Indicates if `x` is already sorted increasingly.


    Returns
    -------

    double
        The value of the inequality index, a number in [0,1].


    See Also
    --------

    genieclust.inequality.devergottini_index : The normalised De Vergottini index
    genieclust.inequality.gini_index : The normalised Gini index


    Notes
    -----

    The normalised Bonferroni [1]_ index is given by:

    .. math::

        B(x_1,\\dots,x_n) = \\frac{
        \\sum_{i=1}^{n}  \\left( n-\\sum_{j=1}^i \\frac{n}{n-j+1} \\right) x_{\\sigma(n-i+1)}
        }{
        (n-1) \\sum_{i=1}^n x_i
        },

    where :math:`\\sigma` is an ordering permutation of :math:`(x_1,\\dots,x_n)`.

    Time complexity: :math:`O(n)` for sorted data.


    References
    ----------

    .. [1]
        C. Bonferroni, *Elementi di Statistica Generale*, Libreria Seber,
        Firenze, 1930


    Examples
    --------

    No inequality (perfect equality):

    >>> round(genieclust.inequality.bonferroni_index(np.r_[2, 2,  2, 2, 2]), 2)
    0.0

    One has it all (total inequality):

    >>> round(genieclust.inequality.bonferroni_index(np.r_[0, 0, 10, 0, 0]), 2)
    1.0

    Give to the poor, take away from the rich:

    >>> round(genieclust.inequality.bonferroni_index(np.r_[7, 0,  3, 0, 0]), 2)
    0.91

    Robinhood even more:

    >>> round(genieclust.inequality.bonferroni_index(np.r_[6, 0,  3, 1, 0]), 2)
    0.83
    """

    if not is_sorted: x = np.sort(x)
    else: x = np.asarray(x, dtype=x.dtype, order="C")  # ensure c_contiguity

    return Cbonferroni_sorted(&x[0], x.shape[0])



cpdef double devergottini_index(np.ndarray[T] x, bint is_sorted=False):
    """
    genieclust.inequality.devergottini_index(x, is_sorted=False)

    Computes the normalised De Vergottini index


    Parameters
    ----------

    x : ndarray
        A vector with non-negative elements.

    is_sorted : bool
        Indicates if `x` is already sorted increasingly.


    Returns
    -------

    double
        The value of the inequality index, a number in [0,1].


    See Also
    --------

    genieclust.inequality.bonferroni_index : The normalised Bonferroni index
    genieclust.inequality.gini_index : The normalised Gini index


    Notes
    -----

    The normalised De Vergottini index is given by:

    .. math::

        V(x_1,\\dots,x_n) = \\frac{1}{\\sum_{i=2}^n \\frac{1}{i}}
        \\left(
        \\frac{ \\sum_{i=1}^n \\left(
            \\sum_{j=i}^{n} \\frac{1}{j}
        \\right) x_{\\sigma(n-i+1)} }{\\sum_{i=1}^{n} x_i} - 1
        \\right),

    where :math:`\\sigma` is an ordering permutation of :math:`(x_1,\\dots,x_n)`.

    Time complexity is :math:`O(n)` for sorted data.


    Examples
    --------

    No inequality (perfect equality):

    >>> round(genieclust.inequality.devergottini_index(np.r_[2, 2,  2, 2, 2]), 2)
    0.0

    One has it all (total inequality):

    >>> round(genieclust.inequality.devergottini_index(np.r_[0, 0, 10, 0, 0]), 2)
    1.0

    Give to the poor, take away from the rich:

    >>> round(genieclust.inequality.devergottini_index(np.r_[7, 0,  3, 0, 0]), 2)
    0.77

    Robinhood even more:

    >>> round(genieclust.inequality.devergottini_index(np.r_[6, 0,  3, 1, 0]), 2)
    0.65
    """

    if not is_sorted: x = np.sort(x)
    else: x = np.asarray(x, dtype=x.dtype, order="C")  # ensure c_contiguity

    return Cdevergottini_sorted(&x[0], x.shape[0])
