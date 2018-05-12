#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

"""
Economic Inequity (Inequality) Measures
Copyright (C) 2018 Marek.Gagolewski.com

See -- among others --:

    Beliakov G., Gagolewski M., James S.,
    Penalty-based and other representations of economic inequality,
    International Journal of Uncertainty,
    Fuzziness and Knowledge-Based Systems 24(Suppl. 1), 2016, pp. 1-23.
    doi:10.1142/S0218488516400018

    Gagolewski M., Cena A., Bartoszuk M., Hierarchical clustering via
    penalty-based aggregation and the Genie approach,
    Lecture Notes in Artificial Intelligence 9880, 2016,
    pp. 191-202. doi:10.1007/978-3-319-45656-0_16


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


cimport numpy as np
import numpy as np
cimport cython
cimport libc.math as cmath

ctypedef fused T:
    np.float64_t
    np.float32_t
    np.int64_t
    np.int32_t
    np.int_t
    np.double_t

cdef T square(T x):
    return x*x


cpdef np.float64_t gini(np.ndarray[T] x, bint is_sorted=False):
    """
    The Normalized Gini index:

    $$
        G(x_1,\dots,x_n) = \frac{
        \sum_{i=1}^{n-1} \sum_{j=i+1}^n |x_i-x_j|
        }{
        (n-1) \sum_{i=1}^n x_i
        }.
    $$

    Time complexity: $O(n)$ for sorted data; it holds:
    $$
        G(x_1,\dots,x_n) = \frac{
        \sum_{i=1}^{n} (n-2i+1) x_{\sigma(n-i+1)}
        }{
        (n-1) \sum_{i=1}^n x_i
        },
    $$
    where $\sigma$ is an ordering permutation of $(x_1,\dots,x_n)$.
    """

    if not is_sorted: x = np.sort(x)
    cdef unsigned int n = x.shape[0]
    cdef np.float64_t s = 0.0, t = 0.0
    cdef unsigned int i

    for i in range(1,n+1):
        t += x[n-i]
        s += (n-2.0*i+1.0)*x[n-i]

    return s/(n-1.0)/t


cpdef np.float64_t bonferroni(np.ndarray[T] x, bint is_sorted=False):
    """
    The Normalized Bonferroni index:
    $$
        B(x_1,\dots,x_n) = \frac{
        \sum_{i=1}^{n}  \left( n-\sum_{j=1}^i \frac{n}{n-j+1} \right) x_{\sigma(n-i+1)}
        }{
        (n-1) \sum_{i=1}^n x_i
        },
    $$
    where $\sigma$ is an ordering permutation of $(x_1,\dots,x_n)$.

    Time complexity: $O(n)$ for sorted data.
    """

    if not is_sorted: x = np.sort(x)
    cdef unsigned int n = x.shape[0]
    cdef np.float64_t s = 0.0, t = 0.0, c = 0.0
    cdef unsigned int i

    for i in range(1,n+1):
        c += n/<np.float64_t>(n-i+1.0)
        t += x[n-i]
        s += (n-c)*x[n-i]

    return s/(n-1.0)/t


#cpdef np.float64_t coefvar(np.ndarray[T] x, bint is_sorted=False):
    #"""
    #Coefficient of variation

        #$$
        #C(x_1,\dots,x_n) = \sqrt{\frac{
        #\sum_{i=1}^{n-1} \sum_{j=i+1}^n (x_i-x_j)^2
        #}{
        #(n-1) \sum_{i=1}^n x_i^2
        #}}.
        #$$

    #Is this an inequity measures BTW?
    #"""

    ## sorting is not necessary
    #cdef unsigned int n = len(x)
    #cdef np.float64_t s = 0.0, t = square(x[0])
    #cdef unsigned int i, j

    #for i in range(n-1):
        #t += square(x[i+1])
        #for j in range(i+1, n):
            #s += square(x[i]-x[j])

    #return cmath.sqrt(s/(n-1.0)/t)


# cpdef np.float64_t vergottini(np.ndarray[T] x, bint is_sorted=False):
# "de Vergottini index
#    x <- sort(x, decreasing=TRUE)
#    n <- length(x)
#    vmax <- sum(1/(2:n))
#    (sum(sapply(1:length(x), function(i) mean(x[1:i])))/sum(x)-1)/vmax
# }



