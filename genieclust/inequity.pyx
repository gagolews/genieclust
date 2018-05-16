# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Economic Inequity (Inequality) Measures

Copyright (C) 2018 Marek.Gagolewski.com
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np
import numpy as np
from libc.math cimport fabs, sqrt
from numpy.math cimport INFINITY
import scipy.spatial.distance
import warnings



ctypedef fused intT:
    np.int64_t
    np.int32_t
    np.int_t

ctypedef fused T:
    np.float64_t
    np.float32_t
    np.int64_t
    np.int32_t
    np.int_t
    np.double_t

ctypedef fused arrayT:
    np.ndarray[np.double_t]
    np.ndarray[np.int_t]

cdef T square(T x):
    return x*x


0

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


    Parameters:
    ----------

    x : ndarray, shape (n,)
        Input vector.

    is_sorted : bool
        Indicates if x is sorted increasingly.


    Returns:
    -------

    index : float
        The value of the inequity index, a number in [0,1].
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


    Parameters:
    ----------

    x : ndarray, shape (n,)
        Input vector.

    is_sorted : bool
        Indicates if x is sorted increasingly.


    Returns:
    -------

    index : float
        The value of the inequity index, a number in [0,1].
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



