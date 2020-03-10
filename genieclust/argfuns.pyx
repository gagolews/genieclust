# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Provides access to the argsort() and argkmin() functions.

Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
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



cimport numpy as np
import numpy as np
from . cimport c_argfuns

ctypedef fused T:
    int
    long
    long long
    ssize_t
    float
    double




cpdef np.ndarray[ssize_t] argsort(T[::1] x, bint stable=True):
    """
    Finds the(*) ordering permutation of a c_contiguous array x

    (*) if stable==True, otherwise it's *an* ordering permutation


    Parameters:
    ----------

    x : c_contiguous 1d array
        an integer or float vector

    stable : bool
        should a stable (a bit slower) sorting algorithm be used?


    Returns:
    -------

    ndarray:
        The ordering permutation.
    """
    cdef ssize_t n = x.shape[0]
    cdef np.ndarray[ssize_t] ret = np.empty(n, dtype=np.intp)
    c_argfuns.Cargsort(&ret[0], &x[0], n, stable)
    return ret




cpdef ssize_t argkmin(T[::1] x, int k):
    """
    Returns the index of the (k-1)-th smallest value in an array x,
    where argkmin(x, 0) == argmin(x), or, more generally,
    argkmin(x, k) == np.argsort(x)[k].

    Run time: O(nk), where n == len(x). Working mem: O(k).

    In practice, very fast for small k and randomly ordered
    or almost sorted (increasingly) data.

    Example timings:                 argkmin(x, k) np.argsort(x)[k]
    (ascending)  n= 100000000, k=   1:      0.060s       4.388s
    (descending)                            0.168s       7.329s
    (random)                                0.073s      26.673s
    (ascending)  n= 100000000, k=   5:      0.060s       4.403s
    (descending)                            0.505s       7.414s
    (random)                                0.072s      26.447s
    (ascending)  n= 100000000, k= 100:      0.061s       4.390s
    (descending)                            8.007s       7.639s
    (random)                                0.075s      27.299s



    Parameters:
    ----------

    x : c_contiguous 1d array
        an integer or float vector

    k : int
        an integer in {0,...,len(x)-1}, preferably small


    Returns:
    -------

    value:
        the (k-1)-th smallest value in x
    """
    cdef ssize_t n = x.shape[0]
    return c_argfuns.Cargkmin(&x[0], n, k, <ssize_t*>0)
