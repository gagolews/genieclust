# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Determine the index of the k-th smallest element in an array

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

cpdef np.int_t argkmin(arrayT x, np.int_t k):
    """
    Returns the index of the (k-1)-th smallest value in an array x,
    where argkmin(x, 0) == argmin(x), or, more generally,
    argkmin(x, k) == np.argsort(x)[k].

    Run time: O(nk), where n == len(x). Working mem: O(k).
    Does not modify x.

    In practice, very fast for small k and randomly ordered
    or almost sorted (increasingly) data.

    Example timings:                 argkmin(x, k) np.argsort(x)[k]
    (ascending)  n=100000000, k=  5:        0.058s           1.448s
    (descending)                            0.572s           2.651s
    (random)                                0.064s          20.049s
    (ascending)  n=100000000, k=100:        0.057s           1.472s
    (descending)                           18.051s           2.662s
    (random)                                0.064s          20.269s


    Parameters:
    ----------

    x : ndarray
        an integer or float vector

    k : int
        an integer in {0,...,len(x)-1}, preferably small


    Returns:
    -------

    val
        the (k-1)-th smallest value in x
    """
    cdef np.int_t n = len(x), i, j, ret
    cdef np.int_t* idx
    if k < 0:  raise Exception("k < 0")
    if k >= n: raise Exception("k >= n")

    k += 1
    idx = <np.int_t*>PyMem_Malloc(k*sizeof(np.int_t))
    for i in range(0, k):
        j = i
        idx[i] = i
        while j > 0 and x[idx[j]] < x[idx[j-1]]:
            idx[j], idx[j-1] = idx[j-1], idx[j] # KISS
            j -= 1

    for i in range(k, n):
        if x[idx[k-1]] <= x[i]:
            continue
        j = k-1
        idx[k-1] = i
        while j > 0 and x[idx[j]] < x[idx[j-1]]:
            idx[j], idx[j-1] = idx[j-1], idx[j] # KISS
            j -= 1

    ret = idx[k-1]
    PyMem_Free(idx)
    return ret
