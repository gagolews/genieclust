# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Provides access to MST-related functions.

Copyright (C) 2018-2019 Marek.Gagolewski.com
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


cdef extern from "c_mst.h":

    cdef cppclass CDistance:
        pass

    cdef cppclass CDistanceSquaredEuclidean:
        CDistanceSquaredEuclidean()
        CDistanceSquaredEuclidean(double* X, ssize_t n, ssize_t d)

    cdef cppclass CDistanceManhattan:
        CDistanceManhattan()
        CDistanceManhattan(double* X, ssize_t n, ssize_t d)

    cdef cppclass CDistanceCosine:
        CDistanceCosine()
        CDistanceCosine(double* X, ssize_t n, ssize_t d)

    cdef cppclass CDistanceCompletePrecomputed:
        CDistanceCompletePrecomputed()
        CDistanceCompletePrecomputed(double* d, ssize_t n)

    ssize_t Cmst_nn(double* dist, ssize_t* ind, ssize_t n, ssize_t k,
             double* mst_d, ssize_t* mst_i)

    void Cmst_complete(CDistance* dist, ssize_t n,
             double* mst_d, ssize_t* mst_i)
