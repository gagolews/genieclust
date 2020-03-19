# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""Noisy k-partition post-processing

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

cimport cython
cimport numpy as np
import numpy as np

from . cimport c_postprocess


cpdef np.ndarray[ssize_t] merge_boundary_points(
        ssize_t[:,::1] mst_i,
        ssize_t[::1] c,
        ssize_t[:,::1] nn_i,
        ssize_t M):
    """A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all "boundary" noise points with their nearest
    "core" points.


    Parameters
    ----------

    mst_i : c_contiguous array
        See genieclust.mst.mst_from_distance()
    c : c_contiguous array of shape (n_samples,)
        c[i] gives the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.
    nn_i : c_contiguous matrix of shape (n_samples,n_neighbors)
        nn_ind[i,:] gives the indexes of the i-th point's
        nearest neighbours.
    M : int
        smoothing factor, M>=2


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {-1, 0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[ssize_t] cl2 = np.array(c, dtype=np.intp)

    c_postprocess.Cmerge_boundary_points(
        &mst_i[0,0], mst_i.shape[0],
        &nn_i[0,0], nn_i.shape[1], M,
        &cl2[0], cl2.shape[0])

    return cl2


cpdef np.ndarray[ssize_t] merge_noise_points(
        ssize_t[:,::1] mst_i,
        ssize_t[::1] c):
    """A noisy k-partition post-processing:
    given a k-partition (with noise points included),
    merges all noise points with their nearest
    clusters.


    Parameters
    ----------

    mst_i : c_contiguous array
        See genieclust.mst.mst_from_distance()
    c : c_contiguous array of shape (n_samples,)
        c[i] gives the cluster id
        (in {-1, 0, 1, ..., k-1} for some k) of the i-th object.
        Class -1 denotes the `noise' cluster.


    Returns
    -------

    c : ndarray, shape (n_samples,)
        A new integer vector c with c[i] denoting the cluster
        id (in {0, ..., k-1}) of the i-th object.
    """
    cdef np.ndarray[ssize_t] cl2 = np.array(c, dtype=np.intp)

    c_postprocess.Cmerge_noise_points(
        &mst_i[0,0], mst_i.shape[0],
        &cl2[0], cl2.shape[0])

    return cl2

